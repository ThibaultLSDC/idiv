from collections import namedtuple
from dataclasses import dataclass
from functools import partial
import optax
import haiku as hk
import jax.numpy as jnp
import jax.random as rd
from jax import jit, value_and_grad as vgrad
from tqdm import tqdm
import jax
import numpy as np

from idiv.models.unet import UNet


# DDPMStates = namedtuple('DDPMStates', ['params', 'state', 'opt_state'])

@dataclass
class DDPMStates:
    params: hk.Params
    state: hk.State
    opt_state: optax.OptState

jax.tree_util.register_pytree_node(
        DDPMStates,
        lambda xs: ((xs.params, xs.state, xs.opt_state), None),
        lambda _, xs:  DDPMStates(*xs)
    )


class DDPM:
    def __init__(self, config, optim: optax.GradientTransformation) -> None:

        self.config = config
        self.optim = optim

        transformed = self.build(config)

        self.init = transformed.init
        self.apply = transformed.apply

        self.betas = jnp.linspace(config.betas_low, config.betas_high, config.T, dtype=jnp.float32)
        self.alphas = 1 - self.betas.copy()
        self.alpha_cumprod = jnp.cumprod(self.alphas)

        self.sigmas = jnp.sqrt((1-self.alpha_cumprod[:-1]) / (1 - self.alpha_cumprod[1:])) \
            * jnp.sqrt(1 - self.alpha_cumprod[1:] / self.alpha_cumprod[:-1])
        self.sigmas = jnp.concatenate((jnp.zeros((1,)), self.sigmas))

        self.T = config.T
        self.ema_weight = config.ema_weight
        self.ema_warmup = config.ema_warmup

    @staticmethod
    def build(config):
        @hk.transform_with_state
        def f(x, t, is_training):
            net = UNet(config.unet_config)
            return net(x, t, is_training)
        return f
 
    def init_model(self, key: rd.KeyArray, batch, t):
        # weights
        params, state = self.init(key, batch, t, is_training=True)
        opt_state = self.optim.init(params)

        # ema
        ema_fn = hk.transform_with_state(lambda x: hk.EMAParamsTree(self.ema_weight, zero_debias=False, warmup_length=self.ema_warmup)(x))
        _, ema_state = ema_fn.init(None, params)

        return params, state, opt_state, ema_fn, ema_state

    @partial(jit, static_argnums=[0, 5])
    def forward(self, params: hk.Params, state: hk.State, x, t, is_training: bool):
        eps, state = self.apply(params, state, None, x, t, is_training)
        return eps, state

    @partial(jit, static_argnums=[0, 5])
    def loss(self, params, state, key: rd.KeyArray, x, is_training: bool):
        bs, h, w, c = x.shape

        key, subkey = rd.split(key)
        t = rd.randint(subkey, (bs,), 0, self.T)
        alphas_cp = jnp.expand_dims(self.alpha_cumprod[t], axis=[1, 2, 3])

        key, subkey = rd.split(key)
        eps = rd.normal(subkey, x.shape)

        x_t = jnp.sqrt(alphas_cp) * x + jnp.sqrt(1 - alphas_cp) * eps

        pred, state = self.forward(params, state, x_t, t, is_training)

        return jnp.mean((pred - eps)**2), state

    @partial(jit, static_argnums=[0])
    def update(self, params, state, opt_state, key, x):

        loss_and_grad = vgrad(self.loss, has_aux=True, allow_int=True)

        (loss, state), grads = loss_and_grad(params, state, key, x, True)

        updates, opt_state = self.optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, state, opt_state, loss

    def fit(
        self,
        key: rd.KeyArray,
        datasets: tuple,
        model_state: DDPMStates=None):

        x_train, x_val = datasets

        if model_state is None:
            t = jnp.ones((x_train.shape[1],), dtype=jnp.float32)
            key, subkey = rd.split(key)
            params, state, opt_state, ema_fn, ema_state = self.init_model(subkey, x_train[0], t)

        for epoch in range(self.config.epochs):
            running_loss = 0.
            for x in tqdm(x_train, desc=f"Epoch {epoch+1}"):
                key, subkey = rd.split(key)
                params, state, opt_state, loss = self.update(params, state, opt_state, subkey, x)
                ema_params, ema_state = jit(ema_fn.apply)(None, ema_state, None, params)
                running_loss += loss / x_train.shape[0]

            print(f"Epoch {epoch+1}/{self.config.epochs} | loss {running_loss:.5f}")

            if epoch % 10 == 0:
                key, subkey = rd.split(key)
                path = f"data/sampling_epoch{epoch+1}.jpg"
                self.log_img(ema_params, state, subkey, path)

        key, subkey = x(params, state, subkey, path)

        return params, state, opt_state
    
    # @partial(jit, static_argnums=0)
    def img_estimate(self, params: hk.Params, state: hk.State, x_t, t):
        eps, state = self.forward(params, state, x_t, t, is_training=False)
        x_0_estimate = (x_t - jnp.sqrt(1 - self.alpha_cumprod[t]) * eps) / jnp.sqrt(self.alpha_cumprod[t])
        return x_0_estimate, eps, state
    
    @partial(jit, static_argnums=0)
    def sample_one(self, params: hk.Params, state: hk.State, x_t, t, key, eta=1.):
        x_0, eps, state = self.img_estimate(params, state, x_t, t)
        noise = rd.normal(key, x_0.shape)
        x_prev = jnp.sqrt(self.alpha_cumprod[t-1]) * x_0 \
            + jnp.sqrt(1 - self.alpha_cumprod[t-1] - eta**2 * self.sigmas[t-1]**2) * eps \
            + eta * self.sigmas[t-1] * noise
        return x_prev, state
    
    # @partial(jit, static_argnums=0)
    def sample(self, params: hk.Params, state: hk.State, key: rd.KeyArray, eta=1., x_T_shape=(1, 64, 64, 3)):
        key, subkey = rd.split(key)
        x_prev = rd.normal(key, x_T_shape)
        self.alpha_cumprod = jnp.concatenate((jnp.ones((1,)), self.alpha_cumprod))
        saved = []
        for t in tqdm(range(self.T, 0, -1), desc='sampling...'):
            key, subkey = rd.split(key)
            x_prev, state = self.sample_one(params, state, x_prev, jnp.array([t]), subkey, eta=eta)
        self.alpha_cumprod = self.alpha_cumprod[1:]
        print('done inside')
        return x_prev
    
    def log_img(self, params, state, key, path):

        x = self.sample(params, state, key)
        print('done sampling')
        y = np.array(x)
        print('done array')
        x_0 = (y.clip(-1, 1) + 1) * 127.5

        print('done with sampling, saving image')

        aaa = x_0[0].astype(np.uint8)
        Image.fromarray(aaa).save(path)


if __name__=='__main__':

    from configs.config import DefaultConfig

    from PIL import Image

    import jax

    img = jnp.array(Image.open('/home/ty/Documents/code/idiv/data/risitas.jpg'), dtype=jnp.float32)

    img = jax.image.resize(img / 127.5 - 1, (64, 64, 3), method='lanczos3')

    batch = jnp.stack([jnp.stack([img for _ in range(64)], axis=0) for _ in range(100)], axis=0)

    datasets = (batch, None)

    seed = 42

    key = rd.PRNGKey(seed)

    config = DefaultConfig.build_from_argv(fallback='configs/default_config/main_config.yaml')

    optim = optax.adam(**config.optim)

    trainer = DDPM(config.test_ddpm, optim)

    trainer.fit(key, datasets)