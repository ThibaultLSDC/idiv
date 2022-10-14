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

from idiv.models.unet import UNet64


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

        self.T = config.T

    @staticmethod
    def build(config):
        @hk.transform_with_state
        def f(x, is_training):
            net = UNet64(config.dim)
            return net(x, is_training)
        return f

    def init_model(self, key: rd.KeyArray, batch):
        params, state = self.init(key, batch, is_training=True)
        opt_state = self.optim.init(params)
        return params, state, opt_state

    def forward(self, params: hk.Params, state: hk.State, x, is_training: bool):
        eps, state = self.apply(params, state, None, x, is_training)
        return eps, state

    @partial(jit, static_argnums=[0, 5])
    def loss(self, params, state, key: rd.KeyArray, x, is_training: bool):
        bs, h, w, c = x.shape

        key, subkey = rd.split(key)
        t = rd.randint(subkey, (bs,), 0, self.T)
        alphas_cp = jnp.expand_dims(self.alpha_cumprod[t], axis=[1, 2, 3])

        key, subkey = rd.split(key)
        eps = rd.normal(subkey, x.shape)

        x_t = jnp.sqrt(alphas_cp) * x + jnp.sqrt(1 + alphas_cp) * eps

        pred, state = self.forward(params, state, x_t, is_training)

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
            key, subkey = rd.split(key)
            params, state, opt_state = self.init_model(subkey, x_train[0])

        for epoch in range(self.config.epochs):
            running_loss = 0.
            for x in tqdm(x_train, desc=f"Epoch {epoch+1}"):
                key, subkey = rd.split(key)
                params, state, opt_state, loss = self.update(params, state, opt_state, subkey, x)
                running_loss += loss / x_train.shape[0]

            print(f"Epoch {epoch+1}/{self.config.epochs} | loss {running_loss:.5f}")

        x = self.sample(params, state, key)
        print('done sampling')
        y = np.array(x)
        print('done array')
        x_0 = (y.clip(-1, 1) + 1) * 127.5

        print('done with sampling, saving image')

        Image.fromarray(x_0[0].astype(np.uint8)).save('data/sampling.jpg')

        return params, state, opt_state
    
    # @partial(jit, static_argnums=0)
    def img_estimate(self, params: hk.Params, state: hk.State, x_t, t):
        eps, state = self.forward(params, state, x_t, is_training=False)
        x_0_estimate = (x_t - jnp.sqrt(1 - self.alpha_cumprod[t]) * eps) / jnp.sqrt(self.alpha_cumprod[t])
        return x_0_estimate, eps, state
    
    # @partial(jit, static_argnums=0)
    def sample_one(self, params: hk.Params, state: hk.State, x_t, t):
        x_0, eps, state = self.img_estimate(params, state, x_t, t)
        x_prev = jnp.sqrt(self.alpha_cumprod[t-1]) * x_0 + jnp.sqrt(1 - self.alpha_cumprod[t-1]) * eps
        return x_prev, state
    
    # @partial(jit, static_argnums=0)
    def sample(self, params: hk.Params, state: hk.State, key: rd.KeyArray, x_T_shape=(1, 64, 64, 3)):
        x_prev = rd.normal(key, x_T_shape)
        self.alpha_cumprod = jnp.concatenate((jnp.ones((1,)), self.alpha_cumprod))
        for t in tqdm(range(self.T, 0, -1), desc='sampling...'):
            x_prev, state = self.sample_one(params, state, x_prev, t)
        self.alpha_cumprod = self.alpha_cumprod[1:]
        print('done inside')
        return x_prev


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