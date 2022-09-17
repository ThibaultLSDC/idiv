import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad as vgrad, jit, vmap, grad
import jax.random as rd

import haiku as hk
import optax

from functools import partial

from tqdm import tqdm

from idiv.models import Encoder, Decoder, Codebook


class VQVAE:
    def __init__(self, config, optim: optax.GradientTransformation) -> None:
        
        self.config = config
        self.optim = optim

        transformed = self.build(config)

        self.init = transformed.init

        self.encode, self.quantize, self.decode = transformed.apply

    @staticmethod
    def build(config):
        @hk.multi_transform_with_state
        def f():
            encoder = Encoder(**config.encoder)
            codebook = Codebook(**config.codebook)
            decoder = Decoder(**config.decoder)

            def encode(x, is_training):
                return encoder(x, is_training)

            def code(x):
                return codebook(x)

            def decode(x, is_training):
                return decoder(x, is_training)
            
            def init(x, is_training):
                latent = encode(x, is_training)
                quantize = code(latent)
                return decode(quantize.quantize, is_training)
            
            return init, (encode, code, decode)

        return f
    
    def init_model(self, key: rd.KeyArray, batch):
        params, state = self.init(key, batch, is_training=True)
        opt_state = self.optim.init(params)
        return params, state, opt_state

    def forward(self, params: hk.Params, state: hk.State, x, is_training: bool):
        latent, state = self.encode(params, state, None, x, is_training)
        print(latent.shape)
        quantize, state = self.quantize(params, state, None, latent)
        print(quantize.quantize.shape)
        res, state = self.decode(params, state, None, quantize.quantize, is_training)
        print(res.shape)
        return {
            'pred': res,
            'state': state,
            'loss_quantization': quantize.loss
        }

    @partial(jit, static_argnums=[0, 4])
    def loss_and_pred(self,
                      params: hk.Params,
                      state: hk.State,
                      x,
                      is_training: bool
                      ):
        output = self.forward(params, state, x, is_training)
        pred = output['pred']
        state = output['state']
        loss_quantization = output['loss_quantization']

        loss_rec = jnp.mean(jnp.sum((x - pred)**2, axis=[1, 2, 3]))

        loss = loss_rec + loss_quantization
    
        return loss, (pred, state)
    
    @partial(jit, static_argnums=[0])
    def update(self,
               params: hk.Params,
               state: hk.State,
               opt_state: optax.OptState,
               x
               ):
        (loss, (pred, state)), grads = vgrad(self.loss_and_pred, has_aux=True)(
        params, state, x, True
        )
        print(loss, pred.shape)
        updates, opt_state = self.optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, state, loss, pred

    def fit(self,
            key: rd.KeyArray,
            datasets: tuple,
            params: hk.Params=None,
            state: hk.State=None,
            opt_state: optax.OptState=None):

        x_train, x_val = datasets

        if params is None or state is None or opt_state is None:
            key, subkey = rd.split(key)
            params, state, opt_state = self.init_model(subkey, x_train[0])
        
        for epoch in range(self.config.epochs):
            running_loss = 0.
            for x in tqdm(x_train, desc=f"Epoch {epoch}"):
                params, state, opt_state, loss, pred = self.update(params, state, opt_state, x)
                running_loss += loss / x_train[0]
        
        print(f"epoch {epoch}/{self.config.epochs} | loss {running_loss: .3f}")

        return params, state, opt_state


if __name__=='__main__':

    from configs.config import DefaultConfig

    from PIL import Image
    
    img = jnp.array(Image.open('/home/ty/Documents/code/jax-rl/data/risitas.jpg'), dtype=jnp.float32)

    img = jax.image.resize(img / 127.5 - 1, (64, 64, 3), method='lanczos3')

    datasets = (jnp.expand_dims(img, axis=[0, 1]), None)

    seed = 42

    key = rd.PRNGKey(seed)

    config = DefaultConfig()

    optim = optax.adam(**config.optim)

    trainer = VQVAE(config.vqvae, optim)

    trainer.fit(key, datasets)