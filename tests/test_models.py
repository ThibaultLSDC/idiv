from collections import namedtuple
from multiprocessing import dummy
from idiv.models import *
import haiku as hk
import jax.numpy as jnp
import jax.random as rd

import pytest


Output = namedtuple('Output', ['forward', 'params', 'state', 'output'])


@pytest.fixture(scope='class')
def encoder():
    @hk.transform_with_state
    def forward(x, is_training):
        net = Encoder(64)
        return net(x, is_training)
    
    key = rd.PRNGKey(0)
    key1, key2 = rd.split(key)

    dummy_batch = jnp.ones((2, 64, 64, 3))

    params, state = forward.init(key1, dummy_batch, is_training=True)

    output, state = forward.apply(params, state, key2, dummy_batch, is_training=True)

    return Output(forward, params, state, output)


@pytest.fixture(scope='class')
def decoder():
    @hk.transform_with_state
    def forward(x, is_training):
        net = Decoder(64)
        return net(x, is_training)
    
    key = rd.PRNGKey(0)
    key1, key2 = rd.split(key)

    dummy_batch = jnp.ones((2, 8, 8, 512))

    params, state = forward.init(key1, dummy_batch, is_training=True)

    output, state = forward.apply(params, state, key2, dummy_batch, is_training=True)

    return Output(forward, params, state, output)


@pytest.fixture(scope='class')
def codebook():
    @hk.transform_with_state
    def forward(x):
        return Codebook(512, 10, 1.)(x)
    
    key = rd.PRNGKey(666)
    dummy = jnp.ones((2, 8, 8, 512))

    params, state = forward.init(key, dummy)
    output, state = forward.apply(params, state, None, dummy)
    
    return Output(forward, params, state, output)


class TestVAE:
    def test_encoder(self, encoder):
        assert encoder.output.shape == (2, 8, 8, 512)

    def test_decoder(self, decoder):
        assert decoder.output.shape == (2, 64, 64, 3)
    
    def test_in_out(self, encoder, decoder):
        batch = jnp.zeros((2, 64, 64, 3))
        key1, key2 = rd.split(rd.PRNGKey(0))

        latent, _ = encoder.forward.apply(
            encoder.params,
            encoder.state,
            key1,
            batch,
            is_training=True)

        output, _ = decoder.forward.apply(
            decoder.params,
            decoder.state,
            key2,
            latent,
            is_training=True)
        
        assert batch.shape == output.shape

    def test_codebook(self, codebook):
        assert codebook.output.quantize.shape == (2, 8, 8, 512)

    def test_full_house(self, encoder, codebook, decoder):
        batch = jnp.zeros((2, 64, 64, 3))
        key1, key2 = rd.split(rd.PRNGKey(0), )

        latent, _ = encoder.forward.apply(
            encoder.params,
            encoder.state,
            key1,
            batch,
            is_training=True)

        quantize, _ = codebook.forward.apply(
            codebook.params,
            codebook.state,
            None,
            latent
        )

        output, _ = decoder.forward.apply(
            decoder.params,
            decoder.state,
            key2,
            quantize.quantize,
            is_training=True)
        
        assert batch.shape == output.shape


@pytest.fixture(scope='class')
def attention() -> Output:
    @hk.transform
    def forward(x, return_attn):
        net = Attention()
        return net(x, return_attn)

    key1, key2 = rd.split(rd.PRNGKey(0))

    dummy = jnp.ones((4, 8, 8, 32))

    params = forward.init(key1, dummy, return_attn=True)
    output = forward.apply(params, key2, dummy, return_attn=True)

    return Output(forward, params, None, output)


@pytest.fixture(scope='class')
def norm() -> Output:
    @hk.transform_with_state
    def f(x, is_training):
        net = ConvBlock()
        return prenorm(x, net, is_training)
    
    key = rd.PRNGKey(0)
    dummy = jnp.ones((4, 32, 32, 16))

    params, state = f.init(key, dummy, True)
    output, state = f.apply(params, state, None, dummy, True)

    return Output(f, params, None, output)


@pytest.fixture(scope='class')
def sinembedding() -> Output:
    @hk.transform
    def forward(x):
        net = SinEmbedding(16)
        return net(x)
    
    key1, key2 = rd.split(rd.PRNGKey(0))

    dummy= jnp.arange(0., 8., 1.)

    params = forward.init(key1, dummy)
    output = forward.apply(params, key2, dummy)

    return Output(forward, params, None, output)


class TestAttention:
    def test_attn_shape(self, attention):
        assert attention.output[0].shape == (4, 8, 8, 32)
        assert attention.output[1].shape == (4, 4, 8**2, 8**2)
    
    def test_prenorm(self, norm):
        param = norm.params['layer_norm']['layer_norm']
        assert param.shape == (1, 1, 1, 16)
        assert norm.output.shape == (4, 32, 32, 16)
        assert jnp.mean(norm.output) <= 1e-5

    def test_sin_embedding(self, sinembedding):
        assert sinembedding.output.shape == (8, 16)