from collections import namedtuple
from typing import Optional
import jax
import haiku as hk
import jax.numpy as jnp


class ConvBlock(hk.Module):
    def __init__(self, dilatation=2):
        super().__init__()
        self.dilatation = dilatation
    
    def __call__(self, x, is_training):
        _, _, _, c = x.shape
        hidden = self.dilatation * c
        init = hk.initializers.VarianceScaling(2.)

        x = jax.nn.relu(hk.Conv2D(hidden, 1, w_init=init)(x))
        x = hk.BatchNorm(True, True, 0.9)(x, is_training)
        x = jax.nn.relu(hk.Conv2D(
            hidden,
            3,
            padding='SAME',
            w_init=init,
            feature_group_count=hidden
            )(x))
        x = hk.BatchNorm(True, True, 0.9)(x, is_training)
        x = hk.Conv2D(c, 1, w_init=init)(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training)

        return x


def residual(f):
    def res(x, *args, **kwargs):
        return jax.nn.relu(f(x, *args, **kwargs) + x)
    return res


class DownSample(hk.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
    
    def __call__(self, x, is_training):
        x = hk.Conv2D(self.out_dim, 2, 2, w_init=hk.initializers.VarianceScaling(2))(x)
        x = jax.nn.relu(hk.BatchNorm(True, True, 0.9)(x, is_training))
        return x


class UpSample(hk.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
    
    def __call__(self, x, is_training):
        x = hk.Conv2DTranspose(self.out_dim, 2, 2, w_init=hk.initializers.VarianceScaling(2))(x)
        x = jax.nn.relu(hk.BatchNorm(True, True, 0.9)(x, is_training))
        return x


class Encoder(hk.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def __call__(self, x, is_training):
        x = hk.Conv2D(self.dim, 1, w_init=hk.initializers.VarianceScaling(2.))(x)
        # 64 64 dim
        x = residual(ConvBlock())(x, is_training)
        x = residual(ConvBlock())(x, is_training)
        x = DownSample(self.dim*2)(x, is_training)
        # 32 32 2*dim
        x = residual(ConvBlock())(x, is_training)
        x = residual(ConvBlock())(x, is_training)
        x = DownSample(self.dim*4)(x, is_training)
        # 16 16 4*dim
        x = residual(ConvBlock())(x, is_training)
        x = residual(ConvBlock())(x, is_training)
        x = DownSample(self.dim*8)(x, is_training)
        # 8 8 8*dim
        x = residual(ConvBlock())(x, is_training)
        x = residual(ConvBlock())(x, is_training)

        return x


class Decoder(hk.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def __call__(self, x, is_training):
        # 8 8 8*dim
        x = residual(ConvBlock())(x, is_training)
        x = residual(ConvBlock())(x, is_training)
        x = UpSample(self.dim*4)(x, is_training)
        # 16 16 4*dim
        x = residual(ConvBlock())(x, is_training)
        x = residual(ConvBlock())(x, is_training)
        x = UpSample(self.dim*4)(x, is_training)
        # 32 32 2*dim
        x = residual(ConvBlock())(x, is_training)
        x = residual(ConvBlock())(x, is_training)
        x = UpSample(self.dim*4)(x, is_training)
        # 64 64 dim
        x = residual(ConvBlock())(x, is_training)
        x = residual(ConvBlock())(x, is_training)

        x = hk.Conv2D(3, 1, w_init=hk.initializers.VarianceScaling(2.))(x)
        
        return x


class Codebook(hk.Module):
    def __init__(self,
                 latent_dim: int,
                 code_size: int,
                 beta: float,
                 name: Optional[str] = None,
                 ):
        super().__init__(name)
        self.latent_dim = latent_dim
        self.code_size = code_size
        self.beta = beta

        initializer = hk.initializers.VarianceScaling(distribution='uniform')
        self.codebook = hk.get_parameter('codebook', (code_size, latent_dim), init=initializer)

    def __call__(self, x):

        Output = namedtuple('Output', ['quantize', 'loss', 'indices'])

        jax.tree_util.register_pytree_node(
            Output,
            lambda xs: (tuple(xs), None),
            lambda _, xs:  Output(*xs)
        )

        # input: bs h w c, codebook: N c

        # using negative scalar product as "distance"
        # bs h w N
        dists = jnp.matmul(x, self.codebook.T)

        # bs h w
        idx = jnp.argmin(dists, axis=-1)

        # bs h w c
        quantize = self.codebook[idx]

        quantize = x + jax.lax.stop_gradient(quantize - x)

        enc_loss = jnp.mean((jax.lax.stop_gradient(quantize) - x)**2)
        com_loss = jnp.mean((quantize - jax.lax.stop_gradient(x))**2)

        loss = enc_loss + self.beta * com_loss
        return Output(quantize, loss, idx)