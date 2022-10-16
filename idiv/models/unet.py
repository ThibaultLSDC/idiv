from typing import Optional
import haiku as hk
from yaecs import Configuration

import jax.nn as nn
import jax.numpy as jnp

from idiv.models.utils import (
    ConvBlock,
    SinEmbedding,
    residual,
    Attention,
    prenorm,
    UpSample,
    DownSample,
    ResBlock,
)


class UNet64(hk.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, x, is_training):

        inititializer = hk.initializers.VarianceScaling(2.)

        x = hk.Conv2D(self.dim, 3, w_init=inititializer)(x) # 64, 64, 64
        x1 = residual(ConvBlock())(x, is_training)
        x = residual(ConvBlock())(x1, is_training)
        x = DownSample(self.dim)(x, is_training) # 32, 32, 64
        x2 = residual(ConvBlock())(x, is_training)
        x = residual(ConvBlock())(x2, is_training)
        x = DownSample(self.dim*2)(x, is_training) # 16, 16, 128
        x = residual(ConvBlock())(x, is_training)
        x3 = prenorm(x, residual(Attention()))
        x = residual(ConvBlock())(x3, is_training)
        x = DownSample(self.dim*4)(x, is_training) # 8, 8, 256
        x = residual(ConvBlock())(x, is_training)
        x4 = prenorm(x, residual(Attention(heads=8)))
        x = residual(ConvBlock())(x4, is_training)
        x = DownSample(self.dim*8)(x, is_training) # 4, 4, 512
        x = residual(ConvBlock())(x, is_training)
        x = residual(ConvBlock())(x, is_training)
        x = UpSample(self.dim*4)(x, is_training) # 8, 8, 256
        x = residual(ConvBlock())(x, is_training)
        x = prenorm(x + x4, residual(Attention(heads=8)))
        x = residual(ConvBlock())(x, is_training)
        x = UpSample(self.dim*2)(x, is_training) # 16, 16, 128
        x = residual(ConvBlock())(x, is_training)
        x = prenorm(x + x3, residual(Attention()))
        x = residual(ConvBlock())(x, is_training)
        x = UpSample(self.dim)(x, is_training) # 32, 32, 64
        x = residual(ConvBlock())(x, is_training)
        x = prenorm(x + x2, residual(ConvBlock()), is_training)
        x = UpSample(self.dim)(x, is_training) # 64, 64, 64
        x = residual(ConvBlock())(x, is_training)
        x = prenorm(x + x1, residual(ConvBlock()), is_training)
        
        x = hk.Conv2D(3, 3)(x)

        return x


class UNet(hk.Module):
    def __init__(self, config: Configuration, name: Optional[str] = None):
        super().__init__(name)
        self.config = config
    
    def __call__(self, x, t, is_training=True):
        dim = self.config.dim
        out_dim = self.config.out_dim
        mults = self.config.mults
        attention_layers = self.config.attention_layers
        resblock_groups = self.config.resblock_groups

        dims = [dim * m for m in mults]
        attn = [m * dim for m in attention_layers]

        time_dim = dim*4

        time_emb = SinEmbedding(dim)(t)
        time_emb = hk.Linear(time_dim)(time_emb)
        time_emb = nn.gelu(time_emb)
        time_emb = hk.Linear(time_dim)(time_emb)

        x = hk.Conv2D(dim, 7, padding=(3, 3))(x)
        r = x.copy()

        res = []
        for d in dims:
            x = ResBlock(d, resblock_groups, time_dim=time_dim)(x, time_emb)
            res.append(x)

            x = ResBlock(d, resblock_groups, time_dim=time_dim)(x, time_emb)
            x = prenorm(x, residual(Attention())) if d in attn else x
            res.append(x)

            if not d==dims[-1]:
                x = DownSample(d)(x, is_training)

        x = ResBlock(dims[-1], resblock_groups, time_dim=time_dim)(x, time_emb)
        x = prenorm(x, residual(Attention()))
        x = ResBlock(dims[-1], resblock_groups, time_dim=time_dim)(x, time_emb)

        for d in reversed(dims):
            x = jnp.concatenate((x, res.pop()), axis=-1)
            x = ResBlock(d, resblock_groups, time_dim=time_dim)(x, time_emb)

            x = jnp.concatenate((x, res.pop()), axis=-1)
            x = ResBlock(d, resblock_groups, time_dim=time_dim)(x, time_emb)
            x = prenorm(x, residual(Attention())) if d in attn else x

            if not d==dims[0]:
                x = UpSample(d)(x, is_training)
        
        x = jnp.concatenate((x, r), axis=-1)
        x = ResBlock(dim, resblock_groups, time_dim)(x, time_emb)
        x = hk.Conv2D(out_dim, 1)(x)
        return x