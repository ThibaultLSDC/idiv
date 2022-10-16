from typing import Optional, Tuple
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from einops import rearrange
from jax.numpy import einsum


def exists(thing):
    return not thing is None


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


class Attention(hk.Module):
    def __init__(self, heads: int=4, head_dim: int=32, scale: float=1.):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.total_dim = heads * head_dim
        self.scale = scale

    def __call__(self, x, return_attn=False):
        bs, height, width, c = x.shape

        initializer = hk.initializers.VarianceScaling(2.)

        # bs, l, k, 3*h*d
        qkv = hk.Conv2D(self.total_dim * 3, 1, w_init=initializer)(x)

        qkv = jnp.split(qkv, 3, axis=-1)

        q, k, v = jax.tree_util.tree_map(lambda t: rearrange(t, 'b x y (h c) -> b h c (x y)', h=self.heads), qkv)

        sim = einsum('b h c i, b h c j -> b h i j', q, k) * self.scale
        attn = jax.nn.softmax(sim, axis=-1)
        output = einsum('b h i j, b h c j -> b h i c', attn, v)
        output = rearrange(output, 'b h (x y) c -> b x y (h c)', x=height, y=width)

        output = hk.Conv2D(c, 1, w_init=initializer)(output)
        if return_attn:
            return output, attn
        else:
            return output


class LayerNorm(hk.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        dim = x.shape[-1]
        eps = 1e-5
        var = jnp.var(x, axis=-1, keepdims=True)
        mean = jnp.mean(x, axis=-1, keepdims=True)

        g = hk.get_parameter('layer_norm', (1, 1, 1, dim), init=jnp.ones)

        return (x - mean) / jnp.sqrt(var + eps) * g


def prenorm(x, f, *args, norm=LayerNorm):
    x = norm()(x)
    return f(x, *args)


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


class SinEmbedding(hk.Module):
    def __init__(self, time_dim):
        super().__init__()
        self.time_dim = time_dim
    
    def __call__(self, t: jnp.DeviceArray):
        assert t.ndim == 1
        half_dim = self.time_dim//2
        emb = np.log(10000) / (half_dim-1)
        emb = jnp.exp(jnp.arange(0, half_dim, step=1) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb


class Block(hk.Module):
    def __init__(self, out_dim: int, groups: int=8, name: Optional[str] = None):
        super().__init__(name)

        self.out_dim = out_dim
        self.groups = groups
    
    def __call__(self, x: jnp.DeviceArray, scale_shift: Optional[Tuple]=None):
        x = hk.Conv2D(self.out_dim, 3, padding=(1, 1))(x)
        x = hk.GroupNorm(self.groups)(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = (1 + scale) * x + shift
        x = jax.nn.silu(x)
        return x


class ResBlock(hk.Module):
    def __init__(self, out_dim: int, groups: int, time_dim: Optional[int]=None, name: Optional[str] = None):
        super().__init__(name)
        self.out_dim = out_dim
        self.groups = groups
        self.time_dim = time_dim
    
    def __call__(self, x, time_emb=None):
        
        scale_shift = None
        if exists(self.time_dim) and exists(time_emb):
            time_emb = hk.Linear(self.out_dim*2)(jax.nn.silu(time_emb))
            time_emb = time_emb[:, None, None, :]
            scale_shift = jnp.split(time_emb, 2, axis=-1)
        
        h = Block(self.out_dim, self.groups)(x, scale_shift=scale_shift)
        h = Block(self.out_dim, self.groups)(h)
        res = hk.Conv2D(self.out_dim, 1)(x)
        return res + h

