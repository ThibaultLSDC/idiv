from functools import partial
import jax.numpy as jnp
import haiku as hk
import jax.random as rd
from tqdm import tqdm
from jax import jit


class DDIMSampler:
    def __init__(self,
                 alpha_cumprod: jnp.DeviceArray,
                 apply_func,
                 T: int,
                 ) -> None:
        self.alphas = alpha_cumprod

        self.sigmas = jnp.sqrt((1-self.alphas[:-1]) / (1 - self.alphas[1:])) \
            * jnp.sqrt(1 - self.alphas[1:] / self.alphas[:-1])

        self.apply = apply_func

        self.T = T

    def forward(self, params, state, x_t, t):
        return self.apply(params, state, None, x_t, t, False)
    
    def x_0_estimate(self, params, state, x_t, t):
        eps, state = self.forward(params, state, x_t, t)
        x_0 = (x_t - jnp.sqrt(1 - self.alphas[t]) * eps) / jnp.sqrt(self.alphas[t])
        return x_0, eps, state

    @partial(jit, static_argnums=0)
    def p_sample(self, key, params, state, x_t, t, t_prev, eta):
        x_0, eps, state = self.x_0_estimate(params, state, x_t, t)
        x_prev = jnp.sqrt(self.alphas[t_prev]) * x_0 \
            + jnp.sqrt(1 - self.alphas[t_prev] - eta**2 * self.sigmas[t]**2) * eps \
            + eta * self.sigmas[t] * rd.normal(key, x_0.shape)
        return x_prev

    def sample(self,
               key,
               params,
               state,
               shape,
               n_steps,
               eta=0.,
               ):

        timesteps = jnp.linspace(0, self.T, int(n_steps), dtype=jnp.int32) 

        timesteps = reversed(list(zip(timesteps[1:], timesteps[:-1])))

        key, subkey = rd.split(key)
        x_t = rd.normal(subkey, shape)

        for t, t_prev in tqdm(timesteps, desc=f"sampling {n_steps} ddim steps, eta={eta}"):
            key, subkey = rd.split(key)
            x_t = self.p_sample(subkey, params, state, x_t, t, t_prev, eta)
        
        return x_t