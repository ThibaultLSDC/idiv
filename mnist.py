from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad as vgrad, jit, vmap, grad
from jax import random

import haiku as hk
import optax

from functools import partial

from tqdm import tqdm

@jit
def normalize(x):
    return 2 * x / 255 - 1


def mk_dataset(key, batch_size):

    train_dataset = MNIST('./data', train=True, transform=ToTensor, download=True)
    test_dataset = MNIST('./data', train=False, transform=ToTensor, download=True)

    x_train = jnp.array(train_dataset.data.numpy(), dtype=jnp.float32)
    y_train = jax.nn.one_hot(jnp.array(train_dataset.targets.numpy()), 10)

    x_test = jnp.array(test_dataset.data.numpy(), dtype=jnp.float32)
    y_test = jax.nn.one_hot(jnp.array(test_dataset.targets.numpy()), 10)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    
    key1, key2 = random.split(key)

    perm_train = random.permutation(key1, (x_train.shape[0] // batch_size) * batch_size)
    perm_test = random.permutation(key1, (x_test.shape[0] // batch_size) * batch_size)

    x_train = x_train[perm_train].reshape((-1, batch_size, 28, 28, 1))
    y_train = y_train[perm_train].reshape((-1, batch_size, 10))
    x_test = x_test[perm_test].reshape((-1, batch_size, 28, 28, 1))
    y_test = y_test[perm_test].reshape((-1, batch_size, 10))

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    return ((normalize(x_train), y_train), (normalize(x_test), y_test))


class Net(hk.Module):
    def __init__(self, name: str = None):
        super().__init__(name)
    
    def __call__(self, x: jnp.ndarray, is_training: bool):
        x = hk.Conv2D(output_channels=64, kernel_shape=3, stride=1, padding='VALID', w_init=hk.initializers.VarianceScaling(2.0))(x)
        x = jax.nn.relu(x)
        x = hk.Conv2D(output_channels=64, kernel_shape=3, stride=1, padding='VALID', w_init=hk.initializers.VarianceScaling(2.0))(x)
        x = jax.nn.relu(x) ## 24*24*64
        if is_training:
            x = hk.dropout(hk.next_rng_key(), 0.5, x)
        x = hk.MaxPool(2, 2, padding='SAME')(x) ## 12*12*64
        x = hk.Conv2D(output_channels=128, kernel_shape=3, stride=1, padding='VALID', w_init=hk.initializers.VarianceScaling(2.0))(x)
        x = jax.nn.relu(x)
        x = hk.Conv2D(output_channels=128, kernel_shape=3, stride=1, padding='VALID', w_init=hk.initializers.VarianceScaling(2.0))(x)
        x = jax.nn.relu(x) ## 8*8*128
        if is_training:
            x = hk.dropout(hk.next_rng_key(), 0.5, x)
        x = hk.MaxPool(2, 2, padding='SAME')(x) ## 4*4*128
        x = hk.Flatten()(x)
        x = hk.Linear(10, w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg'))(x)
        return x


@hk.transform
def forward(x: jnp.ndarray, is_training: bool):
    module = Net()
    return module(x, is_training)


@partial(jit, static_argnums=[4])
def forward_and_loss(params: dict, key: jnp.array, x: jnp.array, y: jnp.array, is_training: bool):
    pred = forward.apply(params, key, x, is_training) # ???????
    loss = jnp.mean(optax.softmax_cross_entropy(pred, y))
    return loss, pred


@partial(jit, static_argnums=[1])
def apply_grads(params: dict, optimizer: optax.GradientTransformation, opt_state: dict, grads: dict):
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, new_opt_state


@jit
def acc_function(pred, y):
    return jnp.mean(jnp.where(jnp.argmax(pred, axis=-1) == jnp.argmax(y, axis=-1), 1, 0))


def fit_model(key, datasets, optimizer: optax.GradientTransformation, epochs):
    key, subkey = random.split(key)
    (x_train, y_train), (x_val, y_val) = datasets

    params = forward.init(subkey, x_train[0], is_training=True)
    opt_state = optimizer.init(params)

    for epoch in range(epochs):
        acc = 0.0
        for x, y in tqdm(zip(x_train, y_train), desc=f"epoch {epoch+1}", total=x_train.shape[0]):
            key, subkey = random.split(key)
            (loss, pred), grads = vgrad(forward_and_loss, has_aux=True)(
                params, subkey, x, y, is_training=True
            )
            params, opt_state = apply_grads(params, optimizer, opt_state, grads)

            acc += acc_function(pred, y) / len(x_train)
        print(f"epoch {epoch+1}/{epochs} | loss {loss: .3f} | acc {acc: .3f}", end='\n')

        acc = 0.0
        for x, y in tqdm(zip(x_val, y_val), desc=f"epoch {epoch+1}", total=x_val.shape[0]):
            key, subkey = random.split(key)
            loss, pred = forward_and_loss(params, subkey, x, y, is_training=False)
            acc += acc_function(pred, y) / x_val.shape[0]
        print(f"epoch {epoch+1}/{epochs} | loss_val {loss: .3f} | acc_val {acc: .3f}", end='\n')



if __name__=='__main__':
    seed = 0
    epochs = 10
    batch_size = 512
    optimizer = optax.adam(learning_rate=1e-3)

    key1, key2 = random.split(random.PRNGKey(seed))
    datasets = mk_dataset(key1, batch_size=batch_size)
    fit_model(key2, datasets=datasets, optimizer=optimizer, epochs=epochs)
    