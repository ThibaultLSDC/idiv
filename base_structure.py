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
    return x / 127.5 - 1


def mk_dataset(key, batch_size):
    # gestion dataset
    train_dataset = MNIST('./data', train=True, transform=ToTensor, download=True)
    test_dataset = MNIST('./data', train=False, transform=ToTensor, download=True)

    x_train = jnp.array(train_dataset.data.numpy(), dtype=jnp.float32)
    y_train = jax.nn.one_hot(jnp.array(train_dataset.targets.numpy()), 10)

    x_test = jnp.array(test_dataset.data.numpy(), dtype=jnp.float32)
    y_test = jax.nn.one_hot(jnp.array(test_dataset.targets.numpy()), 10)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    
    key1, key2 = random.split(key)

    perm_train = random.permutation(key1, (x_train.shape[0] // batch_size) * batch_size)
    perm_test = random.permutation(key2, (x_test.shape[0] // batch_size) * batch_size)

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
            x = hk.dropout(hk.next_rng_key(), 0.5, x) # rng ?
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


class Classifier:
    def __init__(self, optim: optax.GradientTransformation) -> None:
        
        self.optim = optim

        transformed = self.build()

        self.init = transformed.init
        self.apply = transformed.apply

    @staticmethod
    def build():
        def f(x, is_training):
            net = Net()
            return net(x, is_training)

        return hk.transform(f)
    
    def init_state(self, rng: random.KeyArray, batch: jnp.DeviceArray):
        params = self.init(rng, batch, is_training=True)
        opt_state = self.optim.init(params)
        return params, opt_state

    # @partial(jit, static_argnums=[0, 4])
    def forward(self, params: hk.Params, key: random.KeyArray, x: jnp.ndarray, is_training: bool):
        pred = self.apply(params, key, x, is_training)
        return pred
    
    @partial(jit, static_argnums=[0, 5])
    def loss_and_pred(self, params: hk.Params, key: random.KeyArray, x: jnp.ndarray, y: jnp.ndarray, is_training: bool):
        pred = self.forward(params, key, x, is_training)
        loss = optax.softmax_cross_entropy(pred, y).mean()
        return loss, pred
    
    @partial(jit, static_argnums=[0])
    def update(self,
               params: hk.Params,
               opt_state: optax.OptState,
               key: random.KeyArray,
               x: jnp.ndarray,
               y: jnp.ndarray
               ):
        (loss, pred), grads = vgrad(self.loss_and_pred, has_aux=True)(params, key, x, y, True)

        updates, opt_state = self.optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, pred

    # @partial(jit, static_argnums=[0])
    def accuracy(self, pred, y):
        return jnp.mean(jnp.where(jnp.argmax(pred, axis=-1) == jnp.argmax(y, axis=-1), 1, 0))
    
    def fit(self, key: random.KeyArray, datasets: tuple, epochs: int, params=None, opt_state=None):
        
        (x_train, y_train), (x_val, y_val) = datasets

        if params is None or opt_state is None:
            key, subkey = random.split(key)
            params, opt_state = self.init_state(subkey, x_train[0])
            print(type(opt_state))

        for epoch in range(epochs):
            running_acc = 0.0
            running_loss = 0.0
            for x, y in tqdm(zip(x_train, y_train), desc=f"epoch {epoch}", total=x_train.shape[0]):
                key, subkey = random.split(key)
                params, opt_state, loss, pred = self.update(params, opt_state, subkey, x, y)
                running_loss += loss / x_train.shape[0] / x_train.shape[1]
                running_acc += self.accuracy(pred, y) / x_train.shape[0]

            print(f"epoch {epoch}/{epochs} | loss {running_loss: .3f} | acc {running_acc: .3f}", end='\n')

            running_acc = 0.0
            running_loss = 0.0
            for x, y in tqdm(zip(x_val, y_val), desc=f"epoch {epoch}", total=x_val.shape[0]):
                key, subkey = random.split(key)
                loss, pred = self.loss_and_pred(params, subkey, x, y, False)
                running_loss += loss / x_train.shape[0] / x_train.shape[1]
                running_acc += self.accuracy(pred, y) / x_train.shape[0]

            print(f"epoch {epoch}/{epochs} | loss_val {running_loss: .3f} | acc_val {running_acc: .3f}", end='\n')

        return params, opt_state


if __name__=='__main__':

    seed = 42
    epochs = 10
    batch_size = 128

    key1, key2 = random.split(random.PRNGKey(seed))

    optim = optax.adam(1e-3)
    classfifier = Classifier(optim)

    datasets = mk_dataset(key1, batch_size)

    classfifier.fit(key2, datasets, epochs)