import equinox as eqx
import chex
from util import LeakyReLU, Linear
from jax import random as jax_random, jit, numpy as jnp, lax
from jax.nn import softplus

class QNetwork(eqx.Module):
    layers: list[chex.Array]
    activation: eqx.Module

    def __init__(
            self,
            obs_shape: int,
            hidden_layer_sizes: list[int],
            num_actions: int,
            key: chex.PRNGKey,
            activation: eqx.Module = LeakyReLU()
        ):

        self.activation = activation

        self.layers = []
        self.activation = activation
        in_size = obs_shape
        for size in hidden_layer_sizes:
            # Add a linear layer
            key, _key = jax_random.split(key)
            layer = Linear(in_size, size, key=_key)
            self.layers.append(layer)

            # # Add layer norm
            layer_norm = eqx.nn.LayerNorm(size, use_weight=False, use_bias=False)
            self.layers.append(layer_norm)

            # # Add activation function
            self.layers.append(activation)
            in_size = size

        # Final output layer
        key, _key = jax_random.split(key)
        output_layer = Linear(in_size, num_actions, key=_key)
        self.layers.append(output_layer)

    @jit
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def num_actions(self):
        return self.layers[-1].weight.shape[0]


class Actor(eqx.Module):
    layers: list[chex.Array]
    num_actions: float
    mu_layer: chex.Array
    std_layer: chex.Array
    activation: eqx.Module

    def __init__(
            self,
            obs_shape: int,
            hidden_layer_sizes: list[int],
            num_actions: int,
            key: chex.PRNGKey,
            activation: eqx.Module = LeakyReLU()
        ):

        self.activation = activation

        self.layers = []
        self.activation = activation
        in_size = obs_shape
        for size in hidden_layer_sizes:
            # Add a linear layer
            key, _key = jax_random.split(key)
            layer = Linear(in_size, size, key=_key)
            self.layers.append(layer)

            # # Add layer norm
            layer_norm = eqx.nn.LayerNorm(size, use_weight=False, use_bias=False)
            self.layers.append(layer_norm)

            # Add activation function
            self.layers.append(activation)
            in_size = size

        # Final output layers
        self.num_actions = num_actions * 1.0  # Ensure num_actions is a float
        num_actions = int(num_actions)
        key, _mu_key, _std_key = jax_random.split(key, 3)
        self.mu_layer = Linear(in_size, num_actions, key=_mu_key)
        self.std_layer = Linear(in_size, num_actions, key=_std_key)

    @jit
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        
        mu = self.mu_layer(x)
        pre_std = self.std_layer(x)
        std = jnp.where(pre_std >= 20, pre_std, softplus(pre_std))
        return mu, std
    
    @jit
    def sample(self, x, key):
        mu, std = self(x)
        return jax_random.normal(key, mu.shape, dtype=x.dtype) * std + mu

    @jit
    def entropy(self, x):
        _, std = self(x)
        return (
            0.5 * self.num_actions * (1 + jnp.log(2 * jnp.pi))
            + 0.5 * jnp.log((std**2).sum())  # Log of determinant of covariance matrix
        )

    @jit
    def log_prob(self, x, action):
        mu, std = self(x)
        covar = std**2

        return (
            -1/2 * ((action - mu).T * (1 / covar) * (action - mu)).sum(axis=-1)
            - 1/2 * jnp.log(covar.sum())
            - 0.5 * self.num_actions * jnp.log(2 * jnp.pi)
        )


class Critic(eqx.Module):
    layers: list[chex.Array]
    activation: eqx.Module

    def __init__(
            self,
            obs_shape: int,
            hidden_layer_sizes: list[int],
            key: chex.PRNGKey,
            activation: eqx.Module = LeakyReLU()
        ):

        self.activation = activation

        self.layers = []
        self.activation = activation
        in_size = obs_shape
        for size in hidden_layer_sizes:
            # Add a linear layer
            key, _key = jax_random.split(key)
            layer = Linear(in_size, size, key=_key)
            self.layers.append(layer)

            # # Add layer norm
            layer_norm = eqx.nn.LayerNorm(size, use_weight=False, use_bias=False)
            self.layers.append(layer_norm)

            # # Add activation function
            self.layers.append(activation)
            in_size = size

        # Final output layer
        key, _key = jax_random.split(key)
        self.layers.append(Linear(in_size, 1, key=_key))

    @jit
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x
