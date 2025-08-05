import equinox as eqx
import chex
from util import LeakyReLU, Linear
from jax import numpy as jnp, random as jax_random, jit
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
    mu_layer: chex.Array
    std_layer: chex.Array
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

            # Add activation function
            self.layers.append(activation)
            in_size = size

        # Final output layers
        key, _mu_key, _std_key = jax_random.split(key, 3)
        self.mu_layer = Linear(in_size, 1, key=_mu_key)
        self.std_layer = Linear(in_size, 1, key=_std_key)

    @jit
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        
        mu = self.mu_layer(x)
        pre_std = self.std_layer(x)
        std = softplus(pre_std)
        return mu, std


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
        key, _key = jax_random.split(key, 3)
        self.layers.append(Linear(in_size, 1, key=_key))

    @jit
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x
