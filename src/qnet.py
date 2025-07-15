import equinox as eqx
import chex
from util import LeakyReLU, Linear
from jax import numpy as jnp, random as jax_random, jit, lax as jax_lax

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
