import chex
import equinox as eqx
from util import LeakyReLU, Linear, LayerNorm
import jax.random as jax_random
import jax.numpy as jnp
from jax import grad, vmap
import jax.tree as jt


def layer_norm(x):
    return (x - x.mean()) / jnp.sqrt(x.var() + 1e-5)


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
        k1, k2, k3 = jax_random.split(key, 3)
        self.layers = [
            Linear(obs_shape, hidden_layer_sizes, key=k1),
            Linear(hidden_layer_sizes, hidden_layer_sizes, key=k2),
            Linear(hidden_layer_sizes, num_actions, key=k3),
        ]

    def __call__(self, x):
        x = self.layers[0](x)
        x = eqx.nn.LayerNorm(32, use_weight=False, use_bias=False)(x)
        x = LeakyReLU()(x)
        x = self.layers[1](x)
        x = eqx.nn.LayerNorm(32, use_weight=False, use_bias=False)(x)
        x = LeakyReLU()(x)
        return self.layers[2](x)
    
    def num_actions(self):
        return self.layers[-1].weight.shape[0]

    
if __name__ == "__main__":
    q = QNetwork(4, 32, 3, jax_random.PRNGKey(0))
    t = jnp.array([0, 0, 0, 0], dtype=jnp.float32)
    r = -q(t)[0]
    grads = grad(lambda q, t: -q(t)[0])(q, t)
    for grad_ in jt.leaves(grads):
        print(grad_)
