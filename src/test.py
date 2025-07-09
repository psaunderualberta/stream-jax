import chex
import equinox as eqx
from util import LeakyReLU, Linear
import jax.random as jax_random
import jax.numpy as jnp
from jax import grad, vmap
import jax.tree as jt
import pickle as pkl


with open("./data.pkl", "rb") as f:
    w1, b1, w2, b2, w3, b3 = pkl.load(f)


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
            Linear(obs_shape, hidden_layer_sizes, k1),
            eqx.nn.LayerNorm(hidden_layer_sizes, use_weight=False, use_bias=False),
            LeakyReLU(),
            Linear(hidden_layer_sizes, hidden_layer_sizes, k2),
            eqx.nn.LayerNorm(hidden_layer_sizes, use_weight=False, use_bias=False),
            LeakyReLU(),
            Linear(hidden_layer_sizes, num_actions, k3)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def num_actions(self):
        return self.layers[-1].weight.shape[0]

    
if __name__ == "__main__":
    q = QNetwork(4, 32, 2, jax_random.PRNGKey(1))
    t = jnp.array([1, 1, 1, 1], dtype=jnp.float32)
    r = -q(t)[0]
    grads = eqx.filter_grad(lambda q, t: -q(t)[0])(q, t)
    for grad_ in jt.leaves(grads):
        print(grad_)
