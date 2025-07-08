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
    weights: list[chex.Array]
    biases:  list[chex.Array]
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
        self.weights = [
            jnp.array(w1, dtype=jnp.float32),
            jnp.array(w2, dtype=jnp.float32),
            jnp.array(w3, dtype=jnp.float32),
        ]

        self.biases = [
            jnp.array(b1, dtype=jnp.float32),
            jnp.array(b2, dtype=jnp.float32),
            jnp.array(b3, dtype=jnp.float32),
        ]

    def __call__(self, x):
        x = self.weights[0] @ x + self.biases[0]
        x = (x - x.mean()) / jnp.sqrt(x.var() + 1e-5)
        x = LeakyReLU()(x)
        x = self.weights[1] @ x + self.biases[1]
        x = eqx.nn.LayerNorm(32, use_weight=False, use_bias=False)(x)
        x = LeakyReLU()(x)
        return self.weights[2] @ x + self.biases[2]
    
    def num_actions(self):
        return self.layers[-1].weight.shape[0]

    
if __name__ == "__main__":
    q = QNetwork(4, 32, 2, jax_random.PRNGKey(0))
    t = jnp.array([1, 1, 1, 1], dtype=jnp.float32)
    r = -q(t)[0]
    grads = grad(lambda q, t: -q(t)[0])(q, t)
    for grad_ in jt.leaves(grads):
        print(grad_)
