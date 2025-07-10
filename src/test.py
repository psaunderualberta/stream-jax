import chex
import equinox as eqx
from util import LeakyReLU, Linear, ObGD, init_eligibility_trace, update_eligibility_trace
import jax.random as jax_random
import jax.numpy as jnp
from jax import jit, value_and_grad
import jax.tree as jt
import pickle as pkl
from streamq import get_delta


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
        with open("data.pkl", "rb") as f:
            mats = pkl.load(f)
        
        self.layers = [
            Linear(mats[0], mats[1]),
            Linear(mats[2], mats[3]),
            Linear(mats[4], mats[5]),
        ]

    @jit
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = (x - x.mean()) / jnp.sqrt(x.var() + 1e-5)
            x = self.activation(x)
        return self.layers[-1](x)
    
    def num_actions(self):
        return self.layers[-1].weight.shape[0]

    
if __name__ == "__main__":
    q = QNetwork(4, 32, 2, jax_random.PRNGKey(1))
    s = jnp.array([0, 0, 0, 0], dtype=jnp.float32)
    a = jnp.array(1, dtype=jnp.int32)
    reward = jnp.array(1, dtype=jnp.float32)
    sp = jnp.array([-0.6759,  0.7071, -0.6997, -0.7071], dtype=jnp.float32)

    td, grads = value_and_grad(get_delta)(
        q,
        reward,
        0.99,
        False,
        s,
        a,
        sp
    )

    print(td)

    z_w = init_eligibility_trace(q)
    z_w = update_eligibility_trace(z_w, 0.99, 0.8, grads)

    q = ObGD(z_w, q, td, 1.0, 2.0)

    for grad_ in jt.leaves(q):
        print(grad_)
