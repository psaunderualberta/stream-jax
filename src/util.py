from jax import numpy as jnp, lax as jax_lax, tree_util as jtu, jit
from jax import random as jr
import optax
import equinox as eqx

import chex
from typing import Union

__eps = 1e-8


def get_float_dtype():
    """Returns the default float dtype."""
    return jnp.float32


@eqx.filter_jit
def is_none(x):
    return x is None


# @eqx.filter_jit
def ObGD(
    eligibility_trace: chex.Array,
    model: chex.Array,
    delta: Union[float, chex.Array],
    alpha: Union[float, chex.Array],
    kappa: Union[float, chex.Array]
):
    delta_bar = jnp.maximum(jnp.abs(delta), 1.0)
    trace_flat, _ = jtu.tree_flatten(eligibility_trace)
    eligibility_trace_norm = sum(jnp.sum(jnp.abs(x)) for x in jtu.tree_leaves(trace_flat))
    M = alpha * kappa * delta_bar * eligibility_trace_norm
    alpha_ = jnp.minimum(alpha / M, alpha)

    # update in direction of gradient
    def _apply_update(m, u):
        if u is None:
            return m
        else:
            return m + alpha_ * delta * u

    return jtu.tree_map(_apply_update, model, eligibility_trace, is_leaf=is_none)


@jit
def sample_mean_var(
    x: Union[float, chex.Array],
    mu: Union[float, chex.Array],
    p: Union[float, chex.Array],
    n: Union[int, chex.Array]
):
    n += 1
    mu_bar = mu + (x - mu) / n
    p = p + (x - mu) * (x - mu_bar)
    var = jax_lax.select(n >= 2, p / (n - 1), jnp.ones_like(p))
    return mu_bar, p, var, n


@jit
def scale_reward(
    reward: Union[float, chex.Array],
    gamma: Union[float, chex.Array],
    u: Union[float, chex.Array],
    p: Union[float, chex.Array],
    done: Union[bool, chex.Array],
    n: Union[int, chex.Array],
):
    done = done.astype(reward.dtype)
    u = gamma * (1 - done) * u + reward
    _, p, var, n = sample_mean_var(u, 0, p, n)
    reward_scaled = reward / jnp.sqrt(var + __eps)
    return reward_scaled, u, p


@jit
def normalize_observation(
    observation: chex.Array,
    mu: chex.Array,
    p: chex.Array,
    n: Union[int, chex.Array],
):
    mu, var, p, n = sample_mean_var(observation, mu, p, n)
    return (observation - mu) / jnp.sqrt(var + __eps), mu, p


@eqx.filter_jit
def pytree_tile(pytree, value):
    treedef = jtu.tree_structure(pytree)
    values = jnp.ones_like(value) * value
    return jtu.tree_unflatten(treedef, values)


@eqx.filter_jit
def pytree_keys(model, key):
    treedef = jtu.tree_structure(model)
    keys = jr.split(key, treedef.num_leaves)
    return jtu.tree_unflatten(treedef, keys)


@eqx.filter_jit
def sparse_init_linear(in_size, size, sparsity_level, key):
    layer_size = (size, in_size)  # equinox expects (out_size, in_size) for weight shape
    fan_in = 1 / jnp.sqrt(in_size)

    # Generate a random mask for sparsity
    zero_inits = jr.bernoulli(key, sparsity_level, layer_size)

    # Initialize weights with lecun initialization + sparsity
    weights = jr.uniform(key, layer_size, get_float_dtype(), minval=-fan_in, maxval=fan_in)
    weights = jnp.where(zero_inits, 0.0, weights)

    # Initialize bias
    bias = jnp.zeros((size,), dtype=get_float_dtype())
    return weights, bias


def update_eligibility_trace(
    z_w,
    gamma, 
    lambda_,
    new_term
):
    def update_trace(z_w_, new_term_):
        if new_term_ is None:
            return z_w_
        return gamma * lambda_ * z_w_ + new_term_
    return jtu.tree_map(update_trace, z_w, new_term, is_leaf=is_none)


def init_eligibility_trace(
    model: eqx.Module
):

    def fun(model_arr):
        if model_arr is None:
            return model_arr
        return jnp.zeros_like(model_arr)

    return jtu.tree_map(fun, model, is_leaf=is_none)


class Softmax(eqx.Module):
    def __call__(self, x):
        e_x = jnp.exp(x - jnp.max(x))  # for numerical stability
        return e_x / jnp.sum(e_x, axis=-1, keepdims=True)

class ReLU(eqx.Module):
    def __call__(self, x):
        return jnp.maximum(0, x)


class Linear(eqx.Module):
    weight: chex.Array
    bias: chex.Array

    def __init__(self, in_size, out_size, key):
        self.weight, self.bias = sparse_init_linear(in_size, out_size, sparsity_level=0.9, key=key)

    def __call__(self, x):
        return self.weight @ x + self.bias
