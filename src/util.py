from jax import numpy as jnp, lax as jax_lax, tree_util as jtu, jit
from jax import random as jr
import jax
import optax
import equinox as eqx

import chex
from typing import Union

__eps = 1e-5


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
    eligibility_trace_norm = sum(
        jnp.abs(x).sum() for x in jtu.tree_leaves(eligibility_trace)
    )
    M = alpha * kappa * delta_bar * eligibility_trace_norm
    alpha_ = jnp.minimum(alpha / M, alpha)

    # update in direction of gradient
    def _apply_update(m, e):
        if e is None:
            return m
        
        return m - alpha_ * delta * e

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
    mu, p, var, n = sample_mean_var(observation, mu, p, n)
    return (observation - mu) / jnp.sqrt(var + __eps), mu, p


# @eqx.filter_jit
def sparse_init_linear(in_size, out_size, sparsity_level, key):
    layer_size = (out_size, in_size)  # equinox expects (out_size, in_size) for weight shape
    init_bound = 1 / jnp.sqrt(in_size)

    # Generate a random mask for sparsity
    zeros_per_col = jnp.ceil(sparsity_level * in_size).astype(int)

    # Initialize weights with lecun initialization + sparsity
    weights = jr.uniform(key, layer_size, get_float_dtype(), minval=-init_bound, maxval=init_bound)

    # init same as source code
    for col_idx in range(out_size):
        key, _key = jr.split(key)
        zero_idxs = jr.permutation(_key, in_size)[:zeros_per_col]
        weights = weights.at[col_idx, zero_idxs].set(0.0)

    # Initialize bias
    bias = jnp.zeros((out_size,), dtype=get_float_dtype())
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


@eqx.filter_jit
def init_eligibility_trace(
    model: eqx.Module
):
    def fun(model_arr):
        if model_arr is None:
            return model_arr
        return jnp.zeros_like(model_arr)

    return jtu.tree_map(fun, model, is_leaf=is_none)
    


class LeakyReLU(eqx.Module):
    def __call__(self, x):
        return jnp.where(x <= 0, 0.01 * x, x)


class Linear(eqx.Module):
    weight: chex.Array
    bias: chex.Array

    def __init__(self, in_size, out_size, key):
        self.weight, self.bias = sparse_init_linear(in_size, out_size, sparsity_level=0.9, key=key)

    def __call__(self, x):
        return self.weight @ x + self.bias


def linear_epsilon_schedule(start_e, end_e, duration, t):
    slope = (end_e - start_e) / duration
    return  jnp.maximum(slope * t + start_e, end_e)