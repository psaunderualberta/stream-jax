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


class SampleMeanStats(eqx.Module):
    mu: chex.Array
    p: chex.Array
    var: chex.Array
    count: int

    def __init__(self, mu, p, var, count):
        self.mu = mu
        self.p = p
        self.var = var
        self.count = count

    @classmethod
    def new_params(cls, shape):
        mu = jnp.ones(shape, dtype=get_float_dtype())
        p = jnp.zeros(shape, dtype=get_float_dtype())
        var = jnp.zeros(shape, dtype=get_float_dtype())
        count = 1

        return SampleMeanStats(
            mu=mu,
            p=p,
            var=var,
            count=count,
        )


class SampleMeanUpdate(eqx.Module):
    @classmethod
    def update(cls, sample: Union[float, chex.Array], stats: SampleMeanStats):
        new_count = stats.count + 1
        mu_bar = stats.mu + (sample - stats.mu) / new_count
        new_p = stats.p + (sample - stats.mu) * (sample - mu_bar)
        var = jax_lax.select(new_count >= 2, new_p / (new_count - 1), jnp.ones_like(new_p))
        return SampleMeanStats(
            mu=mu_bar,
            p=new_p,
            var=var,
            count=new_count
        )


@jit
def normalize_observation(
    observation: chex.Array,
    observation_stats: SampleMeanStats
):
    new_stats = SampleMeanUpdate.update(observation, observation_stats)
    return (observation - new_stats.mu) / jnp.sqrt(new_stats.var + __eps), new_stats


@jit
def scale_reward(
    reward: Union[float, chex.Array],
    reward_stats: SampleMeanStats,
    reward_trace: Union[float, chex.Array],
    done: bool,
    gamma: Union[float, chex.Array],
):
    done = done.astype(reward.dtype)
    reward_trace = gamma * (1 - done) * reward_trace + reward
    new_stats = SampleMeanUpdate.update(reward_trace, reward_stats)
    reward_scaled = reward / jnp.sqrt(new_stats.var + __eps)
    return reward_scaled, reward_trace, new_stats


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