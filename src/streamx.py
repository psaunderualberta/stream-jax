import equinox as eqx
from flax import struct
from gymnax.environments import environment
import chex
import jax.random as jax_random
from jax import jit, lax as jax_lax
import jax
from util import SampleMeanStats
from typing import Any
from functools import partial

class StreamXTrainState:
    pass

class StreamXAlgorithm(eqx.Module):
    """
    https://refactoring.guru/design-patterns/template-method
    """
    env: environment.Environment
    env_params: environment.EnvParams
    gamma: float = 0.99
    lambda_: float = 0.8
    alpha: float = 1.0
    kappa: float = 2.0
    max_learning_timesteps: int = 5 * 10**6
    eval_callback: callable = struct.field(pytree_node=False, default=None)
    eval_freq: int = struct.field(pytree_node=False, default=4096)
    num_envs_per_eval: int = struct.field(pytree_node=False, default=8)
    

    @partial(jit, static_argnums=(0))
    def jitted_train(
        self,
        input_key: chex.PRNGKey,
        ts: StreamXTrainState,
    ) -> StreamXTrainState:
        

        def inner_loop_body(ts: StreamXTrainState):
            action, ts = self.get_action(ts)

            # Step the environment
            key, ts = self.get_key(ts)
            next_obs, next_state, reward, done, _ = self.step(key, ts.state, action, self.env_params)
            ts = self.step_env(ts, next_obs, next_state, reward, done)

            ts = self.get_delta_and_traces(ts)
            ts = self.update_eligibility_traces(ts)
            ts = self.update_weights(ts)

            key, ts = self.get_key(ts)
            return jax_lax.cond(
                ts.done,
                lambda ts: self.reset_env(ts, *self.env.reset(key, self.env_params)),
                lambda ts: self.next_training_iteration(ts),
                ts
            )

        def inner_loop(ts: StreamXTrainState):
            ts = jax_lax.fori_loop(
                0, self.eval_freq,
                lambda _, ts: inner_loop_body(ts),
                ts
            )

            self.eval_callback(self, ts, self.env, self.env_params)
            return ts
        
        def outer_loop(ts: StreamXTrainState):
            return jax_lax.while_loop(
                lambda ts: ts.current_timestep <= self.max_learning_timesteps,
                inner_loop,
                ts
            )

        ts = ts.replace(key=input_key)
        key, ts = self.get_key(ts)
        obs, state = self.reset(key, self.env_params)
        ts = self.reset_env(ts, obs, state)
        ts = eqx.filter_jit(outer_loop)(ts)
        return ts
            

    def non_jitted_train(
            self,
            key: chex.PRNGKey,
            ts: StreamXTrainState,
    ):
        ts = ts.replace(key=key)
        key, conf = self.get_key(ts); ts = ts.replace(**conf)
        obs, state = self.reset(key, self.env_params)
        # with jax.checking_leaks():
        ts = ts.replace(**self.reset_env(ts, obs))
        while ts.current_timestep <= self.max_learning_timesteps:
            for _ in range(self.eval_freq):
                ts = ts.replace(**self.get_action(ts))
                action = ts.action

                # Step the environment
                key, conf = self.get_key(ts); ts = ts.replace(**conf)
                next_obs, next_state, reward, done, _ = self.step(key, state, action, self.env_params)
                ts = ts.replace(**self.step_env(ts, next_obs, reward, done))

                ts = ts.replace(**self.get_delta_and_traces(ts))
                ts = ts.replace(**self.update_eligibility_traces(ts))
                ts = ts.replace(**self.update_weights(ts))

                if ts.done:
                    obs, state = self.reset(key, self.env_params)
                    print(self.reset_env(ts, obs))
                    ts = ts.replace(**self.reset_env(ts, obs))
                else:
                    ts = ts.replace(**self.next_training_iteration(ts))
                    state = next_state

            # self.eval_callback(self, ts, self.env, self.env_params)

        return ts

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params):
        obs, state, reward, done, info = self.env.step(key, state, action, params)
        obs = obs.astype(float)
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params):
        obs, state = self.env.reset(key, params)
        obs = obs.astype(float)
        return obs, state
    
    # @partial(jit, static_argnums=(0,))
    def get_key(self, ts: StreamXTrainState):
        key, key_ = jax_random.split(ts.key)
        return key_, {"key": key}

    
    def step_env(self, ts: StreamXTrainState, action, env, env_params):
        raise NotImplementedError("step_env")

    def get_action(self, ts: StreamXTrainState) -> tuple[int | float | chex.Array, StreamXTrainState]:
        raise NotImplementedError("get_action")
    
    def normalize_observation(self, ts: StreamXTrainState):
        raise NotImplementedError("normalize_observation")

    def scale_reward(self, ts: StreamXTrainState):
        raise NotImplementedError("scale_reward")

    def get_delta_and_traces(self, ts: StreamXTrainState, gamma: float):
        raise NotImplementedError("get_delta_and_traces")

    def update_eligibility_traces(self, ts: StreamXTrainState, gamma: float, lambda_: float):
        raise NotImplementedError("update_eligibility_traces")

    def update_weights(self, ts: StreamXTrainState, alpha: float, kappa: float):
        raise NotImplementedError("update_weights")
    
    def next_timestep(self, ts: StreamXTrainState):
        return ts.replace(current_timestep=ts.current_timestep + 1)
    
    def increment_episode_num(self, ts: StreamXTrainState):
        return ts.replace(current_episode_num=ts.current_episode_num + 1)
