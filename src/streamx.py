import equinox as eqx
from flax import struct
from gymnax.environments import environment
import chex
import jax.random as jax_random
from util import SampleMeanStats
from typing import Any


class StreamXTrainState:
    pass

class StreamXAlgorithm(eqx.Module):
    """
    https://refactoring.guru/design-patterns/template-method
    """

    gamma: float = 0.99
    lambda_: float = 0.8
    alpha: float = 1.0
    kappa: float = 2.0
    max_learning_timesteps: int = 5 * 10**6
    eval_callback: callable = struct.field(pytree_node=False, default=None)
    eval_freq: int = struct.field(pytree_node=False, default=4096)
    num_envs_per_eval: int = struct.field(pytree_node=False, default=8)
    

    def jitted_train(self, ts: StreamXTrainState):
        self.initialize_eligibility_traces()


    def non_jitted_train(
            self,
            key: chex.PRNGKey,
            ts: StreamXTrainState,
            env: environment.Environment,
            env_params: environment.EnvParams,
    ):
        key, key_ = jax_random.split(key)
        ts = ts.replace(key=key_)
        ts = self.reset_env(ts, env, env_params)
        while ts.current_timestep <= self.max_learning_timesteps:
            eval_freq_counter = 0
            while eval_freq_counter < self.eval_freq:
                action, ts = self.get_action(ts)

                # Step the environment
                ts = self.step_env(ts, action, env, env_params)

                ts = self.get_delta_and_traces(ts)
                ts = self.update_eligibility_traces(ts)
                ts = self.update_weights(ts)

                if ts.done:
                    ts = self.reset_env(ts, env, env_params)
                else:
                    ts = self.next_training_iteration(ts)

                eval_freq_counter += 1

            self.eval_callback(self, ts, env, env_params)

        return ts

    def reset_env(self, ts: StreamXTrainState, env, env_params):
        raise NotImplementedError("reset_env")
    
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
