import equinox as eqx
from flax import struct
from gymnax.environments import environment
import chex
import jax.random as jax_random
from util import SampleMeanStats, normalize_observation, scale_reward
from typing import Any


@struct.dataclass
class StreamXTrainState:
    current_timestep: int = struct.field(pytree_node=True, default=0)
    current_episode_num: int = struct.field(pytree_node=True, default=0)
    reward_stats: SampleMeanStats = struct.field(pytree_node=False, default=SampleMeanStats.new_params(()))
    obs_stats: SampleMeanStats = struct.field(pytree_node=False, default=SampleMeanStats.new_params(()))
    eval_callback: callable = struct.field(pytree_node=False)

    obs: chex.Array = struct.field(pytree_node=False)
    state: environment.EnvState = struct.field(pytree_node=False)


class StreamXAlgorithm(eqx.Module):
    """
    https://refactoring.guru/design-patterns/template-method
    """
    gamma: float = 0.99
    discount_factor: float = 0.99
    max_learning_timesteps: int = 5 * 10**6

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
        obs, state = env.reset(key_, env_params)
        ts = self.pre_learning_initialization(ts, obs, state)
        ts = self.pre_episode_initialization(ts, obs, state)
        while ts.current_timestep <= self.max_learning_timesteps:
            action, ts = self.get_action(ts)

            # Step the environment
            key, step_key = jax_random.split(key)
            next_obs, next_state, reward, done, info = env.step(step_key, ts.state, action, env_params)
            ts = self.update_env_step_outputs(ts, next_obs, next_state, reward, done, info)
            ts = self.post_env_step_hook(ts)

            # normalize observation & reward
            ts = self.normalize_observation(ts)
            ts = self.scale_reward(ts)

            # next_obs, obs_stats = normalize_observation(next_obs, obs_stats)
            # scaled_reward, reward_trace, reward_stats = scale_reward(reward, reward_stats, reward_trace, done, self.gamma)

            ts = self.get_delta_and_traces(ts)
            ts = self.update_eligibility_traces(ts)
            ts = self.update_weights(ts)

            if done:
                key, key_ = jax_random.split(key)
                obs, state = env.reset(key_, env_params)
                ts = self.pre_episode_initialization(ts, obs, state)
                ts = ts.replace(current_episode_num=ts.current_episode_num + 1)
            
            # Increment counter, obs = next_obs, etc.
            ts = ts.replace(current_timestep=ts.current_timestep + 1)
        
        return ts

    def pre_learning_initialization(self, ts: StreamXTrainState, obs: chex.Array, state: environment.EnvState):
        ts = ts.replace(obs_state=SampleMeanStats.new_params(obs.shape))
        return ts

    def pre_episode_initialization(self, ts: StreamXTrainState, obs: chex.Array, state: environment.EnvState):
        raise NotImplementedError("pre_episode_initialization")

    def get_action(self, ts: StreamXTrainState):
        raise NotImplementedError("get_action")
    
    def update_env_step_outputs(
            self,
            ts: StreamXTrainState,
            next_obs: chex.Array,
            next_state: environment.EnvState,
            reward: float,
            done: bool,
            info: dict[Any, Any]
        ):
        raise NotImplementedError("update_env_step_outputs")
    
    def normalize_observation(self, ts: StreamXTrainState):
        raise NotImplementedError("normalize_observation")

    def scale_reward(self, ts: StreamXTrainState):
        raise NotImplementedError("scale_reward")

    def get_delta_and_traces(self, ts: StreamXTrainState):
        raise NotImplementedError("get_delta_and_traces")

    def update_eligibility_traces(self, ts: StreamXTrainState):
        raise NotImplementedError("update_eligibility_traces")

    def update_weights(self, ts: StreamXTrainState):
        raise NotImplementedError("update_weights")
    
    def next_timestep(self, ts: StreamXTrainState):
        return ts.replace(current_timestep=ts.current_timestep + 1)
    
    def increment_episode_num(self, ts: StreamXTrainState):
        return ts.replace(current_episode_num=ts.current_episode_num + 1)

    def post_env_step_hook(self, ts: StreamXTrainState) -> StreamXTrainState:
        return ts