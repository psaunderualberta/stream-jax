import equinox as eqx
from jax import numpy as jnp, random as jax_random, tree as jt, jit, lax as jax_lax, value_and_grad, jax
import chex
from streamq.util import (
    update_eligibility_trace,
    ObGD,
    init_eligibility_trace,
    normalize_observation,
    scale_reward,
    linear_epsilon_schedule,
    SampleMeanStats,
    is_none,
    pytree_if_else,
)
from gymnax.environments import environment, spaces
from gymnax import make
from typing import Any
from streamq.qnet import QNetwork
from flax import struct
from typing import Callable
from util.util import evaluate


@jit
def get_delta(q_network, scaled_reward, gamma, done, obs, action, next_obs):
    q_sp = q_network(next_obs).max()
    q_sa = q_network(obs)[action]
    return (
        scaled_reward
        + (1 - done) * jax_lax.stop_gradient(gamma * q_sp)
        - q_sa
    )


@jit
def q_epsilon_greedy(q_network, state, epsilon: float, key: chex.PRNGKey):
    """Select an action using epsilon-greedy policy."""
    key, eps_key, action_key = jax_random.split(key, 3)

    q_values = q_network(state)
    explore = jax_random.uniform(eps_key) < epsilon
    greedy_action = jnp.argmax(q_values, axis=-1)
    action = jax_lax.select(
        explore,
        jax_random.randint(action_key, (), 0, q_network.num_actions()),
        greedy_action
    )

    q_value = q_values[action]
    explored = action != greedy_action
    return action, q_value, explored


class StreamQTrainState(eqx.Module):
    key: chex.PRNGKey
    done: bool
    obs: chex.PRNGKey
    state: environment.EnvState
    z_w: QNetwork
    q_network: QNetwork
    reward_: float
    reward_trace: float
    global_step: int
    obs_stats: SampleMeanStats
    reward_stats: SampleMeanStats
    length: int

    def replace(self, **kwargs) -> 'StreamQTrainState':
        """Replace attributes in the training state with new values, akin to flax's 'dataclass.replace'"""

        els = list(kwargs.items())
        return eqx.tree_at(
            lambda t: tuple(getattr(t, k) for k, _ in els),
            self,
            tuple(v for _, v in els),
            is_leaf=is_none,
        )


@struct.dataclass
class StreamQ:
    env: environment.Environment = struct.field(pytree_node=False)
    env_params: environment.EnvParams = struct.field(pytree_node=False)
    gamma: float = struct.field(pytree_node=False)
    lambda_: float = struct.field(pytree_node=False)
    alpha: float = struct.field(pytree_node=False)
    kappa: float = struct.field(pytree_node=False)
    start_e: float = struct.field(pytree_node=False)
    end_e: float = struct.field(pytree_node=False)
    stop_exploring_timestep: float = struct.field(pytree_node=False)
    total_timesteps: int = struct.field(pytree_node=False)
    eval_freq: int = struct.field(pytree_node=False, default=5000)
    eval_callback: Any = struct.field(pytree_node=False, default=lambda *_: None)

    @classmethod
    def create(cls, **kwargs) -> "StreamQ":

        return cls(
            env=kwargs['env'],
            env_params=kwargs['env_params'],
            gamma=kwargs['gamma'],
            lambda_=kwargs['lambda_'],
            alpha=kwargs['alpha'],
            kappa=kwargs['kappa'],
            start_e=kwargs['start_e'],
            end_e=kwargs['end_e'],
            stop_exploring_timestep=kwargs['stop_exploring_timestep'],
            total_timesteps=kwargs['total_timesteps'],
        )
    
    def make_act(self, train_state: StreamQTrainState) -> Callable[[chex.Array, chex.PRNGKey], int | float | chex.Array]:
        def act(obs: chex.Array, _: chex.PRNGKey):
            return jnp.argmax(train_state.q_network(obs), axis=-1)
    
        return act

    def make_normalizer(self, train_state: StreamQTrainState) -> Callable[[chex.Array], chex.Array]:
        def normalizer(obs: chex.Array):
            norm_obs, _ = normalize_observation(obs, train_state.obs_stats)
            return norm_obs
        
        return normalizer
    
    def train(self, key: chex.PRNGKey) -> StreamQTrainState:
        @eqx.filter_jit
        def train_iteration(ts: StreamQTrainState):
            # extract carry elements
            key = ts.key
            obs, state = ts.obs, ts.state
            q_network, z_w = ts.q_network, ts.z_w
            obs_stats = ts.obs_stats
            reward_stats = ts.reward_stats
            reward_trace = ts.reward_trace
            global_step = ts.global_step + 1

            key, action_key = jax_random.split(key)

            # Select an action using epsilon-greedy policy.
            eps = linear_epsilon_schedule(self.start_e, self.end_e, self.stop_exploring_timestep, global_step)
            action, q_value, explored = q_epsilon_greedy(q_network, obs, eps, action_key)

            # Step the environment.
            key, step_key = jax_random.split(key)
            next_obs, next_state, reward, done, _ = self.env.step(step_key, state, action, self.env_params)

            # normalize observation & reward
            next_obs, obs_stats = normalize_observation(next_obs, obs_stats)
            scaled_reward, reward_trace, reward_stats = scale_reward(reward, reward_stats, reward_trace, done, self.gamma)

            # Update eligibility trace
            td_error, td_grad = value_and_grad(get_delta)(q_network, scaled_reward, self.gamma, done, obs, action, next_obs)
            z_w = update_eligibility_trace(z_w, self.gamma, self.lambda_, td_grad)

            # Update Q-network using ObGD
            q_network = ObGD(z_w, q_network, td_error, self.alpha, self.kappa)

            # reset eligibility trace if an exploration occurred
            z_w = jt.map(lambda old: jax_lax.select(
                    jnp.logical_or(explored, done),
                    jnp.zeros_like(old),
                    old
                ), z_w
            )

            next_ts = ts.replace(
                key=key,
                done=done,
                obs=next_obs,
                state=next_state,
                z_w=z_w,
                q_network=q_network,
                reward_=ts.reward_ * self.gamma + reward,
                reward_trace=reward_trace,
                global_step=global_step,
                length=ts.length + 1,
                obs_stats=obs_stats,
                reward_stats=reward_stats,
            )

            # IMPORTANT: Gymnax has auto-reset, i.e. if env.step returns True,
            # then obs & state represent the reset environment's observation & state
            # Hence, if the episode terminated, we need to only reset reward_trace, reward_, & length.
            reset_ts = next_ts.replace(
                reward_trace=0.0,
                reward_=0.0,
                length=0
            )

            return pytree_if_else(done, reset_ts, next_ts)

        @eqx.filter_jit
        def eval_iteration(ts: StreamQTrainState):
            eval_result = jax_lax.fori_loop(
                0,
                self.eval_freq,
                lambda _, ts: train_iteration(ts),
                ts,
            )

            return eval_result, self.eval_callback(self, eval_result, ts.key)

        key, key_reset, key_ts, key_net = jax_random.split(key, 4)
        obs, state = self.env.reset(key_reset, self.env_params)
        obs_stats = SampleMeanStats.new_params(obs.shape)
        obs, obs_stats = normalize_observation(obs, obs_stats)

        obs_shape = self.env.observation_space(self.env_params).shape
        num_actions = self.env.action_space(self.env_params).n
        hidden_layer_sizes = [32, 32]  # Example hidden layer sizes
        q_network = QNetwork(obs_shape[0], hidden_layer_sizes, num_actions, key_net)

        reward_stats = SampleMeanStats.new_params(())
        train_state = StreamQTrainState(
            key=key_ts,
            done=False,
            obs=obs,
            state=state,
            z_w=init_eligibility_trace(q_network),
            q_network=q_network,
            reward_=0.0,
            reward_trace=0.0,
            global_step=0,
            length=0,
            obs_stats=obs_stats,
            reward_stats=reward_stats,
        )
        train_result, evaluations = jax_lax.scan(
            lambda ts, _: eval_iteration(ts),
            train_state,
            None,
            length=self.total_timesteps // self.eval_freq
        )

        return train_result, evaluations

if __name__ == "__main__":
    # Example usage
    
    key = jax_random.PRNGKey(1)
    key, key_reset, key_act, key_step = jax_random.split(key, 4)

    # Instantiate the environment & its settings.
    env, env_params = make("MountainCar-v0")
    env_params = env_params.replace(max_steps_in_episode=10_000)

    obs_shape = env.observation_space(env_params).shape[0]
    num_actions = env.action_space(env_params).n
    hidden_layer_sizes = [32, 32]  # Example hidden layer sizes
    q_network = QNetwork(obs_shape, hidden_layer_sizes, num_actions, key_reset)

    def eval_callback(algo: StreamQ, ts: StreamQTrainState, key: chex.PRNGKey):
        act = algo.make_act(ts)
        max_steps = algo.env_params.max_steps_in_episode
        step = ts.global_step

        return ts.reward


    # Run the stream Q-learning algorithm
    q_network = StreamQ(
        env,
        env_params,
        gamma=0.99,
        lambda_=0.8,
        alpha=1.0,
        kappa=2.0,
        start_e=1.0,
        end_e=0.2,
        stop_exploring_timestep=2_000_000,
        total_timesteps=4_000_000,
        eval_freq=1000,
        eval_callback=eval_callback
    ).train(key_act)
