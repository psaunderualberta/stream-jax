import equinox as eqx
from jax import numpy as jnp, random as jax_random, tree as jt, jit, lax as jax_lax, value_and_grad, jax
import chex
from util import (
    LeakyReLU,
    Linear,
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
from transition import Transition
from gymnax.environments import environment, spaces
from gymnax import make
from typing import Any
from visualizer import visualize
from simple_env import RightIsGoodState, RightIsGoodParams, RightIsGoodEnv
from qnet import QNetwork
from flax import struct
from typing import Callable


jax.config.update('jax_default_device', jax.devices('cpu')[0])


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
    global_timestep: int
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
    q_network: QNetwork = struct.field(pytree_node=False)
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
            q_network=kwargs['q_network'],
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
    
    def train(self, key: chex.PRNGKey) -> StreamQTrainState:
        def train_iteration(ts: StreamQTrainState):
            # extract carry elements
            key = ts.key
            obs, state = ts.obs, ts.state
            q_network, z_w = ts.q_network, ts.z_w
            obs_stats = ts.obs_stats
            reward_stats = ts.reward_stats
            reward_trace = ts.reward_trace
            global_timestep = ts.global_timestep + 1

            key, action_key = jax_random.split(key)

            # Select an action using epsilon-greedy policy.
            eps = linear_epsilon_schedule(self.start_e, self.end_e, self.stop_exploring_timestep, global_timestep)
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
                global_timestep=global_timestep,
                length=ts.length + 1,
                obs_stats=obs_stats,
                reward_stats=reward_stats,
            )
            
            # if the episode terminated, we need to reset obs, state, reward_trace, z_w, reward_, length
            key, reset_key = jax_random.split(key)
            reset_obs, reset_state = self.env.reset(key, self.env_params)
            reset_obs, reset_obs_stats = normalize_observation(reset_obs, next_ts.obs_stats)

            reset_ts = next_ts.replace(
                obs=reset_obs,
                state=reset_state,
                obs_stats=reset_obs_stats,
                key=reset_key,
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

        key, key_reset, key_ts = jax_random.split(key, 3)
        obs, state = env.reset(key, env_params)
        obs_stats = SampleMeanStats.new_params(obs.shape)
        obs, obs_stats = normalize_observation(obs, obs_stats)
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
            global_timestep=0,
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
    env, env_params = make("CartPole-v1")
    # env_params = env_params.replace(max_steps_in_episode=10_000)

    obs_shape = env.observation_space(env_params).shape[0]
    num_actions = env.action_space(env_params).n
    hidden_layer_sizes = [32, 32]  # Example hidden layer sizes
    q_network = QNetwork(obs_shape, hidden_layer_sizes, num_actions, key_reset)

    def eval_callback(algo: StreamQ, ts: StreamQTrainState, key: chex.PRNGKey):
        q = ts.q_network
        act = algo.make_act(ts)
        key, key_reset = jax_random.split(key)
        transition = Transition.initial_transition(env, env_params, key_reset)
        transition = transition.replace(
            obs=normalize_observation(transition.obs, ts.obs_stats)[0]
        )

        def loop_body(transition: Transition):
            action = act(transition.obs, key)
            next_obs, next_state, reward, done, _ = algo.env.step(key, transition.state, action, algo.env_params)
            next_obs, _ = normalize_observation(next_obs, ts.obs_stats)

            return Transition(
                obs=next_obs,
                state=next_state,
                reward=transition.reward * algo.gamma + reward,
                done=done,
                has_next_state=False
            )

        transition = jax_lax.while_loop(
            lambda t: jnp.logical_not(t.done),
            loop_body,
            transition
        )

        time_elapsed = ts.global_timestep / algo.total_timesteps * 100
        jax.debug.print(
            "Episodic Return: {:.1f}. Percent elapsed: {:2.2f}%. Epsilon: {:.2f}",
            transition.reward,
            time_elapsed,
            linear_epsilon_schedule(
                algo.start_e, algo.end_e, algo.stop_exploring_timestep, ts.global_timestep
            )
        )
        return transition.reward


    # Run the stream Q-learning algorithm
    q_network = StreamQ(
        q_network,
        env,
        env_params,
        gamma=0.99,
        lambda_=0.8,
        alpha=1.0,
        kappa=2.0,
        start_e=1.0,
        end_e=0.01,
        stop_exploring_timestep=250_000,
        total_timesteps=500_000,
        eval_freq=1000,
        eval_callback=eval_callback
    ).train(key_act)
