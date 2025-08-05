import equinox as eqx
from jax import numpy as jnp, random as jax_random, tree as jt, jit, lax as jax_lax, value_and_grad, grad, jax
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
    divide_pytree,
    ObGD_update
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
import optax


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
def huber_loss_delta(q_network, scaled_reward, gamma, done, obs, action, next_obs):
    delta = get_delta(q_network, scaled_reward, gamma, done, obs, action, next_obs)
    return jax_lax.select(
        jnp.abs(delta) <= 1,
        1/2 * delta ** 2,
        jnp.abs(delta) - 1/2
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
    q_network: QNetwork
    sum_td_grad: QNetwork = struct.field(pytree_node=False)
    reward_: float
    reward_trace: float
    global_timestep: int
    obs_stats: SampleMeanStats
    reward_stats: SampleMeanStats
    length: int
    total_loss: float
    episode_num: int = 0

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
    num_episodes: int = struct.field(pytree_node=False)
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
            q_network = ts.q_network
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
            delta, td_grad = value_and_grad(get_delta)(q_network, scaled_reward, self.gamma, done, obs, action, next_obs)

            sum_td_grad = ObGD_update(td_grad, delta, self.alpha, self.kappa)
            sum_td_grad = eqx.apply_updates(ts.sum_td_grad, td_grad)

            next_ts = ts.replace(
                key=key,
                done=done,
                obs=next_obs,
                state=next_state,
                q_network=q_network,
                reward_=ts.reward_ * self.gamma + reward,
                reward_trace=reward_trace,
                global_timestep=global_timestep,
                length=ts.length + 1,
                obs_stats=obs_stats,
                reward_stats=reward_stats,
                sum_td_grad=sum_td_grad,
                total_loss=ts.total_loss + delta
            )

            return next_ts

        @eqx.filter_jit
        def eval_iteration(ts: StreamQTrainState):
            eval_result = jax_lax.while_loop(
                lambda ts: jnp.logical_not(ts.done),
                train_iteration,
                ts,
            )

            avg_grad = divide_pytree(eval_result.sum_td_grad, eval_result.length)
            q_network = eqx.apply_updates(eval_result.q_network, avg_grad)

            episode_result = eval_result.replace(
                q_network=q_network,
                sum_td_grad=init_eligibility_trace(q_network),
                total_loss=0.0,
                reward_=0.,
                reward_trace=0.0,
                length=0,
                episode_num=eval_result.episode_num + 1,
                done=False,
            )

            return episode_result, self.eval_callback(self, eval_result, ts.key)

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
            sum_td_grad=init_eligibility_trace(q_network),
            q_network=q_network,
            reward_=0.0,
            reward_trace=0.0,
            global_timestep=0,
            length=0,
            obs_stats=obs_stats,
            reward_stats=reward_stats,
            total_loss=0.0,
        )

        train_result, evaluations = jax_lax.scan(
            lambda ts, _: eval_iteration(ts),
            train_state,
            None,
            length=self.num_episodes
        )

        return train_result, evaluations

if __name__ == "__main__":
    # Example usage
    
    key = jax_random.PRNGKey(1)
    key, key_reset, key_act, key_step = jax_random.split(key, 4)

    # Instantiate the environment & its settings.
    env, env_params = make("CartPole-v1")
    # env_params = env_params.replace(max_steps_in_episode=1000)

    obs_shape = env.observation_space(env_params).shape[0]
    num_actions = env.action_space(env_params).n
    hidden_layer_sizes = [64, 64]  # Example hidden layer sizes
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

        jax.debug.print(
            "Episodic Return: {:.1f}. Training Steps: {:2}. Epsilon: {:.2f}",
            transition.reward,
            ts.global_timestep,
            linear_epsilon_schedule(
                algo.start_e, algo.end_e, algo.stop_exploring_timestep, ts.global_timestep
            )
        )
        return transition.reward


    # Run the stream Q-learning algorithm
    result, evals = StreamQ(
        q_network,
        env,
        env_params,
        gamma=0.99,
        lambda_=0.8,
        alpha=1.0,
        kappa=2.0,
        start_e=1.0,
        end_e=0.01,
        stop_exploring_timestep=50_000,
        num_episodes=2000,
        eval_freq=1000,
        eval_callback=eval_callback
    ).train(key_act)

    print(result.global_timestep)
