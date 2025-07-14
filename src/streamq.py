import equinox as eqx
from jax import numpy as jnp, random as jax_random, tree as jt, jit, lax as jax_lax, value_and_grad, jax, vmap
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
)
from gymnax.environments import environment
from gymnax import make
from typing import Any
from flax import struct
from streamx import StreamXAlgorithm, StreamXTrainState
from functools import partial

jax.config.update('jax_default_device', jax.devices('cpu')[0])


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

        self.layers = []
        self.activation = activation
        in_size = obs_shape
        for size in hidden_layer_sizes:
            # Add a linear layer
            key, _key = jax_random.split(key)
            layer = Linear(in_size, size, key=_key)
            self.layers.append(layer)

            # # Add layer norm
            layer_norm = eqx.nn.LayerNorm(size, use_weight=False, use_bias=False)
            self.layers.append(layer_norm)

            # # Add activation function
            self.layers.append(activation)
            in_size = size

        # Final output layer
        key, _key = jax_random.split(key)
        output_layer = Linear(in_size, num_actions, key=_key)
        self.layers.append(output_layer)

    @jit
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    @jit
    def get_action(self, obs: chex.Array):
        q_values = self(obs)
        return jnp.argmax(q_values, axis=-1)
    
    def num_actions(self):
        return self.layers[-1].weight.shape[0]


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


@jit
def get_delta(q_network, reward, gamma, done, s, a, sp):
    q_sp = q_network(sp).max()
    q_sa = q_network(s)[a]
    return (
        reward
        + (1 - done) * jax_lax.stop_gradient(gamma * q_sp)
        - q_sa
    )


class _StreamQEpisodeLoopState(eqx.Module):
    key: chex.PRNGKey
    q_network: QNetwork
    done: bool
    obs: chex.Array
    state: Any
    z_w: Any
    reward_: float
    reward_trace: float
    obs_stats: SampleMeanStats
    reward_stats: SampleMeanStats
    length: int
    global_timestep: int


class _StreamQOuterLoopState(eqx.Module):
    key: chex.PRNGKey
    q_network: QNetwork
    obs: chex.Array
    state: Any
    reward_stats: SampleMeanStats
    obs_stats: SampleMeanStats
    current_timestep: int = 0
    ep_num: int = 0


class StreamQ(eqx.Module):
    q_network: QNetwork
    env: environment.Environment
    env_params: environment.EnvParams
    gamma: float
    lambda_: float
    alpha: float
    kappa: float
    start_e: float
    end_e: float
    stop_exploring_timestep: float
    total_timesteps: int

    def __init__(
        self,
        q_network: QNetwork,
        env: environment.Environment,
        env_params: environment.EnvParams,
        gamma: float,
        lambda_: float,
        alpha: float,
        kappa: float,
        start_e: float,
        end_e: float,
        stop_exploring_timestep: float,
        total_timesteps: int,
    ):
        self.q_network = q_network
        self.env = env
        self.env_params = env_params
        self.gamma = gamma
        self.lambda_ = lambda_
        self.alpha = alpha
        self.kappa = kappa
        self.start_e = start_e
        self.end_e = end_e
        self.stop_exploring_timestep = stop_exploring_timestep
        self.total_timesteps = total_timesteps
    
    def train(self, input_key: chex.PRNGKey, jit_training: bool = False):
        def episode_train_body(carry: _StreamQEpisodeLoopState):
            # extract carry elements
            key = carry.key
            obs, state = carry.obs, carry.state
            q_network, z_w = carry.q_network, carry.z_w
            obs_stats = carry.obs_stats
            reward_stats = carry.reward_stats
            reward_trace = carry.reward_trace
            global_timestep = carry.global_timestep + 1

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

            return _StreamQEpisodeLoopState(
                key=key,
                done=done,
                obs=next_obs,
                state=next_state,
                z_w=z_w,
                q_network=q_network,
                reward_=carry.reward_ * self.gamma + reward,
                reward_trace=reward_trace,
                global_timestep=global_timestep,
                length=carry.length + 1,
                obs_stats=obs_stats,
                reward_stats=reward_stats,
            )

        @eqx.filter_jit
        def train_episode(episode_state: _StreamQEpisodeLoopState):
            return jax_lax.while_loop(
                lambda carry: jnp.logical_not(carry.done),
                episode_train_body,
                episode_state
            )

        def non_train_episode(episode_state: _StreamQEpisodeLoopState):
            while not episode_state.done:
                episode_state = episode_train_body(episode_state)
            return episode_state

        def outer_train(ls: _StreamQOuterLoopState):
            initial_loop_state = _StreamQEpisodeLoopState(
                key=ls.key,
                done=False,
                obs=ls.obs,
                state=ls.state,
                z_w=init_eligibility_trace(ls.q_network),
                q_network=ls.q_network,
                reward_=0.0,
                reward_trace=0.0,
                global_timestep=ls.current_timestep,
                length=0.0,
                obs_stats=ls.obs_stats,
                reward_stats=ls.reward_stats,
            )

            # Run the episode loop
            if jit_training:
                episode_result = train_episode(initial_loop_state)
            else:
                episode_result = non_train_episode(initial_loop_state)

            # extract relevant info from environment, like PRNG key
            key = episode_result.key
            q_network = episode_result.q_network
            current_timestep = ls.current_timestep + episode_result.length
            obs_stats = episode_result.obs_stats
            reward_stats = episode_result.reward_stats

            # reset environment
            key, reset_key = jax_random.split(key)
            obs, state = self.env.reset(reset_key, self.env_params)
            obs, obs_stats = normalize_observation(obs, obs_stats)

            epsilon = linear_epsilon_schedule(self.start_e, self.end_e, self.stop_exploring_timestep, current_timestep)
            jax.debug.print(
                "Ep: {}. Episodic Return: {:.1f}. Percent elapsed: {:2.2f}%. Epsilon: {:.2f}",
                ls.ep_num,
                episode_result.reward_,
                100 * current_timestep / self.total_timesteps,
                epsilon
            )

            return _StreamQOuterLoopState(
                key=key,
                q_network=q_network,
                current_timestep=current_timestep,
                obs=obs,
                state=state,
                reward_stats=reward_stats,
                obs_stats=obs_stats,
                ep_num=ls.ep_num + 1,
            )
        
        key, reset_key = jax_random.split(input_key)
        obs, state = self.env.reset(key, self.env_params)
        obs_stats = SampleMeanStats.new_params(obs.shape)
        obs, obs_stats = normalize_observation(obs, obs_stats)
        reward_stats = SampleMeanStats.new_params(())

        training_result = jax.lax.while_loop(
            lambda loop_state: loop_state.current_timestep < self.total_timesteps,
            outer_train,
            _StreamQOuterLoopState(
                key=key,
                q_network=self.q_network,
                obs=obs,
                state=state,
                reward_stats=reward_stats,
                obs_stats=obs_stats,
            )
        )

        return training_result.q_network

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

    learning_time = 50_000

    def eval_callback(agent, env, env_params):
        rewards = []
        lengths = []

        def test_vmap(ts, key):
            ts = ts.replace(key=key)
            obs, state = env.reset(key, env_params)
            ts = env.reset(ts, obs, state)

            @eqx.filter_jit
            def loop(tup):
                done, obs, state = tup
                action = agent.get_action(obs)

                # Step the environment
                next_obs, next_state, reward, done, _ = env.step(key, state, action, env_params)

                return done

            done = False
            while not done:
                done, ts, _ = loop((False, ts, state))
            # (_, ts) = jax_lax.while_loop(
            #     lambda args: jnp.logical_not(args[0]),
            #     loop,
            #     (False, ts)
            # )

            return ts.reward, ts.episode_length

    
        # keys = jax_random.split(key, algo.num_envs_per_eval)


        # test_vmap = jit(  # vmap
        #     test_vmap,
        #     # in_axes=(None, 0)
        # )  # )

        rewards, lengths = test_vmap(ts, key)

        jax.debug.print("{} | {}", jnp.mean(jnp.array(rewards)), jnp.mean(jnp.array(lengths)))


    StreamQ(
        q_network=q_network,
        env=env,
        env_params=env_params,
        gamma=0.99,
        lambda_=0.8,
        alpha=1.0,
        kappa=2.0,
        start_e=1.0,
        end_e=0.01,
        stop_exploring_timestep=0.5 * learning_time,
        total_timesteps=learning_time,
    ).train(key_reset, jit_training=True)