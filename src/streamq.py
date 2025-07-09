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
)
from gymnax.environments import environment, spaces
from gymnax import make
from typing import Any
from visualizer import visualize


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
        self.layers = []
        self.activation = activation
        in_size = obs_shape
        for size in hidden_layer_sizes:
            # Add a linear layer
            key, _key = jax_random.split(key)
            layer = Linear(in_size, size, key=_key)
            self.layers.append(layer)

            # # Add layer norm
            # layer_norm = eqx.nn.LayerNorm(size, use_weight=False, use_bias=False)
            # self.layers.append(layer_norm)

            # # Add activation function
            # self.layers.append(activation)
            in_size = size

        # Final output layer
        key, _key = jax_random.split(key)
        output_layer = Linear(in_size, num_actions, key=_key)
        self.layers.append(output_layer)

    @jit
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = (x - x.mean()) / jnp.sqrt(x.var() + 1e-5)
            x = self.activation(x)
        return self.layers[-1](x)
    
    def num_actions(self):
        return self.layers[-1].weight.shape[0]


@jit
def get_td_error(q_network, reward, gamma, done, s, a, sp):
    q_sp = q_network(sp).max()
    q_sa = q_network(s)[a]
    return (
        reward
        + jax_lax.stop_gradient(gamma * q_sp * done)
        - q_sa
    )


# @jit
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


def stream_q(
    q_network: QNetwork,
    env: environment.Environment,
    env_params: environment.EnvParams,
    gamma: float,
    lambda_: float,
    alpha: float,
    kappa: float,
    start_e: float,
    end_e: float,
    exploration_fraction: float,
    total_timesteps: int,
    key: chex.PRNGKey,
):

    class LoopState(eqx.Module):
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

    def while_loop_body(carry: LoopState):
        # extract carry elements
        obs, state = carry.obs, carry.state
        q_network, z_w = carry.q_network, carry.z_w
        obs_stats = carry.obs_stats
        reward_stats = carry.reward_stats
        reward_trace = carry.reward_trace
        global_timestep = carry.global_timestep + 1

        key, action_key = jax_random.split(carry.key)

        # Select an action using epsilon-greedy policy.
        eps = linear_epsilon_schedule(start_e, end_e, exploration_fraction * total_timesteps, global_timestep)
        action, q_value, explored = q_epsilon_greedy(q_network, obs, eps, action_key)

        # Step the environment.
        key, step_key = jax_random.split(key)
        next_obs, next_state, reward, done, _ = env.step(step_key, state, action, env_params)

        # normalize observation & reward
        next_obs, obs_stats = normalize_observation(next_obs, obs_stats)
        scaled_reward, reward_trace, reward_stats = scale_reward(reward, reward_stats, reward_trace, done, gamma)

        # Update eligibility trace
        td_error, td_grad = value_and_grad(get_td_error)(q_network, scaled_reward, gamma, done, obs, action, next_obs)
        z_w = update_eligibility_trace(z_w, gamma, lambda_, td_grad)

        # Update Q-network using ObGD
        q_network = ObGD(z_w, q_network, td_error, alpha, kappa)

        # if True:
        #     max_q_s_prime_a_prime = jnp.max(q_network(next_obs), axis=-1)
        #     td_target = scaled_reward + gamma * max_q_s_prime_a_prime * done
        #     delta_bar = td_target - q_network(obs)[action]
        #     if jnp.sign(delta_bar * td_error).item() == -1:
        #         print("Overshooting Detected!")

        # reset eligibility trace if an exploration occurred
        z_w = jt.map(lambda old: jax_lax.select(
                jnp.logical_or(explored, done),
                jnp.zeros_like(old),
                old
            ), z_w
        )

        return LoopState(
            key=key,
            done=done,
            obs=next_obs,
            state=next_state,
            z_w=z_w,
            q_network=q_network,
            reward_=gamma * carry.reward_ + reward,
            reward_trace=reward_trace,
            global_timestep=global_timestep,
            length=carry.length + 1,
            obs_stats=obs_stats,
            reward_stats=reward_stats,
        )

    @eqx.filter_jit
    def jitted_while_loop(episode_state: LoopState):
        return jax_lax.while_loop(
            lambda carry: jnp.logical_not(carry.done),
            while_loop_body,
            episode_state
        )

    def non_jitted_while_loop(episode_state: LoopState):
        while not episode_state.done:
            episode_state = while_loop_body(episode_state)
        return episode_state

    i = 0
    current_timestep = 0
    obs, state = env.reset(key, env_params)
    obs_stats = SampleMeanStats.new_params(obs.shape)
    reward_stats = SampleMeanStats.new_params(())
    reward_trace = 0.0
    while current_timestep < total_timesteps:
        # reset environment
        key, reset_key = jax_random.split(key)
        obs, state = env.reset(reset_key, env_params)
        obs, obs_stats = normalize_observation(obs, obs_stats)

        initial_loop_state = LoopState(
            key=key,
            q_network=q_network,
            done=False,
            obs=obs,
            state=state,
            z_w=init_eligibility_trace(q_network),
            reward_=0,
            reward_trace=reward_trace,
            length=0,
            global_timestep=current_timestep,
            obs_stats=obs_stats,
            reward_stats=reward_stats,
        )

        episode_result = initial_loop_state

        episode_result = non_jitted_while_loop(episode_result)
        # episode_result = jitted_while_loop(episode_result)

        key = episode_result.key
        q_network = episode_result.q_network
        current_timestep += episode_result.length
        obs_stats = episode_result.obs_stats
        reward_stats = episode_result.reward_stats
        reward_trace = episode_result.reward_trace
        
        epsilon = linear_epsilon_schedule(start_e, end_e, exploration_fraction * total_timesteps, current_timestep)
        print(f"Ep: {i}: Epsiodic Return: {episode_result.reward_:.1f}. Percent elapsed: {100 * current_timestep / total_timesteps:2.2f}%. Epsilon: {epsilon:.2f}")
        i += 1
    
    return q_network

if __name__ == "__main__":
    # Example usage
    
    key = jax_random.key(1)
    key, key_reset, key_act, key_step = jax_random.split(key, 4)

    # Instantiate the environment & its settings.
    env, env_params = make("CartPole-v1")

    obs_shape = env.observation_space(env_params).shape[0]
    num_actions = env.action_space(env_params).n
    hidden_layer_sizes = [32, 32]  # Example hidden layer sizes
    q_network = QNetwork(obs_shape, hidden_layer_sizes, num_actions, key_reset)

    # Run the stream Q-learning algorithm
    q_network = stream_q(
        q_network,
        env,
        env_params,
        gamma=0.99,
        lambda_=0.8,
        alpha=1.0,
        kappa=2.0,
        start_e=1.0,
        end_e=0.01,
        exploration_fraction=0.5,
        total_timesteps=50_000,
        key=key_act
    )

    print([x for x in jt.leaves(q_network)])

