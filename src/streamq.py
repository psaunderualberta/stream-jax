import equinox as eqx
from jax import numpy as jnp, random as jax_random, tree as jt, jit, lax as jax_lax
import chex
from util import ReLU, Linear, is_none, update_eligibility_trace, ObGD, init_eligibility_trace, normalize_observation, scale_reward
from gymnax.environments import environment, spaces
from gymnax import make
from typing import Any
from visualizer import visualize


class QNetwork(eqx.Module):
    layers: list[chex.Array]
    activation: eqx.Module

    def __init__(
            self,
            obs_shape: int,
            hidden_layer_sizes: list[int],
            num_actions: int,
            key: chex.PRNGKey,
            activation: eqx.Module = ReLU()
        ):
        self.layers = []
        self.activation = activation
        in_size = obs_shape
        for size in hidden_layer_sizes:
            # Add a linear layer
            layer = Linear(in_size, size, key=key)
            self.layers.append(layer)
            # Add layer norm
            layer_norm = eqx.nn.LayerNorm(size, use_weight=False, use_bias=False)
            self.layers.append(layer_norm)
            # Add activation function
            self.layers.append(activation)
            in_size = size

        # Final output layer
        output_layer = Linear(in_size, num_actions, key=key)
        self.layers.append(output_layer)
        print(self.layers)
    
    @classmethod
    def from_architecture(arch: "QNetwork", key: chex.PRNGKey):
        """Create a new QNetwork with the same architecture as *arch* but with new parameters."""
        input_size = arch.layers[0].in_features
        hidden_layer_sizes = []
        for layer in arch.layers[:-1]:  # Exclude the last layer (output layer)
            hidden_layer_sizes.append(layer.out_features)
        
        num_actions = arch.num_actions()
        return QNetwork(input_size, hidden_layer_sizes, num_actions, key, activation=arch.activation)
        
    @eqx.filter_jit
    def __call__(self, state):
        x = state
        for layer in self.layers:
            x = layer(x)
        return x
    
    def num_actions(self):
        return self.layers[-1].weight.shape[1]

    @eqx.filter_jit
    def get_action(self, state):
        return jnp.argmax(self(state), axis=-1)


@eqx.filter_jit
def get_td_error_grad(q_network, reward, gamma, done, s, a, sp):
    qsp = jax_lax.stop_gradient(jnp.max(q_network(sp), axis=-1))
    qsa = q_network(s)[a]
    return (
        reward
        + gamma * jax_lax.select(done, jnp.zeros_like(qsp), qsp)
        - qsa
    )


@eqx.filter_jit
def _q_state_value(q_network, x):
    """Compute the state value from the Q-values."""
    q_values = q_network(x)
    return jnp.max(q_values, axis=-1)

@eqx.filter_jit
def q_epsilon_greedy(q_network, state, epsilon: float, key: chex.PRNGKey):
    """Select an action using epsilon-greedy policy."""
    key, eps_key, action_key = jax_random.split(key, 3)

    q_values = q_network(state)
    explore = jax_random.uniform(eps_key) < epsilon
    action = jax_lax.select(
        explore,
        jax_random.randint(action_key, (1,), 0, q_network.num_actions()).squeeze(),
        jnp.argmax(q_values, axis=-1)
    )

    q_value = q_values[action]

    return action, q_value, explore


def stream_q(
    q_network: QNetwork,
    env: environment.Environment,
    env_params: environment.EnvParams,
    gamma: float,
    lambda_: float,
    alpha: float,
    kappa: float,
    epsilon_greedy: float,
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
        p_r: float
        p_s: float
        mu_s: float
        length: int
        t: int
        u: float

    current_timestep = 0
    state_value_grad_fun = eqx.filter_grad(_q_state_value)
    def while_loop_body(carry: LoopState):
        # extract carry elements
        obs, state = carry.obs, carry.state
        q_network, z_w = carry.q_network, carry.z_w
        p_r, p_s, mu_s, t = carry.p_r, carry.p_s, carry.mu_s, carry.t
        u = carry.u
        t += 1

        key, action_key = jax_random.split(carry.key)

        # Select an action using epsilon-greedy policy.
        action, q_value, explored = q_epsilon_greedy(q_network, obs, epsilon_greedy, action_key)

        # reset eligibility trace if an exploration occurred
        z_w = jt.map(lambda new, old: jax_lax.select(
                explored,
                new,
                old
            ),
            init_eligibility_trace(q_network),
            z_w
        )

        # Step the environment.
        key, step_key = jax_random.split(key)
        next_obs, next_state, reward, done, _ = env.step(step_key, state, action, env_params)

        next_obs, mu_s, p_s = normalize_observation(next_obs, mu_s, p_s, t)
        scaled_reward, u, p_r = scale_reward(reward, gamma, u, p_r, done, t)

        # Compute td error
        next_state_value = _q_state_value(q_network, next_obs)
        td_error = (
            scaled_reward
            + gamma * jax_lax.select(done, jnp.zeros_like(next_state_value), next_state_value)
            - q_value
        )

        # Update eligibility trace
        state_value_grad = state_value_grad_fun(q_network, obs)
        z_w = update_eligibility_trace(z_w, gamma, lambda_, state_value_grad)

        # Update Q-network using ObGD
        q_network = ObGD(
            z_w,
            q_network,
            td_error,
            alpha,
            kappa
        )

        return LoopState(
            key=key,
            done=done,
            obs=next_obs,
            state=next_state,
            z_w=z_w,
            q_network=q_network,
            reward_=gamma * carry.reward_ + reward,
            length=carry.length + 1,
            p_r=p_r,
            mu_s=mu_s,
            p_s=p_s,
            t=t,
            u=u,
        )

    obs, state = env.reset(key, env_params)
    mu_s, p_s, p_r, t, u = (
        jnp.ones_like(obs),
        jnp.zeros_like(obs),
        0,
        1,
        0,
    )

    i = 0
    while current_timestep < total_timesteps:
        # reset environment
        key, reset_key = jax_random.split(key)
        obs, state = env.reset(reset_key, env_params)
        obs, mu_s, p_s = normalize_observation(obs, mu_s, p_s, t)
        
        initial_loop_state = LoopState(
            key=key,
            q_network=q_network,
            done=False,
            obs=obs,
            state=state,
            z_w=init_eligibility_trace(q_network),
            reward_=0,
            length=0,
            p_r=p_r,
            mu_s=mu_s,
            p_s=p_s,
            t=t,
            u=u,
        )

        state = initial_loop_state

        episode_result = jax_lax.while_loop(
            lambda carry: jnp.logical_not(carry.done),
            while_loop_body,
            initial_loop_state
        )

        key = episode_result.key
        q_network = episode_result.q_network
        current_timestep += episode_result.length
        mu_s, p_r, p_s, t, u = (
            episode_result.mu_s,
            episode_result.p_r,
            episode_result.p_s,
            episode_result.t,
            episode_result.u
        )
        
        print(f"Finished Episode {i} with reward: {episode_result.reward_}")
        i += 1
    
    return q_network

if __name__ == "__main__":
    # Example usage
    
    key = jax_random.key(0)
    key, key_reset, key_act, key_step = jax_random.split(key, 4)

    # Instantiate the environment & its settings.
    env, env_params = make("MountainCar-v0")

    obs_shape = env.observation_space(env_params).shape[0]
    num_actions = env.action_space(env_params).n
    hidden_layer_sizes = [8, 8]  # Example hidden layer sizes
    q_network = QNetwork(obs_shape, hidden_layer_sizes, num_actions, key_reset)

    # Run the stream Q-learning algorithm
    q_network = stream_q(
        q_network,
        env,
        env_params,
        gamma=1.0,
        lambda_=0.8,
        alpha=1.0,
        kappa=2.0,
        epsilon_greedy=0.6,
        total_timesteps=10_000,
        key=key_act
    )

    visualize(env, env_params, q_network, key)