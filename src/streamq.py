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
from gymnax.environments import environment
from gymnax import make
from typing import Any
from flax import struct
from streamx import StreamXAlgorithm, StreamXTrainState

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
    
    def num_actions(self):
        return self.layers[-1].weight.shape[0]


@struct.dataclass
class StreamQState(StreamXTrainState):
    ### stream-q specific non-default arguments
    net: QNetwork = struct.field(pytree_node=True)

    ### stream-q specific default arguments
    obs: chex.Array = struct.field(pytree_node=False, default=None)
    state: environment.EnvState = struct.field(pytree_node=False, default=None)
    next_obs: chex.Array = struct.field(pytree_node=False, default=None)
    next_state: environment.EnvState = struct.field(pytree_node=False, default=None)
    action: int = struct.field(pytree_node=False, default=0)
    explored: bool = struct.field(pytree_node=False, default=False)
    reward: float = struct.field(pytree_node=False, default=0.0)
    done: bool = struct.field(pytree_node=False, default=False)

    # eps-greedy
    start_eps: float = struct.field(pytree_node=False, default=1.0)
    end_eps: float = struct.field(pytree_node=False, default=0.01)
    stop_exploring_timestep: float = struct.field(pytree_node=False, default=0.05)
    delta: float = struct.field(pytree_node=False, default=0)
    td_grad: QNetwork = struct.field(pytree_node=False, default=None)
    zw: QNetwork = struct.field(pytree_node=True, default=None)

    ### Default arguments common 
    current_timestep: int = struct.field(pytree_node=True, default=0)
    current_episode_num: int = struct.field(pytree_node=True, default=0)
    reward_stats: SampleMeanStats = struct.field(pytree_node=False, default=SampleMeanStats.new_params(()))
    obs_stats: SampleMeanStats = struct.field(pytree_node=False, default=SampleMeanStats.new_params(()))
    reward_trace: float = struct.field(pytree_node=False, default=0.0)
    eval_callback: callable = struct.field(pytree_node=False, default=None)
    episode_length: int = struct.field(pytree_node=False, default=0)

    learning_mode: bool = struct.field(pytree_node=False, default=False)
    key: chex.PRNGKey = struct.field(pytree_node=False, default=jax_random.PRNGKey(0))



class StreamQ(StreamXAlgorithm):
    def pre_episode_initialization(self, ts: StreamQState, obs: chex.Array, state: environment.EnvState):
        return ts.replace(
            zw = init_eligibility_trace(ts.net),
            done=False,
            reward_trace=0.0,
            episode_length=0,
        )

    def get_action(self, ts: StreamQState):
        key, eps_key, action_key = jax_random.split(ts.key, 3)
        ts = ts.replace(key=key)

        q_values = q_network(ts.state)
        greedy_action = jnp.argmax(q_values, axis=-1)

        eps = linear_epsilon_schedule(ts.start_eps, ts.end_eps, ts.stop_exploring_timestep, ts.current_timestep)
        explore = jnp.logical_and(
            jnp.logical_not(ts.learning_mode),  # not in learning mode i.e. 
            jax_random.uniform(eps_key) < eps,  # TODO: eplison calculator
        )
        action = jax_lax.select(
            explore,
            jax_random.randint(action_key, (), 0, q_network.num_actions()),
            greedy_action
        )

        q_value = q_values[action]
        explored = action != greedy_action
        return action, q_value, explored
    
    def update_env_step_outputs(
            self,
            ts: StreamQState,
            next_obs: chex.Array,
            next_state: environment.EnvState,
            reward: float,
            done: bool,
            info: dict[Any, Any]
        ):
        return ts.replace(
            next_obs=next_obs,
            next_state=next_state,
            reward=reward,
            done=done,
        )
    
    def normalize_observation(self, ts: StreamQState):
        obs, obs_stats = normalize_observation(ts.obs, ts.obs_stats)
        return ts.replace(
            obs=obs,
            obs_stats=obs_stats,
        )

    def scale_reward(self, ts: StreamQState):
        reward, reward_stats = scale_reward(ts.reward, ts.reward_stats)
        return ts.replace(
            reward=reward,
            reward_stats=reward_stats
        )

    def get_delta_and_traces(self, ts: StreamQState, gamma: float):
        q_sp = ts.net(ts.next_obs).max()
        def get_delta(q, s):
            q_sa = q(s)[ts.action]
            return (
                ts.reward
                + (1 - ts.done) * jax_lax.stop_gradient(gamma * q_sp)
                - q_sa
            )

        td_error, td_grad = value_and_grad(get_delta)(ts.net, ts.obs)
        return ts.replace(
            delta=td_error,
            td_grad=td_grad,
        )

    def update_eligibility_traces(self, ts: StreamQState, gamma: float, lambda_: float):
        new_zw = update_eligibility_trace(ts.zw, gamma, lambda_, ts.td_grad)
        return ts.replace(zw=new_zw)

    def update_weights(self, ts: StreamQState, alpha: float, kappa: float):
        new_net = ObGD(ts.zw, ts.net, ts.delta, alpha, kappa)
        return ts.replace(net=new_net)
    
    def next_timestep(self, ts: StreamQState):
        return ts.replace(current_timestep=ts.current_timestep + 1)
    
    def increment_episode_num(self, ts: StreamQState):
        return ts.replace(current_episode_num=ts.current_episode_num + 1)

    def post_weight_update_hook(self, ts: StreamQState) -> StreamQState:
        new_zw = jt.map(lambda old: jax_lax.select(
                jnp.logical_or(ts.explored, ts.done),
                jnp.zeros_like(old),
                old
            ), ts.zw
        )

        return ts.replace(
            zw=new_zw
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


@jit
def get_delta(q_network, reward, gamma, done, s, a, sp):
    q_sp = q_network(sp).max()
    q_sa = q_network(s)[a]
    return (
        reward
        + (1 - done) * jax_lax.stop_gradient(gamma * q_sp)
        - q_sa
    )


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
    stop_exploring_timestep: float,
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
        key = carry.key
        obs, state = carry.obs, carry.state
        q_network, z_w = carry.q_network, carry.z_w
        obs_stats = carry.obs_stats
        reward_stats = carry.reward_stats
        reward_trace = carry.reward_trace
        global_timestep = carry.global_timestep + 1

        key, action_key = jax_random.split(key)

        # Select an action using epsilon-greedy policy.
        eps = linear_epsilon_schedule(start_e, end_e, stop_exploring_timestep, global_timestep)
        action, q_value, explored = q_epsilon_greedy(q_network, obs, eps, action_key)

        # Step the environment.
        key, step_key = jax_random.split(key)
        next_obs, next_state, reward, done, _ = env.step(step_key, state, action, env_params)

        # normalize observation & reward
        next_obs, obs_stats = normalize_observation(next_obs, obs_stats)
        scaled_reward, reward_trace, reward_stats = scale_reward(reward, reward_stats, reward_trace, done, gamma)

        # Update eligibility trace
        td_error, td_grad = value_and_grad(get_delta)(q_network, scaled_reward, gamma, done, obs, action, next_obs)
        z_w = update_eligibility_trace(z_w, gamma, lambda_, td_grad)

        # Update Q-network using ObGD
        q_network = ObGD(z_w, q_network, td_error, alpha, kappa)

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
            reward_=carry.reward_ * gamma + reward,
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
    key, reset_key = jax_random.split(key)
    obs, state = env.reset(key, env_params)
    obs_stats = SampleMeanStats.new_params(obs.shape)
    obs, obs_stats = normalize_observation(obs, obs_stats)
    reward_stats = SampleMeanStats.new_params(())
    while current_timestep < total_timesteps:
        initial_loop_state = LoopState(
            key=key,
            done=False,
            obs=obs,
            state=state,
            z_w=init_eligibility_trace(q_network),
            q_network=q_network,
            reward_=0.0,
            reward_trace=0.0,
            global_timestep=current_timestep,
            length=0.0,
            obs_stats=obs_stats,
            reward_stats=reward_stats,
        )

        # episode_result = non_jitted_while_loop(initial_loop_state)
        episode_result = jitted_while_loop(initial_loop_state)

        # extract relevant info from environment, like PRNG key
        key = episode_result.key
        q_network = episode_result.q_network
        current_timestep += episode_result.length
        obs_stats = episode_result.obs_stats
        reward_stats = episode_result.reward_stats

        # reset environment
        key, reset_key = jax_random.split(key)
        obs, state = env.reset(reset_key, env_params)
        obs, obs_stats = normalize_observation(obs, obs_stats)

        epsilon = linear_epsilon_schedule(start_e, end_e, stop_exploring_timestep, current_timestep)
        print(f"Ep: {i}. Episodic Return: {episode_result.reward_:.1f}. Percent elapsed: {100 * current_timestep / total_timesteps:2.2f}%. Epsilon: {epsilon:.2f}")

        i += 1

    return q_network

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

    # Run the stream Q-learning algorithm
    # q_network = stream_q(
    #     q_network,
    #     env,
    #     env_params,
    #     gamma=0.99,
    #     lambda_=0.8,
    #     alpha=1.0,
    #     kappa=2.0,
    #     start_e=1.0,
    #     end_e=0.2,
    #     stop_exploring_timestep=2_000_000,
    #     total_timesteps=4_000_000,
    #     key=key_act
    # )

    learning_time = 50_000
    algo = StreamQ(
        gamma=0.99,
        lambda_=0.8,
        alpha=1.0,
        kappa=2.0,
        max_learning_timesteps=50_000
    )

    algo_state = StreamQState(
        net=q_network,
        start_eps=1.0,
        end_eps=0.01,
        stop_exploring_timestep=0.5 * learning_time
    )

    algo.non_jitted_train(
        key_act,
        algo_state,
        env,
        env_params
    )

    print(algo)
