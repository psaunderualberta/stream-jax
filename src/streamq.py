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


jax.config.update('jax_default_device', jax.devices('cpu')[0])


@jit
def get_delta(q_network, transition: Transition, gamma: float):
    q_sp = q_network(transition.next_obs).max()
    q_sa = q_network(transition.obs)[transition.action]
    return (
        transition.reward
        + (1 - transition.done) * jax_lax.stop_gradient(gamma * q_sp)
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


@struct.dataclass
class StreamQTrainState:
    q_network: QNetwork
    z_w: Any
    obs_stats: SampleMeanStats
    reward_stats: SampleMeanStats
    reward_trace: float
    state_transition: Transition
    global_timestep: int
    rng_ts: chex.PRNGKey

    @classmethod
    def create(
        cls,
        q_network: QNetwork,
        obs_stats: SampleMeanStats,
        reward_stats: SampleMeanStats,
        state_transition: Transition,
        global_timestep: int,
        rng_ts: chex.PRNGKey,
    ) -> 'StreamQTrainState':
        """Create a new StreamQTrainState."""
        obs, obs_stats = normalize_observation(state_transition.obs, obs_stats)
        state_transition = state_transition.replace(obs=obs)

        return cls(
            q_network=q_network,
            z_w=init_eligibility_trace(q_network),
            obs_stats=obs_stats,
            reward_stats=reward_stats,
            reward_trace=0.0,
            state_transition=state_transition,
            global_timestep=global_timestep,
            rng_ts=rng_ts
        )

    def start_of_episode(
        self,
        env: environment.Environment,
        env_params: environment.EnvParams,
        reset_rng: chex.PRNGKey
    ) -> 'StreamQTrainState':
        
        init_transition = Transition.initial_transition(
            env, env_params, reset_rng
        )

        return StreamQTrainState.create(
            q_network=self.q_network,
            obs_stats=self.obs_stats,
            reward_stats=self.reward_stats,
            state_transition=init_transition,
            global_timestep=self.global_timestep,
            rng_ts=self.rng_ts
        )

    def next_iteration(self) -> 'StreamQTrainState':
        """Prepare the state for the next iteration."""
        return self.replace(
            global_timestep=self.global_timestep + 1,
            rng_ts=jax_random.split(self.rng_ts)[1],
            state_transition=self.state_transition.step_transition()
        )

    def normalize_transition(self, gamma: float) -> 'StreamQTrainState':
        """Normalize the next observation and reward in the transition."""
        next_obs, obs_stats = normalize_observation(
            self.state_transition.next_obs, self.obs_stats
        )
        scaled_reward, reward_trace, reward_stats = scale_reward(
            self.state_transition.reward,
            self.reward_stats,
            self.reward_trace,
            self.state_transition.done,
            gamma
        )

        new_transition = self.state_transition.replace(
            next_obs=next_obs,
            reward=scaled_reward,
        )

        return StreamQTrainState(
            q_network=self.q_network,
            z_w=self.z_w,
            obs_stats=obs_stats,
            reward_stats=reward_stats,
            reward_trace=reward_trace,
            state_transition=new_transition,
            global_timestep=self.global_timestep,
            rng_ts=self.rng_ts
        )
    
    def new_transition(self, env, env_params, action: chex.Array, rng: chex.PRNGKey) -> 'StreamQTrainState':
        """Create a new transition object based on the current state."""
        next_transition = self.state_transition.populate_transition(env, env_params, action, rng)
        return self.replace(state_transition=next_transition)


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
    eval_callback: Any = struct.field(pytree_node=False, default=lambda *args: None)

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
    
    def make_act(self, train_state: StreamQTrainState, rng: chex.PRNGKey) -> Any:
        pass
    
    def train(self, rng: chex.PRNGKey) -> StreamQTrainState:
        def train_iteration(ts: StreamQTrainState):
            rng, rng_action = jax_random.split(ts.rng_ts)

            # Select an action using epsilon-greedy policy.
            eps = linear_epsilon_schedule(self.start_e, self.end_e, self.stop_exploring_timestep, ts.global_timestep)
            action, q_value, explored = q_epsilon_greedy(ts.q_network, ts.state_transition.obs, eps, rng_action)

            # Step the environment, normalize observation & reward
            rng, rng_step = jax_random.split(rng)
            next_ts = ts.new_transition(
                self.env, self.env_params, action, rng_step
            ).normalize_transition(self.gamma)

            # Update eligibility trace
            td_error, td_grad = value_and_grad(get_delta)(ts.q_network, next_ts.state_transition, self.gamma)
            z_w = update_eligibility_trace(ts.z_w, self.gamma, self.lambda_, td_grad)

            # Update Q-network using ObGD
            q_network = ObGD(ts.z_w, ts.q_network, td_error, self.alpha, self.kappa)

            # reset eligibility trace if an exploration occurred
            done = next_ts.state_transition.done
            z_w = pytree_if_else(explored, init_eligibility_trace(q_network), z_w)

            # move to next training iteration
            rng, rng_reset = jax_random.split(rng)
            next_ts = next_ts.replace(
                q_network=q_network,
                z_w=z_w,
                rng_ts=rng,
            ).next_iteration()

            reset_ts = next_ts.start_of_episode(
                self.env, self.env_params, rng_reset
            )

            # If the episode is done, reset the state transition
            return pytree_if_else(done, reset_ts, next_ts, is_leaf=is_none)

        @eqx.filter_jit
        def eval_iteration(ts: StreamQTrainState):
            eval_result = jax_lax.fori_loop(
                0,
                self.eval_freq,
                lambda _, ts: train_iteration(ts),
                ts,
            )

            return eval_result, self.eval_callback(self, eval_result, ts.rng_ts)

        rng, rng_reset, rng_ts = jax_random.split(rng, 3)
        obs_shape = env.observation_space(self.env_params).shape
        train_state = StreamQTrainState.create(
            q_network=self.q_network,
            obs_stats=SampleMeanStats.new_params(obs_shape),
            reward_stats=SampleMeanStats.new_params(()),
            state_transition=Transition.initial_transition(
                env, env_params, rng_reset
            ),
            global_timestep=0,
            rng_ts=rng_ts
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

    def eval_callback(algo: StreamQ, ts: StreamQTrainState, rng: chex.PRNGKey):
        q = ts.q_network
        state = ts.state_transition.initial_transition(algo.env, algo.env_params, rng)
        obs, _ = normalize_observation(state.obs, ts.obs_stats)
        state = state.replace(obs=obs)

        def loop_body(transition: Transition):
            action = jnp.argmax(q(transition.obs), axis=-1)
            next_obs, next_state, reward, done, _ = algo.env.step(rng, transition.state, action, algo.env_params)
            next_obs, _ = normalize_observation(next_obs, ts.obs_stats)

            return Transition(
                obs=next_obs,
                state=next_state,
                reward=transition.reward * algo.gamma + reward,
                done=done,
                has_next_state=False
            )

        trans = jax_lax.while_loop(
            lambda t: jnp.logical_not(t.done),
            loop_body,
            state
        )

        time_elapsed = ts.global_timestep / algo.total_timesteps * 100
        jax.debug.print(
            "Episodic Return: {:.1f}. Percent elapsed: {:2.2f}%. Epsilon: {:.2f}",
            trans.reward,
            time_elapsed,
            linear_epsilon_schedule(
                algo.start_e, algo.end_e, algo.stop_exploring_timestep, ts.global_timestep
            )
        )
        return trans.reward


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
        stop_exploring_timestep=25_000,
        total_timesteps=50_000,
        eval_freq=1000,
        eval_callback=eval_callback
    ).train(key_act)



# def stream_q(
#     q_network: QNetwork,
#     env: environment.Environment,
#     env_params: environment.EnvParams,
#     gamma: float,
#     lambda_: float,
#     alpha: float,
#     kappa: float,
#     start_e: float,
#     end_e: float,
#     stop_exploring_timestep: float,
#     total_timesteps: int,
#     key: chex.PRNGKey,
# ):

#     class LoopState(eqx.Module):
#         key: chex.PRNGKey
#         q_network: QNetwork
#         done: bool
#         obs: chex.Array
#         state: Any
#         z_w: Any
#         reward_: float
#         reward_trace: float
#         obs_stats: SampleMeanStats
#         reward_stats: SampleMeanStats
#         length: int
#         global_timestep: int

#     def while_loop_body(carry: LoopState):
#         # extract carry elements
#         key = carry.key
#         obs, state = carry.obs, carry.state
#         q_network, z_w = carry.q_network, carry.z_w
#         obs_stats = carry.obs_stats
#         reward_stats = carry.reward_stats
#         reward_trace = carry.reward_trace
#         global_timestep = carry.global_timestep + 1

#         key, action_key = jax_random.split(key)

#         # Select an action using epsilon-greedy policy.
#         eps = linear_epsilon_schedule(start_e, end_e, stop_exploring_timestep, global_timestep)
#         action, q_value, explored = q_epsilon_greedy(q_network, obs, eps, action_key)

#         # Step the environment.
#         key, step_key = jax_random.split(key)
#         next_obs, next_state, reward, done, _ = env.step(step_key, state, action, env_params)

#         # normalize observation & reward
#         next_obs, obs_stats = normalize_observation(next_obs, obs_stats)
#         scaled_reward, reward_trace, reward_stats = scale_reward(reward, reward_stats, reward_trace, done, gamma)

#         # Update eligibility trace
#         td_error, td_grad = value_and_grad(get_delta)(q_network, scaled_reward, gamma, done, obs, action, next_obs)
#         z_w = update_eligibility_trace(z_w, gamma, lambda_, td_grad)

#         # Update Q-network using ObGD
#         q_network = ObGD(z_w, q_network, td_error, alpha, kappa)

#         # reset eligibility trace if an exploration occurred
#         z_w = jt.map(lambda old: jax_lax.select(
#                 jnp.logical_or(explored, done),
#                 jnp.zeros_like(old),
#                 old
#             ), z_w
#         )


#         return LoopState(
#             key=key,
#             done=done,
#             obs=next_obs,
#             state=next_state,
#             z_w=z_w,
#             q_network=q_network,
#             reward_=carry.reward_ * gamma + reward,
#             reward_trace=reward_trace,
#             global_timestep=global_timestep,
#             length=carry.length + 1,
#             obs_stats=obs_stats,
#             reward_stats=reward_stats,
#         )

#     @eqx.filter_jit
#     def jitted_while_loop(episode_state: LoopState):
#         return jax_lax.while_loop(
#             lambda carry: jnp.logical_not(carry.done),
#             while_loop_body,
#             episode_state
#         )

#     def non_jitted_while_loop(episode_state: LoopState):
#         while not episode_state.done:
#             episode_state = while_loop_body(episode_state)
#         return episode_state

#     i = 0
#     current_timestep = 0
#     key, reset_key = jax_random.split(key)
#     obs, state = env.reset(key, env_params)
#     obs_stats = SampleMeanStats.new_params(obs.shape)
#     obs, obs_stats = normalize_observation(obs, obs_stats)
#     reward_stats = SampleMeanStats.new_params(())
#     while current_timestep < total_timesteps:
#         initial_loop_state = LoopState(
#             key=key,
#             done=False,
#             obs=obs,
#             state=state,
#             z_w=init_eligibility_trace(q_network),
#             q_network=q_network,
#             reward_=0.0,
#             reward_trace=0.0,
#             global_timestep=current_timestep,
#             length=0.0,
#             obs_stats=obs_stats,
#             reward_stats=reward_stats,
#         )

#         # episode_result = non_jitted_while_loop(initial_loop_state)
#         episode_result = jitted_while_loop(initial_loop_state)

#         # extract relevant info from environment, like PRNG key
#         key = episode_result.key
#         q_network = episode_result.q_network
#         current_timestep += episode_result.length
#         obs_stats = episode_result.obs_stats
#         reward_stats = episode_result.reward_stats

#         # reset environment
#         key, reset_key = jax_random.split(key)
#         obs, state = env.reset(reset_key, env_params)
#         obs, obs_stats = normalize_observation(obs, obs_stats)

#         epsilon = linear_epsilon_schedule(start_e, end_e, stop_exploring_timestep, current_timestep)
#         print(f"Ep: {i}. Episodic Return: {episode_result.reward_:.1f}. Percent elapsed: {100 * current_timestep / total_timesteps:2.2f}%. Epsilon: {epsilon:.2f}")

#         i += 1

#     return q_network
