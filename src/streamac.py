import equinox as eqx
from jax import numpy as jnp, random as jax_random, tree as jt, jit, lax as jax_lax, value_and_grad, vmap, debug, jax
import chex
from util import (
    update_eligibility_trace,
    ObGD,
    init_eligibility_trace,
    normalize_observation,
    scale_reward,
    linear_epsilon_schedule,
    SampleMeanStats,
    is_none,
    pytree_if_else,
    eval_callback,
)
from gymnax.environments import environment
from gymnax import make
from typing import Any
from nets import Actor, Critic
from flax import struct
from typing import Callable


jax.config.update('jax_default_device', jax.devices('cpu')[0])


@jit
def get_delta(value_network, scaled_reward, gamma, done, obs, next_obs):
    q_sp = value_network(next_obs).squeeze()
    q_sa = value_network(obs).squeeze()
    return (
        scaled_reward
        + (1 - done) * jax_lax.stop_gradient(gamma * q_sp)
        - q_sa
    )

@jit
def compute_actor_update(actor: Actor, obs: chex.Array, action: chex.Array, tau: float, delta: float):
    return -1 * (
        actor.log_prob(obs, action) + tau * jnp.sign(delta) * actor.entropy(obs)
    ).squeeze()


class StreamACTrainState(eqx.Module):
    key: chex.PRNGKey
    done: bool
    obs: chex.PRNGKey
    state: environment.EnvState
    actor: Actor
    critic: Critic
    actor_z_w: Actor
    critic_z_w: Critic
    reward_: float
    reward_trace: float
    global_step: int
    obs_stats: SampleMeanStats
    reward_stats: SampleMeanStats
    length: int

    def replace(self, **kwargs) -> 'StreamACTrainState':
        """Replace attributes in the training state with new values, akin to flax's 'dataclass.replace'"""

        els = list(kwargs.items())
        return eqx.tree_at(
            lambda t: tuple(getattr(t, k) for k, _ in els),
            self,
            tuple(v for _, v in els),
            is_leaf=is_none,
        )


@struct.dataclass
class StreamAC:
    env: environment.Environment = struct.field(pytree_node=False)
    env_params: environment.EnvParams = struct.field(pytree_node=False)
    gamma: float = struct.field(pytree_node=False)
    lambda_: float = struct.field(pytree_node=False)
    alpha: float = struct.field(pytree_node=False)
    policy_kappa: float = struct.field(pytree_node=False)
    value_kappa: float = struct.field(pytree_node=False)
    tau: float = struct.field(pytree_node=False)
    total_timesteps: int = struct.field(pytree_node=False)
    eval_freq: int = struct.field(pytree_node=False, default=5000)
    eval_callback: Any = struct.field(pytree_node=False, default=lambda *_: None)

    @classmethod
    def create(cls, **kwargs) -> "StreamAC":

        return cls(
            env=kwargs['env'],
            env_params=kwargs['env_params'],
            gamma=kwargs['gamma'],
            lambda_=kwargs['lambda_'],
            alpha=kwargs['alpha'],
            policy_alpha=kwargs['policy_alpha'],
            value_kappa=kwargs['value_kappa'],
            tau=kwargs['tau'],
            total_timesteps=kwargs['total_timesteps'],
        )
    
    def make_act(self, train_state: StreamACTrainState) -> Callable[[chex.Array, chex.PRNGKey], int | float | chex.Array]:
        def act(obs: chex.Array, _: chex.PRNGKey):
            norm_obs, _ = normalize_observation(obs, train_state.obs_stats)
            mu, _ = train_state.actor(norm_obs)
            return mu
    
        return act

    def make_normalizer(self, train_state: StreamACTrainState) -> Callable[[chex.Array], chex.Array]:
        def normalizer(obs: chex.Array):
            norm_obs, _ = normalize_observation(obs, train_state.obs_stats)
            return norm_obs
        
        return normalizer
    
    def train(self, key: chex.PRNGKey) -> StreamACTrainState:
        @eqx.filter_jit
        def train_iteration(ts: StreamACTrainState):
            # extract carry elements
            key = ts.key
            obs, state = ts.obs, ts.state
            actor, actor_z_w = ts.actor, ts.actor_z_w
            critic, critic_z_w = ts.critic, ts.critic_z_w
            obs_stats = ts.obs_stats
            reward_stats = ts.reward_stats
            reward_trace = ts.reward_trace
            global_step = ts.global_step + 1

            key, action_key = jax_random.split(key)
            action = actor.sample(obs, action_key)

            # Step the environment.
            key, step_key = jax_random.split(key)
            next_obs, next_state, reward, done, _ = self.env.step(step_key, state, action, self.env_params)

            # normalize observation & reward
            next_obs, obs_stats = normalize_observation(next_obs, obs_stats)
            scaled_reward, reward_trace, reward_stats = scale_reward(reward, reward_stats, reward_trace, done, self.gamma)

            # Update eligibility trace of critic & actor
            td_error, td_grad = value_and_grad(get_delta)(critic, scaled_reward, self.gamma, done, obs, next_obs)
            _, actor_grad = value_and_grad(compute_actor_update)(actor, obs, action, self.tau, td_error)
            critic_z_w = update_eligibility_trace(critic_z_w, self.gamma, self.lambda_, td_grad)
            actor_z_w = update_eligibility_trace(actor_z_w, self.gamma, self.lambda_, actor_grad)

            # Update actor, critic params using ObGD
            critic = ObGD(critic_z_w, critic, td_error, self.alpha, self.value_kappa)
            actor = ObGD(actor_z_w, actor, td_error, self.alpha, self.policy_kappa)

            next_ts = ts.replace(
                key=key,
                done=done,
                obs=next_obs,
                state=next_state,
                actor=actor,
                critic=critic,
                actor_z_w=actor_z_w,
                critic_z_w=critic_z_w,
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
        def eval_iteration(ts: StreamACTrainState):
            eval_result = jax_lax.fori_loop(
                0,
                self.eval_freq,
                lambda _, ts: train_iteration(ts),
                ts,
            )

            return eval_result, self.eval_callback(self, eval_result, ts.key)

        key, key_reset, key_ts = jax_random.split(key, 3)
        obs, state = self.env.reset(key_reset, self.env_params)
        obs_stats = SampleMeanStats.new_params(obs.shape)
        obs, obs_stats = normalize_observation(obs, obs_stats)

        obs_shape = self.env.observation_space(self.env_params).shape
        hidden_layer_sizes = [32, 32]  # Example hidden layer sizes

        key, key_actor, key_critic = jax_random.split(key, 3)
        num_actions = self.env.action_space(self.env_params).shape[0]
        actor = Actor(obs_shape[0], hidden_layer_sizes, num_actions, key_actor)
        critic = Critic(obs_shape[0], hidden_layer_sizes, key_critic)

        reward_stats = SampleMeanStats.new_params(())
        train_state = StreamACTrainState(
            key=key_ts,
            done=False,
            obs=obs,
            state=state,
            actor=actor,
            critic=critic,
            actor_z_w=init_eligibility_trace(actor),
            critic_z_w=init_eligibility_trace(critic),
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
    env, env_params = make("Pendulum-v1")
    # env_params = env_params.replace(max_steps_in_episode=10_000)

    gamma = 0.99
    def evaluate(algo, ts, key):
        keys = jax_random.split(key, 3)
        rewards, lengths = vmap(eval_callback, in_axes=(None, None, 0, None))(algo, ts, keys, gamma)

        debug.print("Global Step: {}, Avg. Reward: {:.2f}, Avg Length: {:.2f}",
                    ts.global_step, jnp.mean(rewards), jnp.mean(lengths))

        return rewards, lengths


    # Run the stream Q-learning algorithm
    streamac = StreamAC(
        env,
        env_params,
        gamma=gamma,
        lambda_=0.8,
        alpha=1.0,
        policy_kappa=3.0,
        value_kappa=2.0,
        tau=0.01,
        total_timesteps=1_000_000,
        eval_freq=5000,
        eval_callback=evaluate
    ).train(key_act)
