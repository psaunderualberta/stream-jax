from gymnax.visualize import Visualizer
import jax
import jax.numpy as jnp

def visualize(env, env_params, model, key):
    state_seq, reward_seq = [], []
    key, key_reset = jax.random.split(key)
    obs, env_state = env.reset(key_reset, env_params)
    while True:
        state_seq.append(env_state)
        key, key_step = jax.random.split(key)
        action = model.get_action(obs)
        next_obs, next_env_state, reward, done, info = env.step(
            key_step, env_state, action, env_params
        )
        reward_seq.append(reward)
        if done:
            break

        obs = next_obs
        env_state = next_env_state

    cum_rewards = jnp.cumsum(jnp.array(reward_seq))
    vis = Visualizer(env, env_params, state_seq, cum_rewards)
    vis.animate(f"docs/anim.gif")