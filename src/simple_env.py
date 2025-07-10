"""JAX implementation of CartPole-v1 OpenAI gym environment."""

from typing import Any

import jax
import jax.numpy as jnp
import equinox as eqx
from gymnax.environments import environment, spaces


class RightIsGoodState(eqx.Module, environment.EnvState):
    x: jax.Array
    time: int


class RightIsGoodParams(eqx.Module, environment.EnvParams):
    init_range: int = 100
    max_steps_in_episode: int = 500  # v0 had only 200 steps!


class RightIsGoodEnv(environment.Environment[RightIsGoodState, RightIsGoodParams]):
    """Super Shrimple Environment
    Move right -> reward = +1
    Move left -> reward = -1
    """

    def __init__(self):
        super().__init__()
        self.obs_shape = (1,)

    @property
    def default_params(self) -> RightIsGoodParams:
        # Default environment parameters for CartPole-v1
        return RightIsGoodParams()

    def step_env(
        self,
        key: jax.Array,
        state: RightIsGoodState,
        action: int | float | jax.Array,
        params: RightIsGoodParams,
    ) -> tuple[jax.Array, RightIsGoodState, jax.Array, jax.Array, dict[Any, Any]]:
        """Performs step transitions in the environment."""

        # map action: [0, 1] -> [-1, 1]
        reward = 2 * action - 1
        new_x = state.x + reward

        # Update state dict and evaluate termination conditions
        state = RightIsGoodState(
            x=new_x.astype(state.x.dtype),
            time=state.time + 1,
        )

        done = self.is_terminal(state, params)

        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            jnp.array(reward),
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: jax.Array, params: RightIsGoodParams
    ) -> tuple[jax.Array, RightIsGoodState]:
        """Performs resetting of environment."""
        init_state = jax.random.randint(key, (1,), -params.init_range, params.init_range+1)
        state = RightIsGoodState(
            x=init_state,
            time=0,
        )
        return self.get_obs(state), state

    def get_obs(self, state: RightIsGoodState, params=None, key=None) -> jax.Array:
        """Applies observation function to state."""
        return state.x

    def is_terminal(self, state: RightIsGoodState, params: RightIsGoodParams) -> jax.Array:
        """Check whether state is terminal."""
        # Check termination criteria
        return state.time > params.max_steps_in_episode

    @property
    def name(self) -> str:
        """Environment name."""
        return "RightIsGood-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: RightIsGoodParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(2)

    def observation_space(self, params: RightIsGoodParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(-jnp.inf, jnp.inf, (1,), dtype=jnp.int32)

    def state_space(self, params: RightIsGoodParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "x": spaces.Box(-jnp.inf, jnp.inf, (), jnp.int32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )