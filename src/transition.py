from gymnax.environments import EnvState, EnvState, environment
import chex
import equinox as eqx
from util import is_none

class Transition(eqx.Module):
    obs: chex.Array
    state: EnvState
    action: chex.Array
    next_obs: chex.Array
    next_state: EnvState
    reward: float
    done: bool
    has_next_state: bool

    def __init__(
        self,
        obs: chex.Array = None,
        state: EnvState = None,
        action: chex.Array = None,
        next_obs: chex.Array = None,
        next_state: EnvState = None,
        reward: float = 0.0,
        done: bool = False,
        has_next_state: bool = False
    ):
        self.obs = obs
        self.state = state
        self.action = action
        self.next_obs = next_obs
        self.next_state = next_state
        self.reward = reward
        self.done = done
        self.has_next_state = has_next_state

    def replace(self, **kwargs) -> 'Transition':
        """Replace attributes in the transition with new values."""

        els = list(kwargs.items())

        return eqx.tree_at(
            lambda t: tuple(getattr(t, k) for k, _ in els),
            self,
            tuple(v for _, v in els),
            is_leaf=is_none,
        )

    def take_action(
        self,
        env: environment.Environment,
        env_params: environment.EnvParams,
        action: chex.Array,
        rng: chex.PRNGKey
    ) -> 'Transition':
        """Create a new transition object based on the current state."""
        next_obs, next_state, reward, done, _ = env.step(rng, self.state, action, env_params)

        return Transition(
            obs=self.obs,
            state=self.state,
            action=action,
            next_obs=next_obs,
            next_state=next_state,
            reward=reward,
            done=done,
            has_next_state=True
        )
    
    def step_transition(self) -> 'Transition':
        """Step the transition to the next state."""
        return Transition(
            obs=self.next_obs,
            state=self.next_state,
            has_next_state=False,
        )


    @classmethod
    def initial_transition(
        cls,
        env: environment.Environment,
        env_params: environment.EnvParams,
        rng: chex.PRNGKey
    ) -> 'Transition':
        """Create an initial transition object for the start of an episode."""
        obs, state = env.reset(rng, env_params)
        
        return cls(
            obs=obs,
            state=state,
            has_next_state=False,
        )
