import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any
import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any
from gymnax.environments import environment, spaces
from brax import envs
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
from envs.ant_maze import AntMaze
from envs.humanoid_maze import HumanoidMaze
from envs.simple_maze import SimpleMaze
from envs.manipulation.arm_binpick_hard import ArmBinpickHard
from envs.manipulation.arm_binpick_easy import ArmBinpickEasy



class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class BraxGymnaxWrapper:
    def __init__(self, config):
        env = create_env(config)
        # env = AntMaze(backend="spring", maze_layout_name=env_name[4:], include_goal_in_obs=False)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)

    def reset(self, key, params=None):
        state = self._env.reset(key)
        return state.obs, state

    def step(self, key, state, action, params=None):
        next_state = self._env.step(state, action)
        return next_state.obs, next_state, next_state.reward, next_state.done > 0.5, {}

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )

from gymnax.environments import spaces, environment
class AtariGymWrapper:
    def __init__(self, env):
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size)

    def reset(self, key, params=None):
        state = self._env.reset(key)
        # import pdb;pdb.set_trace()
        return state[0], state[1]

    def step(self, key, state, action, params=None):
        next_state = self._env.step(state, action)
        new_states, rew, done, info = next_state
        return new_states, rew, done, info

    def observation_space(self, params=None):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self.observation_size),
        )

    def action_space(self, params=None):
        return spaces.Discrete(self.action_size)
    


class BatchEnvWrapper(GymnaxWrapper):
    """Batches reset and step functions"""

    def __init__(self, env, num_envs: int):
        super().__init__(env)

        self.num_envs = num_envs

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, state, reward, done, info = self.step_fn(rngs, state, action, params)

        return obs, state, reward, done, info


class AutoResetEnvWrapper(GymnaxWrapper):
    """Provides standard auto-reset functionality, providing the same behaviour as Gymnax-default."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):

        rng, _rng = jax.random.split(rng)
        obs_st, state_st, reward, done, info = self._env.step(
            _rng, state, action, params
        )

        rng, _rng = jax.random.split(rng)
        obs_re, state_re = self._env.reset(_rng, params)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree_util.tree_map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return obs, state

        obs, state = auto_reset(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


class OptimisticResetVecEnvWrapper(GymnaxWrapper):
    """
    Provides efficient 'optimistic' resets.
    The wrapper also necessarily handles the batching of environment steps and resetting.
    reset_ratio: the number of environment workers per environment reset.  Higher means more efficient but a higher
    chance of duplicate resets.
    """

    def __init__(self, env, num_envs: int, reset_ratio: int):
        super().__init__(env)

        self.num_envs = num_envs
        self.reset_ratio = reset_ratio
        assert (
            num_envs % reset_ratio == 0
        ), "Reset ratio must perfectly divide num envs."
        self.num_resets = self.num_envs // reset_ratio

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs_st, state_st, reward, done, info = self.step_fn(rngs, state, action, params)

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_resets)
        obs_re, state_re = self.reset_fn(rngs, params)

        rng, _rng = jax.random.split(rng)
        reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)

        being_reset = jax.random.choice(
            _rng,
            jnp.arange(self.num_envs),
            shape=(self.num_resets,),
            p=done,
            replace=False,
        )
        reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(self.num_resets))

        obs_re = obs_re[reset_indexes]
        state_re = jax.tree_util.tree_map(lambda x: x[reset_indexes], state_re)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree_util.tree_map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return state, obs

        state, obs = jax.vmap(auto_reset)(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


@struct.dataclass
class LogEnvState:
    env_state: Any
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int
    handle: Any


class AtariLogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env, num_envs):
        super().__init__(env)
        self.num_envs = num_envs

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        obs, handle = self._env.reset(key, params)
        float_init = jnp.zeros(shape=(self.num_envs, ), dtype=jnp.float32)
        int_init = jnp.zeros(shape=(self.num_envs, ), dtype=jnp.int32)
        state = LogEnvState(obs, float_init, int_init, float_init, int_init, int_init, handle)
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float],
        params=None,
    ):
        obs, reward, done, info, new_handle = self._env.step(
            key, state.env_state, state.handle, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=obs,
            handle=new_handle,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info

class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0.0, 0, 0.0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float],
        params=None,
    ):
        obs, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=obs,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info



def create_env(config) -> object:
    env_name = config["ENV_NAME"]
    if "maze" in env_name:
        if "ant" in env_name: 
            # Possible env_name = {'ant_u_maze', 'ant_big_maze', 'ant_hardest_maze'}
            env = AntMaze(backend=config["BACKEND"] or "spring", maze_layout_name=env_name[4:], include_goal_in_obs=config["INCLUDE_GOAL_IN_OBS"])
        elif "humanoid" in env_name:
            # Possible env_name = {'humanoid_u_maze', 'humanoid_big_maze', 'humanoid_hardest_maze'}
            env = HumanoidMaze(backend=config["BACKEND"] or "spring", maze_layout_name=env_name[9:], include_goal_in_obs=config["INCLUDE_GOAL_IN_OBS"])
        else:
            # Possible env_name = {'simple_u_maze', 'simple_big_maze', 'simple_hardest_maze'}
            env = SimpleMaze(backend=config["BACKEND"]or "spring", maze_layout_name=env_name[7:])
    elif env_name == "arm_binpick_easy":
        env = ArmBinpickEasy(backend=config["BACKEND"] or "mjx", include_goal_in_obs=config["INCLUDE_GOAL_IN_OBS"])
    elif env_name == "arm_binpick_hard":
        env = ArmBinpickHard(backend=config["BACKEND"] or "mjx", include_goal_in_obs=["INCLUDE_GOAL_IN_OBS"])
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    return env