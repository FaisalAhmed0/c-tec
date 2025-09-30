import gym as old_gym
from gym.vector import SyncVectorEnv
from gym.spaces import Box


# from stable_baselines3.common.env_util import make_vec_env

import gymnasium_robotics
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
import os
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import numpy as np
from gymnasium import Wrapper

# from gymnasium.vector import SyncVectorEnv


from gymnasium.wrappers import FilterObservation
from gymnasium.wrappers import FlattenObservation
# import gymnasi
def make_vec_env(
    env_id,
    n_envs: int = 1,
    seed = None,
    start_index = 0,
    monitor_dir = None,
    wrapper_class = None,
    env_kwargs= None,
    vec_env_cls= None,
    vec_env_kwargs = None,
    monitor_kwargs = None,
    wrapper_kwargs = None,
) :
    """
    Create a wrapped, monitored ``VecEnv``.
    By default it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.

    :param env_id: either the env ID, the env class or a callable returning an env
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
        Note: the wrapper specified by this parameter will be applied after the ``Monitor`` wrapper.
        if some cases (e.g. with TimeLimit wrapper) this can lead to undesired behavior.
        See here for more details: https://github.com/DLR-RM/stable-baselines3/issues/894
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :param wrapper_kwargs: Keyword arguments to pass to the ``Wrapper`` class constructor.
    :return: The wrapped environment
    """
    env_kwargs = env_kwargs or {}
    vec_env_kwargs = vec_env_kwargs or {}
    monitor_kwargs = monitor_kwargs or {}
    wrapper_kwargs = wrapper_kwargs or {}
    assert vec_env_kwargs is not None  # for mypy

    def make_env(rank: int):
        def _init() -> gym.Env:
            # For type checker:
            assert monitor_kwargs is not None
            assert wrapper_kwargs is not None
            assert env_kwargs is not None
                # if the render mode was not specified, we set it to `rgb_array` as default.
            kwargs = {"render_mode": "rgb_array"}
            kwargs.update(env_kwargs)
            try:
                env = gym.make(env_id, **kwargs)  # type: ignore[arg-type]
                env = FilterObservation(env, filter_keys=["observation", "achieved_goal"])
                env = FlattenObservation(env)
                env = GymnasiumToGymWrapper(env)
            except TypeError:
                env = gym.make(env_id, **env_kwargs)
                env = FilterObservation(env, filter_keys=["observation", "achieved_goal"])
                env = FlattenObservation(env)
            if seed is not None:
                # Note: here we only seed the action space
                # We will seed the env at the next reset
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None and monitor_dir is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    vec_env = SubprocVecEnv([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)
    # Prepare the seeds for the first reset
    # vec_env.seed(seed)
    return vec_env


class GymnasiumToGymWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.old_observation_space = env.observation_space
        self.old_action_space = env.action_space
        self.env = env
        self.observation_space = Box(self.old_observation_space.low, self.old_observation_space.high, shape=self.observation_space.shape, dtype=self.old_observation_space.dtype)
        self.action_space = Box(self.old_action_space.low, self.old_action_space.high, shape=self.old_action_space.shape, dtype=self.old_action_space.dtype)

    def reset(self, **kwargs):
        """Gym pre-0.26 compatible reset (returns just obs)"""
        obs, _ = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        """Gym pre-0.26 compatible step (returns obs, reward, done, info)"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info
    
    def seed(self, *args):
        return self.env.seed(*args)

if __name__ == "__main__":
    def make_Env():
        def create_env():
            env = gym.make("AntMaze_Large-v3", render_mode="rgb_array")
            env = FilterObservation(env, filter_keys=["observation", "achieved_goal"])
            env = FlattenObservation(env)
            env = GymnasiumToGymWrapper(env)
            return env
        return create_env
    # import pdb;pdb.set_trace()
    venv = make_vec_env("AntMaze_Large-v3", n_envs=8, seed=1)
    # venv = SyncVectorEnv([make_Env() for _ in range(8)])
    import pdb;pdb.set_trace()
    
    
