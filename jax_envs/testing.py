import jax 
import jax.numpy as jnp
import numpy as np
from ant_maze import AntMaze
from brax.envs.wrappers.gym import GymWrapper, VectorGymWrapper
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import os
from brax.envs.wrappers.training import VmapWrapper
from stable_baselines3.common.monitor import Monitor
from gym.spaces import Box
from src.env.subproc_vec_env import BraxGymDummyVecEnv
os.environ["JAX_PLATFORM_NAME"] = "cpu" # all brax environments should be on cpu, this is what the codebase expect!


def make_brax_gym_vec_env(
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
    env_kwargs = env_kwargs or {}
    vec_env_kwargs = vec_env_kwargs or {}
    monitor_kwargs = monitor_kwargs or {}
    wrapper_kwargs = wrapper_kwargs or {}
    assert vec_env_kwargs is not None  # for mypy

    def make_env(rank: int):
        def _init():
            # For type checker:
            assert monitor_kwargs is not None
            assert wrapper_kwargs is not None
            assert env_kwargs is not None
                # if the render mode was not specified, we set it to `rgb_array` as default.
            kwargs = {"render_mode": "rgb_array"}
            kwargs.update(env_kwargs)
            try:
                env = AntMaze(backend="spring", maze_layout_name=env_layout)
                env = VmapWrapper(env, 16)
                env = VectorGymWrapper(env)
            except TypeError:
                env = AntMaze(backend="spring", maze_layout_name=env_layout)
                env = VmapWrapper(env, 16)
                env = VectorGymWrapper(env)
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
    import pdb;pdb.set_trace()
    env = make_env(0)()
    vec_env = BraxGymDummyVecEnv(env, 16)
    
    # Prepare the seeds for the first reset
    # vec_env.seed(seed)
    return vec_env



if __name__ == "__main__":
    env_layout = "big_maze"
    make_brax_gym_vec_env(None)
    # env = AntMaze(backend="spring", maze_layout_name=env_layout)
    # rng = jax.random.key(1)
    # env = GymWrapper(env)
    import pdb;pdb.set_trace()


