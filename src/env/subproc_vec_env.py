import warnings
from typing import Callable, List, Optional, Union, Sequence, Dict

import gym
import numpy as np
from numpy import ndarray
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env.base_vec_env import tile_images, VecEnvStepReturn
from stable_baselines3.common.vec_env.subproc_vec_env import _flatten_obs
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info
from collections import OrderedDict
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
import time
import warnings
from typing import Optional, Tuple

import numpy as np

class BraxGymDummyVecEnv(VecEnv):
    def __init__(self, envs, num_envs):
        self.envs = envs
        env_fns = []
        VecEnv.__init__(self, len(env_fns), envs.observation_space, envs.action_space)
        self.num_envs = num_envs
        obs_space = envs.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)
        self.num_envs = num_envs

        self.buf_obs = OrderedDict([(k, np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k])) for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.metadata = envs.metadata

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        # import pdb;pdb.set_trace()
        return self.envs.step(self.actions)

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        self.envs.seed(seed)
        return np.array([seed])

    def reset(self):
        return np.array(self.envs.reset())

    def close(self) -> None:
        self.envs.close()

    def get_images(self) -> Sequence[np.ndarray]:
        return None

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via ``BaseVecEnv.render()``.
        Otherwise (if ``self.num_envs == 1``), we pass the render call directly to the
        underlying environment.

        Therefore, some arguments such as ``mode`` will have values that are valid
        only when ``num_envs == 1``.

        :param mode: The rendering type.
        """
        return None
    
    def get_attr(self, attr_name: str, indices = None) :
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name: str, value, indices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices = None, **method_kwargs) :
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def env_is_wrapped(self, wrapper_class, indices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]


class CustomSubprocVecEnv(SubprocVecEnv):

    def __init__(self, 
                 env_fns: List[Callable[[], gym.Env]], 
                 start_method: Optional[str] = None):
        super().__init__(env_fns, start_method)
        self.can_see_walls = True
        self.image_noise_scale = 0.0
        self.image_rng = None  # to be initialized with run id in ppo_rollout.py

    def set_seeds(self, seeds: List[int] = None) -> List[Union[None, int]]:
        self.seeds = seeds
        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", int(seeds[idx])))
        return [remote.recv() for remote in self.remotes]

    def get_seeds(self) -> List[Union[None, int]]:
        return self.seeds

    def send_reset(self, env_id: int) -> None:
        self.remotes[env_id].send(("reset", None))

    def invisibilize_obstacles(self, obs):
        # Algorithm A5 in the Technical Appendix
        # For MiniGrid envs only
        obs = np.copy(obs)
        for r in range(len(obs[0])):
            for c in range(len(obs[0][r])):
                # The color of Walls is grey
                # See https://github.com/Farama-Foundation/gym-minigrid/blob/20384cfa59d7edb058e8dbd02e1e107afd1e245d/gym_minigrid/minigrid.py#L215-L223
                # COLOR_TO_IDX['grey']: 5
                if obs[1][r][c] == 5 and 0 <= obs[0][r][c] <= 2:
                    obs[1][r][c] = 0
                # OBJECT_TO_IDX[0,1,2]: 'unseen', 'empty', 'wall'
                if 0 <= obs[0][r][c] <= 2:
                    obs[0][r][c] = 0
        return obs

    def add_noise(self, obs):
        # Algorithm A4 in the Technical Appendix
        # Add noise to observations
        obs = obs.astype(np.float64)
        obs_noise = self.image_rng.normal(loc=0.0, scale=self.image_noise_scale, size=obs.shape)
        return obs + obs_noise

    def recv_obs(self, env_id: int) -> ndarray:
        obs = VecTransposeImage.transpose_image(self.remotes[env_id].recv())
        if not self.can_see_walls:
            obs = self.invisibilize_obstacles(obs)
        if self.image_noise_scale > 0:
            obs = self.add_noise(obs)
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs_arr, rews, dones, infos = zip(*results)
        obs_arr = _flatten_obs(obs_arr, self.observation_space).astype(np.float64)
        for idx in range(len(obs_arr)):
            if not self.can_see_walls:
                obs_arr[idx] = self.invisibilize_obstacles(obs_arr[idx])
            if self.image_noise_scale > 0:
                obs_arr[idx] = self.add_noise(obs_arr[idx])
        return obs_arr, np.stack(rews), np.stack(dones), infos

    def get_first_image(self) -> Sequence[np.ndarray]:
        for pipe in self.remotes[:1]:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(("render", "rgb_array"))
        imgs = [pipe.recv() for pipe in self.remotes[:1]]
        return imgs

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        try:
            # imgs = self.get_images()
            imgs = self.get_first_image()
        except NotImplementedError:
            warnings.warn(f"Render not defined for {self}")
            return

        # Create a big image by tiling images from subprocesses
        bigimg = tile_images(imgs[:1])
        if mode == "human":
            import cv2  # pytype:disable=import-error
            cv2.imshow("vecenv", bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == "rgb_array":
            return bigimg
        else:
            raise NotImplementedError(f"Render mode {mode} is not supported by VecEnvs")
        



class BraxGymVecMonitor(VecEnvWrapper):
    """
    A vectorized monitor wrapper for *vectorized* Gym environments,
    it is used to record the episode reward, length, time and other data.

    Some environments like `openai/procgen <https://github.com/openai/procgen>`_
    or `gym3 <https://github.com/openai/gym3>`_ directly initialize the
    vectorized environments, without giving us a chance to use the ``Monitor``
    wrapper. So this class simply does the job of the ``Monitor`` wrapper on
    a vectorized level.

    :param venv: The vectorized environment
    :param filename: the location to save a log file, can be None for no log
    :param info_keywords: extra information to log, from the information return of env.step()
    """

    def __init__(
        self,
        venv: VecEnv,
        filename: Optional[str] = None,
        info_keywords: Tuple[str, ...] = (),
    ):
        # Avoid circular import
        from stable_baselines3.common.monitor import Monitor, ResultsWriter

        # This check is not valid for special `VecEnv`
        # like the ones created by Procgen, that does follow completely
        # the `VecEnv` interface
        try:
            is_wrapped_with_monitor = venv.env_is_wrapped(Monitor)[0]
        except AttributeError:
            is_wrapped_with_monitor = False

        if is_wrapped_with_monitor:
            warnings.warn(
                "The environment is already wrapped with a `Monitor` wrapper"
                "but you are wrapping it with a `VecMonitor` wrapper, the `Monitor` statistics will be"
                "overwritten by the `VecMonitor` ones.",
                UserWarning,
            )

        VecEnvWrapper.__init__(self, venv)
        self.episode_returns = None
        self.episode_lengths = None
        self.episode_count = 0
        self.t_start = time.time()

        env_id = None
        if hasattr(venv, "spec") and venv.spec is not None:
            env_id = venv.spec.id

        if filename:
            self.results_writer = ResultsWriter(
                filename, header={"t_start": self.t_start, "env_id": env_id}, extra_keys=info_keywords
            )
        else:
            self.results_writer = None
        self.info_keywords = info_keywords

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return obs
    
    def step_async(self, actions: np.ndarray) -> None:
        self.action = actions
        # print("here 2")
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        # print("here")
        obs, rewards, dones, infos = self.venv.step(self.action)
        self.episode_returns += np.array(rewards)
        self.episode_lengths += 1
        # import pdb;pdb.set_trace()
        num_envs = len(dones)

        # Step 1: Convert batched dict-of-arrays into a list of dicts (one per env)
        new_infos = [{} for _ in range(num_envs)]
        for key, values in infos.items():
            for i in range(num_envs):
                new_infos[i][key] = values[i]

        for i in range(len(dones)):
            if dones[i]: # Extract info dict for env i by gathering all keys at index i
                info = new_infos[i].copy()
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {
                    "r": episode_return,
                    "l": episode_length,
                    "t": round(time.time() - self.t_start, 6)
                }
                info["episode"] = episode_info
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0

                if self.results_writer:
                    self.results_writer.write_row(episode_info)

                new_infos[i] = info  # replace with updated info
        return np.array(obs), np.array(rewards), np.array(dones), new_infos

    def close(self) -> None:
        if self.results_writer:
            self.results_writer.close()
        return self.venv.close()
