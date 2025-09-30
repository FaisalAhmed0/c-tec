import os
import warnings
from typing import Any, Callable, Dict, Optional, Type, Union, Sequence
import multiprocessing as mp

import envpool
import gym
import numpy as np

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecEnvWrapper, VecMonitor
from envpool.python.protocol import EnvPool
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, tile_images
from collections import Counter
import matplotlib.pyplot as plt

# From Stable Baseline 3
# https://github.com/DLR-RM/stable-baselines3/blob/18f4e3ace084a2fd3e0a3126613718945cf3e5b5/stable_baselines3/common/env_util.py

from packaging import version

is_legacy_gym = version.parse(gym.__version__) < version.parse("0.26.0")


class DiscretizedDensity:
    def __init__(self, axes=None, bin_width=0.5, goal_dim=2, run_folder=None):
        self._axes = axes
        self._bin_width = bin_width
        self.goal_dim = goal_dim
        self.counter = Counter()
        self.all_observations = np.array([[0]*goal_dim])

    def discretize(self, obs):
        if self._axes:
            obs = np.array([obs[i] for i in self._axes])
        obs = obs / self._bin_width
        obs = np.floor(obs).astype(np.int64)
        if self._axes or obs.shape[-1] > 1:
            obs = tuple(obs)
        return obs

    def update_count(self, batch_obs, env_step=0, save_state=True):
        # import pdb;pdb.set_trace()
        batch_obs = batch_obs.reshape(-1, batch_obs.shape[-1])
        batch_obs = batch_obs[:, :self.goal_dim]
        # batch_obs = np.array(batch_obs)
        # Vectorize the discretization process over the batch
        batch_obs = batch_obs / self._bin_width
        batch_obs = np.floor(batch_obs).astype(np.int64)
        
        if self._axes:
            batch_obs = batch_obs[:, self._axes]  # Select the relevant axes if necessary

        # Convert the batch of discretized observations to a list of tuples
        obs_tuples = [tuple(obs) for obs in batch_obs]
        
        # Update the counter with the batch of discretized observations
        self.counter.update(obs_tuples)
        # import pdb;pdb.set_trace()
        

    def compute_log_prob(self, obs):
        obs_d = self.discretize(obs)
        count = self.counter.get(obs_d, 1)
        total_count = sum(self.counter.values())
        prob = count / total_count if total_count > 0 else 0
        log_prob = np.log(prob + 1e-8)
        return log_prob

    def entropy(self):
        count_values = np.array(list(self.counter.values()))
        total_count = np.sum(count_values)
        if total_count == 0:
            return 0
        prob = count_values / total_count
        log_prob = np.log(prob + 1e-8)
        entropy = -(log_prob * prob).sum()
        return entropy


    def num_states(self):
        return len(self.counter)

    def visualize_scatter(self, title ,save_fig=False, env_step=0):
        data = self.all_observations.reshape(-1, self.goal_dim)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(title)
        im = ax.scatter(*(data.T)[:, 1:], alpha=0.005)
        plt.xlim((np.min(data), np.max(data)))
        plt.ylim((np.min(data), np.max(data)))
        if save_fig and self.run_folder:
            fig_path = f"{self.visual_path}/states_scatter_env_step_{env_step}.png"
            plt.savefig(fig_path, dpi=200)
        fig.canvas.draw()
        scatter = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close("all")
        return scatter
    
    def visualize_2d_histogram(self, title, save_fig=False, env_step=0):
        data = self.all_observations.reshape(-1, self.goal_dim)
        fig = plt.figure()
        plt.title(title)
        plt.hist2d(data[:, 0], data[:, 1], bins=(
            np.linspace(-1, 25.5, 15), np.linspace(-1, 25.5, 15)))
        
        if save_fig and self.run_folder:
            fig_path = f"{self.visual_path}/states_2d_hist_env_step_{env_step}.png"
            plt.savefig(fig_path, dpi=200)

        fig.canvas.draw()
        hist = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (3,))
        plt.close("all")
        return hist

class EnvPoolVecAdapter(VecEnvWrapper):
    """
    Convert EnvPool object to a Stable-Baselines3 (SB3) VecEnv.
    :param venv: The envpool object.
    """

    def __init__(self, venv: EnvPool):
        # Retrieve the number of environments from the config
        venv.num_envs = venv.spec.config.num_envs
        super().__init__(venv=venv)
        self.venv.obs = None

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def reset(self) -> VecEnvObs:
        if is_legacy_gym:
            obs = self.venv.reset()
        else:
            obs = self.venv.reset()[0]
        self.venv.obs = obs
        return obs

    def seed(self, seed: Optional[int] = None) -> None:
        # You can only seed EnvPool env by calling envpool.make()
        pass

    def step_wait(self) -> VecEnvStepReturn:
        if is_legacy_gym:
            obs, rewards, dones, info_dict = self.venv.step(self.actions)
        else:
            obs, rewards, terms, truncs, info_dict = self.venv.step(self.actions)
            dones = terms + truncs

        infos = []
        # Convert dict to list of dict
        # and add terminal observation
        for i in range(self.num_envs):
            infos.append(
                {
                    key: info_dict[key][i]
                    for key in info_dict.keys()
                    if isinstance(info_dict[key], np.ndarray)
                }
            )
            if dones[i]:
                infos[i]["terminal_observation"] = obs[i]
                if is_legacy_gym:
                    obs[i] = self.venv.reset(np.array([i]))
                else:
                    obs[i] = self.venv.reset(np.array([i]))[0]
        self.venv.obs = obs
        return obs, rewards, dones, infos

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        if self.venv.obs is None:
            return

        try:
            imgs = self.venv.obs
        except NotImplementedError:
            warnings.warn(f"Render not defined for {self}")
            return

        # Create a big image by tiling images from subprocesses
        bigimg = tile_images(imgs[:1])

        bigimg_size = bigimg.shape[-1]
        bigimg = bigimg[-1].reshape(bigimg_size, bigimg_size)

        # grayscale to fake-RGB
        bigimg = np.stack((bigimg,) * 3, axis=-1)

        if mode == "human":
            import cv2  # pytype:disable=import-error
            cv2.imshow("vecenv", bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == "rgb_array":
            return bigimg
        else:
            raise NotImplementedError(f"Render mode {mode} is not supported by VecEnvs")

