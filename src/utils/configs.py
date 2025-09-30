import os
import time

import torch as th
import random
import wandb
from torch import nn
from gym_minigrid.wrappers import ImgObsWrapper, FullyObsWrapper, ReseedWrapper
from procgen import ProcgenEnv
import numpy as np
# import src.env.mujoco_custom.halfcheetah_vel_sparse
# import src.env.dmc
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from gymnasium.wrappers import FilterObservation
from gymnasium.wrappers import FlattenObservation
from datetime import datetime
from src.algo.common_models.cnns import BatchNormCnnFeaturesExtractor, LayerNormCnnFeaturesExtractor, \
    CnnFeaturesExtractor, MLPFeatureExtractor
from src.env.subproc_vec_env import CustomSubprocVecEnv, BraxGymDummyVecEnv, BraxGymVecMonitor
from src.utils.enum_types import EnvSrc, NormType, ModelType, DecayType
from wandb.integration.sb3 import WandbCallback

from src.utils.loggers import LocalLogger
from src.utils.video_recorder import VecVideoRecorder
from gym.wrappers import NormalizeObservation, TimeLimit

## For brax/jax-based environments
from src.env.jax_envs.ant_maze import AntMaze
from src.env.jax_envs.humanoid_maze import HumanoidMaze
from src.env.jax_envs.arm_binpick_hard import ArmBinpickHard
from brax.envs.wrappers.gym import GymWrapper, VectorGymWrapper
from brax.envs.wrappers.training import VmapWrapper
import jax

import crafter

def create_brax_env(env_name) -> object:
    if "maze" in env_name:
        if "ant" in env_name: 
            # Possible env_name = {'ant_u_maze', 'ant_big_maze', 'ant_hardest_maze'}
            env = AntMaze(backend="spring", maze_layout_name=env_name[4:], include_goal_in_obs=False)
        elif "humanoid" in env_name:
            # Possible env_name = {'humanoid_u_maze', 'humanoid_big_maze', 'humanoid_hardest_maze'}
            env = HumanoidMaze(backend="spring", maze_layout_name=env_name[9:], include_goal_in_obs=False)
    elif env_name == "arm_binpick_hard":
        env = ArmBinpickHard(backend="mjx", include_goal_in_obs=False)
        # import pdb;pdb.set_trace()  
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    return env


class TrainingConfig():
    def __init__(self):
        self.dtype = th.float32
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    def init_meta_info(self):
        self.file_path = __file__
        self.model_name = os.path.basename(__file__)
        self.start_time = time.time()
        self.start_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def init_env_name(self, game_name: str, project_name: str):
        env_name = game_name
        self.env_source = EnvSrc.get_enum_env_src(self.env_source)
        if self.env_source == EnvSrc.MiniGrid and not game_name.startswith('MiniGrid-'):
            env_name = f'MiniGrid-{game_name}'
            env_name += '-v0'
        if self.env_source == EnvSrc.MiniWorld and not game_name.startswith('MiniWorld-'):
            env_name = f'MiniWorld-{game_name}'
            env_name += '-v0'
        if self.env_source == EnvSrc.MuJoCo:
            env_name = game_name
        if self.env_source == EnvSrc.DMC:
            env_name = f'dmc/{game_name}-v1'
        if self.env_source == EnvSrc.GYMN_ROBOTS:
            env_name = game_name
        if self.env_source == EnvSrc.BRAX:
            env_name = game_name
        self.env_name = env_name
        self.project_name = env_name if project_name is None else project_name

    def init_logger(self):
        self.log_dir = os.path.join(self.log_dir, self.env_name, self.int_rew_source, self.exp_name, str(self.run_id))
        os.makedirs(self.log_dir, exist_ok=True)
        
        if self.use_wandb:
            self.wandb_run = wandb.init(
                # dir=str(self.log_dir),
                name=f'{self.exp_name}_{self.run_id}',
                # entity='thu_jsbsim',  # your project name on wandb
                project="CTEC_ETD_Reward",
                settings=wandb.Settings(start_method="fork"),
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                monitor_gym=True,  # auto-upload the videos of agents playing the game
                save_code=True,  # optional
                config=vars(self),
            )
        else:
            self.wandb_run = None


        if self.write_local_logs:
            self.local_logger = LocalLogger(self.log_dir)
            print(f'Writing local logs at {self.log_dir}')
        else:
            self.local_logger = None

        print(f'Starting run {self.run_id}')

    def init_values(self):
        if self.clip_range_vf <= 0:
            self.clip_range_vf = None

    def close(self):
        if self.wandb_run is not None:
            self.wandb_run.finish()

    def get_wrapper_class(self):
        if self.env_source == EnvSrc.MiniGrid:
            if self.fully_obs:
                wrapper_class = lambda x: ImgObsWrapper(FullyObsWrapper(x))
            else:
                wrapper_class = lambda x: ImgObsWrapper(x)

            if self.fixed_seed >= 0 and self.env_source == EnvSrc.MiniGrid:
                assert not self.fully_obs
                _seeds = [self.fixed_seed]
                wrapper_class = lambda x: ImgObsWrapper(ReseedWrapper(x, seeds=_seeds))
            return wrapper_class
        elif self.env_source == EnvSrc.MiniWorld:
            # TODO: FullyObsWrapper / ReseedWrapper for MiniWorld
            pass
        elif self.env_source == EnvSrc.MuJoCo:
            wrapper_class = lambda x: NormalizeObservation(x)
            return wrapper_class
        elif self.env_source == EnvSrc.GYMN_ROBOTS:
            pass
        elif self.env_source == EnvSrc.Crafter:
            wrapper_class = lambda x: crafter.Recorder(
            x, f"{self.log_dir}/stats",
            save_stats=True,
            save_video=False,
            save_episode=False,
            )
            return wrapper_class
        else:
            return None

    def get_venv(self, wrapper_class=None):
        if self.env_source == EnvSrc.MiniGrid:
            venv = make_vec_env(
                self.env_name,
                wrapper_class=wrapper_class,
                vec_env_cls=CustomSubprocVecEnv,
                n_envs=self.num_processes,
                monitor_dir=self.log_dir,
            )
        elif self.env_source == EnvSrc.ProcGen:
            venv = ProcgenEnv(
                num_envs=self.num_processes,
                env_name=self.env_name,
                rand_seed=self.run_id,
                num_threads=self.procgen_num_threads,
                distribution_mode=self.procgen_mode,
            )
            venv = VecMonitor(venv=venv)
        elif self.env_source == EnvSrc.Crafter:
            venv = make_vec_env(
                'CrafterReward-v1',
                wrapper_class=wrapper_class,
                vec_env_cls=CustomSubprocVecEnv,
                n_envs=self.num_processes,
                monitor_dir=self.log_dir,
            )
        elif self.env_source == EnvSrc.MiniWorld:
            venv = make_vec_env(
                self.env_name,
                n_envs=self.num_processes,
                seed=self.run_id,
                env_kwargs={'image_noise_scale': self.image_noise_scale},
                vec_env_cls=SubprocVecEnv,
                monitor_dir=self.log_dir,
            )
        # elif self.env_source == EnvSrc.PandaGym:
        #     venv = make_vec_env(
        #         self.env_name,
        #         n_envs=self.num_processes,
        #         seed=self.run_id,
        #         vec_env_cls=SubprocVecEnv,
        #         monitor_dir=self.log_dir,
        #     )
        elif self.env_source == EnvSrc.MuJoCo:
            venv = make_vec_env(
                self.env_name,
                n_envs=self.num_processes,
                wrapper_class=wrapper_class,
                seed=self.run_id,
                vec_env_cls=SubprocVecEnv,
                monitor_dir=self.log_dir,
            )
        elif self.env_source == EnvSrc.DMC:
            venv = make_vec_env(
                self.env_name,
                n_envs=self.num_processes,
                wrapper_class=wrapper_class,
                seed=self.run_id,
                vec_env_cls=SubprocVecEnv,
                monitor_dir=self.log_dir,
                env_kwargs={
                    'frame_skip': 2,
                }
            )
        elif self.env_source == EnvSrc.GYMN_ROBOTS:
            print("create the gymnasium robotics env")
            # import pdb;pdb.set_trace()
            venv = make_gymn_robotics_vec_env(self.env_name,n_envs=self.num_processes,seed=self.run_id,vec_env_cls=SubprocVecEnv,monitor_dir=self.log_dir,)
        elif self.env_source == EnvSrc.BRAX:
            print("create the brax robotics env")
            # import pdb;pdb.set_trace()
            venv = make_brax_gym_vec_env(self.env_name,n_envs=self.num_processes,seed=self.run_id,vec_env_cls=SubprocVecEnv,monitor_dir=self.log_dir,)
        else:
            raise NotImplementedError

        if (self.record_video == 2) or \
                (self.record_video == 1 and self.run_id == 0):
            _trigger = lambda x: x > 0 and x % (self.n_steps * self.rec_interval) == 0
            venv = VecVideoRecorder(
                venv,
                os.path.join(self.log_dir, 'videos'),
                record_video_trigger=_trigger,
                video_length=self.video_length,
            )
        # import pdb;pdb.set_trace()
        return venv

    def get_callbacks(self):
        if self.use_wandb:
            callbacks = CallbackList([
                WandbCallback(
                    gradient_save_freq=50,
                    verbose=1,
                )])
        else:
            callbacks = CallbackList([])
        return callbacks

    def get_optimizer(self):
        if self.optimizer.lower() == 'adam':
            optimizer_class = th.optim.Adam
            optimizer_kwargs = dict(
                eps=self.optim_eps,
                betas=(self.adam_beta1, self.adam_beta2),
            )
        elif self.optimizer.lower() == 'rmsprop':
            optimizer_class = th.optim.RMSprop
            optimizer_kwargs = dict(
                eps=self.optim_eps,
                alpha=self.rmsprop_alpha,
                momentum=self.rmsprop_momentum,
            )
        else:
            raise NotImplementedError
        return optimizer_class, optimizer_kwargs

    def get_activation_fn(self):
        if self.activation_fn.lower() == 'relu':
            activation_fn = nn.ReLU
        elif self.activation_fn.lower() == 'gelu':
            activation_fn = nn.GELU
        elif self.activation_fn.lower() == 'elu':
            activation_fn = nn.ELU
        else:
            raise NotImplementedError

        if self.cnn_activation_fn.lower() == 'relu':
            cnn_activation_fn = nn.ReLU
        elif self.cnn_activation_fn.lower() == 'gelu':
            cnn_activation_fn = nn.GELU
        elif self.cnn_activation_fn.lower() == 'elu':
            cnn_activation_fn = nn.ELU
        else:
            raise NotImplementedError
        return activation_fn, cnn_activation_fn

    def cast_enum_values(self):
        self.policy_cnn_norm = NormType.get_enum_norm_type(self.policy_cnn_norm)
        self.policy_mlp_norm = NormType.get_enum_norm_type(self.policy_mlp_norm)
        self.policy_gru_norm = NormType.get_enum_norm_type(self.policy_gru_norm)

        self.model_cnn_norm = NormType.get_enum_norm_type(self.model_cnn_norm)
        self.model_mlp_norm = NormType.get_enum_norm_type(self.model_mlp_norm)
        self.model_gru_norm = NormType.get_enum_norm_type(self.model_gru_norm)

        self.int_rew_source = ModelType.get_enum_model_type(self.int_rew_source)
        if self.int_rew_source == ModelType.DEIR and not self.use_model_rnn:
            print('\nWARNING: Running DEIR without RNNs\n')
        if self.int_rew_source in [ModelType.DEIR, ModelType.PlainDiscriminator]:
            assert self.n_steps * self.num_processes >= self.batch_size
        self.int_rew_decay = DecayType.get_enum_decay_type(self.int_rew_decay)

    def get_cnn_kwargs(self, cnn_activation_fn=nn.ReLU):
        features_extractor_common_kwargs = dict(
            features_dim=self.features_dim,
            activation_fn=cnn_activation_fn,
            model_type=self.policy_cnn_type,
        )

        model_features_extractor_common_kwargs = dict(
            features_dim=self.model_features_dim,
            activation_fn=cnn_activation_fn,
            model_type=self.model_cnn_type,
        )

        if self.policy_cnn_norm == NormType.BatchNorm:
            policy_features_extractor_class = BatchNormCnnFeaturesExtractor
        elif self.policy_cnn_norm == NormType.LayerNorm:
            policy_features_extractor_class = LayerNormCnnFeaturesExtractor
        elif self.policy_cnn_norm == NormType.NoNorm:
            policy_features_extractor_class = CnnFeaturesExtractor
        else:
            raise ValueError

        if self.model_cnn_norm == NormType.BatchNorm:
            model_cnn_features_extractor_class = BatchNormCnnFeaturesExtractor
        elif self.model_cnn_norm == NormType.LayerNorm:
            model_cnn_features_extractor_class = LayerNormCnnFeaturesExtractor
        elif self.model_cnn_norm == NormType.NoNorm:
            model_cnn_features_extractor_class = CnnFeaturesExtractor
        else:
            raise ValueError

        return policy_features_extractor_class, \
            features_extractor_common_kwargs, \
            model_cnn_features_extractor_class, \
            model_features_extractor_common_kwargs


    def get_mlp_kwargs(self, cnn_activation_fn=nn.ReLU):
        features_extractor_common_kwargs = dict(
            features_dim=self.features_dim,
            activation_fn=cnn_activation_fn,
        )

        model_features_extractor_common_kwargs = dict(
            features_dim=self.model_features_dim,
            activation_fn=cnn_activation_fn,
        )

        policy_features_extractor_class = MLPFeatureExtractor
        model_cnn_features_extractor_class = MLPFeatureExtractor

        return policy_features_extractor_class, \
            features_extractor_common_kwargs, \
            model_cnn_features_extractor_class, \
            model_features_extractor_common_kwargs





import gymnasium_robotics
import os
from typing import Any, Callable, Dict, Optional, Type, Union

from gym.vector import SyncVectorEnv
from gym.spaces import Box

import gymnasium as gym

from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from gymnasium import Wrapper

class GymnasiumToGymWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.old_observation_space = env.observation_space
        self.old_action_space = env.action_space
        self.env = env
        self.observation_space = Box(self.old_observation_space.low, self.old_observation_space.high, shape=self.observation_space.shape, dtype=self.old_observation_space.dtype)
        self.action_space = Box(self.old_action_space.low, self.old_action_space.high, shape=self.old_action_space.shape, dtype=self.old_action_space.dtype)
        self.max_time_steps = 1000

    def reset(self, **kwargs):
        """Gym pre-0.26 compatible reset (returns just obs)"""
        obs, _ = self.env.reset(**kwargs)
        self.t = 0
        return obs

    def step(self, action):
        """Gym pre-0.26 compatible step (returns obs, reward, done, info)"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.t += 1
        done = terminated or truncated or self.t >= self.max_time_steps
        return obs, reward, done, info
    
    def seed(self, *args):
        return self.env.seed(*args)
    

def make_gymn_robotics_vec_env(
    env_id: Union[str, Callable[..., gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
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
                env = GymnasiumToGymWrapper(env)
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
    # import pdb;pdb.set_trace()
    return vec_env


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
    # jax.config.update('jax_default_device', jax.devices("cpu")[0])
    
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
                env = create_brax_env(env_id)
                env = VmapWrapper(env, n_envs)
                env = VectorGymWrapper(env)
            except TypeError:
                env = create_brax_env(env_id)
                env = VmapWrapper(env, n_envs)
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
            env.step_async = lambda x: x
            env = BraxGymVecMonitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init
    
    envs = make_env(0)()
    # import pdb;pdb.set_trace()
    vec_env = BraxGymDummyVecEnv(envs, num_envs=n_envs)
    vec_env.observation_space = Box(-np.inf, np.inf, shape=(vec_env.observation_space.shape[-1], ))
    vec_env.action_space = Box(-np.inf, np.inf, shape=(vec_env.action_space.shape[-1], ))
    # import pdb;pdb.set_trace()
    # wramp up reset and step methods
    # vec_env.reset()
    # vec_env.step(vec_env.action_space.sample())
    return vec_env