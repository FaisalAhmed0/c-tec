import csv_logger
import pickle
import json
import jax
import jax.numpy as jnp
import os
import numpy as np
import argparse
import wandb
import matplotlib.pyplot as plt
import csv

from datetime import datetime
from collections import Counter
from brax.io import html
from absl import logging
from etils import epath
from typing import Any
from collections import namedtuple
### 
from envs.ant import Ant
from envs.half_cheetah import Halfcheetah
from envs.reacher import Reacher
from envs.pusher import Pusher, PusherReacher
from envs.pusher2 import Pusher2
from envs.ant_ball import AntBall
from envs.ant_maze import AntMaze
from envs.humanoid import Humanoid
from envs.humanoid_maze import HumanoidMaze
from envs.ant_push import AntPush
from envs.manipulation.arm_reach import ArmReach
from envs.manipulation.arm_grasp import ArmGrasp
from envs.manipulation.arm_push_easy import ArmPushEasy
from envs.manipulation.arm_push_hard import ArmPushHard
from envs.manipulation.arm_binpick_easy import ArmBinpickEasy
from envs.manipulation.arm_binpick_hard import ArmBinpickHard
from envs.simple_maze import SimpleMaze






### Logging and checkpointing utils
def make_csv_logger(csv_path, header):
    log_level = ['logs_a']
    logger_ = csv_logger.CsvLogger(
        filename=csv_path,
        delimiter=',',
        level=logging.INFO,
        add_level_names=log_level,
        max_size=1e+9,
        add_level_nums=None,
        header=header,
    )
    return logger_

def load_params(path: str):
    with epath.Path(path).open('rb') as fin:
        buf = fin.read()
    return pickle.loads(buf)

def save_params(path: str, params: Any):
    """Saves parameters in flax format."""
    with epath.Path(path).open('wb') as fout:
        fout.write(pickle.dumps(params))

def save_args(args, path):
    # convert to a dictionary 
    args_dict = vars(args)
    for k in args_dict:
        if isinstance(args_dict[k], jax.Array):
            args_dict[k] = args_dict[k].tolist()
    # save the file 
    file_path = os.path.join(path, 'args.json') 
    with open(file_path, 'w') as f:
        json.dump(args_dict, f)

def save_buffer_sample(sample, path, global_step):
    file_path = os.path.join(path, f"buffer_sample_{global_step}.npz")
    jnp.savez(file_path, observation=sample.observation, action=sample.action)

class Simple_CSV_logger:
    def __init__(self, path, header):
        self.path = path
        self.header = header

        # If file doesn't exist, create it with header
        if not os.path.exists(self.path):
            with open(self.path, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.header)
                writer.writeheader()

    def log(self, data):
        # Write a new row using the dictionary
        data_ = {}
        for key in self.header:
            if key in data:
                data_[key] = data[key].item() if isinstance(data[key], jnp.ndarray) else data[key]
        with open(self.path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.header)
            writer.writerow(data_)


### CTEC Utils
def gamma_schedule(args, current_timestep, total_timesteps):
    if args.gamma_schedule == "linear":
        # linear growth
        frac = 1.0 - (current_timestep - 1.0) / total_timesteps
        gamma = frac * args.gamma_schedule_start + (1.0 - frac) * args.gamma_schedule_end
        args.discounting_cl = gamma
        return gamma
    elif args.gamma_schedule == "exponential":
        # Exponential growth: gamma = start * (end/start)^(t/T)
        t = current_timestep - 1.0
        T = total_timesteps
        gamma = args.gamma_schedule_start * (args.gamma_schedule_end / args.gamma_schedule_start) ** (t/T)
        args.discounting_cl = gamma
        return gamma
    else:
        return args.discounting_cl
    



### Metric utils
class MetricsRecorder:
    def __init__(self, num_timesteps):
        self.x_data = []
        self.y_data = {}
        self.y_data_err = {}
        self.times = [datetime.now()]

        self.max_x, self.min_x = num_timesteps * 1.1, 0

    def record(self, num_steps, metrics):
        self.times.append(datetime.now())
        self.x_data.append(num_steps)

        for key, value in metrics.items():
            if key not in self.y_data:
                self.y_data[key] = []
                self.y_data_err[key] = []

            self.y_data[key].append(value)
            self.y_data_err[key].append(metrics.get(f"{key}_std", 0))

    def log_wandb(self):
        data_to_log = {}
        for key, value in self.y_data.items():
            data_to_log[key] = value[-1]
        data_to_log["step"] = self.x_data[-1]
        wandb.log(data_to_log, step=self.x_data[-1])

    def plot_progress(self):
        num_plots = len(self.y_data)
        num_rows = (num_plots + 1) // 2  # Calculate number of rows needed for 2 columns

        fig, axs = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))

        for idx, (key, y_values) in enumerate(self.y_data.items()):
            row = idx // 2
            col = idx % 2

            axs[row, col].set_xlim(self.min_x, self.max_x)
            axs[row, col].set_xlabel("# environment steps")
            axs[row, col].set_ylabel(key)
            axs[row, col].errorbar(self.x_data, y_values, yerr=self.y_data_err[key])
            axs[row, col].set_title(f"{key}: {y_values[-1]:.3f}")

        # Hide any empty subplots
        for idx in range(num_plots, num_rows * 2):
            row = idx // 2
            col = idx % 2
            axs[row, col].axis("off")
        plt.tight_layout()
        plt.show()

    def print_progress(self):
        for idx, (key, y_values) in enumerate(self.y_data.items()):
            print(f"step: {self.x_data[-1]}, {key}: {y_values[-1]:.3f} +/- {self.y_data_err[key][-1]:.3f}")

    def print_times(self):
        print(f"time to jit: {self.times[1] - self.times[0]}")
        print(f"time to train: {self.times[-1] - self.times[1]}")

class DiscretizedDensity:
    def __init__(self, axes=None, bin_width=1.0, goal_dim=2, run_folder=None):
        self._axes = axes
        self._bin_width = bin_width
        self.goal_dim = goal_dim
        self.counter = Counter()
        self.all_observations = np.array([[0]*goal_dim])
        self.run_folder = run_folder
        if run_folder:
            # create path for saving state coverage visuals
            visual_path = f"{run_folder}/visuals/state_coverage"
            visited_states_path = f"{run_folder}/visited_states"
            os.makedirs(visited_states_path)
            os.makedirs(visual_path, exist_ok=True)
            self.visual_path = visual_path
            self.visited_states_path = visited_states_path
        

    def discretize(self, obs):
        if self._axes:
            obs = np.array([obs[i] for i in self._axes])
        obs = obs / self._bin_width
        obs = np.floor(obs).astype(np.int64)
        if self._axes or obs.shape[-1] > 1:
            obs = tuple(obs)
        return obs

    def update_count(self, batch_obs, env_step=0):
        batch_obs = batch_obs.reshape(-1, self.goal_dim)
        batch_obs = np.array(batch_obs)

        print("saved new future states")
        np.savez_compressed(f"{self.visited_states_path}/{env_step}", data=batch_obs)

        # Vectorize the discretization process over the batch
        batch_obs = batch_obs / self._bin_width
        batch_obs = np.floor(batch_obs).astype(np.int64)
        
        if self._axes:
            batch_obs = batch_obs[:, self._axes]  # Select the relevant axes if necessary

        # Convert the batch of discretized observations to a list of tuples
        obs_tuples = [tuple(obs) for obs in batch_obs]
        
        # Update the counter with the batch of discretized observations
        self.counter.update(obs_tuples)
        
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
    
def knn_average_distance(observation):
    k = 12
    dists = jnp.sum((observation[:, None, :] - observation[None, :, :]) ** 2, axis=-1)

    # Sort the distances for each point
    sorted_dists = jnp.sort(dists, axis=-1)

    # Initialize a zero array to store the cumulative distances for different k values
    cumulative_knn_dists = jnp.zeros_like(dists[:, 0])

    # Iterate over the ks to compute the k-NN distances and accumulate them

    knn_dists = sorted_dists[:, 1:k+1]  # First distance is to the point itself, so skip it
    mean_knn_dists = jnp.mean(knn_dists, axis=-1)  # Mean distance to k nearest neighbors
    mean_knn_dists

    return mean_knn_dists



### Env utils    
def create_eval_env(args: argparse.Namespace) -> object:
    if not args.eval_env:
        return None
    
    eval_arg = argparse.Namespace(**vars(args))
    eval_arg.env_name = args.eval_env
    return create_env(eval_arg)

def get_env_config(args: argparse.Namespace):
    legal_envs = ["reacher", "cheetah", "pusher_easy", "pusher_hard", "pusher_reacher", "pusher2",
                  "ant", "ant_push", "ant_ball", "humanoid", "arm_reach", "arm_grasp", 
                  "arm_push_easy", "arm_push_hard", "arm_binpick_easy", "arm_binpick_hard"]
    if args.env_name not in legal_envs and "maze" not in args.env_name:
        raise ValueError(f"Unknown environment: {args.env_name}")

    args_dict = vars(args)
    Config = namedtuple("Config", [*args_dict.keys()])
    config = Config(*args_dict.values())
    
    return config

def create_env(args: argparse.Namespace) -> object:
    env_name = args.env_name
    if env_name == "reacher":
        env = Reacher(backend=args.backend or "generalized")
    elif env_name == "ant":
        env = Ant(backend=args.backend or "spring", include_goal_in_obs=args.include_goal_in_obs)
    elif env_name == "ant_ball":
        env = AntBall(backend=args.backend or "spring")
    elif env_name == "ant_push":
        # This is stable only in mjx backend
        # assert args.backend == "mjx"
        env = AntPush(backend=args.backend or "mjx", include_goal_in_obs=args.include_goal_in_obs)
    elif "maze" in env_name:
        if "ant" in env_name: 
            # Possible env_name = {'ant_u_maze', 'ant_big_maze', 'ant_hardest_maze'}
            env = AntMaze(backend=args.backend or "spring", maze_layout_name=env_name[4:], include_goal_in_obs=args.include_goal_in_obs)
        elif "humanoid" in env_name:
            # Possible env_name = {'humanoid_u_maze', 'humanoid_big_maze', 'humanoid_hardest_maze'}
            env = HumanoidMaze(backend=args.backend or "spring", maze_layout_name=env_name[9:], include_goal_in_obs=args.include_goal_in_obs)
        else:
            # Possible env_name = {'simple_u_maze', 'simple_big_maze', 'simple_hardest_maze'}
            env = SimpleMaze(backend=args.backend or "spring", maze_layout_name=env_name[7:])
    elif env_name == "cheetah":
        env = Halfcheetah()
    elif env_name == "pusher_easy":
        env = Pusher(backend=args.backend or "generalized", kind="easy", include_goal_in_obs=args.include_goal_in_obs)
    elif env_name == "pusher_hard":
        env = Pusher(backend=args.backend or "generalized", kind="hard", include_goal_in_obs=args.include_goal_in_obs)
    elif env_name == "pusher_reacher":
        env = PusherReacher(backend=args.backend or "generalized")
    elif env_name == "pusher2":
        env = Pusher2(backend=args.backend or "generalized")
    elif env_name == "humanoid":
        env = Humanoid(backend=args.backend or "spring")
    elif env_name == "arm_reach":
        env = ArmReach(backend=args.backend or "mjx", include_goal_in_obs=args.include_goal_in_obs)
    elif env_name == "arm_grasp":
        env = ArmGrasp(backend=args.backend or "mjx", include_goal_in_obs=args.include_goal_in_obs)
    elif env_name == "arm_push_easy":
        env = ArmPushEasy(backend=args.backend or "mjx", include_goal_in_obs=args.include_goal_in_obs)
    elif env_name == "arm_push_hard":
        env = ArmPushHard(backend=args.backend or "mjx", include_goal_in_obs=args.include_goal_in_obs)
    elif env_name == "arm_binpick_easy":
        env = ArmBinpickEasy(backend=args.backend or "mjx", include_goal_in_obs=args.include_goal_in_obs)
    elif env_name == "arm_binpick_hard":
        env = ArmBinpickHard(backend=args.backend or "mjx", include_goal_in_obs=args.include_goal_in_obs)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    return env

def render(inf_fun_factory, params, env, exp_dir, exp_name, seed=1, rollout_len=5000, timestep=None):
    inference_fn = inf_fun_factory(params)
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(inference_fn)

    rollout = []

    rng = jax.random.PRNGKey(seed=seed)
    state = jit_env_reset(rng=rng)

    for i in range(rollout_len):
        rollout.append(state.pipeline_state)

        rng, step_rng = jax.random.split(rng)
        act_rng, reset_rng = jax.random.split(step_rng)

        act, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_env_step(state, act)

        if i % 1000 == 0:
            state = jit_env_reset(rng=reset_rng)

    url = html.render(env.sys.tree_replace({"opt.timestep": env.dt}), rollout, height=1024)
    with open(os.path.join(exp_dir, f"{exp_name}.html"), "w") as file:
        file.write(url)
    wandb.log({"render": wandb.Html(url)})
    
    
    
    
### Normalization utils
@jax.jit
def update_rms(state, x):
    count, mean, M2 = state
    count_new = count + 1.0
    delta = x - mean
    mean_new = mean + delta / count_new
    delta2 = x - mean_new
    M2_new = M2 + delta * delta2
    std_new = jnp.sqrt(M2_new / count_new)
    return (count_new, mean_new, M2_new), (mean_new, std_new)

# Function to compute incremental mean and std over a 1D stream of data.
def incremental_mean_std(data):
    # Initialize state: count=0, mean=0, M2=0.
    # Using data[0] to create a zero of the same shape as a sample.
    init_state = (0.0, jnp.zeros_like(data[0]), jnp.zeros_like(data[0]))
    # Use lax.scan to perform the updates over the data stream.
    final_state, (means, stds) = lax.scan(update_rms, init_state, data)
    return means, stds