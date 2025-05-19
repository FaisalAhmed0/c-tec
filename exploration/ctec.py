import os
os.environ['MUJOCO_GL'] = 'osmesa'
import jax
import flax
import jax.random
import tyro
import time
import optax
import wandb
import pickle
import random
import wandb_osh
import numpy as np
import flax.linen as nn
import jax.numpy as jnp
import json
import functools
import model_utils as sac_networks
import math
import csv_logger

from absl import logging
from copy import deepcopy
from etils import epath
from dataclasses import dataclass 
from collections import namedtuple
from typing import Any, Generic, Tuple, TypeVar,Union, Generic, NamedTuple, Sequence
from wandb_osh.hooks import TriggerWandbSyncHook
from flax.training.train_state import TrainState
from flax.linen.initializers import variance_scaling
from evaluator import CrlEvaluator, ActorCrlEvaluator

from brax.training.acme import specs
from brax.training.acme import running_statistics
from brax.training import types
from brax.training.types import Params, Policy
from brax.training.acme.types import NestedArray
from brax import envs
from brax.training import pmap as brax_pmap
from brax.training.agents.sac import losses as sac_losses
from brax.training.replay_buffers_test import jit_wrap
from brax.training import gradients
from brax.training.types import PRNGKey
from brax.v1 import envs as envs_v1
from brax.io import model

from utils import MetricsRecorder, create_env, create_eval_env, DiscretizedDensity, Simple_CSV_logger, get_env_config, knn_average_distance, render

from buffers import QueueBase
from losses import make_contrastive_critic_loss as make_contrastive_loss
from models import ContrastiveCritic
from wonderwords import RandomWord



Metrics = types.Metrics

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = np.random.randint(2**31)
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "exploration"
    wandb_entity: str = None
    wandb_mode: str = 'online'
    wandb_dir: str = '.'
    wandb_group: str = '.'
    capture_video: bool = False
    checkpoint: bool = True
    run_name_suffix: str = ""
    num_videos: int = 30

    #environment specific arguments
    env_name: str = "ant_hardest_maze"
    episode_length: int = 1000
    # to be filled in runtime
    obs_dim: int = 0
    goal_start_idx: int = 0
    goal_end_idx: int = 0

    # Algorithm specific arguments
    num_timesteps: int = 10_000_000
    num_epochs: int = 50
    num_envs: int = 512
    num_eval_envs: int = 5
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    rnd_lr: float = 3e-4
    batch_size: int = 256
    discounting: float = 0.99
    use_dense_reward: bool = False
    tau = 0.005
    logsumexp_penalty_coeff: float = 0.1
    entropy_reg: bool = True

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    agent_number_hiddens: int = 2
    agent_hidden_dim: int = 256
    
    unroll_length: int  = 62
    reward_scaling: float = 1.0
    use_her: bool = False
    """Use hindsight experince replay"""
    multiplier_num_sgd_steps: int = 1
    deterministic_eval: bool = False
    action_repeat: int = 1
    num_evals: int = 50
    backend: str = None
    eval_env: str = None
    render_agent: bool = False
    include_goal_in_obs: bool = True
    rnd_number_hiddens: int = 2
    rnd_hidden_dim: int = 256
    rnd_embed_dim: int = 512
    rnd_observation_dim: int = 0 # to be specified at run_time
    layer_norm: bool = False
    agent_activation: str = "nn.relu"
    activation: str = "nn.relu"
    rnd_reward_rms: bool = False
    # to be filled in runtime
    env_steps_per_actor_step : int = 0
    """number of env steps per actor step (computed in runtime)"""
    num_prefill_env_steps : int = 0
    """number of env steps to fill the buffer before starting training (computed in runtime)"""
    num_prefill_actor_steps : int = 0
    """number of actor steps to fill the buffer before starting training (computed in runtime)"""
    num_training_steps_per_epoch : int = 0
    """the number of training steps per epoch(computed in runtime)"""
    render_freq: int = 12*5
    rnd_observation_dim: int = 0
    rnd_goal_indices: object = None

    ## CRL related params
    crl_goal_indices: object = None
    crl_observation_dim: int = 0 # if > 0 use for debugging
    use_complete_future_state: bool = False
    crl_observation_dim: int = 0 # if > 0 use for debugging
    crl_goal_indices: object = None
    noise_std: float = 0.1
    da: bool = False
    sa_projector: bool = False
    g_projector: bool = False
    fix_temp: bool = False
    temp_value:float = 1
    spectral_norm: bool = False
    use_diag_q: bool = False
    logsumexp_penalty_coeff: float = 0.1
    l2_penalty_coeff: float = 0.0
    random_goals: float = 0.0 # poportion of random goals in the actor loss
    energy_fn: str = "l1"
    contr_loss: str = "infonce"
    repr_dim: int = 64
    normalize_repr: bool = True
    temp_scaling: bool = True
    model: str = "crl_sac"
    contrastive_number_hiddens: int = 2
    contrastive_hidden_dim: int = 256
    use_deep_encoder: bool = False
    discounting_cl: float = 0.99
    layer_norm_crl: bool = False
    future_state_rwd_sampling: str = "geometric"
    gamma_schedule: str = None
    gamma_schedule_start: float = 0.1
    gamma_schedule_end: float = 1.0
    save_all_crl_ckpts: bool = False
    ema: float=0.999
    save_replay_data: bool = False

ReplayBufferState = Any
_PMAP_AXIS_NAME = "i"





def gamma_schedule(args, current_timestep, total_timesteps):
    if args.gamma_schedule == "linear":
        frac = 1.0 - (current_timestep - 1.0) / total_timesteps
        gamma = frac * args.gamma_schedule_start + (1.0 - frac) * args.gamma_schedule_end
        args.discounting_cl = gamma
        return gamma
    elif args.gamma_schedule == "exponential":
        # Exponential decay: gamma = start * (end/start)^(t/T)
        t = current_timestep - 1.0
        T = total_timesteps
        gamma = args.gamma_schedule_start * (args.gamma_schedule_end / args.gamma_schedule_start) ** (t/T)
        args.discounting_cl = gamma
        return gamma
    else:
        return args.discounting_cl



# Transition = types.Transition
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]
Transition = types.Transition
Sample = TypeVar("Sample")

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

def actor_step(
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    extra_fields: Sequence[str] = (),
) -> Tuple[State, Transition]:
    """Collect data."""
    # print(env_fstate.obs.shape)
    # input()
    actions, policy_extras = policy(env_state.obs, key)
    nstate = env.step(env_state, actions)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=env_state.obs,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.obs,
        extras={"policy_extras": policy_extras, "state_extras": state_extras},
    )

def actor_step_render(
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    extra_fields: Sequence[str] = (),
) -> Tuple[State, Transition]:
    """Collect data."""
    actions, policy_extras = policy(env_state.obs[None, :], key)
    nstate = env.step(env_state, actions)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=env_state.obs,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.obs,
        extras={"policy_extras": policy_extras, "state_extras": state_extras},
    )

# utility functions for RND
def normalize(arr: jax.Array, mean: jax.Array, std: jax.Array, eps: float = 1e-8) -> jax.Array:
    return jnp.clip((arr - mean) / (std + eps), -5., 5.)


def crl_reward(contrastive_network, contrastive_params, transition: Transition, args, key_critic):
    state = transition.observation[:, :, :args.obs_dim]
    action = transition.action
    future_state = transition.extras["future_state_for_rwd"]
    future_reward = transition.extras["future_reward"]
    

    # import pdb;pdb.set_trace()

    random_goal_mask = jax.random.bernoulli(key_critic, args.random_goals, shape=(future_state.shape[0], 1, 1))
    future_rolled = jnp.roll(future_state, 1, axis=0)
    future_state = jnp.where(random_goal_mask, future_rolled, future_state)

    goal = future_state[:, :, args.crl_goal_indices]
    

    
    
    sa_repr, g_repr, _ = contrastive_network.apply(contrastive_params, state, action, goal, key_critic, args.da, train=False)

    similarity_method = {
            "l2": lambda sa_repr, g_repr: -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1)),
            "l2_no_sqrt":  lambda sa_repr, g_repr: -jnp.sum((sa_repr - g_repr) ** 2, axis=-1),
            "l1":  lambda sa_repr, g_repr: -jnp.sum(jnp.abs(sa_repr - g_repr), axis=-1),
            "dot": lambda sa_repr, g_repr: jnp.einsum("hik,hik->hi", sa_repr, g_repr), # if the vectors are normalized then this the cosine 
        }
    # import pdb;pdb.set_trace()
    sm = similarity_method[args.energy_fn](sa_repr, g_repr)
    # reward = -sm.mean(axis=-1)
    reward = -sm
    # import pdb;pdb.set_trace()
    return  jax.lax.stop_gradient(reward)
    

# Jax RMS
class RMS:
    """Running mean and standard deviation."""
    def __init__(self, epsilon=1e-4, shape=(1,)):
        self.M = jnp.zeros(shape)
        self.S = jnp.ones(shape)
        self.n = epsilon

    def __call__(self, x):
        bs = x.shape[0]
        delta = jnp.mean(x, axis=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + jnp.var(x, axis=0) * bs +
                 jnp.square(delta) * self.n * bs /
                 (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S



@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    policy_optimizer_state: optax.OptState
    policy_params: Params
    q_optimizer_state: optax.OptState
    q_params: Params
    contrastive_optimizer_state: optax.OptState
    contrastive_params: Params
    target_q_params: Params
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray
    alpha_optimizer_state: optax.OptState
    alpha_params: Params
    normalizer_params: running_statistics.RunningStatisticsState
    mean_coverage: jnp.ndarray
    contrastive_params_EMA: Params


class Transition(NamedTuple):
    """Container for a transition."""

    observation: NestedArray
    next_observation: NestedArray
    action: NestedArray
    reward: NestedArray
    discount: NestedArray
    extras: NestedArray = ()  # pytype: disable=annotation-type-mismatch  # jax-ndarray


class TrajectoryUniformSamplingQueue(QueueBase[Sample], Generic[Sample]):
    """Implements an uniform sampling limited-size replay queue BUT WITH TRAJECTORIES."""

    def sample_internal(self, buffer_state: ReplayBufferState) -> Tuple[ReplayBufferState, Sample]:
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"Data shape expected by the replay buffer ({self._data_shape}) does "
                f"not match the shape of the buffer state ({buffer_state.data.shape})"
            )
        key, sample_key, shuffle_key = jax.random.split(buffer_state.key, 3)
        # NOTE: this is the number of envs to sample but it can be modified if there is OOM
        shape = self.num_envs

        # Sampling envs idxs
        envs_idxs = jax.random.choice(sample_key, jnp.arange(self.num_envs), shape=(shape,), replace=False)

        @functools.partial(jax.jit, static_argnames=("rows", "cols"))
        def create_matrix(rows, cols, min_val, max_val, rng_key):
            rng_key, subkey = jax.random.split(rng_key)
            start_values = jax.random.randint(subkey, shape=(rows,), minval=min_val, maxval=max_val)
            row_indices = jnp.arange(cols)
            matrix = start_values[:, jnp.newaxis] + row_indices
            return matrix

        @jax.jit
        def create_batch(arr_2d, indices):
            return jnp.take(arr_2d, indices, axis=0, mode="wrap")

        create_batch_vmaped = jax.vmap(create_batch, in_axes=(1, 0))

        matrix = create_matrix(
            shape,
            self.episode_length,
            buffer_state.sample_position,
            buffer_state.insert_position - self.episode_length,
            sample_key,
        )

        batch = create_batch_vmaped(buffer_state.data[:, envs_idxs, :], matrix)
        transitions = self._unflatten_fn(batch)
        return buffer_state.replace(key=key), transitions

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["config", "env", "apply_fn"])
    def flatten_crl_fn(config, env, transition: Transition, sample_key: PRNGKey, goal_indicies, contrastive_params, apply_fn) -> Transition:
        goal_key, transition_key = jax.random.split(sample_key)
        
        # Because it's vmaped transition obs.shape is of shape (transitions,obs_dim)
        seq_len = transition.observation.shape[0]
        arrangement = jnp.arange(seq_len)
        is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
        discount = config.discounting_cl ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)

        # Sample goal indices for computing the contrastive reward
        if config.future_state_rwd_sampling == "geometric":
            print("sample from the geometric distribution")
            probs_for_rwd = is_future_mask * discount 
        elif config.future_state_rwd_sampling == "uniform":
            print("sample from the uniform distribution")
            discount = 1 ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
            probs_for_rwd = is_future_mask * discount
        elif config.future_state_rwd_sampling == "inv_geometric":
            print("sample from inverse geometric")
            probs_for_rwd = is_future_mask * discount
            probs_for_rwd = jnp.flip(probs_for_rwd, axis=-1)
        elif config.future_state_rwd_sampling == "gaussian":
            print("sample from gaussian distribution")
            mean = 1.0 / (1.0 - discount)
            std = 1.0
            # Generate gaussian probabilities for future states
            diff = jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
            probs_for_rwd = jnp.exp(-0.5 * ((diff - mean) / std) ** 2)
            # Only consider future states and normalize
            probs_for_rwd = probs_for_rwd * is_future_mask
        elif "sim_score" in config.future_state_rwd_sampling:
            '''
            # 1. take the future states and convert them to goals
            # 2. get the state and goal representations
            # 3. compute the score
            '''
            future_state = transition.observation
            future_state_goal = future_state[:, goal_indicies]
            state_rep, goal_rep, _ = apply_fn(contrastive_params, transition.observation[:, :env.state_dim], transition.action, future_state_goal, sample_key, args.da, train=False)
            score = -jnp.sum(jnp.abs(state_rep[:, None, :] - goal_rep[None, :, :]), axis=-1)
            score = score * is_future_mask
            # sample accotding to the negative similarity score
            if "neg" in config.future_state_rwd_sampling:
                probs_for_rwd = jax.lax.stop_gradient(jnp.exp(-score))
            elif "pos" in config.future_state_rwd_sampling:
                probs_for_rwd = jax.lax.stop_gradient(jnp.exp(score))
            
        # sample goals for training the contrastive model
        probs = is_future_mask * discount
        
        single_trajectories = jnp.concatenate([transition.extras["state_extras"]["seed"][:, jnp.newaxis].T] * seq_len, axis=0)
        probs_for_rwd = probs_for_rwd * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
        probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5

        goal_index = jax.random.categorical(goal_key, jnp.log(probs))
        future_state = jnp.take(transition.observation, goal_index[:-1], axis=0)
        # print(future_state.shape)
        # import pdb;pdb.set_trace()
        future_action = jnp.take(transition.action, goal_index[:-1], axis=0)
        
        goal = future_state[:, goal_indicies]
        future_state = future_state[:, :env.state_dim]
        state = transition.observation[:-1, :env.state_dim]
        new_obs = jnp.concatenate([state, goal], axis=1)
        future_reward = jnp.take(transition.reward, goal_index[:-1], axis=0)

        rwd_goal_index = jax.random.categorical(goal_key, jnp.log(probs_for_rwd))
        future_state_for_rwd = jnp.take(transition.observation, rwd_goal_index[:-1], axis=0)

        # import pdb;pdb.set_trace()
        extras = {
            "policy_extras": {},
            "state_extras": {
                "truncation": jnp.squeeze(transition.extras["state_extras"]["truncation"][:-1]),
                "seed": jnp.squeeze(transition.extras["state_extras"]["seed"][:-1]),
            },
            "state": state,
            "future_state": future_state,
            "future_action": future_action,
            "future_reward": future_reward,
            "future_state_for_rwd": future_state_for_rwd
        }

        return transition._replace(
            observation=jnp.squeeze(new_obs),
            action=jnp.squeeze(transition.action[:-1]),
            reward=jnp.squeeze(transition.reward[:-1]),
            discount=jnp.squeeze(transition.discount[:-1]),
            extras=extras,
        )





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

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]


                   
def main(args):
    sgd_to_env = (
        args.num_envs
        * args.episode_length
        * args.multiplier_num_sgd_steps
        / args.batch_size
    ) / (args.num_envs * args.unroll_length)
    print(f"SGD steps per env steps: {sgd_to_env}")
    args.sgd_to_env = sgd_to_env

    args.num_evals_after_init = max(args.num_evals - 1, 1)
    args.env_steps_per_actor_step = args.num_envs * args.unroll_length
    args.num_prefill_actor_steps = args.min_replay_size // args.unroll_length + 1
    args.num_prefill_env_steps = args.num_prefill_actor_steps *args.env_steps_per_actor_step
    args.num_training_steps_per_epoch = -(
        -(args.num_timesteps - args.num_prefill_env_steps) // (args.num_evals_after_init * args.env_steps_per_actor_step)
    )
    print(f"env_steps_per_actor_step: {args.env_steps_per_actor_step}")
    print("Num_prefill_actor_steps: ", args.num_prefill_actor_steps)
    print(f"Number of training steps per epoch: {args.num_training_steps_per_epoch}")



    # run_name = f"{args.env_name}"

    rnd_rms = RMS()

    scratch_path = os.getenv("SCRATCH")
    runs_path = os.path.join(scratch_path, "crl_runs")  
    os.makedirs(runs_path, exist_ok=True)
    exp_dir = os.path.join(args.model, args.env_name, args.run_name_suffix)   
    # /exp_dir = os.path.join(runs_path, exp_dir)  
    # os.makedirs(exp_dir, exist_ok=True)
    word = RandomWord().word()
    uid = f"{int(time.time())}_{word}"
    while os.path.exists(f"runs/{exp_dir}/{uid}"):
        word = RandomWord().word()
        uid = f"{int(time.time())}_{word}"
    run_dir = f"{runs_path}/{exp_dir}/{uid}"
    ckpt_dir = run_dir + '/ckpt'
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    # import pdb;pdb.set_trace()

    

    process_id = jax.process_index()
    local_devices_to_use = jax.local_device_count()
    if args.min_replay_size >= args.num_timesteps:
        raise ValueError("No training will happen because min_replay_size >= num_timesteps")
    
    if args.max_replay_size is None:
        max_replay_size = args.num_timesteps

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)
    rng, key = jax.random.split(rng)


    global_key, local_key = jax.random.split(rng)
    local_key = jax.random.fold_in(local_key, process_id)
    

    local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)
    

    # Environment setup    
    env = create_env(args)
    eval_env = create_eval_env(args)
    if args.render_agent:
        eval_env_render = create_env(args)
        # eval_env_render.step = jax.jit(eval_env_render.step)
    config = get_env_config(args)
    v_randomization_fn = None
    randomization_fn = None
    if randomization_fn is not None:
        v_randomization_fn = functools.partial(
            randomization_fn,
            rng=jax.random.split(key, args.num_envs // jax.process_count() // local_devices_to_use),
        )

    if isinstance(env, envs.Env):
        wrap_for_training = envs.training.wrap
    else:
        wrap_for_training = envs_v1.wrappers.wrap_for_training

    env = wrap_for_training(
        env,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        randomization_fn=v_randomization_fn,
    )
    # import pdb;pdb.set_trace()  

    obs_size = env.observation_size
    action_size = env.action_size
    print(f"Obs size: {obs_size}")
    print(f"action size: {action_size}")
    # Env init
    env_keys = jax.random.split(env_key, args.num_envs // jax.process_count())
    env_keys = jnp.reshape(env_keys, (local_devices_to_use, -1) + env_keys.shape[1:])
    env_state = jax.pmap(env.reset)(env_keys)
    args.obs_dim = env.state_dim
    args.action_dim = env.action_size


    # Network setup
    network_factory = sac_networks.make_sac_networks
    def pre_process(x,y):
        if x.ndim > 1:
            x = x[:, :env.state_dim]
        else:
            x = x[:env.state_dim]
        return x
    # make sac networks and optimizers
    normalize_fn = lambda x, y: x
    agent_hidden_dims = [args.agent_hidden_dim]*args.agent_number_hiddens
    sac_network = network_factory(
        observation_size=env.state_dim, action_size=action_size, preprocess_observations_fn=pre_process, layer_norm=args.layer_norm, activation=eval(args.agent_activation), hidden_layer_sizes=agent_hidden_dims
    )
    
    # import pdb;pdb.set_trace()
    make_policy = sac_networks.make_inference_fn(sac_network)

    alpha_optimizer = optax.adam(learning_rate=args.alpha_lr)

    policy_optimizer = optax.adam(learning_rate=args.actor_lr)
    q_optimizer = optax.adam(learning_rate=args.critic_lr)

    ## TODO: Should we condition the Q-function on the future state?

    if args.crl_observation_dim > 0:
        args.crl_goal_indices = jnp.arange(args.crl_observation_dim)
    else:
        args.crl_goal_indices = jnp.arange(env.state_dim) if args.use_complete_future_state else env.goal_indices


    if args.crl_observation_dim == 0:
        args.crl_observation_dim = env.state_dim if args.use_complete_future_state else env.goal_indices.shape[-1]

    # Make the contrastive critic
    contrastive_network = ContrastiveCritic(args)
    contrastive_optimizer = optax.adam(learning_rate=args.critic_lr)
    
    # create the transition object
    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))
    dummy_transition = Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=dummy_obs,
        next_observation=dummy_obs,
        action=dummy_action,
        reward=0.0,
        discount=0.0,
        extras={
            "state_extras": {
                "truncation": 0.0,
                "seed": 0.0,
            },
            "policy_extras": {},
        },
    )

    # create replay buffer
    replay_buffer = jit_wrap(
        TrajectoryUniformSamplingQueue(
            max_replay_size=args.max_replay_size,
            dummy_data_sample=dummy_transition,
            sample_batch_size=args.batch_size,
            num_envs=args.num_envs,
            episode_length=args.episode_length,
        )
    )
    
    # create losses and update functions
    alpha_loss, critic_loss, actor_loss = sac_losses.make_losses(
        sac_network=sac_network, reward_scaling=args.reward_scaling, discounting=args.discounting, action_size=action_size
    )
    # contrastive_loss = make_contrastive_loss(contrastive_network)
    alpha_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
        alpha_loss, alpha_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )
    critic_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
        critic_loss, q_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )
    actor_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
        actor_loss, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )
    @flax.struct.dataclass
    class CRLNetworks:
        critic_network: nn.Module
    crl_networks = CRLNetworks(
        critic_network=contrastive_network
    )
    # TODO: add function for contrastive reward, modify the training state, replace rnd reward with the critic reward
    contrastive_loss = make_contrastive_loss(crl_networks, args)
    contrastive_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
        contrastive_loss, contrastive_optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
    

    print(f"Experiment directory (run_dir) is: {run_dir}")
    # import pdb;pdb.set_trace()
    args.run_dir = run_dir
    args.ckpt_dir = ckpt_dir
    # args.run_name = run_name
    # import pdb;pdb.set_trace()
    save_args(args, run_dir)
    if args.track:
        if args.wandb_group ==  '.':
            args.wandb_group = None
            
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            group=args.wandb_group,
            # dir=args.wandb_dir,
            config=vars(args),
            name="_".join(exp_dir.split("/")),
            monitor_gym=True,
            save_code=True,
        )

        if args.wandb_mode == 'offline':
            wandb_osh.set_log_level("ERROR")
            trigger_sync = TriggerWandbSyncHook()


    density = DiscretizedDensity(goal_dim=env.goal_indices.shape[-1], bin_width=0.5, run_folder=run_dir)


    ######## Methods #######
    def _unpmap(v):
        return jax.tree_util.tree_map(lambda x: x[0], v)
    def _init_training_state(
    key: PRNGKey,
    obs_size: int,
    future_obs_size: int,
    local_devices_to_use: int,
    sac_network: sac_networks.SACNetworks,
    contrastive_network: nn.Module,
    alpha_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
    contrastive_optimizer: optax.GradientTransformation,
) -> TrainingState:
        """Inits the training state and replicates it over devices."""
        key_policy, key_q, key_contrastive = jax.random.split(key, num=3)
        log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
        alpha_optimizer_state = alpha_optimizer.init(log_alpha)
        dummy_state = jnp.zeros((1, obs_size))
        dummy_action = jnp.zeros((1, args.action_dim))
        dummy_future_state = jnp.zeros((1, future_obs_size))

        policy_params = sac_network.policy_network.init(key_policy)
        policy_optimizer_state = policy_optimizer.init(policy_params)
        q_params = sac_network.q_network.init(key_q)
        q_optimizer_state = q_optimizer.init(q_params)
        contrastive_params = contrastive_network.init(key_contrastive, dummy_state, dummy_action, dummy_future_state, key_contrastive, False)
        contrastive_optimizer_state = contrastive_optimizer.init(contrastive_params)

        normalizer_params = running_statistics.init_state(specs.Array((obs_size,), jnp.dtype("float32")))

        training_state = TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            q_optimizer_state=q_optimizer_state,
            q_params=q_params,
            contrastive_optimizer_state=contrastive_optimizer_state,
            contrastive_params=contrastive_params,
            target_q_params=q_params,
            gradient_steps=jnp.zeros(()),
            env_steps=jnp.zeros(()),
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=log_alpha,
            normalizer_params=normalizer_params,
            mean_coverage=jnp.zeros(()),
            contrastive_params_EMA=contrastive_params
        )
        return jax.device_put_replicated(training_state, jax.local_devices()[:local_devices_to_use])

    def sgd_step(
        carry: Tuple[TrainingState, PRNGKey], transitions: Transition
    ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
        training_state, key = carry

        key, key_alpha, key_critic, key_actor = jax.random.split(key, 4)

        alpha_loss, alpha_params, alpha_optimizer_state = alpha_update(
            training_state.alpha_params,
            training_state.policy_params,
            training_state.normalizer_params,
            transitions,
            key_alpha,
            optimizer_state=training_state.alpha_optimizer_state,
        )
        alpha = jnp.exp(training_state.alpha_params) * args.entropy_reg
        critic_loss, q_params, q_optimizer_state = critic_update(
            training_state.q_params,
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.target_q_params,
            alpha,
            transitions,
            key_critic,
            optimizer_state=training_state.q_optimizer_state,
        )
        # print(transitions.observation.shape)
        # import pdb;pdb.set_trace()
        actor_loss, policy_params, policy_optimizer_state = actor_update(
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.q_params,
            alpha,
            transitions,
            key_actor,
            optimizer_state=training_state.policy_optimizer_state,
        )
        # contrastive_loss, contrastive_params, contrastive_optimizer_state = rnd_update(
        #     training_state.contrastive_params, 
        #     training_state.normalizer_params,
        #     transitions,
        #     args.rnd_goal_indices,
        #     obs_rms.M,
        #     obs_rms.S,
        #     optimizer_state=training_state.contrastive_optimizer_state
        # )

        (contrastive_loss, contrastive_metrics), contrastive_params, contrastive_optimizer_state  = contrastive_update(
            training_state.contrastive_params,
            transitions, 
            key_critic,
            optimizer_state=training_state.contrastive_optimizer_state)

        # calculate the knn covarege metric        
        coverage = knn_average_distance(transitions.extras["future_state"][:,env.goal_indices])
        training_state = training_state.replace(mean_coverage=(training_state.mean_coverage + coverage.mean()))
        actions_mean = transitions.action.mean()
        actions_std = transitions.action.std()


        new_target_q_params = jax.tree_util.tree_map(
            lambda x, y: x * (1 - args.tau) + y * args.tau, training_state.target_q_params, q_params
        )

        # print(training_state.mean_coverage/training_state.gradient_steps)
        metrics = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "contrastive_loss": contrastive_loss,
            "alpha": jnp.exp(alpha_params),
            "contrastive_reward_mean": transitions.reward.mean(),
            "contrastive_reward_max": transitions.reward.max(),
            "contrastive_reward_min": transitions.reward.min(),
            "mean_coverage": training_state.mean_coverage/training_state.gradient_steps,
            "knn_coverage": coverage.mean(),
            "grad_steps": training_state.gradient_steps,
            "contrastive_loss": contrastive_loss,
            "actions_mean": actions_mean,
            "actions_std": actions_std,
        }
        metrics.update(contrastive_metrics)

        # update the EMA
        updated_ema = jax.tree_util.tree_map(
            lambda x, y: args.ema * x + (1-args.ema) * y, training_state.contrastive_params_EMA, contrastive_params
        )
        
        new_training_state = TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            q_optimizer_state=q_optimizer_state,
            q_params=q_params,
            contrastive_optimizer_state=contrastive_optimizer_state,
            contrastive_params=contrastive_params,
            target_q_params=new_target_q_params,
            gradient_steps=training_state.gradient_steps + 1,
            env_steps=training_state.env_steps,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            normalizer_params=training_state.normalizer_params,
            mean_coverage=training_state.mean_coverage,
            contrastive_params_EMA=updated_ema
            
        )
        return (new_training_state, key), metrics

    def get_experience(
        normalizer_params: running_statistics.RunningStatisticsState,
        policy_params: Params,
        env_state: Union[envs.State, envs_v1.State],
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[
        running_statistics.RunningStatisticsState,
        Union[envs.State, envs_v1.State],
        ReplayBufferState,
    ]:
        policy = make_policy((normalizer_params, policy_params))

        @jax.jit
        def f(carry, unused_t):
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            env_state, transition = actor_step(
                env,
                env_state,
                policy,
                current_key,
                extra_fields=(
                    "truncation",
                    "seed",
                ),
            )
            return (env_state, next_key), transition

        (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=args.unroll_length)

        # normalizer_params = running_statistics.update(
        #     normalizer_params,
        #     jax.tree_util.tree_map(
        #         lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
        #     ).observation,  # so that batch size*unroll_length is the first dimension
        #     pmap_axis_name=_PMAP_AXIS_NAME,
        # )
        buffer_state = replay_buffer.insert(buffer_state, data)
        return normalizer_params, env_state, buffer_state

    def training_step(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, Union[envs.State, envs_v1.State], ReplayBufferState, Metrics]:
        experience_key, training_key = jax.random.split(key)
        normalizer_params, env_state, buffer_state = get_experience(
            training_state.normalizer_params,
            training_state.policy_params,
            env_state,
            buffer_state,
            experience_key,
        )
        training_state = training_state.replace(
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + args.env_steps_per_actor_step,
        )

        training_state, buffer_state, metrics = additional_sgds(training_state, buffer_state, training_key)
        return training_state, env_state, buffer_state, metrics

    def prefill_replay_buffer(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            new_normalizer_params, env_state, buffer_state = get_experience(
                training_state.normalizer_params,
                training_state.policy_params,
                env_state,
                buffer_state,
                key,
            )
            new_training_state = training_state.replace(
                normalizer_params=new_normalizer_params,
                env_steps=training_state.env_steps + args.env_steps_per_actor_step,
            )
            return (new_training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=args.num_prefill_actor_steps,
        )[0]
    def additional_sgds(
        training_state: TrainingState,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, ReplayBufferState, Metrics]:
        experience_key, training_key, sampling_key = jax.random.split(key, 3)
        buffer_state, transitions = replay_buffer.sample(buffer_state)

        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        # import pdb;pdb.set_trace()

        transitions = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, None, 0, 0, None, None, None))(
            config, env, transitions, batch_keys, args.crl_goal_indices, training_state.contrastive_params, crl_networks.critic_network.apply
        )

        # Shuffle transitions and reshape them into (number_of_sgd_steps, batch_size, ...)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )
        permutation = jax.random.permutation(experience_key, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, args.batch_size) + x.shape[1:]),
            transitions,
        )

        crl_rewards = crl_reward(crl_networks.critic_network, training_state.contrastive_params, transitions, args, key)
        transitions = transitions._replace(
            reward=crl_rewards
        )
        # print((transitions.reward == rescaled_rnd_rewards).all())

        (training_state, _), metrics = jax.lax.scan(sgd_step, (training_state, training_key), transitions)
        return training_state, buffer_state, metrics

    def scan_additional_sgds(n, ts, bs, a_sgd_key):

        def body(carry, unsued_t):
            ts, bs, a_sgd_key = carry
            new_key, a_sgd_key = jax.random.split(a_sgd_key)
            ts, bs, metrics = additional_sgds(ts, bs, a_sgd_key)
            return (ts, bs, new_key), metrics

        return jax.lax.scan(body, (ts, bs, a_sgd_key), (), length=n)

    def training_epoch(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        def f(carry, unused_t):
            ts, es, bs, k = carry
            k, new_key, a_sgd_key = jax.random.split(k, 3)
            ts, es, bs, metrics = training_step(ts, es, bs, k)
            (ts, bs, a_sgd_key), _ = scan_additional_sgds(args.multiplier_num_sgd_steps - 1, ts, bs, a_sgd_key)
            return (ts, es, bs, new_key), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=args.num_training_steps_per_epoch,
        )
        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, env_state, buffer_state, metrics

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
        training_state: TrainingState, env_state: envs.State, buffer_state: ReplayBufferState, key: PRNGKey
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        nonlocal training_walltime
        t = time.time()
        (training_state, env_state, buffer_state, metrics) = training_epoch(
            training_state, env_state, buffer_state, key
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (args.env_steps_per_actor_step * args.num_training_steps_per_epoch) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return training_state, env_state, buffer_state, metrics  # pytype: disable=bad-return-type  # py311-upgrade
    
    prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME)
    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)


    
    # Training state init
    training_state = _init_training_state(
        key=global_key,
        obs_size=args.obs_dim,
        future_obs_size=args.crl_observation_dim,
        local_devices_to_use=local_devices_to_use,
        sac_network=sac_network,
        contrastive_network=contrastive_network,
        alpha_optimizer=alpha_optimizer,
        policy_optimizer=policy_optimizer,
        q_optimizer=q_optimizer,
        contrastive_optimizer=contrastive_optimizer
    )
    del global_key

    # Replay buffer init
    buffer_state = jax.pmap(replay_buffer.init)(jax.random.split(rb_key, local_devices_to_use))

    if not eval_env:
        eval_env = env
    if randomization_fn is not None:
        v_randomization_fn = functools.partial(randomization_fn, rng=jax.random.split(eval_key, num_eval_envs))

    evaluator = ActorCrlEvaluator(
        eval_env,
        functools.partial(make_policy, deterministic=args.deterministic_eval),
        num_eval_envs=args.num_eval_envs,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        key=eval_key,
    )

    # Run initial eval
    metrics_recorder = MetricsRecorder(args.num_timesteps)

    def ensure_metric(metrics, key):
        if key not in metrics:
            metrics[key] = 0
        else:
            if math.isnan(metrics[key]):
                print(f"Metric: {key} is Nan")
                # raise Exception(f"Metric: {key} is Nan")

    metrics_to_collect = [
        "eval/episode_reward",
        "eval/episode_success",
        "eval/episode_success_any",
        "eval/episode_success_hard",
        "eval/episode_success_easy",
        "eval/episode_reward_dist",
        "eval/episode_reward_near",
        "eval/episode_reward_ctrl",
        "eval/episode_dist",
        "eval/episode_reward_survive",
        "training/actor_loss",
        "training/critic_loss",
        "training/contrastive_loss",
        "training/contrastive_reward_mean",
        "training/contrastive_reward_max",
        "training/contrastive_reward_min",
        "training/knn_coverage",
        "training/mean_coverage",
        "training/sps",
        "training/entropy",
        "training/alpha",
        "training/alpha_loss",
        "training/entropy",
        "training/grad_steps",
        "training/num_visited_unique_state",
        "training/visited_state_entorpy",
        "training/categorical_accuracy",
        "training/logits_pos",
        "training/logits_neg",
        "training/logsumexp",
        "training/binary_accuracy",
        "training/actions_mean",
        "training/actions_std",
        "training/gamma"
    ]

    # import pdb;pdb.set_trace()
    csv_logger_path = os.path.join(run_dir, "logs.csv") 
    metrics_to_collect_logger = metrics_to_collect.copy()
    metrics_to_collect_logger.append("training_steps")
    _logger = Simple_CSV_logger(csv_logger_path, header=metrics_to_collect_logger)

    def progress(num_steps, metrics):
        for key in metrics_to_collect:
            ensure_metric(metrics, key)
        metrics_recorder.record(
            num_steps,
            {key: value for key, value in metrics.items() if key in metrics_to_collect},
        )
        if args.track:
            metrics_recorder.log_wandb()
        metrics_recorder.print_progress()

    metrics = {}
    if process_id == 0 and args.num_evals > 1:
        metrics = evaluator.run_evaluation(
            _unpmap((training_state.normalizer_params, training_state.policy_params)), training_metrics={}
        )
        logging.info(metrics)
        progress(0, metrics)

    # Create and initialize the replay buffer.
    t = time.time()
    prefill_key, local_key = jax.random.split(local_key)
    prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_keys
    )

    new_state = buffer_state.replace(
            data=buffer_state.data[0],
            key=buffer_state.key[0],
            insert_position= buffer_state.insert_position[0],
            sample_position= buffer_state.sample_position[0],
        )
        ## Testing the discretized density for state coverage
    _, sample = replay_buffer.sample(new_state)
    # obs_rms(sample.observation[:, :, args.rnd_goal_indices].reshape(-1, args.rnd_observation_dim))
    # import pdb;pdb.set_trace()

    

    replay_size = jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
    logging.info("replay size after prefill %s", replay_size)
    assert replay_size >= args.min_replay_size
    training_walltime = time.time() - t

    current_step = 0

    # rendering_epochs =[int(t) for t in np.linspace(0, args.num_evals_after_init, 15)][1:]
    # print("rendering_epochs")
    # print(rendering_epochs)
    videos_indices = np.linspace(0, args.num_evals_after_init-1, args.num_videos).astype(int)
    print("videos_indices")
    print(videos_indices)
    
    # training loop!
    # import pdb;pdb.set_trace()
    for epoch in range(args.num_evals_after_init):
        print(f"epcoh: {epoch}")
        if epoch in videos_indices and args.render_agent:
            print("rendering")
            render(make_policy, _unpmap((training_state.normalizer_params, training_state.policy_params)), eval_env_render, run_dir, args.exp_name, seed=args.seed, timestep=current_step)


        logging.info("step %s", current_step)
        logging.info("epoch %s", epoch)

        # Optimization
        epoch_key, local_key = jax.random.split(local_key)
        epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
        (training_state, env_state, buffer_state, training_metrics) = training_epoch_with_timing(
            training_state, env_state, buffer_state, epoch_keys
        )
        current_step = int(_unpmap(training_state.env_steps))

        new_state = buffer_state.replace(
            data=buffer_state.data[0],
            key=buffer_state.key[0],
            insert_position= buffer_state.insert_position[0],
            sample_position= buffer_state.sample_position[0],
        )
        ## Testing the discretized density for state coverage
        _, sample = replay_buffer.sample(new_state)
        # import pdb;pdb.set_trace()
        path = os.path.join(run_dir, "buffer_data")
        os.makedirs(path, exist_ok=True)
        # save_buffer_sample(sample, path, current_step)
        density.update_count(sample.observation[:, :, env.goal_indices], current_step)
        coverage_metrics = {
            "num_visited_unique_state": density.num_states(),
            "visited_state_entorpy": density.entropy()
        }
        
        
        # import pdb;pdb.set_trace()
        # obs_rms(sample.observation[:, :, args.rnd_goal_indices].reshape(-1, args.rnd_observation_dim))
        for k in coverage_metrics:
            training_metrics[f"training/{k}"] = coverage_metrics[k]

        gamma_schedule(args, current_step, args.num_timesteps)
        training_metrics[f"training/gamma"] = args.discounting_cl

        # Eval and logging
        if process_id == 0:
            if ckpt_dir and args.checkpoint:
                # Save current policy.
                print("Saved Model params")
                params = _unpmap((training_state.normalizer_params, training_state.policy_params))
                path = f"{ckpt_dir}/sac_final.pkl"
                model.save_params(path, params)
                params = _unpmap((training_state.contrastive_params))
                if args.save_all_crl_ckpts:
                     path = f"{ckpt_dir}/sac_crl_{current_step}.pkl"
                else:
                    path = f"{ckpt_dir}/sac_crl_final.pkl"
                model.save_params(path, params)
                params = _unpmap(training_state.contrastive_params_EMA)
                path = f"{ckpt_dir}/sac_crl_ema.pkl"
                model.save_params(path, params)
                # import pdb;pdb.set_trace()


            # Run evals.
            metrics = evaluator.run_evaluation(
                _unpmap((training_state.normalizer_params, training_state.policy_params)), training_metrics
            )
            metrics["epoch"] = epoch
            logging.info(metrics)
            # import pdb;pdb.set_trace()
            logger_metrics = deepcopy(metrics)
            logger_metrics["training_steps"] = current_step
            _logger.log(logger_metrics)
            progress(current_step, metrics)

    total_steps = current_step
    assert total_steps >= args.num_timesteps

    params = _unpmap((training_state.normalizer_params, training_state.policy_params))

    # If there was no mistakes the training_state should still be identical on all
    # devices.
    brax_pmap.assert_is_replicated(training_state)
    logging.info("total steps: %s", total_steps)
    brax_pmap.synchronize_hosts()



    
if __name__ == "__main__":
    args = tyro.cli(Args)
    # import pdb; pdb.set_trace()
    main(args)







    











    
    
        
# (50000000 - 1024 x 1000) / 50 x 1024 x 62 = 15        #number of actor steps per epoch (which is equal to the number of training steps)
# 1024 x 999 / 256 = 4000                               #number of gradient steps per actor step 
# 1024 x 62 / 4000 = 16                                 #ratio of env steps per gradient step