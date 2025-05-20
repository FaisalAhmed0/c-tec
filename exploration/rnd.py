import os
os.environ['MUJOCO_GL'] = 'osmesa'
import jax
import flax
import jax.random
import tyro
import time
import optax
import wandb
import random
import wandb_osh
import numpy as np
import flax.linen as nn
import jax.numpy as jnp
import functools
import model_utils as sac_networks
import math

from args import RND_args
from absl import logging
from jax import lax
from copy import deepcopy
from typing import Any, Tuple, TypeVar,Union, NamedTuple, Sequence
from wandb_osh.hooks import TriggerWandbSyncHook
from evaluator import ActorCrlEvaluator

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
from utils import MetricsRecorder, create_env, create_eval_env,\
      DiscretizedDensity, Simple_CSV_logger, get_env_config, \
        knn_average_distance, render, update_rms, \
        load_params, save_params, save_args, save_buffer_sample
from intrinsic_rewards import rnd_reward
from buffers import SacTrajectoryUniformSamplingQueue
from losses import make_rnd_loss
from models import RND, StateEncoder
from wonderwords import RandomWord
from buffers import QueueBase, Generic
Metrics = types.Metrics

ReplayBufferState = Any
_PMAP_AXIS_NAME = "i"

# Transition = types.Transition
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]
Transition = types.Transition
Sample = TypeVar("Sample")

def actor_step(
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    extra_fields: Sequence[str] = (),
) -> Tuple[State, Transition]:
    """Collect data."""
    # print(env_state.obs.shape)
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



@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    policy_optimizer_state: optax.OptState
    policy_params: Params
    q_optimizer_state: optax.OptState
    q_params: Params
    rnd_optimizer_state: optax.OptState
    rnd_params: Params
    target_q_params: Params
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray
    alpha_optimizer_state: optax.OptState
    alpha_params: Params
    normalizer_params: running_statistics.RunningStatisticsState
    mean_coverage: jnp.ndarray
    rnd_rms_state: Any
    rnd_obs_rms_state: Any


class Transition(NamedTuple):
    """Container for a transition."""

    observation: NestedArray
    next_observation: NestedArray
    action: NestedArray
    reward: NestedArray
    discount: NestedArray
    extras: NestedArray = ()  # pytype: disable=annotation-type-mismatch  # jax-ndarray




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



    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}"

    
    scratch_path = os.getenv("SCRATCH")
    runs_path = os.path.join(scratch_path, "crl_runs")  
    os.makedirs(runs_path, exist_ok=True)
    exp_dir = os.path.join(args.model, args.env_name, args.run_name_suffix)   
    # /exp_dir = os.path.join(runs_path, exp_dir)  
    os.makedirs(exp_dir, exist_ok=True)
    word = RandomWord().word()
    uid = f"{int(time.time())}_{word}"
    while os.path.exists(f"runs/{exp_dir}/{uid}"):
        word = RandomWord().word()
        uid = f"{int(time.time())}_{word}"
    run_dir = f"{runs_path}/{exp_dir}/{uid}"
    ckpt_dir = run_dir + '/ckpt'
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    # 

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

    obs_size = env.observation_size
    action_size = env.action_size
    print(f"Obs size: {obs_size}")
    print(f"action size: {action_size}")
    # Env init
    env_keys = jax.random.split(env_key, args.num_envs // jax.process_count())
    env_keys = jnp.reshape(env_keys, (local_devices_to_use, -1) + env_keys.shape[1:])
    env_state = jax.pmap(env.reset)(env_keys)


    # Network setup
    network_factory = sac_networks.make_sac_networks
    # make sac networks and optimizers
    normalize_fn = lambda x, y: x
    agent_hidden_dims = [args.agent_hidden_dim]*args.agent_number_hiddens
    sac_network = network_factory(
        observation_size=obs_size, action_size=action_size, preprocess_observations_fn=normalize_fn, layer_norm=args.layer_norm, activation=eval(args.activation), hidden_layer_sizes=agent_hidden_dims
    )
    # 
    make_policy = sac_networks.make_inference_fn(sac_network)

    alpha_optimizer = optax.adam(learning_rate=args.alpha_lr)

    policy_optimizer = optax.adam(learning_rate=args.actor_lr)
    q_optimizer = optax.adam(learning_rate=args.critic_lr)

    # make rnd model and optimizer
    args.rnd_observation_dim = env.state_dim if args.use_complete_future_state else env.goal_indices.shape[-1]

    rnd_network = RND(args=args)
    rnd_optimizer = optax.adam(learning_rate=args.rnd_lr)

    args.rnd_goal_indices = jnp.arange(env.state_dim) if args.use_complete_future_state else env.goal_indices
    
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
        SacTrajectoryUniformSamplingQueue(
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
    rnd_loss = make_rnd_loss(rnd_network)
    alpha_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
        alpha_loss, alpha_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )
    critic_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
        critic_loss, q_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )
    actor_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
        actor_loss, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )
    rnd_update = gradients.gradient_update_fn(
        rnd_loss, rnd_optimizer,  pmap_axis_name=_PMAP_AXIS_NAME
    )

    print(f"Experiment directory (run_dir) is: {run_dir}")
    # 
    args.run_dir = run_dir
    args.ckpt_dir = ckpt_dir
    args.run_name = run_name
    # 
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
            name=run_name,
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
    rnd_network: nn.Module,
    alpha_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
    rnd_optimizer: optax.GradientTransformation,
) -> TrainingState:
        """Inits the training state and replicates it over devices."""

        # rms state for rnd reward and obs normalization
        rnd_rms_state = (0.0, jnp.zeros(1,), jnp.zeros((1, )))
        rnd_obs_rms_state = (0.0, jnp.zeros((future_obs_size,)), jnp.zeros((future_obs_size, )))

        key_policy, key_q, key_rnd = jax.random.split(key, num=3)
        log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
        alpha_optimizer_state = alpha_optimizer.init(log_alpha)
        dummy_state = jnp.zeros((1, future_obs_size))

        policy_params = sac_network.policy_network.init(key_policy)
        policy_optimizer_state = policy_optimizer.init(policy_params)
        q_params = sac_network.q_network.init(key_q)
        q_optimizer_state = q_optimizer.init(q_params)
        rnd_params = rnd_network.init(key_rnd, dummy_state, rnd_obs_rms_state[1], rnd_obs_rms_state[2])
        rnd_optimizer_state = rnd_optimizer.init(rnd_params)

        normalizer_params = running_statistics.init_state(specs.Array((obs_size,), jnp.dtype("float32")))

        

        training_state = TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            q_optimizer_state=q_optimizer_state,
            q_params=q_params,
            rnd_optimizer_state=rnd_optimizer_state,
            rnd_params=rnd_params,
            target_q_params=q_params,
            gradient_steps=jnp.zeros(()),
            env_steps=jnp.zeros(()),
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=log_alpha,
            normalizer_params=normalizer_params,
            mean_coverage=jnp.zeros(()),
            rnd_rms_state = rnd_rms_state,
            rnd_obs_rms_state = rnd_obs_rms_state,
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
        alpha = jax.lax.stop_gradient(jnp.exp(training_state.alpha_params)) * args.entropy_reg
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
        # 
        actor_loss, policy_params, policy_optimizer_state = actor_update(
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.q_params,
            alpha,
            transitions,
            key_actor,
            optimizer_state=training_state.policy_optimizer_state,
        )
        # update the obs rms
        # TODO: continute working on this
        # 
        rnd_obs_rms_state, (means, stds) = lax.scan(update_rms, training_state.rnd_obs_rms_state, transitions.next_observation.reshape(-1, obs_size)[:, args.rnd_goal_indices] )
        # 
        rnd_loss, rnd_params, rnd_optimizer_state = rnd_update(
            training_state.rnd_params, 
            training_state.normalizer_params,
            transitions,
            args.rnd_goal_indices,
            means[-1],
            stds[-1],
            optimizer_state=training_state.rnd_optimizer_state
        )

        # calculate the knn covarege metric        
        coverage = knn_average_distance(transitions.observation[:,env.goal_indices])
        training_state = training_state.replace(mean_coverage=(training_state.mean_coverage + coverage.mean()), rnd_obs_rms_state=rnd_obs_rms_state)


        new_target_q_params = jax.tree_util.tree_map(
            lambda x, y: x * (1 - args.tau) + y * args.tau, training_state.target_q_params, q_params
        )

        # print(training_state.mean_coverage/training_state.gradient_steps)
        metrics = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "rnd_loss": rnd_loss,
            "alpha": jnp.exp(alpha_params),
            "rnd_reward_mean": transitions.reward.mean(),
            "rnd_reward_max": transitions.reward.max(),
            "rnd_reward_min": transitions.reward.min(),
            "mean_coverage": training_state.mean_coverage/training_state.gradient_steps,
            "knn_coverage": coverage.mean(),
            "grad_steps": training_state.gradient_steps
        }

        new_training_state = TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            q_optimizer_state=q_optimizer_state,
            q_params=q_params,
            rnd_optimizer_state=rnd_optimizer_state,
            rnd_params=rnd_params,
            target_q_params=new_target_q_params,
            gradient_steps=training_state.gradient_steps + 1,
            env_steps=training_state.env_steps,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            normalizer_params=training_state.normalizer_params,
            mean_coverage=training_state.mean_coverage,
            rnd_rms_state=training_state.rnd_rms_state,
            rnd_obs_rms_state=rnd_obs_rms_state,
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

        normalizer_params = running_statistics.update(
            normalizer_params,
            jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
            ).observation,  # so that batch size*unroll_length is the first dimension
            pmap_axis_name=_PMAP_AXIS_NAME,
        )
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
        transitions = jax.vmap(SacTrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, None, 0, 0))(
            config, env, transitions, batch_keys
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


        # Compute RND reward, and replace the task reward with it.
        rnd_rewards, rnd_rms_state = rnd_reward(rnd_network, training_state.rnd_params, transitions, args.rnd_goal_indices, training_state.rnd_rms_state, training_state.rnd_obs_rms_state,
                                                args.rwd_rms)
        # replace the reward with rnd reward
        # 
        transitions = transitions._replace(
            reward=rnd_rewards
        )
        
        # print((transitions.reward == rescaled_rnd_rewards).all())

        (training_state, _), metrics = jax.lax.scan(sgd_step, (training_state, training_key), transitions)
        training_state = training_state.replace(
            rnd_rms_state = rnd_rms_state
        )
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
    ######## Methods #######

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
        obs_size=obs_size,
        future_obs_size=args.rnd_observation_dim,
        local_devices_to_use=local_devices_to_use,
        sac_network=sac_network,
        rnd_network=rnd_network,
        alpha_optimizer=alpha_optimizer,
        policy_optimizer=policy_optimizer,
        q_optimizer=q_optimizer,
        rnd_optimizer=rnd_optimizer
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
        "training/rnd_loss",
        "training/rnd_reward_mean",
        "training/rnd_reward_max",
        "training/rnd_reward_min",
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
        "training/obs_mean",
        "training/obs_std",
        "training/rnd_rms_mean",
        "training/rnd_rms_std",
    ]

    # 
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
    # 

    

    replay_size = jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
    logging.info("replay size after prefill %s", replay_size)
    assert replay_size >= args.min_replay_size
    training_walltime = time.time() - t

    current_step = 0

    videos_indices = np.linspace(0, args.num_evals_after_init-1, args.num_videos).astype(int)
    print("videos_indices")
    print(videos_indices)
    
    # training loop!
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
        density.update_count(sample.observation[:, :, env.goal_indices], current_step)
        coverage_metrics = {
            "num_visited_unique_state": density.num_states(),
            "visited_state_entorpy": density.entropy()
        }

        
        for k in coverage_metrics:
            training_metrics[f"training/{k}"] = coverage_metrics[k]


        training_metrics["training/obs_mean"] = training_state.rnd_obs_rms_state[1][-1].mean().item()
        training_metrics["training/obs_std"] = training_state.rnd_obs_rms_state[2][-1].mean().item()
        training_metrics["training/rnd_rms_mean"] = training_state.rnd_rms_state[1][-1].item()
        training_metrics["training/rnd_rms_std"] = training_state.rnd_rms_state[2][-1].item()


        # Eval and logging
        if process_id == 0:
            if ckpt_dir and args.checkpoint:
                # Save current policy.
                params = _unpmap((training_state.normalizer_params, training_state.policy_params))
                path = f"{ckpt_dir}_sac_{current_step}.pkl"
                model.save_params(path, params)

            # Run evals.
            metrics = evaluator.run_evaluation(
                _unpmap((training_state.normalizer_params, training_state.policy_params)), training_metrics
            )
            metrics["epoch"] = epoch
            logging.info(metrics)
            # 
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
    args = tyro.cli(RND_args)
    main(args)