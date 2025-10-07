import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from craftax.craftax_env import make_craftax_env_from_name

import wandb
from typing import NamedTuple
from jax import lax

from flax.training import orbax_utils
from models.actor_critic import ActorCritic, ActorCriticGaussian
from utils import create_csv_logger
from ctec_ppo_rnn import ScannedRNN
from wrappers import BraxGymnaxWrapper
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

from utils import wandb_bar_chart
from logz.batch_logging import batch_log, create_log_dict
from models.actor_critic import (
    ActorCritic,
    ActorCriticConv,
)
from models.contrastive_model import ContrastiveModel, EmpowermentModel
from losses import contrastive_losses
from models.icm import ICMEncoder, ICMForward, ICMInverse
from models.etd_models import ETDModel
from wonderwords import RandomWord
import json
from args import etd_ppo_args

from jax_wrappers import (
    LogWrapper,
    BraxGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)
from utils import DiscretizedDensity


# Code adapted from the original implementation made by Chris Lu
# Original code located at https://github.com/luchris429/purejaxrl

density = DiscretizedDensity(bin_width=0.5)

def save_args(args_dict, path):
    # convert to a dictionary 
    for k in args_dict:
        if isinstance(args_dict[k], jax.Array):
            args_dict[k] = args_dict[k].tolist()
    # save the file 
    file_path = os.path.join(path, 'args.json') 
    with open(file_path, 'w') as f:
        json.dump(args_dict, f)

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward_e: jnp.ndarray
    reward_i: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    ## some PPO configs
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    # print(config["NUM_UPDATES"])
    # quit()
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    print(config["MINIBATCH_SIZE"])
    env, env_params = BraxGymnaxWrapper(config), None
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    config["GOAL_DIM"] = 2 if "ant" in config["ENV_NAME"] else 3
    


    # # Wrap with a batcher, maybe using optimistic resets
    # if config["USE_OPTIMISTIC_RESETS"]:
    #     env = OptimisticResetVecEnvWrapper(
    #         env,
    #         num_envs=config["NUM_ENVS"],
    #         reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
    #     )
    # else:
    # env = AutoResetEnvWrapper(env)
    # env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

    # learning rate annealing
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac
    
    def mrn_distance(x, y):
        eps = 1e-6
        d = x.shape[-1]
        x_prefix = x[..., :d // 2]
        x_suffix = x[..., d // 2:]
        y_prefix = y[..., :d // 2]
        y_suffix = y[..., d // 2:]
        max_component = jnp.max(jax.nn.relu(x_prefix - y_prefix), axis=-1)
        l2_component = jnp.sqrt(jnp.square(x_suffix - y_suffix).sum(axis=-1) + eps)
        return max_component + l2_component

    similarity_methods = {
            "l2": lambda sa_repr, g_repr: -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1)),
            "l2_no_sqrt":  lambda sa_repr, g_repr: -jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1),
            "l1":  lambda sa_repr, g_repr: -jnp.sum(jnp.abs(sa_repr[:, None, :] - g_repr[None, :, :]), axis=-1),
            "dot": lambda sa_repr, g_repr: jnp.einsum("ik,jk->ij", sa_repr, g_repr), # if the vectors are normalized then this the cosine 
        }
    similarity_methods_for_rwd = {
            "l2": lambda sa_repr, g_repr: -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1)),
            "l2_no_sqrt": lambda sa_repr, g_repr: -(jnp.sum((sa_repr - g_repr) ** 2, axis=-1)),
            "l1":  lambda sa_repr, g_repr: -jnp.sum(jnp.abs(sa_repr - g_repr), axis=-1),
            "dot": lambda sa_repr, g_repr: jnp.einsum("ik,jk->i", sa_repr, g_repr), # if the vectors are normalized then this the cosine 
        }
    # import pdb;pdb.set_trace()
    similarity_method = similarity_methods[config["SIMILARITY_MEASURE"]]
    similarity_method_for_rwd = similarity_methods_for_rwd[config["SIMILARITY_MEASURE"]]
    csv_logger_path = os.path.join(config["RUN_DIR"], "logs.csv") 
    csv_logger = create_csv_logger(config["ENV_NAME"], csv_logger_path)

    

    def sample_future_state(rng, obs, dones):
        """
        Process a single environment trajectory.
        
        Parameters:
        rng   : PRNGKey (used for sampling)
        obs   : Array of shape (num_steps, feature_dim)
        dones : Boolean array of shape (num_steps,)
        
        Returns:
        future_obs: Array of shape (num_steps, feature_dim)
        """
        # obs = trajcectory.obs
        # dones = trajcectory.done
        max_steps = obs.shape[0]
        gamma = config["GAMMA_CL"]

        # Ensure the last step is terminal.
        dones = dones.at[-1].set(1)
        future_obs = jnp.zeros_like(obs)
        all_indices = jnp.arange(max_steps)
        rngs = jax.random.split(rng, max_steps)

        # Loop over time steps.
        for i in range(max_steps):
            # Find the first index j >= i where dones[j] is True.
            # This is valid because dones[-1] is set to True.
            first_done_after_i = i + jnp.argmax(dones[i:])

            # Create a mask: valid future indices are between i and first_done_after_i (inclusive).
            mask = (all_indices >= i) & (all_indices <= first_done_after_i)

            # Compute discounted probabilities for each time step relative to i.
            diff = all_indices - i
            probs = gamma ** diff

            # Zero out probabilities for indices that are not in the valid range.
            probs = jnp.where(mask, probs, 0.0)

            # Normalize probabilities.
            probs = probs / jnp.sum(probs)

            # Sample one future timestep (scalar) using the computed probabilities.
            future_timestep = jax.random.choice(rngs[0], all_indices, p=probs, shape=())

            # Set the future observation for time i.
            future_obs = future_obs.at[i].set(obs[future_timestep])
        
        return future_obs
        
        
        

    def train(rng):
        # INIT NETWORK, depending on the environment observation type
        network = ActorCriticGaussian(env.action_space(env_params).shape[0])
        etd_network = ETDModel(config)
        # import pdb;pdb.set_trace()
        
        if config["USE_EMPOWERMENT"]:
            emp_network = EmpowermentModel(config)
        # import pdb;pdb.set_trace()

        crl_state = {
            "crl_model": None
        }
        emp_state = {
            "emp_model": None
        }
        

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # Exploration state
        ex_state = {
            "icm_encoder": None,
            "icm_forward": None,
            "icm_inverse": None,
            "e3b_matrix": None,
        }
        obs_shape = env.observation_space(env_params).shape
    
        action_shape = env.action_space(env_params).shape[0]
        dones = jnp.zeros((1, config["NUM_ENVS"]))
        # import pdb;pdb.set_trace()
        init_hstate = ScannedRNN.initialize_carry( config["NUM_ENVS"], config["LAYER_SIZE"])
        dummy_obs = jnp.zeros((1, *obs_shape))
        dummy_future_obs = jnp.zeros((1, *obs_shape))
        dummy_action = jnp.zeros((1, action_shape))
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        crl_params = etd_network.init(_rng, dummy_obs, dummy_action, dummy_future_obs, jnp.zeros((1, config["NUM_ENVS"])),  None)
        tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["CRL_LR"], eps=1e-5),
            )
        crl_state["crl_model"] = TrainState.create(
            apply_fn=etd_network.apply,
            params=crl_params,
            tx=tx

        )
        if config["USE_EMPOWERMENT"]:
            emp_params = emp_network.init(_rng, dummy_obs, dummy_action, dummy_obs)
            tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(config["CRL_LR"], eps=1e-5),
                )
            emp_state["emp_model"] = TrainState.create(
                apply_fn=emp_network.apply,
                params=emp_params,
                tx=tx
            )
        # import pdb;pdb.set_trace()

        
        # if you are using icm exploration reward
        if config["TRAIN_ICM"]:
            obs_shape = env.observation_space(env_params).shape
            assert len(obs_shape) == 1, "Only configured for 1D observations"
            obs_shape = obs_shape[0]

            # Encoder
            icm_encoder_network = ICMEncoder(
                num_layers=3,
                output_dim=config["ICM_LATENT_SIZE"],
                layer_size=config["ICM_LAYER_SIZE"],
            )
            rng, _rng = jax.random.split(rng)
            icm_encoder_network_params = icm_encoder_network.init(
                _rng, jnp.zeros((1, obs_shape))
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["ICM_LR"], eps=1e-5),
            )
            ex_state["icm_encoder"] = TrainState.create(
                apply_fn=icm_encoder_network.apply,
                params=icm_encoder_network_params,
                tx=tx,
            )

            # Forward
            icm_forward_network = ICMForward(
                num_layers=3,
                output_dim=config["ICM_LATENT_SIZE"],
                layer_size=config["ICM_LAYER_SIZE"],
                num_actions=env.num_actions,
            )
            rng, _rng = jax.random.split(rng)
            icm_forward_network_params = icm_forward_network.init(
                _rng, jnp.zeros((1, config["ICM_LATENT_SIZE"])), jnp.zeros((1,))
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["ICM_LR"], eps=1e-5),
            )
            ex_state["icm_forward"] = TrainState.create(
                apply_fn=icm_forward_network.apply,
                params=icm_forward_network_params,
                tx=tx,
            )

            # Inverse
            icm_inverse_network = ICMInverse(
                num_layers=3,
                output_dim=env.num_actions,
                layer_size=config["ICM_LAYER_SIZE"],
            )
            rng, _rng = jax.random.split(rng)
            icm_inverse_network_params = icm_inverse_network.init(
                _rng,
                jnp.zeros((1, config["ICM_LATENT_SIZE"])),
                jnp.zeros((1, config["ICM_LATENT_SIZE"])),
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["ICM_LR"], eps=1e-5),
            )
            ex_state["icm_inverse"] = TrainState.create(
                apply_fn=icm_inverse_network.apply,
                params=icm_inverse_network_params,
                tx=tx,
            )

            if config["USE_E3B"]:
                ex_state["e3b_matrix"] = (
                    jnp.repeat(
                        jnp.expand_dims(
                            jnp.identity(config["ICM_LATENT_SIZE"]), axis=0
                        ),
                        config["NUM_ENVS"],
                        axis=0,
                    )
                    / config["E3B_LAMBDA"]
                )

        # INIT ENV
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    ex_state,
                    rng,
                    update_step,
                    crl_state,
                    init_hstate
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward_e, done, info = env.step(rng_step, env_state, action, env_params)

                reward_i = jnp.zeros(config["NUM_ENVS"])

                reward = reward_e + reward_i

                transition = Transition(
                    done=done,
                    action=action,
                    value=value,
                    reward=reward,
                    reward_i=reward_i,
                    reward_e=reward_e,
                    log_prob=log_prob,
                    obs=last_obs,
                    next_obs=obsv,
                    info=info,
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    ex_state,
                    rng,
                    update_step,
                    crl_state,
                    init_hstate
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            # traj_batch shape is (num_steps, num_envs, feature_size)
            #TODO: you should sample the futuere state here
            sample_future_vmap = jax.vmap(sample_future_state, in_axes=(None, 1, 1), out_axes=1)
            rng = runner_state[4]
            future_obs_batch = sample_future_vmap(rng, traj_batch.obs, traj_batch.done)
            
            # add_fuutre_states(traj_batch)
            

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                ex_state,
                rng,
                update_step,
                crl_state,
                initial_hstate
            ) = runner_state
            _, last_val = network.apply(train_state.params, last_obs)
            last_done = traj_batch.done[-1]

            def _calculate_gae(traj_batch, future_obs, last_val, last_done, init_hstate):
                @jax.jit
                @jax.jit
                def mc_crl_reward(trans_batch, gamma):
                    trans_batch, future_obs = trans_batch
                    state = trans_batch.obs
                    action = trans_batch.action
                    dones = trans_batch.done
                    T, N, D = state.shape
                    deltas_desc = jnp.arange(T-1, 0, -1)
                    def compute_min_distance(env_state, prev_states):
                        def body(carry, t):
                            _ = carry
                            current_state = env_state[t][None, :]        # (1, D)
                            phi_x, phi_y, c_y, _ = etd_network.apply(crl_state["crl_model"].params, prev_states, action, env_state, None, None)
                            dist = mrn_distance(current_state, prev_states)  # (T-1,)

                            # mask states after t
                            mask = jnp.arange(prev_states.shape[0]) < t
                            dist_masked = jnp.where(mask, dist, jnp.inf)
                            # running minimum
                            min_distance = jnp.min(dist_masked)
                            return None, min_distance
                        init = jnp.inf
                        _, min_distances = lax.scan(body, None, jnp.arange(T))
                        min_distances = min_distances.at[0].set(0.0)
                        return min_distances
                    reward = jax.vmap(compute_min_distance, in_axes=(1, 1,), out_axes=1)(state, state)
                    return reward

                # def crl_reward(transition, future_obs):
                #     action_onehot = transition.action
                #     phi_x, phi_y, c_y, _ = etd_network.apply(crl_state["crl_model"].params, transition.obs, action_onehot, future_obs, None, None)
                #     import pdb;pdb.set_trace()
                #     rwd = -similarity_method_for_rwd(phi_x, phi_y)
                #     return jax.lax.stop_gradient(rwd)

                # def get_crl_repr(carry, transition_batch):
                #     H = config["NUM_STEPS"]
                #     # import pdb;pdb.set_trace()
                #     transition, future_obs = transition_batch
                #     dicounted_future_reprs, obs_action_rep, next_done, time_step_counter, init_hidden = carry
                #     info = transition.info
                #     done = transition.done
                #     # action_onehot = jax.nn.one_hot(transition.action, num_classes=action_shape)
                #     action_onehot = transition.action
                #     if config["USE_RNN"]:
                #         obs_inpt = transition.obs[np.newaxis, :]
                #         action_inpt = action_onehot[np.newaxis, :]
                #         future_obs_inpt = future_obs[np.newaxis, :]
                #         done_inpt = done[np.newaxis, :]
                #     else:
                #         obs_inpt = transition.obs
                #         action_inpt = action_onehot
                #         future_obs_inpt = future_obs
                #         done_inpt = done
                #     if config["USE_SINGLE_SAMPLE"]:
                #         obs_action_rep, future_obs_rep, log_temp, init_hidden = etd_network.apply(crl_state["crl_model"].params, obs_inpt, action_inpt, future_obs_inpt, done_inpt, init_hidden[0])
                #         dicounted_future_reprs = future_obs_rep
                #     else:
                #         # import pdb;pdb.set_trace()  
                #         obs_action_rep, future_obs_rep, log_temp, init_hidden = etd_network.apply(crl_state["crl_model"].params, obs_inpt, action_inpt, future_obs_inpt, done_inpt, init_hidden[0])
                #         # gamma_cl_reward
                #         dicounted_future_reprs = future_obs_rep + config["GAMMA_CL_REWARD"] * dicounted_future_reprs * (1 - next_done[:, None])
                #         time_step_counter = time_step_counter - 1
                #         # if the episode is done, reset the counter, we use this expression to avoid jax related errors, when using boolean indexing directly
                #         time_step_counter = (time_step_counter * (1-next_done)) + (next_done * H)
                #         # if config["USE_NORM_CONSTANT"]:
                #         #     normalization_constant = (1 - config["GAMMA_CL"]**(H - time_step_counter[:, None] )) / (1 - config["GAMMA_CL"])
                #         #     dicounted_future_reprs = dicounted_future_reprs * normalization_constant
                #     # import pdb;pdb.set_trace()
                #     # import pdb;pdb.set_trace()  
                #     if config["USE_RNN"]:
                #         dicounted_future_reprs = dicounted_future_reprs[0]
                #         obs_action_rep = obs_action_rep[0]
                #     return (jax.lax.stop_gradient(dicounted_future_reprs), obs_action_rep, done, time_step_counter, init_hidden[None, :]), (jax.lax.stop_gradient(obs_action_rep), jax.lax.stop_gradient(dicounted_future_reprs), time_step_counter)
                
                def crl_reward(obs_action_rep, future_obs_rep):
                    rwd = -similarity_method(obs_action_rep, future_obs_rep).diagonal()
                    # import pdb;pdb.set_trace()
                    return jax.lax.stop_gradient(rwd)
                
                def emp_reward(transition, future_obs):
                    action_onehot = transition.action
                    # action_onehot = jax.nn.one_hot(transition.action, num_classes=action_shape)
                    obs_action_rep, obs_rep, future_obs_rep, future_obs_rep2, log_temp = emp_network.apply(emp_state["emp_model"].params, transition.obs, action_onehot, future_obs)
                    rwd = (similarity_method(obs_action_rep, future_obs_rep).diagonal() - similarity_method(obs_rep, future_obs_rep2).diagonal())    
                    # import pdb;pdb.set_trace()
                    return jax.lax.stop_gradient(rwd)
                
                def _get_advantages(gae_and_next_value, transition_batch):
                    
                    transition, future_obs, crl_rwd_mc = transition_batch
                    # import pdb;pdb.set_trace()
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    if config["USE_EMPOWERMENT"]:
                        emp_rewards = emp_reward(transition, future_obs)
                        reward = (config["CRL_REWARD_COEF"] * emp_rewards) + config["TASK_REWARD_COEF"] * reward
                    else:
                        crl_rewards = lax.stop_gradient(crl_rwd_mc)
                        # import pdb;pdb.set_trace()
                        reward = (config["CRL_REWARD_COEF"] * crl_rewards) + config["TASK_REWARD_COEF"] * reward
                    # import pdb;pdb.set_trace()
                    # delta = crl_rewards + config["GAMMA"] * next_value * (1 - done) - value
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    # import pdb;pdb.set_trace()
                    return (reward, value), gae

                # _, o = jax.lax.scan(
                #     get_crl_repr,
                #     (jnp.zeros((config["NUM_ENVS"], config["REPR_DIM"])), jnp.zeros((config["NUM_ENVS"], config["REPR_DIM"])), last_done, jnp.ones(config["NUM_ENVS"])*config["GEOM_TRUNC"], init_hstate),
                #     (traj_batch, future_obs),
                #     reverse=True,
                #     unroll=16,
                # )
                # obs_action_rep, dicounted_future_reprs, time_step_counter = o
                
                # import pdb; pdb.set_trace()
                crl_rwd_mc = mc_crl_reward((traj_batch, future_obs), config["GAMMA_CL_REWARD"])
                adv_info, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    (traj_batch, future_obs, crl_rwd_mc),
                    reverse=True,
                    unroll=16,
                )
                # import pdb;pdb.set_trace()
                return advantages, advantages + traj_batch.value, adv_info[0]

            advantages, targets, crl_rewards = _calculate_gae(traj_batch, future_obs_batch, last_val, last_done, initial_hstate)
            

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    train_state, crl_state = train_state
                    batch_info, batch_with_time_dim_info = batch_info
                    traj_batch, advantages, targets, future_obs_batch = batch_info
                    # import pdb;pdb.set_trace()
                    init_hstate, traj_batch_with_time_dim, _, _, future_obs_batch_with_time_dim = batch_with_time_dim_info

                    import functools

                    # @functools.partial(jax.jit, static_argnums=(2, 3, 4))
                    def _td_crl_loss(model_params,
                                    actor_params,
                                    traj_batch,
                                    future_obs,
                                    init_hstate,
                                    rng):
                            # import pdb;pdb.set_trace()
                            # ——— prepare the batch ———
                            # one-hot actions
                            # action_oh = jax.nn.one_hot(traj_batch.action, num_classes=action_shape)  # (N, A)
                            action_oh = traj_batch.action
                            
                            # sample next action in one shot
                            pi, _ = network.apply(actor_params, traj_batch.next_obs)
                            next_act = pi.sample(seed=rng)
                            next_act_oh = next_act
                            # next_act_oh = jax.nn.one_hot(next_act, num_classes=action_shape)
                            
                            # random swap mask
                            mask = jax.random.bernoulli(rng, p=1.0, shape=(future_obs.shape[0], 1))
                            future_shift = jnp.roll(future_obs, 1, axis=0)
                            future_obs = jnp.where(mask, future_shift, future_obs)
                            
                            # stack everything into a single big batch of size 3*N
                            N = traj_batch.obs.shape[0]
                            obs_all      = jnp.concatenate([traj_batch.obs,
                                                            traj_batch.obs,
                                                            traj_batch.next_obs],      axis=0)  # (3N, …)
                            act_all      = jnp.concatenate([action_oh,
                                                            action_oh,
                                                            next_act_oh],              axis=0)
                            next_obs_all = jnp.concatenate([traj_batch.next_obs,
                                                            future_obs,
                                                            future_obs],              axis=0)
                            done_all     = jnp.concatenate([traj_batch.done,
                                                            traj_batch.done,
                                                            traj_batch.done],          axis=0)
                            hstate_all   = jnp.repeat(init_hstate[0][None, ...], 3*N, axis=0)      # (3N, …)
                            
                            # ——— single contrastive apply ———
                            # returns something of shape (3N, …)
                            all_outputs = etd_network.apply(
                                model_params,
                                obs_all,
                                act_all,
                                next_obs_all,
                                done_all,
                                hstate_all
                            )
                            # unpack and reshape back to (3, N, …)
                            sa_repr, g_repr, _, _ = all_outputs
                            sa_repr      = sa_repr.reshape(3, N, -1)   # first axis indexes which of the 3 modes
                            g_repr       = g_repr.reshape(3, N, -1)
                            
                            # positive: mode 0→2
                            obs_act_rep     = sa_repr[0]
                            obs_act_neg_rep = sa_repr[1]
                            sa_w            = sa_repr[2]
                            
                            next_rep        = g_repr[0]
                            rand_rep        = g_repr[1]
                            g_w             = g_repr[2]
                            
                            # ——— build logits ———
                            logits_pos = similarity_method(obs_act_rep, next_rep)        # (N, N?)
                            logits_neg = similarity_method(obs_act_neg_rep, rand_rep)
                            logits_w   = similarity_method(sa_w, g_w)                    # (N, N)
                            
                            # ——— pre-allocate identity once outside or cache it here ———
                            I = jnp.eye(N)
                            
                            # ——— compute losses ———
                            loss_pos = optax.softmax_cross_entropy(logits_pos, I)
                            
                            # import pdb;pdb.set_trace()
                            w = jax.nn.softmax(logits_w, axis=1)
                            w = jax.lax.stop_gradient(w)
                            # expand w to match the logit shape if needed
                            # w_exp = w[..., None].repeat(logits_neg.shape[-1], axis=-1)
                            loss_neg = optax.softmax_cross_entropy(logits_neg, w)
                            # import pdb;pdb.set_trace()
                            
                            loss = (1 - config["GAMMA_CL"]) * loss_pos + config["GAMMA_CL"] * loss_neg
                            # loss = loss_pos
                            return jnp.mean(loss)

                    
                    # update the contrastive model
                    def _crl_loss(model_params, traj_batch, future_obs, init_hstate):
                        # import pdb;pdb.set_trace()
                        # TODO: fix this for the RNN case
                        # if config["USE_RNN"]:
                        #     import pdb;pdb.set_trace()
                        # action_onehot = jax.nn.one_hot(traj_batch.action, num_classes=action_shape)
                        action_onehot = traj_batch.action
                        phi_x, phi_y, c_y, init_hstate = etd_network.apply(model_params, traj_batch.obs, traj_batch.action, future_obs, traj_batch.done, init_hstate[0])
                        phi_x = phi_x.reshape(-1, config["REPR_DIM"])
                        phi_y = phi_y.reshape(-1, config["REPR_DIM"])
                        dist = mrn_distance(phi_x[:, None], phi_y[None, :])
                        c_y = c_y.reshape(-1, 1)
                        logits = c_y.T - dist
                        loss = contrastive_losses()[config["CONTRASTIVE_LOSS"]](logits, config["UPDATE_PROPORTION"], _rng)
                        # loss = contrastive_losses()[config["CONTRASTIVE_LOSS"]](sim)
                        # import pdb;pdb.set_trace()
                        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=-1)
                        loss += config["LOGSUMEXP_PENALTY_COEFF"] * jnp.mean(logsumexp**2)
                        return loss
                    
                    def _emp_loss(model_params, traj_batch, future_obs):
                        # import pdb;pdb.set_trace()
                        action_onehot = jax.nn.one_hot(traj_batch.action, num_classes=action_shape)
                        obs_action_rep, obs_repr, future_obs_rep, future_obs_rep2, log_temp = emp_network.apply(model_params, traj_batch.obs, action_onehot, future_obs)
                        sim1 = similarity_method(obs_action_rep, future_obs_rep)
                        sim2 = similarity_method(obs_repr, future_obs_rep2)
                        loss1 = contrastive_losses()[config["CONTRASTIVE_LOSS"]](sim1)
                        loss2 = contrastive_losses()[config["CONTRASTIVE_LOSS"]](sim2)
                        loss = (loss1 + loss2)/2
                        # import pdb;pdb.set_trace()
                        return loss
                        
                    
                    # Policy/value network
                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)
                        # import pdb;pdb.set_trace()

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)

                    if config["USE_EMPOWERMENT"]:
                        emp_grad_fn = jax.value_and_grad(_emp_loss, has_aux=False)
                        emp_loss, emp_grad = emp_grad_fn(emp_state["emp_model"].params, traj_batch, future_obs_batch)
                        emp_state["emp_model"] = emp_state["emp_model"].apply_gradients(grads=emp_grad)
                        losses = (total_loss, emp_loss)

                    else:
                        # update the contrastive model
                        # import pdb;pdb.set_trace()
                        
                        if config["USE_TD_CRL_LOSS"]:
                            # import pdb;pdb.set_trace()
                            '''
                            _td_crl_loss(model_params,
                                    actor_params,
                                    action_shape: int,
                                    GAMMA_CL: float,
                                    traj_batch,
                                    future_obs,
                                    init_hstate,
                                    rng)
                            '''
                            crl_grad_fn = jax.value_and_grad(_td_crl_loss, has_aux=False)
                            crl_loss, crl_grad = crl_grad_fn(crl_state["crl_model"].params, train_state.params, traj_batch, future_obs_batch, init_hstate, _rng)
                        else:
                            crl_grad_fn = jax.value_and_grad(_crl_loss, has_aux=False)
                            crl_loss, crl_grad = crl_grad_fn(crl_state["crl_model"].params, traj_batch, future_obs_batch, init_hstate)
                        crl_state["crl_model"] = crl_state["crl_model"].apply_gradients(grads=crl_grad)
                            
                        losses = (total_loss, crl_loss)
                    
                    # ex_state["icm_forward"] = ex_state["icm_forward"].apply_gradients(
                    #     grads=icm_forward_grad
                    # )



                    
                    # import pdb;pdb.set_trace()
                    return (train_state, crl_state), losses

                (
                    train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                    crl_state,
                    init_hstate
                ) = update_state
                # import pdb;pdb.set_trace()
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets, future_obs_batch)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                # import pdb;pdb.set_trace()
                # For RNN contrastive model
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                # init_hstate = init_hstate[None, :]
                batch = (init_hstate, traj_batch, advantages, targets, future_obs_batch)

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches_with_time = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )
                # import pdb;pdb.set_trace()
                train_state, losses = jax.lax.scan(
                    _update_minbatch, (train_state, crl_state), (minibatches, minibatches_with_time), 
                )
                train_state, crl_state = train_state
                # import pdb;pdb.set_trace()
                update_state = (
                    train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                    crl_state,
                    initial_hstate
                )
                # print(705)
                # print(initial_hstate.shape)
                # import pdb;pdb.set_trace()
                return update_state, losses

            # init_hstate = initial_hstate[None, :]  # TBH
            update_state = (
                train_state,
                traj_batch,
                advantages,
                targets,
                rng,
                crl_state,
                init_hstate[None, :]
            )
            # print("calling update_epoch")
            # import pdb;pdb.set_trace()
            # print(722)
            # print(init_hstate.shape)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )

            train_state = update_state[0]
            crl_state = update_state[-2]
            
            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )
            # import pdb;pdb.set_trace()
            # metric["ppo_total_loss"] = loss_info[0].mean()
            metric["crl_loss"] = loss_info[1].mean()
            # import pdb;pdb.set_trace()  
            metric["task_reward"] = traj_batch.reward.mean()
            metric["crl_reward"] = crl_rewards.mean()
            # import pdb;pdb.set_trace()
            
            # import pdb;pdb.set_trace()  

            rng = update_state[-3]

            # UPDATE EXPLORATION STATE
            def _update_ex_epoch(update_state, unused):
                def _update_ex_minbatch(ex_state, traj_batch):
                    def _inverse_loss_fn(
                        icm_encoder_params, icm_inverse_params, traj_batch
                    ):
                        latent_obs = ex_state["icm_encoder"].apply_fn(
                            icm_encoder_params, traj_batch.obs
                        )
                        latent_next_obs = ex_state["icm_encoder"].apply_fn(
                            icm_encoder_params, traj_batch.next_obs
                        )

                        action_pred_logits = ex_state["icm_inverse"].apply_fn(
                            icm_inverse_params, latent_obs, latent_next_obs
                        )
                        true_action = jax.nn.one_hot(
                            traj_batch.action, num_classes=action_pred_logits.shape[-1]
                        )

                        bce = -jnp.mean(
                            jnp.sum(
                                action_pred_logits
                                * true_action
                                * (1 - traj_batch.done[:, None]),
                                axis=1,
                            )
                        )

                        return bce * config["ICM_INVERSE_LOSS_COEF"]

                    inverse_grad_fn = jax.value_and_grad(
                        _inverse_loss_fn,
                        has_aux=False,
                        argnums=(
                            0,
                            1,
                        ),
                    )
                    inverse_loss, grads = inverse_grad_fn(
                        ex_state["icm_encoder"].params,
                        ex_state["icm_inverse"].params,
                        traj_batch,
                    )
                    icm_encoder_grad, icm_inverse_grad = grads
                    ex_state["icm_encoder"] = ex_state["icm_encoder"].apply_gradients(
                        grads=icm_encoder_grad
                    )
                    ex_state["icm_inverse"] = ex_state["icm_inverse"].apply_gradients(
                        grads=icm_inverse_grad
                    )

                    def _forward_loss_fn(icm_forward_params, traj_batch):
                        latent_obs = ex_state["icm_encoder"].apply_fn(
                            ex_state["icm_encoder"].params, traj_batch.obs
                        )
                        latent_next_obs = ex_state["icm_encoder"].apply_fn(
                            ex_state["icm_encoder"].params, traj_batch.next_obs
                        )

                        latent_next_obs_pred = ex_state["icm_forward"].apply_fn(
                            icm_forward_params, latent_obs, traj_batch.action
                        )

                        error = (latent_next_obs - latent_next_obs_pred) * (
                            1 - traj_batch.done[:, None]
                        )
                        return (
                            jnp.square(error).mean() * config["ICM_FORWARD_LOSS_COEF"]
                        )

                    forward_grad_fn = jax.value_and_grad(
                        _forward_loss_fn, has_aux=False
                    )
                    forward_loss, icm_forward_grad = forward_grad_fn(
                        ex_state["icm_forward"].params, traj_batch
                    )
                    ex_state["icm_forward"] = ex_state["icm_forward"].apply_gradients(
                        grads=icm_forward_grad
                    )

                    losses = (inverse_loss, forward_loss)
                    return ex_state, losses

                (ex_state, traj_batch, rng) = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), traj_batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                ex_state, losses = jax.lax.scan(
                    _update_ex_minbatch, ex_state, minibatches
                )
                update_state = (ex_state, traj_batch, rng)
                return update_state, losses

            if config["TRAIN_ICM"]:
                ex_update_state = (ex_state, traj_batch, rng)
                ex_update_state, ex_loss = jax.lax.scan(
                    _update_ex_epoch,
                    ex_update_state,
                    None,
                    config["EXPLORATION_UPDATE_EPOCHS"],
                )
                metric["icm_inverse_loss"] = ex_loss[0].mean()
                metric["icm_forward_loss"] = ex_loss[1].mean()
                metric["reward_i"] = traj_batch.reward_i.mean()
                metric["reward_e"] = traj_batch.reward_e.mean()

                ex_state = ex_update_state[0]
                rng = ex_update_state[-1]

            # wandb logging
            if config["DEBUG"] and config["USE_WANDB"]:
                # import pdb;pdb.set_trace()
                def callback(metric, update_step, traj_batch):
                    observations = traj_batch.obs.reshape(-1,traj_batch.obs.shape[-1] )
                    density.update_count(observations[:, :config["GOAL_DIM"]])
                    if update_step % 10 == 0:
                        metric["state_counts"] =  density.num_states()
                        to_log = create_log_dict(metric, config)
                        agg_logs = batch_log(update_step, to_log, config)
                        csv_logger.log(agg_logs)   
                    # labels  = []
                    # values = []
                    # import pdb;pdb.set_trace()
                    # for m in metric:
                    #     if "Achievements" in m:
                    #         label = m[m.index("Achievements") + len("Achievements") + 1:]
                    #         labels.append(label)
                    #         values.append(metric[m].item())
                    # # import pdb;pdb.set_trace()
                    # wandb_bar_chart(labels, values)
                    # import pdb;pdb.set_trace()

                jax.debug.callback(
                    callback,
                    metric,
                    update_step,
                    traj_batch
                )

            runner_state = (
                train_state,
                env_state,
                last_obs,
                ex_state,
                rng,
                update_step + 1,
                crl_state,
                init_hstate[None, :]
            )
            # print(721)
            # print(init_hstate.shape)
            # import pdb;pdb.set_trace()
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            ex_state,
            _rng,
            0,
            crl_state,
            init_hstate[None, :]
        )
        # print(920)
        # print(init_hstate[None, :].shape)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state , "metric": metric}

    return train


def run_ppo(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=config["ENV_NAME"]
            + "-" + config["MODEL"] + "-"
            + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
            + "M",
            mode="online"
        )

    rng = jax.random.PRNGKey(config["SEED"]) 
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    scratch_path = os.getenv("SCRATCH")
    runs_path = os.path.join(scratch_path, "crl_runs")  
    os.makedirs(runs_path, exist_ok=True)

    exp_dir = os.path.join(config["MODEL"], config["ENV_NAME"], config["RUN_NAME_SUFFIX"])
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
    config["RUN_DIR"] = run_dir
    config["CHECKPOINT_DIR"] = ckpt_dir
    print("Experiment directory: ", run_dir)
    # import pdb;pdb.set_trace()
    save_args(config, run_dir)

    train_jit = jax.jit(make_train(config))
    train_vmap = jax.vmap(train_jit)

    t0 = time.time()
    out = train_vmap(rngs) # run the training on parallel across the random seeds, rngs.shape[0] = number of random seed
    t1 = time.time()
    print("Time to run experiment", t1 - t0)
    print("SPS: ", config["TOTAL_TIMESTEPS"] / (t1 - t0))
    metric = out["metric"]
    labels  = []
    values = []
    for m in metric:
        if "Achievements" in m:
            label = m[m.index("Achievements") + len("Achievements") + 1:]
            labels.append(label)
            values.append(metric[m].mean().item())
    wandb_bar_chart(labels, values)

    if config["USE_WANDB"]:

        def _save_network(rs_index, dir_name):
            train_states = out["runner_state"][rs_index]
            train_state = jax.tree.map(lambda x: x[0], train_states)
            orbax_checkpointer = PyTreeCheckpointer()
            options = CheckpointManagerOptions(max_to_keep=1, create=True)
            path = os.path.join(wandb.run.dir, dir_name)
            checkpoint_manager = CheckpointManager(path, orbax_checkpointer, options)
            print(f"saved runner state to {path}")
            save_args = orbax_utils.save_args_from_target(train_state)
            checkpoint_manager.save(
                config["TOTAL_TIMESTEPS"],
                train_state,
                save_kwargs={"save_args": save_args},
            )

        if config["SAVE_POLICY"]:
            _save_network(0, "policies")

            # import pdb;pdb.set_trace()
            from utils import visualize_agent
            # import pdb;pdb.set_trace()  
            video_path = visualize_agent(wandb.run.dir)
            wandb.log({"Agent_Visual": wandb.Image(video_path)})


if __name__ == "__main__":
    args, reset_args = etd_ppo_args(sys)
    if args.jit:
        run_ppo(args)
    else:
        with jax.disable_jit():
            run_ppo(args)