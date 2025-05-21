import argparse
import os
# os.environ["XLA_FLAGS"] = "--xla_gpu_disable_ptxas_verbose"
import sys

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import time
# from argparse
from args import ctec_rnn_args

from flax.training import orbax_utils
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

import wandb
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Dict
from flax.training.train_state import TrainState
import distrax
import functools

import json
from utils import init_state, update_corr_state,compute_correlation, update_rms
from utils import wandb_bar_chart
from utils import visualize_agent_rnn
from utils import create_csv_logger
from utils import save_args
from wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)
from logz.batch_logging import create_log_dict, batch_log

from craftax.craftax_env import make_craftax_env_from_name
from models.contrastive_model import ContrastiveModel, EmpowermentModel
from losses import contrastive_losses
from wonderwords import RandomWord

# Code adapted from the original implementation made by Chris Lu
# Original code located at https://github.com/luchris429/purejaxrl


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        
        actor_mean = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    print("mini_batch_size", config["MINIBATCH_SIZE"])
    print("mini_batch_size for contrastive training", config["MINIBATCH_SIZE"] * config["UPDATE_PROPORTION"])
    # import pdb;pdb.set_trace()

    # Create environment
    env = make_craftax_env_from_name(
        config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
    )
    env_params = env.default_params

    # Wrap with some extra logging
    env = LogWrapper(env)

    # Wrap with a batcher, maybe using optimistic resets
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac
    
    def gamma_schedule(config, count):
        if config["gamma_schedule".upper()] == "linear":
            frac = 1.0 - (count - 1.0) / config["NUM_UPDATES"]
            gamma = frac * config["gamma_schedule_start".upper()] + (1.0 - frac) * config["gamma_schedule_end".upper()]
            config["GAMMA_CL"] = gamma
            return gamma
        elif config["gamma_schedule".upper()] == "exponential":
            # Exponential decay: gamma = start * (end/start)^(t/T)
            t = count - 1.0
            T = config["NUM_UPDATES"]
            gamma = config["gamma_schedule_start".upper()] * (config["gamma_schedule_end".upper()] / config["gamma_schedule_start".upper()]) ** (t/T)
            config["GAMMA_CL"] = gamma
            return gamma
        else:
            return config["GAMMA_CL"]
    
    
    
    similarity_methods = {
            "l2": lambda sa_repr, g_repr: -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1)),
            "l2_no_sqrt":  lambda sa_repr, g_repr: -jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1),
            "l1":  lambda sa_repr, g_repr: -jnp.sum(jnp.abs(sa_repr[:, None, :] - g_repr[None, :, :]), axis=-1),
            "dot": lambda sa_repr, g_repr: jnp.einsum("ik,jk->ij", sa_repr, g_repr), # if the vectors are normalized then this the cosine 
        }
    similarity_methods_rnn = {
            "l2": lambda sa_repr, g_repr: -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1)),
            "l2_no_sqrt":  lambda sa_repr, g_repr: -jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1),
            "l1":  lambda sa_repr, g_repr: -jnp.sum(jnp.abs(sa_repr[:, None, :] - g_repr[None, :, :]), axis=-1),
            "dot": lambda sa_repr, g_repr: jnp.einsum("hik,hjk->hij", sa_repr, g_repr), # if the vectors are normalized then this the cosine 
        }
    # import pdb;pdb.set_trace()
    similarity_method = similarity_methods[config["SIMILARITY_MEASURE"]]
    similarity_method_rnn = similarity_methods_rnn[config["SIMILARITY_MEASURE"]]
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
        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env_params).n, config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["LAYER_SIZE"]
        )
        network_params = network.init(_rng, init_hstate, init_x)
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

        contrastive_network = ContrastiveModel(config)
        if config["USE_EMPOWERMENT"]:
            emp_network = EmpowermentModel(config)
        # import pdb;pdb.set_trace()

        crl_state = {
            "crl_model": None
        }
        emp_state = {
            "emp_model": None
        }

        obs_shape = env.observation_space(env_params).shape[0]
        action_shape = env.action_space(env_params).n
        if config["USE_RNN"]:
            dummy_obs = jnp.zeros((1, config["NUM_ENVS"], obs_shape))
            dummy_future_obs = jnp.zeros((1, config["NUM_ENVS"], obs_shape))
            dummy_action = jnp.zeros((1, config["NUM_ENVS"], action_shape))
        else:
            dummy_obs = jnp.zeros((1, obs_shape))
            dummy_future_obs = jnp.zeros((1, obs_shape))
            dummy_action = jnp.zeros((1, action_shape))
        crl_params = contrastive_network.init(_rng, dummy_obs, dummy_action, dummy_future_obs, jnp.zeros((1, config["NUM_ENVS"])),  init_hstate)
        tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), # I am clipping the grad norm, is that necessary?
                optax.adam(config["CRL_LR"], eps=1e-5), # also what if we used default eps value?
            )
        crl_state["crl_model"] = TrainState.create(
            apply_fn=contrastive_network.apply,
            params=crl_params,
            tx=tx

        )
        if config["USE_EMPOWERMENT"]:
            emp_params = emp_network.init(_rng, dummy_obs, dummy_action, dummy_obs)
            tx = optax.chain(
                    # optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(config["CRL_LR"]),
                )
            emp_state["emp_model"] = TrainState.create(
                apply_fn=emp_network.apply,
                params=emp_params,
                tx=tx
            )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["LAYER_SIZE"]
        )
        online_correlation_state = init_state()
        # state for the mean and std caclucations
        intr_rms_state = (0.0, jnp.zeros(1,), jnp.zeros((1, )))
        extr_rms_state = (0.0, jnp.zeros(1,), jnp.zeros((1, )))

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    online_correlation_state,
                    intr_rms_state,
                    extr_rms_state,
                    hstate,
                    rng,
                    update_step,
                    crl_state
                ) = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(
                    _rng, env_state, action, env_params
                )
                transition = Transition(
                    last_done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    done,
                    online_correlation_state,
                    intr_rms_state,
                    extr_rms_state,
                    hstate,
                    rng,
                    update_step,
                    crl_state
                )
                return runner_state, transition

            initial_hstate = runner_state[-4]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            sample_future_vmap = jax.vmap(sample_future_state, in_axes=(None, 1, 1), out_axes=1)
            rng = runner_state[-3]
            future_obs_batch = sample_future_vmap(rng, traj_batch.obs, traj_batch.done)

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                online_correlation_state,
                intr_rms_state,
                extr_rms_state,
                hstate,
                rng,
                update_step,
                crl_state
            ) = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

            def _calculate_gae(traj_batch, future_obs, last_val, last_done, intr_rms_state, extr_rms_state, init_hidden):
                def get_crl_repr(carry, transition_batch):
                    H = config["NUM_STEPS"]
                    # import pdb;pdb.set_trace()
                    transition, future_obs = transition_batch
                    dicounted_future_reprs, obs_action_rep, next_done, time_step_counter, init_hidden = carry
                    info = transition.info
                    done = transition.done
                    action_onehot = jax.nn.one_hot(transition.action, num_classes=action_shape)
                    if config["USE_RNN"]:
                        obs_inpt = transition.obs[np.newaxis, :]
                        action_inpt = action_onehot[np.newaxis, :]
                        future_obs_inpt = future_obs[np.newaxis, :]
                        done_inpt = done[np.newaxis, :]
                    else:
                        obs_inpt = transition.obs
                        action_inpt = action_onehot
                        future_obs_inpt = future_obs
                        done_inpt = done
                    if config["USE_SINGLE_SAMPLE"]:
                        obs_action_rep, future_obs_rep, log_temp, init_hidden = contrastive_network.apply(crl_state["crl_model"].params, obs_inpt, action_inpt, future_obs_inpt, done_inpt, init_hidden)
                        dicounted_future_reprs = future_obs_rep
                    else:
                        # import pdb;pdb.set_trace()  
                        obs_action_rep, future_obs_rep, log_temp, init_hidden = contrastive_network.apply(crl_state["crl_model"].params, obs_inpt, action_inpt, future_obs_inpt, done_inpt, init_hidden)
                        # gamma_cl_reward
                        dicounted_future_reprs = future_obs_rep + config["GAMMA_CL_REWARD"] * dicounted_future_reprs * (1 - next_done[:, None])
                        time_step_counter = time_step_counter - 1
                        # if the episode is done, reset the counter, we use this expression to avoid jax related errors, when using boolean indexing directly
                        time_step_counter = (time_step_counter * (1-next_done)) + (next_done * H)
                        # if config["USE_NORM_CONSTANT"]:
                        #     normalization_constant = (1 - config["GAMMA_CL"]**(H - time_step_counter[:, None] )) / (1 - config["GAMMA_CL"])
                        #     dicounted_future_reprs = dicounted_future_reprs * normalization_constant
                    # import pdb;pdb.set_trace()
                    # import pdb;pdb.set_trace()  
                    if config["USE_RNN"]:
                        dicounted_future_reprs = dicounted_future_reprs[0]
                        obs_action_rep = obs_action_rep[0]
                    return (jax.lax.stop_gradient(dicounted_future_reprs), obs_action_rep, done, time_step_counter, init_hidden), (jax.lax.stop_gradient(obs_action_rep), jax.lax.stop_gradient(dicounted_future_reprs), time_step_counter)
                        


                def crl_reward(obs_action_rep, future_obs_rep):
                    rwd = -similarity_method(obs_action_rep, future_obs_rep).diagonal()
                    # import pdb;pdb.set_trace()
                    return jax.lax.stop_gradient(rwd)
                
                def emp_reward(transition, future_obs):
                    action_onehot = jax.nn.one_hot(transition.action, num_classes=action_shape)
                    obs_action_rep, obs_rep, future_obs_rep, future_obs_rep2, log_temp = emp_network.apply(emp_state["emp_model"].params, transition.obs, action_onehot, future_obs)
                    rwd = (similarity_method(obs_action_rep, future_obs_rep).diagonal() - similarity_method(obs_rep, future_obs_rep2).diagonal())    
                    # import pdb;pdb.set_trace()
                    return jax.lax.stop_gradient(rwd)
                
                def _get_advantages(carry, transition_batch):
                    transition, future_obs, obs_action_rep, future_obs_rep = transition_batch
                    # import pdb;pdb.set_trace()  
                    gae, next_value, next_done, _, _ , intr_rms_state, extr_rms_state= carry
                    done, value, task_reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    if config["USE_EMPOWERMENT"]:
                        # TODO: update the method to work the same as crl_reward(...)
                        emp_rewards = emp_reward(transition, future_obs)
                        reward = (emp_rewards * config["CRL_REWARD_COEF"]) + config["TASK_REWARD_COEF"] * task_reward
                    else:
                        # import pdb;pdb.set_trace()
                        crl_rewards = crl_reward(obs_action_rep, future_obs_rep)
                        if config["RWD_RMS"]:
                            intr_rms_state, (means, stds) = jax.lax.scan(update_rms, intr_rms_state, crl_rewards)
                            crl_rewards = crl_rewards / stds[-1]
                        reward = (crl_rewards * config["CRL_REWARD_COEF"]) + config["TASK_REWARD_COEF"] * task_reward + config["LIFE_REWARD"]
                        if config["USE_RELATIVE_SCALE"]:
                            intr_rms_state, (intr_means, intr_stds) = jax.lax.scan(update_rms, intr_rms_state, jnp.abs(crl_rewards))
                            extr_rms_state, (extr_means, extr_stds) = jax.lax.scan(update_rms, extr_rms_state, jnp.abs(task_reward))
                            final_intr_mean = intr_rms_state[1]
                            final_extr_mean = extr_rms_state[1]
                            scale = config["RELATIVE_SCALE"] * (final_extr_mean[-1]/final_intr_mean[-1])
                            reward = task_reward + scale * crl_rewards
                            # import pdb;pdb.set_trace()
                        
                        # import pdb;pdb.set_trace()  
                    delta = (
                        reward + config["GAMMA"] * next_value * (1 - next_done) - value
                    )
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
                    )
                    # import pdb;pdb.set_trace()
                    return (gae, value, done, crl_rewards, task_reward.astype(float), intr_rms_state, extr_rms_state), gae

                # import pdb;pdb.set_trace()
                # init_hidden = flax.utils.replicate(initial_hstate, future_obs.shape[0])
                _, o = jax.lax.scan(
                    get_crl_repr,
                    (jnp.zeros((config["NUM_ENVS"], config["REPR_DIM"])), jnp.zeros((config["NUM_ENVS"], config["REPR_DIM"])), last_done, jnp.ones(config["NUM_ENVS"])*config["GEOM_TRUNC"], init_hidden),
                    (traj_batch, future_obs),
                    reverse=True,
                    unroll=16,
                )
                # import pdb;pdb.set_trace()
                obs_action_rep, dicounted_future_reprs, time_step_counter = o
                # import pdb;pdb.set_trace()
                # the constant is per time_step, so we need to compute for each time_step.
                if config["USE_NORM_CONSTANT"] and not config["USE_SINGLE_SAMPLE"]:
                        normalization_constant = (1 - config["GAMMA_CL_REWARD"]**(config["GEOM_TRUNC"] - time_step_counter)) / (1 - config["GAMMA_CL_REWARD"])
                        dicounted_future_reprs = dicounted_future_reprs * normalization_constant[:, :, None]

                
                # import pdb;pdb.set_trace()
                adv_info, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, last_done, jnp.zeros_like(last_val), jnp.zeros_like(last_val), intr_rms_state, extr_rms_state),
                    (traj_batch, future_obs, obs_action_rep, dicounted_future_reprs),
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value, adv_info

            advantages, targets, adv_info = _calculate_gae(traj_batch, future_obs_batch, last_val, last_done, intr_rms_state, extr_rms_state, initial_hstate)
            intr_rms_state = adv_info[-2]
            extr_rms_state = adv_info[-1]



            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    train_state, crl_state = train_state
                    init_hstate, traj_batch, advantages, targets, future_obs_batch = batch_info

                    # update the contrastive model
                    def _crl_loss(model_params, traj_batch, future_obs, init_hstate):
                        # import pdb;pdb.set_trace()
                        action_onehot = jax.nn.one_hot(traj_batch.action, num_classes=action_shape)
                        if config["USE_RNN"]:
                            # import pdb;pdb.set_trace()
                            obs_in = traj_batch.obs
                            action_in = action_onehot
                            future_obs = future_obs
                            dones_in = traj_batch.done
                        else:
                            obs_in = traj_batch.obs.reshape(-1, obs_shape)
                            action_in = action_onehot.reshape(-1, action_shape)
                            future_obs = future_obs.reshape(-1, obs_shape)
                            dones_in = traj_batch.done.reshape(-1, 1)
                        obs_action_rep, future_obs_rep, log_temp, init_hstate = contrastive_network.apply(model_params, obs_in, action_in, future_obs, dones_in, init_hstate[0])
                        if config["USE_RNN"]:
                            obs_action_rep = obs_action_rep.reshape(-1, config["REPR_DIM"])
                            future_obs_rep = future_obs_rep.reshape(-1, config["REPR_DIM"])
                        sim = similarity_method(obs_action_rep, future_obs_rep)
                        loss = contrastive_losses()[config["CONTRASTIVE_LOSS"]](sim, config["UPDATE_PROPORTION"], _rng)
                        # add the regularization term
                        logsumexp = jax.nn.logsumexp(sim + 1e-6, axis=-1)
                        loss += config["LOGSUMEXP_PENALTY_COEFF"] * jnp.mean(logsumexp**2)
                        # import pdb;pdb.set_trace()
                        return loss
                    
                    def _emp_loss(model_params, traj_batch, future_obs):
                        # import pdb;pdb.set_trace()
                        action_onehot = jax.nn.one_hot(traj_batch.action, num_classes=action_shape)
                        obs_in = traj_batch.obs.reshape(-1, obs_shape)
                        action_in = action_onehot.reshape(-1, action_shape)
                        future_obs = future_obs.reshape(-1, obs_shape)
                        obs_action_rep, obs_repr, future_obs_rep, future_obs_rep2, log_temp = emp_network.apply(model_params, obs_in, action_in, future_obs)
                        sim1 = similarity_method(obs_action_rep, future_obs_rep)
                        sim2 = similarity_method(obs_repr, future_obs_rep2)
                        loss1 = contrastive_losses()[config["CONTRASTIVE_LOSS"]](sim1)
                        loss2 = contrastive_losses()[config["CONTRASTIVE_LOSS"]](sim2)
                        loss = loss1 + loss2
                        # import pdb;pdb.set_trace()
                        return loss

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params, init_hstate[0], (traj_batch.obs, traj_batch.done)
                        )
                        log_prob = pi.log_prob(traj_batch.action)

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
                        train_state.params, init_hstate, traj_batch, advantages, targets
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
                        crl_grad_fn = jax.value_and_grad(_crl_loss, has_aux=False)
                        crl_loss, crl_grad = crl_grad_fn(crl_state["crl_model"].params, traj_batch, future_obs_batch, init_hstate)
                        crl_state["crl_model"] = crl_state["crl_model"].apply_gradients(grads=crl_grad)
                        losses = (total_loss, crl_loss)
                    return (train_state, crl_state), (total_loss, losses)

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                    crl_state
                ) = update_state

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets, future_obs_batch)

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree.map(
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

                train_state, (total_loss, losses) = jax.lax.scan(
                    _update_minbatch, (train_state, crl_state), minibatches
                )
                train_state, crl_state = train_state
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                    crl_state
                )
                return update_state, (total_loss, losses)

            init_hstate = initial_hstate[None, :]  # TBH
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
                crl_state
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            crl_state = update_state[-1]
            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )
            # import pdb;pdb.set_trace()
            metric["crl_loss"] = loss_info[1][1].mean()
            metric["task_reward"] = traj_batch.reward.mean()
            metric["crl_reward"] = adv_info[3].mean()
            # import pdb;pdb.set_trace()  
            online_correlation_state = update_corr_state(online_correlation_state, metric["task_reward"], metric["crl_reward"]) 
            corr = compute_correlation(online_correlation_state)    
            metric["task_intrinisc_correlation"] = corr
            # relative scale
            final_intr_mean = intr_rms_state[1]
            final_extr_mean = extr_rms_state[1]
            metric["relative_scale"] = final_extr_mean.mean() / final_intr_mean.mean()
            config["GAMMA_CL"] = gamma_schedule(config, update_step + 1)
            metric["gamma_cl"] = config["GAMMA_CL"]    
            

            rng = update_state[-2]
            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, update_step):
                    if update_step % 10 == 0:
                        to_log = create_log_dict(metric, config)
                        agg_logs = batch_log(update_step, to_log, config)
                        csv_logger.log(agg_logs)    
                        
                    # import pdb;pdb.set_trace()
                jax.debug.callback(callback, metric, update_step)

            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                online_correlation_state,
                intr_rms_state,
                extr_rms_state,
                hstate,
                rng,
                update_step + 1,
                crl_state
            )
            # update the gamma value
            
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            online_correlation_state,
            intr_rms_state,
            extr_rms_state,
            init_hstate,
            _rng,
            0,
            crl_state
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return train


def run_ppo(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=config["ENV_NAME"]
            + "-PPO_RNN-CRL"
            + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
            + "M",
            save_code=False
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    # create a directory for logging
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
    out = train_vmap(rngs)
    t1 = time.time()
    print("Time to run experiment", t1 - t0)
    print("SPS: ", config["TOTAL_TIMESTEPS"] / (t1 - t0))
    metric = out["metric"]
    labels  = []
    values = []
    # import pdb;pdb.set_trace()
    for m in metric:
        if "Achievements" in m:
            label = m[m.index("Achievements") + len("Achievements") + 1:]
            labels.append(label)
            values.append(metric[m].mean().item())
    # import pdb;pdb.set_trace()
    wandb_bar_chart(labels, values)
    # import pdb;pdb.set_trace()

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

            visualize_agent_rnn(wandb.run.dir, args)


if __name__ == "__main__":
    args, reset_args = ctec_rnn_args(sys)

    if args.jit:
        run_ppo(args)
    else:
        with jax.disable_jit():
            run_ppo(args)
