import argparse
import os
import sys
from args import intr_baselines_rnn_args
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import time
import json
from flax.training import orbax_utils
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)
from utils import create_csv_logger
import wandb
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Dict
from flax.training.train_state import TrainState
import distrax
import functools

from utils import update_rms
from utils import wandb_bar_chart
from wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)
from logz.batch_logging import create_log_dict, batch_log

from craftax.craftax_env import make_craftax_env_from_name
from models.rnd import RNDNetwork, ActorCriticRND
from models.icm import ICMEncoder, ICMForward, ICMInverse
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
    next_obs: jnp.ndarray
    info: jnp.ndarray
    reward_e: jnp.ndarray
    reward_i: jnp.ndarray

def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

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
    
    csv_logger_path = os.path.join(config["RUN_DIR"], "logs.csv") 
    csv_logger = create_csv_logger(config["ENV_NAME"], csv_logger_path)

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

        # Exploration state
        ex_state = {
            "rnd_model": None,
            "icm_encoder": None,
            "icm_forward": None,
            "icm_inverse": None,
            "e3b_matrix": None,
        }

        if config["USE_RND"]:
            obs_shape = env.observation_space(env_params).shape
            assert len(obs_shape) == 1, "Only configured for 1D observations"
            obs_shape = obs_shape[0]

            # Random network
            rnd_random_network = RNDNetwork(
                num_layers=3,
                output_dim=config["RND_OUTPUT_SIZE"],
                layer_size=config["RND_LAYER_SIZE"],
            )
            rng, _rng = jax.random.split(rng)
            rnd_random_network_params = rnd_random_network.init(
                _rng, jnp.zeros((1, obs_shape))
            )

            # Distillation Network
            rnd_distillation_network = RNDNetwork(
                num_layers=3,
                output_dim=config["RND_OUTPUT_SIZE"],
                layer_size=config["RND_LAYER_SIZE"],
            )
            rng, _rng = jax.random.split(rng)
            rnd_distillation_network_params = rnd_distillation_network.init(
                _rng, jnp.zeros((1, obs_shape))
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["RND_LR"], eps=1e-5),
            )
            ex_state["rnd_distillation_network"] = TrainState.create(
                apply_fn=rnd_distillation_network.apply,
                params=rnd_distillation_network_params,
                tx=tx,
            )
        elif config["USE_ICM"]:
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
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["LAYER_SIZE"]
        )
        # state for the mean and std caclucations
        rms_state = (0.0, jnp.zeros(1,), jnp.zeros((1, )))

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    ex_state,
                    last_done,
                    rms_state,
                    hstate,
                    rng,
                    update_step,
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
                # import pdb;pdb.set_trace()
                reward_i = jnp.zeros(config["NUM_ENVS"])
                reward_e = reward
                if config["USE_RND"]:
                    random_pred = rnd_random_network.apply(
                        rnd_random_network_params, obsv
                    )
                    # import pdb;pdb.set_trace()

                    distill_pred = ex_state["rnd_distillation_network"].apply_fn(
                        ex_state["rnd_distillation_network"].params, obsv
                    )
                    error = (random_pred - distill_pred) * (1 - done[:, None])
                    mse = jnp.square(error).mean(axis=-1)
                    if config["RWD_RMS"]:
                        rms_state, (means, stds) = jax.lax.scan(update_rms, rms_state, mse)
                        mse = mse/(stds[-1] + 1e-8)
                        
                    reward_i = mse * config["RND_REWARD_COEFF"]
                elif config["USE_ICM"]:
                    latent_obs = ex_state["icm_encoder"].apply_fn(
                        ex_state["icm_encoder"].params, last_obs
                    )
                    latent_next_obs = ex_state["icm_encoder"].apply_fn(
                        ex_state["icm_encoder"].params, obsv
                    )

                    latent_next_obs_pred = ex_state["icm_forward"].apply_fn(
                        ex_state["icm_forward"].params, latent_obs, action
                    )
                    error = (latent_next_obs - latent_next_obs_pred) * (
                        1 - done[:, None]
                    )
                    mse = jnp.square(error).mean(axis=-1)

                    reward_i = mse * config["ICM_REWARD_COEFF"]
                    if config["USE_E3B"]:
                        # Embedding is (NUM_ENVS, 128)
                        # e3b_matrix is (NUM_ENVS, 128, 128)
                        us = jax.vmap(jnp.matmul)(ex_state["e3b_matrix"], latent_obs)
                        bs = jax.vmap(jnp.dot)(latent_obs, us)

                        def update_c(c, b, u):
                            return c - (1.0 / (1 + b)) * jnp.outer(u, u)

                        updated_cs = jax.vmap(update_c)(ex_state["e3b_matrix"], bs, us)
                        new_cs = (
                            jnp.repeat(
                                jnp.expand_dims(
                                    jnp.identity(config["ICM_LATENT_SIZE"]), axis=0
                                ),
                                config["NUM_ENVS"],
                                axis=0,
                            )
                            / config["E3B_LAMBDA"]
                        )
                        ex_state["e3b_matrix"] = jnp.where(
                            done[:, None, None], new_cs, updated_cs
                        )

                        e3b_bonus = jnp.where(
                            done, jnp.zeros((config["NUM_ENVS"],)), bs
                        )

                        reward_i = e3b_bonus * config["E3B_REWARD_COEFF"]

                reward = (reward_e * config["TASK_REWARD_COEF"]) + reward_i + config["LIFE_REWARD"]

                transition = Transition(
                    last_done, action, value, reward, log_prob, last_obs, obsv, info, reward_e, reward_i
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    ex_state,
                    done,
                    rms_state,
                    hstate,
                    rng,
                    update_step,
                )
                return runner_state, transition

            initial_hstate = runner_state[-3]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                ex_state,
                last_done,
                rms_state,
                hstate,
                rng,
                update_step,
            ) = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done = carry
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = (
                        reward + config["GAMMA"] * next_value * (1 - next_done) - value
                    )
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
                    )
                    return (gae, value, done), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, last_done),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

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
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets)

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

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            init_hstate = initial_hstate[None, :]  # TBH
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )
            rng = update_state[-1]

            # UPDATE EXPLORATION STATE
            def _update_ex_epoch(update_state, unused):
                def _update_ex_minbatch(ex_state, traj_batch):
                    rnd_loss = 0

                    if config["USE_RND"]:

                        def _rnd_loss_fn(rnd_distillation_params, traj_batch):
                            random_network_out = rnd_random_network.apply(
                                rnd_random_network_params, traj_batch.next_obs
                            )

                            distillation_network_out = ex_state[
                                "rnd_distillation_network"
                            ].apply_fn(rnd_distillation_params, traj_batch.next_obs)

                            error = (random_network_out - distillation_network_out) * (
                                1 - traj_batch.done[:, None]
                            )
                            return jnp.square(error).mean() * config["RND_LOSS_COEFF"]

                        rnd_grad_fn = jax.value_and_grad(_rnd_loss_fn, has_aux=False)
                        rnd_loss, rnd_grad = rnd_grad_fn(
                            ex_state["rnd_distillation_network"].params, traj_batch
                        )
                        ex_state["rnd_distillation_network"] = ex_state[
                            "rnd_distillation_network"
                        ].apply_gradients(grads=rnd_grad)

                        losses = (rnd_loss,)
                    elif config["USE_ICM"]:
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

            # if config["USE_RND"]:
            ex_update_state = (ex_state, traj_batch, rng)
            ex_update_state, ex_loss = jax.lax.scan(
                _update_ex_epoch,
                ex_update_state,
                None,
                config["EXPLORATION_UPDATE_EPOCHS"],
            )
            if config["USE_RND"]:
                metric["rnd_loss"] = ex_loss[0].mean()
                metric["reward_i"] = traj_batch.reward_i.mean()
                metric["reward_e"] = traj_batch.reward_e.mean()
            elif config["USE_ICM"]: 
                metric["icm_inverse_loss"] = ex_loss[0].mean()
                metric["icm_forward_loss"] = ex_loss[1].mean()
                metric["reward_i"] = traj_batch.reward_i.mean()
                metric["reward_e"] = traj_batch.reward_e.mean()

            ex_state = ex_update_state[0]
            rng = ex_update_state[-1]


            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, update_step):
                    if update_step % 10 == 0:
                        to_log = create_log_dict(metric, config)
                        agg_logs = batch_log(update_step, to_log, config)
                        csv_logger.log(agg_logs)   

                jax.debug.callback(callback, metric, update_step)

            runner_state = (
                train_state,
                env_state,
                last_obs,
                ex_state,
                last_done,
                rms_state,
                hstate,
                rng,
                update_step + 1,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            ex_state,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            rms_state,
            init_hstate,
            _rng,
            0,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return train


def save_args(args_dict, path):
    # convert to a dictionary 
    for k in args_dict:
        if isinstance(args_dict[k], jax.Array):
            args_dict[k] = args_dict[k].tolist()
    # save the file 
    file_path = os.path.join(path, 'args.json') 
    with open(file_path, 'w') as f:
        json.dump(args_dict, f)

def run_ppo(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=config["ENV_NAME"]
            + "-PPO_RNN-"
            + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
            + "M",
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    if config["USE_RND"]:
        model_name = "rnd_ppo_rnn"
    elif config["USE_ICM"]: 
        model_name = "icm_ppo_rnn"
        if config["USE_E3B"]:   
            model_name = "e3b_ppo_rnn"
    else:
        assert False, "No model name provided, you should specify the intrinsic reward baseline model: '--use_rnd', '--use_icm'  or '--use_icm --use_e3b'"

    config["MODEL"] = model_name
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


if __name__ == "__main__":
    args, reset_args = intr_baselines_rnn_args(sys)

    if args.jit:
        run_ppo(args)
    else:
        with jax.disable_jit():
            run_ppo(args)