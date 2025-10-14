import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
import wandb
from flax.training.train_state import TrainState
import distrax
import gymnax
from ctec_craftax.models import contrastive_model
from wrappers import LogWrapper, FlattenObservationWrapper, LogEnvState
import envpool
import jax.lax as lax
from flax.linen.initializers import variance_scaling, orthogonal, constant  
from dataclasses import field
from typing import Any, Callable, Sequence, Tuple
from losses import contrastive_losses
import tyro


ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
bias_init = nn.initializers.zeros

class AtariCNN(nn.Module):
    @nn.compact
    def __call__(self, obs):
        x = obs
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4))(x)
        x = nn.relu(x)

        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)

        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=512)(x)
        x = nn.relu(x)

        embedding = x.reshape(x.shape[0], -1)
        return embedding

##### Contrastive learning models
class CRL_MLP(nn.Module):
    layer_sizes: list[int]
    use_layer_norm: bool
    activation_crl: nn.activation = field(default=nn.relu)
    activate_final: bool = False
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()

    @nn.compact
    def __call__(self, state, train=False):
        
        hidden = state
        # import pdb;pdb.set_trace()
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = nn.Dense(
            hidden_size,
            name=f"hidden_{i}",
            kernel_init=self.kernel_init,
            use_bias=True,
                        )(hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                if self.use_layer_norm:
                    hidden = nn.LayerNorm()(hidden)
                hidden = self.activation_crl(hidden)

        return hidden

class SA_encoder(nn.Module):
    config: object

    def setup(self):
        # Initialize the temperature parameter (starting with 1.0, can be adjusted)
        self.log_temperature = self.param('temperature', lambda key: jnp.zeros(()))

    @nn.compact
    def __call__(self, s , a):
        config = self.config
        if config["USE_ACTION_IN_CL"]:
            x = jnp.concatenate([s, a], axis=-1)
        else:
            x = s
        # create the model
        # import pdb;pdb.set_trace()
        layer_sizes = [config["CONTRASTIVE_HIDDEN_DIM"]]*config["CONTRASTIVE_NUMBER_HIDDENS"] + [config["REPR_DIM"]]
        encoder = CRL_MLP(layer_sizes, config["USE_LAYER_NORM"], eval(config["ACTIVATION_CRL"]))
        x = encoder(x)

        if config["USE_NORMALIZE_REPR"]:
            x = x / (jnp.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
            if config["FIX_TEMP"]:
                x = x / config["TEMP_VALUE"]
            else:
                x = x / jnp.exp(self.log_temperature)

        return x

class S_encoder(nn.Module):
    config: object

    @nn.compact
    def __call__(self, s):
        config = self.config
        x = s
        # create the model
        layer_sizes = [config["CONTRASTIVE_HIDDEN_DIM"]]*config["CONTRASTIVE_NUMBER_HIDDENS"] + [config["REPR_DIM"]]
        encoder = CRL_MLP(layer_sizes, config["USE_LAYER_NORM"], eval(config["ACTIVATION_CRL"]))
        x = encoder(x)

        if config["USE_NORMALIZE_REPR"]:
            x = x / (jnp.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        return x

class ContrastiveModelConv(nn.Module):
    config: object

    @nn.compact
    def __call__(self, obs, action, future_obs, dones):
        config = self.config
        cnn = AtariCNN()
        state_embedding = cnn(obs)
        future_state_embedding = cnn(future_obs)
        
        # update the mean and the std of the observations
        sa_encoder = SA_encoder(config)
        s_encoder = S_encoder(config)
        obs_action_rep = sa_encoder(state_embedding, action)
        future_obs_rep = s_encoder(future_state_embedding)

        return obs_action_rep, future_obs_rep, sa_encoder.log_temperature

##### PPO Agent model

class ActorCriticConv(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    layer_width: int = 512

    @nn.compact
    def __call__(self, obs):
        cnn = AtariCNN()
        embedding = cnn(obs)
        actor_mean = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    similarity_methods = {
        "l2": lambda sa_repr, g_repr: -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1)),
        "l2_no_sqrt":  lambda sa_repr, g_repr: -jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1),
        "l1":  lambda sa_repr, g_repr: -jnp.sum(jnp.abs(sa_repr[:, None, :] - g_repr[None, :, :]), axis=-1),
        "dot": lambda sa_repr, g_repr: jnp.einsum("ik,jk->ij", sa_repr, g_repr), # if the vectors are normalized then this the cosine 
    } # for the contrastive loss
    
    similarity_methods_for_rwd = {
            "l2": lambda sa_repr, g_repr: -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1)),
            "l2_no_sqrt": lambda sa_repr, g_repr: -(jnp.sum((sa_repr - g_repr) ** 2, axis=-1)),
            "l1":  lambda sa_repr, g_repr: -jnp.sum(jnp.abs(sa_repr - g_repr), axis=-1),
            "dot": lambda sa_repr, g_repr: jnp.einsum("ik,jk->i", sa_repr, g_repr), # if the vectors are normalized then this the cosine 
        } # for computing the c-tec reward
    
    similarity_method = similarity_methods[config["SIMILARITY_MEASURE"]]
    similarity_method_for_rwd = similarity_methods_for_rwd[config["SIMILARITY_MEASURE"]]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    if config["ATARI_ENV"]:
        env = envpool.make_gym(config["ENV_NAME"], num_envs=config["NUM_ENVS"], episodic_life=True,
        reward_clip=True,)
    else:
        env, env_params = gymnax.make(config["ENV_NAME"])
        env = FlattenObservationWrapper(env)
        env = LogWrapper(env)

    if config["USE_WANDB"]:
        wandb.init(
            project="pure-jax-rl",
            mode="online", 
            config=config,
        )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

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
        if config["ATARI_ENV"]:
            network = ActorCriticConv(
                env.action_space.n, activation=config["ACTIVATION"]
            )
            obs_shape = env.observation_space.shape
            action_shape = env.action_space.n
            contrastive_network = ContrastiveModelConv(config)
        else:
            network = ActorCritic(
                env.action_space(env_params).n, activation=config["ACTIVATION"]
            )
        rng, _rng = jax.random.split(rng)
        if config["ATARI_ENV"]:
            init_x = jnp.zeros(env.observation_space.shape)
        else:
            init_x = jnp.zeros(env.observation_space(env_params).shape)

        dummy_obs = jnp.zeros((1, *obs_shape))
        dummy_future_obs = jnp.zeros((1, *obs_shape))
        dummy_action = jnp.zeros((1, action_shape))

        # initilize models
        network_params = network.init(_rng, init_x)
        crl_params = contrastive_network.init(_rng, dummy_obs, dummy_action, dummy_future_obs, jnp.zeros((1, config["NUM_ENVS"])))
        print("=== All models have been initialized === ")
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

        tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), # I am clipping the grad norm, is that necessary?
                optax.adam(config["CRL_LR"], eps=1e-5), # also what if we used default eps value?
            )

        crl_state = {
            "crl_model": None
        }
        crl_state["crl_model"] = TrainState.create(
            apply_fn=contrastive_network.apply,
            params=crl_params,
            tx=tx

        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        if config["ATARI_ENV"]:
            internal_env_state, recv, send, step = env.xla()
            obsv, info = env.reset()
            # import pdb;pdb.set_trace()
            float_zeros = jnp.zeros(shape=(config["NUM_ENVS"], ), dtype=jnp.float32)
            int_zeros = jnp.zeros(shape=(config["NUM_ENVS"], ), dtype=jnp.int32)
            env_state = LogEnvState(
            env_state=internal_env_state,
            episode_returns=float_zeros,
            episode_lengths=int_zeros,
            returned_episode_returns=float_zeros,
            returned_episode_lengths=int_zeros,
            timestep=int_zeros,
        )
            # import pdb;pdb.set_trace()
        else:
            obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)


    
            

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng, crl_state, update_step = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                if config["ATARI_ENV"]:
                    # import pdb;pdb.set_trace()
                    internal_env_state, (obsv, reward, term, trunc, info) = step(env_state.env_state, action)
                    done = jnp.logical_or(term, trunc)
                    # import pdb;pdb.set_trace()
                    new_episode_return = env_state.episode_returns + reward
                    new_episode_length = env_state.episode_lengths + 1
                    env_state = LogEnvState(
                        env_state=internal_env_state,
                        episode_returns=new_episode_return * (1 - done),
                        episode_lengths=new_episode_length * (1 - done),
                        returned_episode_returns=env_state.returned_episode_returns * (1 - done)+ new_episode_return * done,
                        returned_episode_lengths=env_state.returned_episode_lengths * (1 - done)
                        + new_episode_length * done,
                        timestep=env_state.timestep + 1,
                    )
                    info["returned_episode_returns"] = env_state.returned_episode_returns
                    info["returned_episode_lengths"] = env_state.returned_episode_lengths
                    info["timestep"] = env_state.timestep
                    info["returned_episode"] = done
                    # print("logging seems to work")
                    # import pdb;pdb.set_trace()
                else:
                    obsv, env_state, reward, done, info = jax.vmap(
                        env.step, in_axes=(0, 0, 0, None)
                    )(rng_step, env_state, action, env_params)
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng, crl_state, update_step)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            sample_future_vmap = jax.vmap(sample_future_state, in_axes=(None, 1, 1), out_axes=1)
            future_obs_batch = sample_future_vmap(runner_state[-3], traj_batch.obs, traj_batch.done)
            
            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng, crl_state, update_step = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, future_obs, last_val):
                @jax.jit
                def mc_crl_reward(trans_batch, gamma):
                    trans_batch, future_obs = trans_batch
                    
                    state = trans_batch.obs
                    action = trans_batch.action
                    dones = trans_batch.done

                    T, N = state.shape[:2]
                    deltas_desc = jnp.arange(T-1, 0, -1)
                    def one_time(_, t):
                        s_t = lax.dynamic_index_in_dim(state, t, axis=0, keepdims=False)
                        a_t = lax.dynamic_index_in_dim(action, t, axis=0, keepdims=False)
                        a_t = jax.nn.one_hot(a_t, num_classes=action_shape)
                        
                        done = lax.dynamic_index_in_dim(dones, t, axis=0, keepdims=False)
                        def accumulate(r, delta):
                            
                            k, valid = t + delta, ((t + delta) < T)
                            s_k = lax.dynamic_index_in_dim(state, jnp.minimum(k, T-1),
                                                        axis=0, keepdims=False)
                            obs_action_rep, future_obs_rep, log_temp = contrastive_network.apply(crl_state["crl_model"].params, s_t, a_t, s_k, None)
                            d2  = similarity_method_for_rwd(obs_action_rep, future_obs_rep)    # (N,)

                            d2 = jnp.where(~done, d2*valid, 0.0)
                            return d2 + gamma * r, None
                        r_t, _ = lax.scan(accumulate, jnp.zeros((N,)), deltas_desc)
                        norm = (1.0 - gamma ** (T - t)) / (1.0 - gamma) if config["USE_NORM_CONSTANT"] else 1
                        return _, norm*r_t
                    _, reward_rev = lax.scan(one_time, None, jnp.arange(T-1, -1, -1))
                    return reward_rev[::-1]


                
                def _get_advantages(carry, transition):
                    gae, next_value, _, _ = carry
                    transition, ctec_mc_reward = transition
                    done, value, task_reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    crl_rewards = ctec_mc_reward
                    reward = (crl_rewards * config["CRL_REWARD_COEF"]) + config["TASK_REWARD_COEF"] * task_reward
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value, crl_rewards, task_reward.astype(float)), gae

                crl_rwd_mc = -1 * jax.lax.stop_gradient(mc_crl_reward((traj_batch, future_obs), config["GAMMA_CL_REWARD"]))
                # import pdb;pdb.set_trace()
                adv_info, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, jnp.zeros_like(last_val), jnp.zeros_like(last_val)),
                    (traj_batch, crl_rwd_mc),
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value, adv_info

            advantages, targets, adv_info = _calculate_gae(traj_batch, future_obs_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    metrics = {}
                    train_state, crl_state = train_state
                    traj_batch, advantages, targets, future_obs_batch = batch_info

                    # update the contrastive model
                    def _crl_loss(model_params, traj_batch, future_obs):
                        
                        action_onehot = jax.nn.one_hot(traj_batch.action, num_classes=action_shape)
                        obs_in = traj_batch.obs.reshape(-1, *obs_shape)
                        action_in = action_onehot.reshape(-1, action_shape)
                        future_obs = future_obs.reshape(-1, *obs_shape)
                        dones_in = traj_batch.done.reshape(-1, 1)
                        obs_action_rep, future_obs_rep, log_temp = contrastive_network.apply(model_params, obs_in, action_in, future_obs, dones_in)
                        sim = similarity_method(obs_action_rep, future_obs_rep)
                        #TODO: Add the contrastive losses
                        # import pdb;pdb.set_trace()
                        loss = contrastive_losses()[config["CONTRASTIVE_LOSS"]](sim, config["UPDATE_PROPORTION"], _rng)
                        # add the regularization term
                        logsumexp = jax.nn.logsumexp(sim + 1e-6, axis=-1)
                        loss += config["LOGSUMEXP_PENALTY_COEFF"] * jnp.mean(logsumexp**2)
                        
                        return loss

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
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
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    crl_grad_fn = jax.value_and_grad(_crl_loss, has_aux=False)
                    crl_loss, crl_grad = crl_grad_fn(crl_state["crl_model"].params, traj_batch, future_obs_batch)
                    crl_state["crl_model"] = crl_state["crl_model"].apply_gradients(grads=crl_grad)
                    losses = (total_loss, crl_loss)
                    return (train_state, crl_state), (total_loss, losses)

                train_state, traj_batch, advantages, targets, rng, crl_state = update_state
                rng, _rng = jax.random.split(rng)
                # Batching and Shuffling
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets, future_obs_batch)
                # import pdb;pdb.set_trace()
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # Mini-batch Updates
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, (total_loss, losses) = jax.lax.scan(
                    _update_minbatch, (train_state, crl_state), minibatches
                )
                train_state, crl_state = train_state
                update_state = (train_state, traj_batch, advantages, targets, rng, crl_state)
                return update_state, (total_loss, losses)
            # Updating Training State and Metrics:
            update_state = (train_state, traj_batch, advantages, targets, rng, crl_state)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            # import pdb;pdb.set_trace()
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-2]
            metric["crl_loss"] = loss_info[1][1].mean()
            metric["task_reward"] = traj_batch.reward.mean()
            metric["crl_reward"] = adv_info[2].mean()
            metric["crl_reward_max"] = adv_info[2].max()
            metric["crl_reward_min"] = adv_info[2].min()
            metric["crl_value"] = adv_info[1].mean()
            metric["crl_value_max"] = adv_info[1].max()
            metric["crl_value_min"] = adv_info[1].min()
            # Debugging mode
            if config.get("DEBUG"):
                def callback(info, update_step):
                    logs = {}
                    if info["returned_episode"].sum() > 0:
                        return_values = (info["returned_episode_returns"][info["returned_episode"]]).mean()
                    else:
                        return_values = 0
                    if info["returned_episode"].sum() > 0:
                        episdoe_length = (info["returned_episode_lengths"][info["returned_episode"]]).mean()
                    else:
                        episdoe_length = 0
                    timesteps = info["timestep"].max() * config["NUM_ENVS"]
                    logs["charts/episdoe_length"] = episdoe_length
                    logs["losses/crl_loss"] = info["crl_loss"].item()
                    logs["charts/task_reward"] = info["task_reward"].item()
                    logs["charts/crl_reward"] = info["crl_reward"].item()
                    logs["charts/crl_reward_max"] = info["crl_reward_max"].item()
                    logs["charts/crl_reward_min"] = info["crl_reward_min"].item()
                    logs["charts/crl_value"] = info["crl_value"].item()
                    logs["charts/crl_value_max"] = info["crl_value_max"].item()
                    logs["charts/crl_value_min"] = info["crl_value_min"].item()
                    logs["charts/mean_episode_return"] = return_values
                    if config["USE_WANDB"]:
                        wandb.log(logs, step=timesteps)
                    
        
                # import pdb;pdb.set_trace()
                jax.debug.callback(callback, metric, update_step)

            runner_state = (train_state, env_state, last_obs, rng, crl_state, update_step+1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng, crl_state, 0)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class Args:
        lr: float = 2.5e-4
        seed: int = 0
        num_envs: int = 512
        num_steps: int = 64
        total_timesteps: float = 50e6
        update_epochs: int = 4
        num_minibatches: int = 8
        gamma: float = 0.99
        gae_lambda: float = 0.8
        clip_eps: float = 0.1
        ent_coef: float = 0.01
        vf_coef: float = 0.5
        max_grad_norm: float = 1.0
        activation: str = "tanh"
        env_name: str = "Pong-v5"
        anneal_lr: bool = True
        debug: bool = True
        atari_env: bool = True
        use_wandb: bool = True
        similarity_measure: str = "l2"
        use_action_in_cl: bool = True
        fix_temp: bool = False
        temp_value: float = 1.0
        contrastive_hidden_dim: int = 2048
        contrastive_number_hiddens: int = 4
        repr_dim: int = 64
        use_layer_norm: bool = False
        activation_crl: str = "nn.relu"
        use_normalize_repr: bool = True
        crl_lr: float = 3e-4
        gamma_cl: float = 0.99
        gamma_cl_reward: float = 0.99
        crl_reward_coef: float = 1.0
        task_reward_coef: float = 0.0
        use_norm_constant: bool = False
        contrastive_loss: str = "infonce"
        update_proportion: int = 1
        logsumexp_penalty_coeff: float = 0.0

    config = tyro.cli(Args)
    config = {k.upper(): v for k, v in config.__dict__.items()}
    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)