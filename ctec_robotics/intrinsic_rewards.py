import jax
from jax import lax
import jax.numpy as jnp
from brax.training import types
from typing import Any
from utils import update_rms

Transition = types.Transition

def crl_reward(contrastive_network, contrastive_params, transition: Transition, args, key_critic):
    state = transition.observation[:, :, :args.obs_dim]
    action = transition.action
    future_state = transition.extras["future_state_for_rwd"]

    random_goal_mask = jax.random.bernoulli(key_critic, args.random_goals, shape=(future_state.shape[0], 1, 1))
    future_rolled = jnp.roll(future_state, 1, axis=0)
    future_state = jnp.where(random_goal_mask, future_rolled, future_state)
    goal = future_state[:, :, args.crl_goal_indices]

    if args.use_mono_critic:
        # TODO: figure out how to use add another function to the module and use it instead of using __call__
        sm = contrastive_network.apply(contrastive_params, state, action, goal, method=contrastive_network.compute_intr_rwd).squeeze()
    else:
        sa_repr, g_repr, _ = contrastive_network.apply(contrastive_params, state, action, goal, key_critic, args.da, train=False)

        similarity_method = {
                "l2": lambda sa_repr, g_repr: -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1)),
                "l2_no_sqrt":  lambda sa_repr, g_repr: -jnp.sum((sa_repr - g_repr) ** 2, axis=-1),
                "l1":  lambda sa_repr, g_repr: -jnp.sum(jnp.abs(sa_repr - g_repr), axis=-1),
                "dot": lambda sa_repr, g_repr: jnp.einsum("hik,hik->hi", sa_repr, g_repr), # if the vectors are normalized then this the cosine 
            }
        
        sm = similarity_method[args.energy_fn](sa_repr, g_repr)
    reward = -sm
    return  jax.lax.stop_gradient(reward)


def apt_reward(contrastive_network, contrastive_params, transition: Transition, args, key_critic):
    state = transition.observation[:, :, args.crl_goal_indices]
    action = transition.action * 0 # zero the action out, apt learns only state representations

    s_repr, _, _ = contrastive_network.apply(contrastive_params, state, action, state, key_critic, args.da, train=False)
    
    k = 12
    dists = jnp.sum((s_repr[:, :, None, :] - s_repr[:, None, :, :]) ** 2, axis=-1)

    sorted_dists = jnp.sort(dists, axis=-1)

    knn_dists = sorted_dists[:, :, 1:k+1]  # First distance is to the point itself, so skip it
    mean_knn_dists = jnp.mean(knn_dists, axis=-1)  # Mean distance to k nearest neighbors
    reward = mean_knn_dists
    
    return  jax.lax.stop_gradient(reward)



def rnd_reward(rnd_network, rnd_params, transition: Transition, goal_inds: jax.Array, rwd_rms_state: Any, rnd_obs_rms_state: Any, rwd_rms=False):
    next_state = transition.next_observation[:, :, goal_inds]
    pred, target = rnd_network.apply(rnd_params, next_state, rnd_obs_rms_state[1], rnd_obs_rms_state[2])
    rwd = jax.lax.stop_gradient(jnp.mean(jnp.square(pred - target), axis=-1))
    if rwd_rms:
        eps = 1e-8
        rwd_rms_state, (means, stds) = lax.scan(update_rms, rwd_rms_state, rwd.reshape(-1))
        rwd  = rwd / (stds[-1] + eps)
        
    return rwd, rwd_rms_state




def icm_reward(icm_network, icm_params, transition: Transition, goal_inds: jax.Array, icm_rms_state: Any, rwd_rms=False):
    obs_t = transition.observation[:, :, goal_inds]
    action_t = transition.action
    next_obs = transition.next_observation[:, :, goal_inds]
    next_obs_latent_hat, _, next_obs_latent = icm_network.apply(icm_params, obs_t, next_obs, action_t)
    rwd = jax.lax.stop_gradient(jnp.mean(jnp.square(next_obs_latent_hat - next_obs_latent), axis=-1))
    if rwd_rms:
        eps = 1e-8
        icm_rms_state, (means, stds) = jax.lax.scan(update_rms, icm_rms_state, rwd.reshape(-1))
        rwd  = rwd / (stds[-1] + eps)
    # import pdb;pdb.set_trace()
    return rwd, icm_rms_state


# prediction error based on forward dynamics
def fd_reward(icm_network, icm_params, transition: Transition, goal_inds: jax.Array, fwd_rms_state: Any, rwd_rms=False):
    # obs_t = transition.observation[:, :, goal_inds]
    # action_t = transition.action
    # next_obs = transition.next_observation[:, :, goal_inds]
    obs_t = transition.observation
    action_t = transition.action
    next_obs = transition.next_observation
    next_obs_prediction = icm_network.apply(icm_params, obs_t, action_t)
    # import pdb;pdb.set_trace()
    rwd = jax.lax.stop_gradient(jnp.mean(jnp.square(next_obs - next_obs_prediction), axis=-1))
    if rwd_rms:
        def update_rms_scan(state, x_b):
            # state: tuple of 3 (), x_b: (B,)
            def body_fn(carry, x_i):
                return update_rms(carry, x_i)
            final_state, (means, stds) = jax.lax.scan(body_fn, state, x_b)
            # Return the last mean and std, with the final updated state
            return final_state, (means[-1], stds[-1])
        eps = 1e-8
        vmap_update = jax.vmap(update_rms_scan, in_axes=(0, 0))
        new_fwd_rms_state, (final_means, final_stds) = vmap_update(fwd_rms_state, rwd)
        rwd  = rwd / (final_stds[:, None] + eps)
    else:
        new_fwd_rms_state = fwd_rms_state
    return rwd, new_fwd_rms_state