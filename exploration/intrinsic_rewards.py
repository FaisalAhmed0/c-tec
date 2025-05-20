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