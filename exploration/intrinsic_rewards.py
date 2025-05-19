import jax
import jax.numpy as jnp
from brax.training import types

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