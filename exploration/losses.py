##### This python file contains several implementations of contrastive losses
import jax.numpy as jnp
from optax import sigmoid_binary_cross_entropy
import jax




### Contrastive related losses
def log_softmax(logits, axis, resubs):
            if not resubs:
                I = jnp.eye(logits.shape[0])
                big = 100
                eps = 1e-6
                return logits, -jax.nn.logsumexp(logits - big * I + eps, axis=axis, keepdims=True)
            else:
                return logits, -jax.nn.logsumexp(logits, axis=axis, keepdims=True)

def binary(logits):
    loss = jnp.mean(
            sigmoid_binary_cross_entropy(logits, labels=jnp.eye(logits.shape[0]))
        )  # shape[0] - is a batch size
    return loss

def symmetric_infonce(logits):
    resubs = True
    l_align1, l_unify1 = log_softmax(logits, axis=1, resubs=resubs)
    l_align2, l_unify2 = log_softmax(logits, axis=0, resubs=resubs)
    l_align = l_align1 + l_align2
    l_unif = l_unify1 + l_unify2
    loss = -jnp.mean(jnp.diag(l_align1 + l_unify1) + jnp.diag(l_align2 + l_unify2))
    return loss
    
def infonce(logits):
    resubs = True
    l_align, l_unif = log_softmax(logits, axis=1, resubs=resubs)
    loss = -jnp.mean(jnp.diag(l_align + l_unif))
    return loss

def infonce_backward(logits):
    resubs = True
    l_align, l_unif = log_softmax(logits, axis=0, resubs=resubs)
    loss = -jnp.mean(jnp.diag(l_align + l_unif))
    return loss

def flatnce(logits):
    # from https://arxiv.org/pdf/2107.01152
    logits_flat = logits - jnp.diag(logits)[:, None]
    clogits = jax.nn.logsumexp(logits_flat, axis=1)
    l_align = clogits
    l_unif = jnp.sum(logits_flat, axis=-1)
    loss = jnp.exp(clogits - jax.lax.stop_gradient(clogits)).mean()
    return loss
        
def flatnce_backward(logits):
    # same as flatnce but with axis=0 like for infonce_backward
    logits_flat = logits - jnp.diag(logits)
    clogits = jax.nn.logsumexp(logits_flat, axis=0)
    l_align = clogits
    l_unif = jnp.sum(logits_flat, axis=-1)
    loss = jnp.exp(clogits - jax.lax.stop_gradient(clogits)).mean()
    return loss
        
def fb(logits):
    batch_size = logits.shape[0]
    I = jnp.eye(batch_size)
    l_align = -jnp.diag(logits)  # shape = (batch_size,)
    l_unif = 0.5 * jnp.sum(logits**2 * (1 - I) / (batch_size - 1), axis=-1)  # shape = (batch_size,)
    loss = (l_align + l_unif).mean()  # shape = ()
    return loss
        
def dpo(logits):
    # This is based on DPO loss
    # It aims to drive positive and negative logits further away from each other
    positive = jnp.diag(logits)
    diffs = positive[:, None] - logits
    loss = -jnp.mean(jax.nn.log_sigmoid(diffs))
    return loss

def ipo(logits):
    # This is based on IPO loss
    # It aims to have difference between positive and negative logits == 1
    positive = jnp.diag(logits)
    diffs = positive[:, None] - logits
    loss = jnp.mean((diffs - 1) ** 2)
    return loss

def sppo(logits):
    # This is based on SPPO loss
    # It aims to have positive logits == 1 and negative == -1
    batch_size = logits.shape[0]
    target = -jnp.ones(batch_size) + 2* jnp.eye(batch_size)

    diff = (logits - target) ** 2
    
    # We scale positive logits by batch size to have symmetry w.r.t. negative logits
    scale = jnp.ones((batch_size, batch_size))
    scale = jnp.fill_diagonal(scale, batch_size, inplace=False)

    loss = jnp.mean(diff * scale)
    return loss

def contrastive_losses():
     return {
          "binary": binary, 
          "symmetric_infonce": symmetric_infonce,
          "infonce": infonce,
          "infonce_backward": infonce_backward,
          "flatnce": flatnce,
          "flatnce_backward": flatnce_backward,
          "fb": fb,
          "dpo": dpo,
          "ipo": ipo,
          "sppo": sppo
          
     }
def make_contrastive_critic_loss(crl_networks, args): 
    critic_network = crl_networks.critic_network
    def critic_loss(critic_params, transitions, key):
        obs = transitions.observation[:, :args.obs_dim]
        action = transitions.action
        future_obs = transitions.observation[:, args.obs_dim:]

        sa_repr, g_repr, log_temp = critic_network.apply(critic_params, obs, action, future_obs,  key, args.da)

        ################ Energy function ################
        similarity_method = {
            "l2": lambda sa_repr, g_repr: -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1)),
            "l2_no_sqrt":  lambda sa_repr, g_repr: -jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1),
            "l1":  lambda sa_repr, g_repr: -jnp.sum(jnp.abs(sa_repr[:, None, :] - g_repr[None, :, :]), axis=-1),
            "dot": lambda sa_repr, g_repr: jnp.einsum("ik,jk->ij", sa_repr, g_repr), # if the vectors are normalized then this the cosine 
        }
        ################ Energy function ################

        logits = similarity_method[args.energy_fn](sa_repr, g_repr)

        ################ Contrastive losses ################
        critic_loss = contrastive_losses()[args.contr_loss](logits)
        ################ Contrastive losses ################

        # logsumexp regularisation
        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        critic_loss += args.logsumexp_penalty_coeff * jnp.mean(logsumexp**2)

        # l2 regularisation
        l2_loss = (jnp.mean(sa_repr**2) + jnp.mean(g_repr**2))
        critic_loss += args.l2_penalty_coeff * l2_loss


        I = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
        logits_mean = logits.mean(axis=-1).mean()
        logits_std = logits.std(axis=-1).mean()

        metrics = {
            "categorical_accuracy": jnp.mean(correct),
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "logits_mean":logits_mean,
            "logits_std": logits_std,
            "logsumexp": logsumexp.mean(),
            "critic_loss": critic_loss,
            "temparture": jax.lax.stop_gradient(jnp.exp(log_temp))
        }
        return critic_loss, metrics
    return critic_loss

