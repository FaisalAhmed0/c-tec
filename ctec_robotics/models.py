import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling
from dataclasses import field
import jax
from typing import Any, Callable
from flax.linen import SpectralNorm

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]

lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
bias_init = nn.initializers.zeros

## Contrastive models

class CRL_MLP(nn.Module):
    '''
    An MLP for the contrastive encoders
    '''
    layer_sizes: list[int] # list of number of units in each hidden layer, len(layer_sizes) = # hidden layers
    layer_norm: bool # boolean for using layer normalization
    activation: nn.activation = field(default=nn.relu)
    activate_final: bool = False # an activation for the output layer
    spectral_norm: bool = False #  boolean for spectral normalization (for smoothness contrastint)
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()

    @nn.compact
    def __call__(self, state, train=True):
        
        hidden = state
        for i, hidden_size in enumerate(self.layer_sizes):
            if self.spectral_norm:
                hidden = SpectralNorm(nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=True,
                            ))(hidden, update_stats=train)
            else:
                hidden = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=True,
                            )(hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                if self.layer_norm:
                    hidden = nn.LayerNorm()(hidden)
                hidden = self.activation(hidden)
        return hidden
    

class SA_encoder(nn.Module):
    '''
    State action encoder
    '''
    args: object

    def setup(self):
        # Initialize the temperature parameter (starting with 1.0, can be adjusted)
        self.log_temperature = self.param('temperature', lambda key: jnp.zeros(()))

    @nn.compact
    def __call__(self, s , a, key, augment=False, train=False):
        args = self.args
        x = jnp.concatenate([s, a], axis=-1)
        layer_sizes = [args.contrastive_hidden_dim]*args.contrastive_number_hiddens + [args.repr_dim]
        encoder = CRL_MLP(layer_sizes, args.layer_norm_crl, eval(args.activation), spectral_norm=args.spectral_norm)
        x = encoder(x, train=train)
        # if enabled, normalize the representations
        if args.normalize_repr:
            x = x / (jnp.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
            if args.fix_temp:
                x = x / args.temp_value
            else:
                x = x / jnp.exp(self.log_temperature)
        return x
    
class S_encoder(nn.Module):
    '''
    State action encoder
    '''
    args: object

    def setup(self):
        # Initialize the temperature parameter (starting with 1.0, can be adjusted)
        self.log_temperature = self.param('temperature', lambda key: jnp.zeros(()))

    @nn.compact
    def __call__(self, s , a, key, augment=False, train=False):
        args = self.args
        x = s
        layer_sizes = [args.contrastive_hidden_dim]*args.contrastive_number_hiddens + [args.repr_dim]
        encoder = CRL_MLP(layer_sizes, args.layer_norm_crl, eval(args.activation), spectral_norm=args.spectral_norm)
        x = encoder(x, train=train)
        # if enabled, normalize the representations
        if args.normalize_repr:
            x = x / (jnp.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
            if args.fix_temp:
                x = x / args.temp_value
            else:
                x = x / jnp.exp(self.log_temperature)
        return x
    

class G_encoder(nn.Module):
    '''
    Future state or "Goal" encoder
    '''
    args: object

    @nn.compact
    def __call__(self, s, key, augment=False, train=False):
        args = self.args
        x = s
        layer_sizes = [args.contrastive_hidden_dim]*args.contrastive_number_hiddens + [args.repr_dim]
        encoder = CRL_MLP(layer_sizes, args.layer_norm_crl,eval(args.activation), spectral_norm=args.spectral_norm)
        x = encoder(x, train=train)
        # if enabled, normalize the representations
        if args.normalize_repr:
            x = x / (jnp.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        return x
    


class ContrastiveCritic(nn.Module):
    '''
    Temporal contrastive learning model
    '''
    args: object

    def setup(self):
        args = self.args
        
        self.obs_action_encoder = SA_encoder(args)
        self.future_obs_encoder = G_encoder(args)

    def __call__(self, obs, action, future_obs,key, augment=False, train=True):
        # encode the state and action
        obs_action_rep = self.obs_action_encoder(obs, action, key, augment=augment, train=train)
        # encode the futuere state
        future_obs_rep = self.future_obs_encoder(future_obs, key,augment=augment, train=train)
        return obs_action_rep, future_obs_rep, self.obs_action_encoder.log_temperature
    


class Aptcritic(nn.Module):
    '''
    contrastive learning model for APT baseline
    '''
    args: object

    def setup(self):
        args = self.args
    
        self.obs_encoder = S_encoder(args)

    def __call__(self, obs, action, obs_aug,key, augment=False, train=True):
        args = self.args

        obs_repr = self.obs_encoder(obs, action, key, augment=augment, train=train)
        obs_aug_rep = self.obs_encoder(obs_aug, action, key,augment=augment, train=train)
        if args.fix_temp:
            obs_aug_rep = obs_aug_rep * args.temp_value
        else:
            # remove the temporature from the second representations to avoid duplicate multiplication
            obs_aug_rep = obs_aug_rep * jnp.exp(self.obs_encoder.log_temperature)
        
        return obs_repr, obs_aug_rep, self.obs_encoder.log_temperature
    




### RND models
class StateEncoder(nn.Module):
    # hidden_dim: int
    # output_dim: int
    layer_sizes: list[int]
    layer_norm: bool
    activation: nn.activation = field(default=nn.relu)

    @nn.compact
    def __call__(self, state):
        hidden = state
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=jax.nn.initializers.lecun_uniform(),
                use_bias=True,
            )(hidden)
            if i != len(self.layer_sizes) - 1:
                if self.layer_norm:
                    hidden = nn.LayerNorm()(hidden)
                hidden = self.activation(hidden)

        return hidden
    
class RND(nn.Module):
    args: object 

    def setup(self):
        args = self.args
        hidden_sizes = [args.rnd_hidden_dim]*args.rnd_number_hiddens + [args.rnd_embed_dim]
        self.predictor = StateEncoder(layer_sizes=hidden_sizes, layer_norm=args.layer_norm, activation=eval(args.activation))
        self.target = StateEncoder(layer_sizes=hidden_sizes, layer_norm=args.layer_norm, activation=eval(args.activation))
        
    def __call__(self, obs, obs_mean, obs_std):
        args = self.args
        if args.obs_rms:
            obs = self.normalize(obs, obs_mean, obs_std)
        pred = self.predictor(obs)
        target = self.target(obs)
        return pred, jax.lax.stop_gradient(target)
    
    def normalize(self, arr: jax.Array, mean: jax.Array, std: jax.Array, eps: float = 1e-8) -> jax.Array:
        # utils for observation normalization
        return jnp.clip((arr - mean) / (std + eps), -5., 5.)
    



### ICM models
class MLP(nn.Module):
    layer_sizes: list[int]
    layer_norm: bool
    activation: nn.activation = field(default=nn.relu)
    activate_final: bool = False

    @nn.compact
    def __call__(self, state):
        hidden = state
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=jax.nn.initializers.lecun_uniform(),
                use_bias=True,
            )(hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                if self.layer_norm:
                    hidden = nn.LayerNorm()(hidden)
                hidden = self.activation(hidden)

        return hidden
        

class ICM(nn.Module):
    args: object 

    def setup(self):
        args = self.args
        # setup the state encoder, forward model and backward model
        hidden_sizes_state_endoer = [args.icm_hidden_dim]*args.icm_number_hiddens + [args.icm_embed_dim]
        hidden_sizes_forward_model = [args.icm_hidden_dim]*(args.icm_number_hiddens-1) + [args.icm_embed_dim]
        hidden_sizes_backward_model = [args.icm_hidden_dim]*(args.icm_number_hiddens-1) + [args.action_dim]
        self.obs_encoder = MLP(layer_sizes=hidden_sizes_state_endoer, layer_norm=args.layer_norm, activation=eval(args.activation))
        self.forward_model = MLP(layer_sizes=hidden_sizes_forward_model, layer_norm=args.layer_norm, activation=eval(args.activation))
        self.backward_model = MLP(layer_sizes=hidden_sizes_backward_model, layer_norm=args.layer_norm, activation=eval(args.activation), activate_final=nn.tanh)


    def __call__(self, obs, next_obs, action):
        args = self.args
        # get the observation latent
        obs_latent = self.obs_encoder(obs)
        next_obs_latent = self.obs_encoder(next_obs)
        # forward prediction
        next_obs_latent_hat = self.forward_model(jnp.concatenate([obs_latent, action], axis=-1))
        # backward prediction 
        action_hat = self.backward_model(jnp.concatenate([obs_latent, next_obs_latent], axis=-1))

        return next_obs_latent_hat, action_hat, next_obs_latent

    def encode(self, obs):
        return self.obs_encoder(obs)
