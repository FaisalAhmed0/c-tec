import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling, orthogonal, constant  
from dataclasses import field
import jax
from brax.training import types
import dataclasses
from typing import Any, Callable, Sequence, Tuple
import functools
import numpy as np

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
bias_init = nn.initializers.zeros



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
        rnn_state = jnp.where(resets[:, np.newaxis],self.initialize_carry(ins.shape[0], ins.shape[1]),rnn_state,)
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


def residual_block(x, width, normalize, activation):
    identity = x
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = x + identity
    return x



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


class SA_encoder_deep(nn.Module):
    config: object

    def setup(self):
        # Initialize the temperature parameter (starting with 1.0, can be adjusted)
        self.log_temperature = self.param('temperature', lambda key: jnp.zeros(()))

    @nn.compact
    def __call__(self, s , a):
        config = self.config

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if config["USE_LAYER_NROM"]:
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x
        
        activation = eval(config["ACTIVATION_CRL"])

        config = self.config
        if config["USE_ACTION_IN_CL"]:
            x = jnp.concatenate([s, a], axis=-1)
        else:
            x = s
        # create the model
        # import pdb;pdb.set_trace()

        x = nn.Dense(config["CONTRASTIVE_HIDDEN_DIM"], kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)

        for i in range(config["CONTRASTIVE_NUMBER_HIDDENS"] // 4):
            x = residual_block(x, config["CONTRASTIVE_HIDDEN_DIM"], normalize, activation)

        #Final layer
        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)

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
    

class S_encoder_deep(nn.Module):
    config: object

    @nn.compact
    def __call__(self, s , a):
        config = self.config

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if config["USE_LAYER_NROM"]:
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x
        
        activation = eval(config["ACTIVATION_CRL"])

        config = self.config
        x = s
        # create the model
        # import pdb;pdb.set_trace()

        x = nn.Dense(config["CONTRASTIVE_HIDDEN_DIM"], kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)

        for i in range(config["CONTRASTIVE_NUMBER_HIDDENS"] // 4):
            x = residual_block(x, config["CONTRASTIVE_HIDDEN_DIM"], normalize, activation)

        #Final layer
        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        
        if config["USE_NORMALIZE_REPR"]:
            x = x / (jnp.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

        return x

    

class ContrastiveModel(nn.Module):
    config: object

    # def setup(self):
    #     config = self.config
    #     # setup the state encoder, forward model and backward model
    #     self.obs_action_encoder = SA_encoder(config)
    #     self.future_obs_encoder = S_encoder(config)

    @nn.compact
    def __call__(self, obs, action, future_obs, dones, hidden_state):
        config = self.config
        if config["USE_RNN"]:
                embedding = nn.Dense(
                self.config["LAYER_SIZE"],
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                )(obs)
                embedding = nn.relu(embedding)
                rnn_in = (embedding, dones)
                hidden_state, embedding = ScannedRNN()(hidden_state, rnn_in)
                # import pdb;pdb.set_trace()  
        else:
            embedding = obs
        # update the mean and the std of the observations
        sa_encoder = SA_encoder(config)
        s_encoder = S_encoder(config)
        obs_action_rep = sa_encoder(embedding, action)
        future_obs_rep = s_encoder(future_obs)

        return obs_action_rep, future_obs_rep, sa_encoder.log_temperature, hidden_state
    

class EmpowermentModel(nn.Module):
    config: object

    def setup(self):
        config = self.config
        if config["USE_CRL_DEEP_MODEL"]:
            self.obs_action_encoder = SA_encoder_deep(config)
            self.obs_encoder = S_encoder_deep(config)
            self.future_obs_encoder1 = S_encoder_deep(config)
            self.future_obs_encoder2 = S_encoder_deep(config)
        else:
            # setup the state encoder, forward model and backward model
            self.obs_action_encoder = SA_encoder(config)
            self.obs_encoder = S_encoder(config)
            self.future_obs_encoder1 = S_encoder(config)
            self.future_obs_encoder2 = S_encoder(config)

    def __call__(self, obs, action, future_obs):
        config = self.config

        
        # compute the representations from the first contrastive model
        obs_action_rep = self.obs_action_encoder(obs, action)
        future_obs_rep1 = self.future_obs_encoder1(future_obs)
        # compute the representations from the second contrastive model
        obs_rep = self.obs_encoder(obs)
        future_obs_rep2 = self.future_obs_encoder2(future_obs)        

        return obs_action_rep, obs_rep, future_obs_rep1, future_obs_rep2, self.obs_action_encoder.log_temperature