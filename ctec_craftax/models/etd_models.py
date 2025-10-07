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




class MLP(nn.Module):
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


class PotentialNetwork(nn.Module):
    config: object

    def setup(self):
        # Initialize the temperature parameter (starting with 1.0, can be adjusted)
        self.log_temperature = self.param('temperature', lambda key: jnp.zeros(()))

    @nn.compact
    def __call__(self, s):
        config = self.config
        x = s
        # create the model
        # import pdb;pdb.set_trace()
        layer_sizes = [config["CONTRASTIVE_HIDDEN_DIM"]]*config["CONTRASTIVE_NUMBER_HIDDENS"] + [1]
        encoder = MLP(layer_sizes, config["USE_LAYER_NORM"], eval(config["ACTIVATION_CRL"]))
        x = encoder(x)

        if config["USE_NORMALIZE_REPR"]:
            x = x / (jnp.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
            if config["FIX_TEMP"]:
                x = x / config["TEMP_VALUE"]
            else:
                x = x / jnp.exp(self.log_temperature)

        return x


class Encoder(nn.Module):
    config: object

    def setup(self):
        # Initialize the temperature parameter (starting with 1.0, can be adjusted)
        self.log_temperature = self.param('temperature', lambda key: jnp.zeros(()))

    @nn.compact
    def __call__(self, s):
        config = self.config
        x = s
        # create the model
        # import pdb;pdb.set_trace()
        layer_sizes = [config["CONTRASTIVE_HIDDEN_DIM"]]*config["CONTRASTIVE_NUMBER_HIDDENS"] + [config["REPR_DIM"]]
        encoder = MLP(layer_sizes, config["USE_LAYER_NORM"], eval(config["ACTIVATION_CRL"]))
        x = encoder(x)

        if config["USE_NORMALIZE_REPR"]:
            x = x / (jnp.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
            if config["FIX_TEMP"]:
                x = x / config["TEMP_VALUE"]
            else:
                x = x / jnp.exp(self.log_temperature)

        return x

    

class ETDModel(nn.Module):
    config: object

    @nn.compact
    def __call__(self, obs, action, future_obs, dones, hidden_state):
        config = self.config
        # update the mean and the std of the observations
        potential_network = PotentialNetwork(config)
        encoder = Encoder(config)
        phi_x = encoder(obs)
        phi_y = encoder(future_obs)
        c_y = potential_network(future_obs)

        return phi_x, phi_y, c_y, hidden_state
    