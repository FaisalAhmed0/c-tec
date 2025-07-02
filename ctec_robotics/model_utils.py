"""Network definitions."""
import dataclasses
from typing import Any, Callable, Sequence, Tuple
from brax.training import types
from flax import linen
import jax
import jax.numpy as jnp
from brax.training import distribution
from brax.training import networks
from brax.training import types
import flax
from flax import linen
from brax.training.types import PRNGKey

@flax.struct.dataclass
class SACNetworks:
  policy_network: networks.FeedForwardNetwork
  q_network: networks.FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]

class MLP(linen.Module):
    """MLP module."""

    layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True
    layer_norm: bool = False
    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = linen.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                if self.layer_norm:
                    hidden = linen.LayerNorm()(hidden)
                hidden = self.activation(hidden)
        return hidden




@dataclasses.dataclass
class FeedForwardNetwork:
  init: Callable[..., Any]
  apply: Callable[..., Any]


########### For general SAC based methods ###########
def make_policy_network(
    param_size: int,
    obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    layer_norm: bool = False) -> FeedForwardNetwork:
  """Creates a policy network."""
  policy_module = MLP(
      layer_sizes=list(hidden_layer_sizes) + [param_size],
      activation=activation,
      kernel_init=kernel_init,
      layer_norm=layer_norm)

  def apply(processor_params, policy_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    return policy_module.apply(policy_params, obs)

  dummy_obs = jnp.zeros((1, obs_size))
  return FeedForwardNetwork(
      init=lambda key: policy_module.init(key, dummy_obs), apply=apply)


def make_q_network(
    obs_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    layer_norm: bool = False,
    n_critics: int = 2) -> FeedForwardNetwork:
  """Creates a value network."""

  class QModule(linen.Module):
    """Q Module."""
    n_critics: int

    @linen.compact
    def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
      hidden = jnp.concatenate([obs, actions], axis=-1)
      res = []
      for _ in range(self.n_critics):
        q = MLP(
            layer_sizes=list(hidden_layer_sizes) + [1],
            activation=activation,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            layer_norm=layer_norm)(
                hidden)
        res.append(q)
      return jnp.concatenate(res, axis=-1)

  q_module = QModule(n_critics=n_critics)

  def apply(processor_params, q_params, obs, actions):
    obs = preprocess_observations_fn(obs, processor_params)
    return q_module.apply(q_params, obs, actions)

  dummy_obs = jnp.zeros((1, obs_size))
  dummy_action = jnp.zeros((1, action_size))
  return FeedForwardNetwork(
      init=lambda key: q_module.init(key, dummy_obs, dummy_action), apply=apply)


def make_inference_fn(sac_networks):
  """Creates params and inference function for the SAC agent."""

  def make_policy(params: types.PolicyParams,
                  deterministic: bool = False) -> types.Policy:

    def policy(observations: types.Observation,
               key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
      logits = sac_networks.policy_network.apply(*params, observations)
      if deterministic:
        return sac_networks.parametric_action_distribution.mode(logits), {}
      return sac_networks.parametric_action_distribution.sample(
          logits, key_sample), {}

    return policy

  return make_policy


def make_inference_fn_with_repr(sac_networks, crl_network, args):
  """Creates params and inference function for the SAC agent."""

  def make_policy(params: types.PolicyParams,
                  deterministic: bool = False) -> types.Policy:

    def policy(observations: types.Observation,
               key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
      zeros_action = jnp.zeros(shape=(observations.shape[:-1] + (args.action_dim, )))
      sa_repr, g_repr, _ = crl_network.apply(params[2], observations[:, :args.obs_dim], zeros_action, observations[:, args.obs_dim:], key_sample, False)
      observations_inpt = jax.lax.stop_gradient(jnp.concatenate([sa_repr, g_repr], axis=1))
      logits = sac_networks.policy_network.apply(*params[:2], observations_inpt)
      if deterministic:
        return sac_networks.parametric_action_distribution.mode(logits), {}
      return sac_networks.parametric_action_distribution.sample(
          logits, key_sample), {}

    return policy

  return make_policy


def make_inference_fn_discrete(sac_networks):
  """Creates params and inference function for the SAC agent."""

  def make_policy(params: types.PolicyParams,
                  deterministic: bool = False) -> types.Policy:

    def policy(observations: types.Observation,
               key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
      logits = sac_networks.policy_network.apply(*params, observations)
      # import pdb;pdb.set_trace()
      # Actions are sampled correctly!, checked
      if deterministic:
        return CategoricalDistribution(logits).mode(), {}
      return CategoricalDistribution(logits).sample(logits, key_sample), {} 
      # return sac_networks.parametric_action_distribution.sample(
      #     logits, key_sample), {}

    return policy

  return make_policy

def make_sac_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu, 
    layer_norm=False) -> SACNetworks:
  """Make SAC networks."""
  # import pdb;pdb.set_trace()  
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size)
  policy_network = make_policy_network(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation, 
      layer_norm=layer_norm)
  q_network = make_q_network(
      observation_size,
      action_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation, 
      layer_norm=layer_norm)
  return SACNetworks(
      policy_network=policy_network,
      q_network=q_network,
      parametric_action_distribution=parametric_action_distribution)
