import jax
import time
import numpy as np
import jax.numpy as jnp
import flax.linen as nn

from brax import envs
from envs.ant import Ant
from typing import NamedTuple
from collections import namedtuple

import time
import jax
import numpy as np
from brax.training import acting
from brax.training.types import Metrics
from brax.training.types import PolicyParams


def make_inference_fn(crl_networks):
    """Creates params and inference function for the CRL agent."""
    def make_policy(params, deterministic: bool = False):
        def policy(obs, key_sample):
            logits = crl_networks.policy_network.apply(*params[:2], obs)
            if deterministic:
                action = crl_networks.parametric_action_distribution.mode(logits)
            else:
                action = crl_networks.parametric_action_distribution.sample(logits, key_sample)
            return action, {}
        return policy
    return make_policy

def generate_unroll(actor_step, training_state, env, env_state, unroll_length, extra_fields=()):
  """Collect trajectories of given unroll_length."""

  @jax.jit
  def f(carry, unused_t):
    state = carry
    nstate, transition = actor_step(training_state, env, state, extra_fields=extra_fields)
    return nstate, transition

  final_state, data = jax.lax.scan(f, env_state, (), length=unroll_length)
  return final_state, data

def generate_unroll_custom_actor_step(normalizer_params, actor_step, env, env_state, unroll_length, crl_networks, contrastive_params, policy_params, key, args, extra_fields=()):
  """Collect trajectories of given unroll_length."""
  make_policy = make_inference_fn(crl_networks)
  policy = make_policy((normalizer_params, policy_params))
  @jax.jit
  def f(carry, unused_t):
    env_state, current_key = carry
    current_key, next_key = jax.random.split(current_key)
    env_state, transition = actor_step(
                env,
                env_state,
                policy,
                current_key,
                crl_networks.critic_network,
                contrastive_params,
                args,
                extra_fields=(
                    "truncation",
                    "seed",
                ),
            )
    # nstate, transition = actor_step(training_state, env, state, extra_fields=extra_fields)
    return (env_state, next_key), transition

  final_state, data = jax.lax.scan(f, (env_state, key), (), length=unroll_length)
  return final_state[0], data

class CrlEvaluator():
    '''
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    contrastive_model,
    contrastive_params:Params,
    args,
    extra_fields: Sequence[str] = (),
    '''
    def __init__(self, actor_step, 
                        eval_env,
                        num_eval_envs, 
                        episode_length, 
                        key, 
                        crl_networks, 
                        training_state,
                        args):

      self._key = key
      self._eval_walltime = 0.

      eval_env = envs.training.EvalWrapper(eval_env)

      

      

      def generate_eval_unroll(training_state, key):
        reset_keys = jax.random.split(key, num_eval_envs)
        eval_first_state = eval_env.reset(reset_keys)
        normalizer_params,policy_params,critic_params = training_state
        return generate_unroll_custom_actor_step(normalizer_params,
                                                  actor_step,
                                                  eval_env, 
                                                  eval_first_state, 
                                                  episode_length, 
                                                  crl_networks, 
                                                  critic_params, 
                                                  policy_params,
                                                  key, args, extra_fields=())[0]

      self._generate_eval_unroll = jax.jit(generate_eval_unroll)
      self._steps_per_unroll = episode_length * num_eval_envs

    def run_evaluation(self, training_state, training_metrics, aggregate_episodes = True):
      """Run one epoch of evaluation."""
      self._key, unroll_key = jax.random.split(self._key)

      t = time.time()
      eval_state = self._generate_eval_unroll(training_state, unroll_key)
      eval_metrics = eval_state.info["eval_metrics"]
      eval_metrics.active_episodes.block_until_ready()
      epoch_eval_time = time.time() - t
      metrics = {}
      aggregating_fns = [
          (np.mean, ""),
          # (np.std, "_std"),
          # (np.max, "_max"),
          # (np.min, "_min"),
      ]

      for (fn, suffix) in aggregating_fns:
          metrics.update(
              {
                  f"eval/episode_{name}{suffix}": (
                      fn(eval_metrics.episode_metrics[name]) if aggregate_episodes else eval_metrics.episode_metrics[name]
                  )
                  for name in ['reward', 'success', 'success_easy', 'dist', 'distance_from_origin']
              }
          )

      # We check in how many env there was at least one step where there was success
      if "success" in eval_metrics.episode_metrics:
          metrics["eval/episode_success_any"] = np.mean(
              eval_metrics.episode_metrics["success"] > 0.0
          )

      metrics["eval/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
      metrics["eval/epoch_eval_time"] = epoch_eval_time
      metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
      self._eval_walltime = self._eval_walltime + epoch_eval_time
      metrics = {"eval/walltime": self._eval_walltime, **training_metrics, **metrics}
      return metrics
    


  # This is an evaluator that behaves in the exact same way as brax Evaluator,
# but additionally it aggregates metrics with max, min.
# It also logs in how many episodes there was any success.
class ActorCrlEvaluator(acting.Evaluator):
    def run_evaluation(
        self,
        policy_params: PolicyParams,
        training_metrics: Metrics,
        aggregate_episodes: bool = True,
    ) -> Metrics:
        """Run one epoch of evaluation."""
        self._key, unroll_key = jax.random.split(self._key)

        t = time.time()
        eval_state = self._generate_eval_unroll(policy_params, unroll_key)
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}
        aggregating_fns = [
            (np.mean, ""),
            (np.std, "_std"),
            (np.max, "_max"),
            (np.min, "_min"),
        ]

        for (fn, suffix) in aggregating_fns:
            metrics.update(
                {
                    f"eval/episode_{name}{suffix}": (
                        fn(value) if aggregate_episodes else value
                    )
                    for name, value in eval_metrics.episode_metrics.items()
                }
            )

        # We check in how many env there was at least one step where there was success
        if "success" in eval_metrics.episode_metrics:
            metrics["eval/episode_success_any"] = np.mean(
                eval_metrics.episode_metrics["success"] > 0.0
            )

        metrics["eval/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics["eval/epoch_eval_time"] = epoch_eval_time
        metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {"eval/walltime": self._eval_walltime, **training_metrics, **metrics}

        return metrics  # pytype: disable=bad-return-type  # jax-ndarray