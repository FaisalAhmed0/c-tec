import jax
import flax
import functools
import jax.numpy as jnp

from jax import flatten_util
from brax.training.types import PRNGKey
from brax.training.replay_buffers import ReplayBuffer, ReplayBufferState
from typing import Generic, TypeVar, Tuple, Any

Sample = TypeVar("Sample")
ReplayBufferState = Any
### Generic
@flax.struct.dataclass
class ReplayBufferState:
  """Contains data related to a replay buffer."""

  data: jnp.ndarray
  insert_position: jnp.ndarray
  sample_position: jnp.ndarray
  key: PRNGKey

class QueueBase(ReplayBuffer[ReplayBufferState, Sample], Generic[Sample]):
    """Base class for limited-size FIFO reply buffers.

    Implements an `insert()` method which behaves like a limited-size queue.
    I.e. it adds samples to the end of the queue and, if necessary, removes the
    oldest samples form the queue in order to keep the maximum size within the
    specified limit.

    Derived classes must implement the `sample()` method.
    """

    def __init__(
        self,
        max_replay_size: int,
        dummy_data_sample: Sample,
        sample_batch_size: int,
        num_envs: int,
        episode_length: int,
    ):
        self._flatten_fn = jax.vmap(jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0]))

        dummy_flatten, self._unflatten_fn = flatten_util.ravel_pytree(dummy_data_sample)
        self._unflatten_fn = jax.vmap(jax.vmap(self._unflatten_fn))
        data_size = len(dummy_flatten)

        self._data_shape = (max_replay_size, num_envs, data_size)
        self._data_dtype = dummy_flatten.dtype
        self._sample_batch_size = sample_batch_size
        self._size = 0
        self.num_envs = num_envs
        self.episode_length = episode_length

    def init(self, key: PRNGKey) -> ReplayBufferState:
        return ReplayBufferState(
            data=jnp.zeros(self._data_shape, self._data_dtype),
            sample_position=jnp.zeros((), jnp.int32),
            insert_position=jnp.zeros((), jnp.int32),
            key=key,
        )

    def check_can_insert(self, buffer_state, samples, shards):
        """Checks whether insert operation can be performed."""
        assert isinstance(shards, int), "This method should not be JITed."
        insert_size = jax.tree_util.tree_flatten(samples)[0][0].shape[0] // shards
        if self._data_shape[0] < insert_size:
            raise ValueError(
                "Trying to insert a batch of samples larger than the maximum replay"
                f" size. num_samples: {insert_size}, max replay size"
                f" {self._data_shape[0]}"
            )
        self._size = min(self._data_shape[0], self._size + insert_size)

    def insert_internal(
        self, buffer_state: ReplayBufferState, samples: Sample
    ) -> ReplayBufferState:
        """Insert data in the replay buffer.

        Args:
          buffer_state: Buffer state
          samples: Sample to insert with a leading batch size.

        Returns:
          New buffer state.
        """
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"buffer_state.data.shape ({buffer_state.data.shape}) "
                f"doesn't match the expected value ({self._data_shape})"
            )

        update = self._flatten_fn(samples)
        data = buffer_state.data

        # If needed, roll the buffer to make sure there's enough space to fit
        # `update` after the current position.
        position = buffer_state.insert_position
        roll = jnp.minimum(0, len(data) - position - len(update))
        data = jax.lax.cond(roll, lambda: jnp.roll(data, roll, axis=0), lambda: data)
        position = position + roll

        # Update the buffer and the control numbers.
        data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
        position = (position + len(update)) % (len(data) + 1)
        sample_position = jnp.maximum(0, buffer_state.sample_position + roll)

        return buffer_state.replace(
            data=data,
            insert_position=position,
            sample_position=sample_position,
        )
    def sample_internal(
        self, buffer_state: ReplayBufferState
    ) -> Tuple[ReplayBufferState, Sample]:
        raise NotImplementedError(f"{self.__class__}.sample() is not implemented.")

    def size(self, buffer_state: ReplayBufferState) -> int:
        return (
            buffer_state.insert_position - buffer_state.sample_position
        )  # pytype: disable=bad-return-type  # jax-ndarray
    
    
    
### Trajectories Buffer    
class TrajectoryUniformSamplingQueue(QueueBase[Sample], Generic[Sample]):
    """Implements an uniform sampling limited-size replay queue BUT WITH TRAJECTORIES."""

    def sample_internal(self, buffer_state: ReplayBufferState) -> Tuple[ReplayBufferState, Sample]:
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"Data shape expected by the replay buffer ({self._data_shape}) does "
                f"not match the shape of the buffer state ({buffer_state.data.shape})"
            )
        key, sample_key, shuffle_key = jax.random.split(buffer_state.key, 3)
        # NOTE: this is the number of envs to sample but it can be modified if there is OOM
        shape = self.num_envs

        # Sampling envs idxs
        envs_idxs = jax.random.choice(sample_key, jnp.arange(self.num_envs), shape=(shape,), replace=False)

        @functools.partial(jax.jit, static_argnames=("rows", "cols"))
        def create_matrix(rows, cols, min_val, max_val, rng_key):
            rng_key, subkey = jax.random.split(rng_key)
            start_values = jax.random.randint(subkey, shape=(rows,), minval=min_val, maxval=max_val)
            row_indices = jnp.arange(cols)
            matrix = start_values[:, jnp.newaxis] + row_indices
            return matrix

        @jax.jit
        def create_batch(arr_2d, indices):
            return jnp.take(arr_2d, indices, axis=0, mode="wrap")

        create_batch_vmaped = jax.vmap(create_batch, in_axes=(1, 0))

        matrix = create_matrix(
            shape,
            self.episode_length,
            buffer_state.sample_position,
            buffer_state.insert_position - self.episode_length,
            sample_key,
        )

        batch = create_batch_vmaped(buffer_state.data[:, envs_idxs, :], matrix)
        transitions = self._unflatten_fn(batch)
        return buffer_state.replace(key=key), transitions

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["config", "env", "apply_fn"])
    def flatten_crl_fn(config, env, transition, sample_key: PRNGKey, goal_indicies, contrastive_params, apply_fn):
        goal_key, transition_key = jax.random.split(sample_key)
        
        # Because it's vmaped transition obs.shape is of shape (transitions,obs_dim)
        seq_len = transition.observation.shape[0]
        arrangement = jnp.arange(seq_len)
        is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
        discount = config.discounting_cl ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)

        # Sample goal indices for computing the contrastive reward
        if config.future_state_rwd_sampling == "geometric":
            print("sample from the geometric distribution")
            probs_for_rwd = is_future_mask * discount 
        elif config.future_state_rwd_sampling == "uniform":
            print("sample from the uniform distribution")
            discount = 1 ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
            probs_for_rwd = is_future_mask * discount
        elif config.future_state_rwd_sampling == "inv_geometric":
            print("sample from inverse geometric")
            probs_for_rwd = is_future_mask * discount
            probs_for_rwd = jnp.flip(probs_for_rwd, axis=-1)
        elif config.future_state_rwd_sampling == "gaussian":
            print("sample from gaussian distribution")
            mean = 1.0 / (1.0 - discount)
            std = 1.0
            # Generate gaussian probabilities for future states
            diff = jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
            probs_for_rwd = jnp.exp(-0.5 * ((diff - mean) / std) ** 2)
            # Only consider future states and normalize
            probs_for_rwd = probs_for_rwd * is_future_mask
        elif "sim_score" in config.future_state_rwd_sampling:
            '''
            # 1. take the future states and convert them to goals
            # 2. get the state and goal representations
            # 3. compute the score
            '''
            future_state = transition.observation
            future_state_goal = future_state[:, goal_indicies]
            state_rep, goal_rep, _ = apply_fn(contrastive_params, transition.observation[:, :env.state_dim], transition.action, future_state_goal, sample_key, args.da, train=False)
            score = -jnp.sum(jnp.abs(state_rep[:, None, :] - goal_rep[None, :, :]), axis=-1)
            score = score * is_future_mask
            # sample accotding to the negative similarity score
            if "neg" in config.future_state_rwd_sampling:
                probs_for_rwd = jax.lax.stop_gradient(jnp.exp(-score))
            elif "pos" in config.future_state_rwd_sampling:
                probs_for_rwd = jax.lax.stop_gradient(jnp.exp(score))
            
        # sample goals for training the contrastive model
        probs = is_future_mask * discount
        
        single_trajectories = jnp.concatenate([transition.extras["state_extras"]["seed"][:, jnp.newaxis].T] * seq_len, axis=0)
        probs_for_rwd = probs_for_rwd * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
        probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5

        goal_index = jax.random.categorical(goal_key, jnp.log(probs))
        future_state = jnp.take(transition.observation, goal_index[:-1], axis=0)
        future_action = jnp.take(transition.action, goal_index[:-1], axis=0)
        
        goal = future_state[:, goal_indicies]
        future_state = future_state[:, :env.state_dim]
        state = transition.observation[:-1, :env.state_dim]
        new_obs = jnp.concatenate([state, goal], axis=1)
        future_reward = jnp.take(transition.reward, goal_index[:-1], axis=0)

        rwd_goal_index = jax.random.categorical(goal_key, jnp.log(probs_for_rwd))
        future_state_for_rwd = jnp.take(transition.observation, rwd_goal_index[:-1], axis=0)

        
        extras = {
            "policy_extras": {},
            "state_extras": {
                "truncation": jnp.squeeze(transition.extras["state_extras"]["truncation"][:-1]),
                "seed": jnp.squeeze(transition.extras["state_extras"]["seed"][:-1]),
            },
            "state": state,
            "future_state": future_state,
            "future_action": future_action,
            "future_reward": future_reward,
            "future_state_for_rwd": future_state_for_rwd
        }

        return transition._replace(
            observation=jnp.squeeze(new_obs),
            action=jnp.squeeze(transition.action[:-1]),
            reward=jnp.squeeze(transition.reward[:-1]),
            discount=jnp.squeeze(transition.discount[:-1]),
            extras=extras,
        )