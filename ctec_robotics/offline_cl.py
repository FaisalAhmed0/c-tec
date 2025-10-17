from dataclasses import dataclass
from typing import NamedTuple 
import tyro
import torch
from torch.utils.data import Dataset, DataLoader
import os
import jax.numpy as jnp
import argparse
import json
import numpy as np
from models import ContrastiveCritic
from losses import make_contrastive_critic_loss as make_contrastive_loss
import flax
import flax.linen as nn
import jax
import matplotlib.pyplot as plt
from tqdm import tqdm
from brax.training import gradients
from brax.training.acme.types import NestedArray
import optax
from brax.training.types import Params, PRNGKey
import wandb
_PMAP_AXIS_NAME = "i"

similarity_method = {
            "l2": lambda sa_repr, g_repr: -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1)),
            "l2_no_sqrt":  lambda sa_repr, g_repr: -jnp.sum((sa_repr - g_repr) ** 2, axis=-1),
            "l1":  lambda sa_repr, g_repr: -jnp.sum(jnp.abs(sa_repr - g_repr), axis=-1),
            "dot": lambda sa_repr, g_repr: jnp.einsum("hik,hik->hi", sa_repr, g_repr), # if the vectors are normalized then this the cosine 
        }


class Transition(NamedTuple):
    """Container for a transition."""

    observation: NestedArray
    action: NestedArray
    all_observations: NestedArray


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    contrastive_optimizer_state: optax.OptState
    contrastive_params: Params

@dataclass
class Args:
    exper_path: str = "/network/scratch/f/faisal.mohamed/crl_runs/ctec_sac/ant_hardest_maze//1760646002_nickname"
    epochs: int = 10
    batch_size: int = 256
    wandb_project_name: str = "offline_cl"
    wandb_entity: str = None
    wandb_mode: str = 'online'
    track: bool = False
    viusal_freq: int = 5
    crl_goal_indices: object = None
    crl_observation_dim: int = 0 # if > 0 use for debugging
    use_complete_future_state: bool = False
    crl_observation_dim: int = 0 # if > 0 use for debugging
    crl_goal_indices: object = None
    noise_std: float = 0.1
    da: bool = False
    sa_projector: bool = False
    g_projector: bool = False
    fix_temp: bool = False
    temp_value:float = 1
    spectral_norm: bool = False
    use_diag_q: bool = False
    logsumexp_penalty_coeff: float = 0.1
    l2_penalty_coeff: float = 0.0
    random_goals: float = 0.0 # poportion of random goals in the actor loss
    energy_fn: str = "l1"
    contr_loss: str = "infonce"
    repr_dim: int = 64
    normalize_repr: bool = True
    temp_scaling: bool = True
    model: str = "ctec_sac"
    contrastive_number_hiddens: int = 2
    contrastive_hidden_dim: int = 256
    use_deep_encoder: bool = False
    discounting_cl: float = 0.99
    layer_norm_crl: bool = False


def load_args(path: str):
    with open(path, 'rb') as fin:
        return dict_to_args(json.load(fin))
    

def dict_to_args(d: dict) -> argparse.Namespace:
    """
    Convert a dictionary back to an argparse.Namespace.
    (Note: if the dict values were converted to JSON-friendly types, 
    they will remain as such; converting back to a JAX array would require extra steps.)
    """
    return argparse.Namespace(**d)



class TorchDataSet(Dataset):
    """
    PyTorch Dataset for loading .npz files (with JAX arrays) from a directory.
    Converts arrays to torch tensors for use with DataLoader.
    """
    def __init__(self, npz_files_directory, args, number_of_file=50):
        self.args = args
        npz_files = [f for f in os.listdir(npz_files_directory) if f.endswith('.npz')]
        if not npz_files:
            raise ValueError(f"No .npz files found in {npz_files_directory}")
        data_dict = {}
        npz_files = np.random.choice(npz_files, size=number_of_file, replace=False)
        for file in tqdm(npz_files, desc="Loading npz files", unit="file"):
            file_path = os.path.join(npz_files_directory, file)
            with np.load(file_path) as data:
                for key in data:
                    arr = data[key]
                    if key not in data_dict:
                        data_dict[key] = [arr]
                    else:
                        data_dict[key].append(arr)
            # break
        # Concatenate arrays for each key
        for key in data_dict:
            data_dict[key] = np.concatenate(data_dict[key], axis=0)
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        self.length = data_dict[self.keys[0]].shape[0]
        
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Return a dict of torch tensors for each key
        observations = self.data_dict["observation"][idx]
        actions = self.data_dict["action"][idx]
        self.obs_shape = observations.shape[-1]
        # rewards = self.data_dict["reward"][idx]
        # selection random observation, action and a future observation
        traj_length = observations.shape[0]
        rand_ind = np.random.choice(traj_length)
        obs = observations[rand_ind]
        action = actions[rand_ind]   
        # sample future observation 
        indices_range = np.arange(rand_ind, traj_length)
        gammas = self.args.discounting_cl ** (indices_range - rand_ind)
        gammas = gammas / np.sum(gammas)
        future_ind = np.random.choice(indices_range, p=gammas)
        future_obs = observations[future_ind][:2]
        obs[self.args.obs_dim:] = future_obs
        # import pdb;pdb.set_trace()
        all_observations = observations.reshape(-1, self.obs_shape)
        return {"observation": (obs), "action": (action), "future_observation": (future_obs), "all_observations": all_observations}    

    def sample_random_obs(self, batch_size):
        all_obs = self.data_dict["observation"].reshape(-1,self.obs_shape)
        # import pdb;pdb.set_trace()  
        random_inds = np.random.choice(all_obs.shape[0], size=(batch_size, ), replace=False)
        return all_obs[random_inds][:,:2]

    

    


def create_contrastive_model_and_optimizer(args):
    # TODO: fix the MonolithicCriticLoss
    # if args.use_mono_critic:
    #     contrastive_network = MonolithicCritic(args)
    # else:
    contrastive_network = ContrastiveCritic(args)
    contrastive_optimizer = optax.adam(learning_rate=args.critic_lr)
    return contrastive_network, contrastive_optimizer



    


def main(args):
    assert args.exper_path != ""
    # load the training args
    training_args_path = os.path.join(args.exper_path, "args.json")
    buffer_data_path = os.path.join(args.exper_path, "buffer_data")
    training_args = load_args(training_args_path)
    dataset = TorchDataSet(buffer_data_path, training_args)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    
    contrastive_network, contrastive_optimizer = create_contrastive_model_and_optimizer(training_args)

    if args.track:
        wandb.init(project=args.wandb_project_name, 
                   entity=args.wandb_entity, 
                   mode=args.wandb_mode)
        


    @flax.struct.dataclass
    class CRLNetworks:
        critic_network: nn.Module
    crl_networks = CRLNetworks(
        critic_network=contrastive_network
    )
    # TODO: fix the MonolithicCriticLoss
    # if training_args.use_mono_critic:
    #     contrastive_loss_fn = make_mono_critic_loss(crl_networks, training_args)
    # else:
    contrastive_loss_fn = make_contrastive_loss(crl_networks, training_args)


    key = jax.random.key(training_args.seed)
    key, key_critic = jax.random.split(key, 2)
    # import pdb;pdb.set_trace()
    def _init_training_state(
    key: PRNGKey,
    obs_size: int,
    future_obs_size: int,
    local_devices_to_use: int,
    contrastive_network: nn.Module,
    contrastive_optimizer: optax.GradientTransformation,
) -> TrainingState:
        """Inits the training state and replicates it over devices."""
        key_policy, key_q, key_contrastive = jax.random.split(key, num=3)
        dummy_state = jnp.zeros((1, obs_size))
        dummy_action = jnp.zeros((1, training_args.action_dim))
        dummy_future_state = jnp.zeros((1, future_obs_size))

        contrastive_params = contrastive_network.init(key_contrastive, dummy_state, dummy_action, dummy_future_state, key_contrastive, False)
        contrastive_optimizer_state = contrastive_optimizer.init(contrastive_params)
        # initialize the state for the GC-policy

        training_state = TrainingState(
            contrastive_optimizer_state=contrastive_optimizer_state,
            contrastive_params=contrastive_params,
        )
        return training_state
    
    training_state = _init_training_state(key, training_args.obs_dim, training_args.crl_observation_dim, 1, contrastive_network, contrastive_optimizer)

    def visualize(training_state, transitions, epoch):
        number_of_samples = 50000
        random_future_obs = dataset.sample_random_obs(number_of_samples)
        batch_size = transitions.observation.shape[0]
        random_ind = np.random.choice(batch_size, replace=False)
        # import pdb;pdb.set_trace()
        obs = transitions.observation[random_ind]
        initial_observation_tiled = jnp.repeat(obs[None, :training_args.obs_dim], axis=0, repeats=number_of_samples)
        num_actions = 1
        ctec_reward = []
        for i in range(num_actions):
            random_action_ind = np.random.choice(batch_size, replace=False)
            # print(random_action_ind)
            action = transitions.action[random_action_ind]
            initial_action_tiled = jnp.repeat(action[None, :], axis=0, repeats=number_of_samples)
            all_future_observations = random_future_obs[:, training_args.crl_goal_indices]
            # import pdb;pdb.set_trace()
            sa_repr, future_repr, _ = contrastive_network.apply(training_state.contrastive_params,initial_observation_tiled, initial_action_tiled, all_future_observations, key_critic, False, train=False)
            logits = similarity_method[training_args.energy_fn](sa_repr, future_repr)
            ctec_reward.append(logits)
            # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        ctec_reward = -1 * jnp.stack(ctec_reward)
        # pair_wise_dist = jnp.sqrt(((actions_logits[:, None, :] - actions_logits[None, :, : ] ) ** 2))
        # pair_wise_dist = pair_wise_dist.reshape(num_actions*num_actions, -1)
        # # import pdb;pdb.set_trace()
        fig = plt.figure(figsize=(20, 20))
        plt.title(f"Pairwise differences between similarity score of different actions")
        for j in range(num_actions*num_actions):
            plt.subplot(num_actions,num_actions,j+1)
            obs_2d = obs[:2]
            plt.scatter(all_future_observations[:, 0], all_future_observations[:, 1], c=ctec_reward[j], cmap="jet", alpha=0.3)
            plt.colorbar()
            plt.scatter(obs_2d[0], obs_2d[1], color="black", s=100)
            
        fig.canvas.draw()
        # # Get the width and height of the figure in pixels
        width, height = fig.canvas.get_width_height()
        # # Extract the RGB buffer as a string
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape((height, width, 3))
        if args.track:
            wandb.log({f"ctec reward": wandb.Image(image)})
        plt.close(fig)
        # import pdb;pdb.set_trace()

    def train_step(training_state, transitions, key_critic):

        (loss, metrics), grads = jax.value_and_grad(contrastive_loss_fn, has_aux=True)(training_state.contrastive_params, transitions, key_critic)
        updates, opt_state = contrastive_optimizer.update(grads, training_state.contrastive_optimizer_state)
        new_contrastive_params = optax.apply_updates(training_state.contrastive_params, updates)
        training_state = training_state.replace(contrastive_params=new_contrastive_params, contrastive_optimizer_state=opt_state)

        return loss, metrics, training_state
    
    transitions = next(iter(dataloader))
    transitions = Transition(observation=jnp.asarray(transitions["observation"].numpy()),action=jnp.asarray(transitions["action"].numpy()),all_observations=jnp.asarray(transitions["all_observations"].numpy()) )

    # import pdb;pdb.set_trace()
    jitted_trianing_step = jax.jit(train_step)
    global_iterations_counter = 0
    for epoch in range(args.epochs):
        iteration_conter = 0
        average_loss = 0
        for transitions in dataloader:
            global_iterations_counter += 1
            iteration_conter += 1
            transitions = Transition(observation=jnp.asarray(transitions["observation"].numpy()),action=jnp.asarray(transitions["action"].numpy()),all_observations=jnp.asarray(transitions["all_observations"].numpy()) )
            loss, metrics, training_state = jitted_trianing_step(training_state, transitions, key_critic)
            average_loss += loss
            if args.track:
                wandb.log({"contrastive_loss_per_iteration": loss, })
        average_loss /=  iteration_conter
        print(f"Epoch: {epoch}, Loss: {average_loss}")
        if args.track:
            wandb.log({"contrastive_avg_loss_per_epoch": average_loss, })
        # import pdb;pdb.set_trace()
        if epoch % args.viusal_freq == 0:
            visualize(training_state, transitions, epoch)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
    