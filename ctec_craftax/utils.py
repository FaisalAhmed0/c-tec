import argparse
import os
import sys
import wandb
import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
import optax
import yaml
import json
from models.actor_critic import ActorCriticRNN, ScannedRNN
from wrappers import AutoResetEnvWrapper
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)
import orbax.checkpoint as ocp

from models.actor_critic import ActorCriticConv, ActorCritic
from craftax.craftax_classic.renderer import render_craftax_pixels
import imageio
import jax.lax as lax
import csv

def create_csv_logger(env_name, path):
    metrics_to_collect = ["achievements", "episode_return", "max_return_percentage"]
    achievements_classes = [
    "collect_wood",
    "place_table",
    "eat_cow",
    "collect_sapling",
    "collect_drink",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_plant",
    "defeat_zombie",
    "collect_stone",
    "place_stone",
    "eat_plant",
    "defeat_skeleton",
    "make_stone_pickaxe",
    "make_stone_sword",
    "wake_up",
    "place_furnace",
    "collect_coal",
    "collect_iron",
    "collect_diamond",
    "make_iron_pickaxe",
    "make_iron_sword"
    ]
    achievements_hard = [
    "collect_wood",
    "place_table",
    "eat_cow",
    "collect_sapling",
    "collect_drink",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_plant",
    "defeat_zombie",
    "collect_stone",
    "place_stone",
    "eat_plant",
    "defeat_skeleton",
    "make_stone_pickaxe",
    "make_stone_sword",
    "wake_up",
    "place_furnace",
    "collect_coal",
    "collect_iron",
    "collect_diamond",
    "make_iron_pickaxe",
    "make_iron_sword",
    "make_arrow",
    "make_torch",
    "place_torch",
    "collect_sapphire",
    "collect_ruby",
    "make_diamond_pickaxe",
    "make_diamond_sword",
    "make_iron_armour",
    "make_diamond_armour",
    "enter_gnomish_mines",
    "enter_dungeon",
    "enter_sewers",
    "enter_vault",
    "enter_troll_mines",
    "enter_fire_realm",
    "enter_ice_realm",
    "enter_graveyard",
    "defeat_gnome_warrior",
    "defeat_gnome_archer",
    "defeat_orc_solider",
    "defeat_orc_mage",
    "defeat_lizard",
    "defeat_kobold",
    "defeat_knight",
    "defeat_archer",
    "defeat_troll",
    "defeat_deep_thing",
    "defeat_pigman",
    "defeat_fire_elemental",
    "defeat_frost_troll",
    "defeat_ice_elemental",
    "damage_necromancer",
    "defeat_necromancer",
    "eat_bat",
    "eat_snail",
    "find_bow",
    "fire_bow",
    "learn_fireball",
    "cast_fireball",
    "learn_iceball",
    "cast_iceball",
    "open_chest",
    "drink_potion",
    "enchant_sword",
    "enchant_armour"
]

    if "Classic" in env_name:
        metrics_to_collect += [f"Achievements/{a}" for a in achievements_classes]
    else:
        metrics_to_collect += [f"Achievements/{a}" for a in achievements_hard]
        
    _logger = Simple_CSV_logger(path, header=metrics_to_collect)
    return _logger

class Simple_CSV_logger:
    def __init__(self, path, header):
        self.path = path
        self.header = header

        # If file doesn't exist, create it with header
        if not os.path.exists(self.path):
            with open(self.path, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.header)
                writer.writeheader()

    def log(self, data):
        # Write a new row using the dictionary
        data_ = {}
        for key in self.header:
            if key in data: 
                if isinstance(data[key], jnp.ndarray):
                    data_value = data[key].item()
                elif isinstance(data[key], np.ndarray):
                    data_value = data[key].item()
                else:
                    data_value = data[key]
                data_[key] = data_value
        with open(self.path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.header)
            writer.writerow(data_)

def visualize_agent(path):
    with open(os.path.join(path, "config.yaml")) as f:
        raw_config = yaml.load(f, Loader=yaml.Loader)

        config = {}
        for key, value in raw_config.items():
            if isinstance(value, dict) and "value" in value:
                config[key] = value["value"]

    config["NUM_ENVS"] = 1

    orbax_checkpointer = PyTreeCheckpointer()
    options = CheckpointManagerOptions(max_to_keep=1, create=True)
    # import pdb;pdb.set_trace()
    checkpoint_manager = CheckpointManager(os.path.join(path, "policies"), orbax_checkpointer, options)

    is_classic = False

    if config["ENV_NAME"] == "Craftax-Symbolic-v1":
        from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
        from craftax.craftax.constants import Action

        env = CraftaxSymbolicEnv(CraftaxSymbolicEnv.default_static_params())
        network = ActorCritic(len(Action), config["LAYER_SIZE"])
    elif config["ENV_NAME"] == "Craftax-Pixels-v1":
        from craftax.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv
        from craftax.craftax.constants import Action

        env = CraftaxPixelsEnv(CraftaxPixelsEnv.default_static_params())
        network = ActorCriticConv(len(Action), config["LAYER_SIZE"])
    elif config["ENV_NAME"] == "Craftax-Classic-Symbolic-v1":
        from craftax.craftax_classic.envs.craftax_symbolic_env import (
            CraftaxClassicSymbolicEnv,
        )
        from craftax.craftax_classic.constants import Action

        env = CraftaxClassicSymbolicEnv(
            CraftaxClassicSymbolicEnv.default_static_params()
        )
        network = ActorCritic(len(Action), config["LAYER_SIZE"])
        is_classic = True
    elif config["ENV_NAME"] == "Craftax-Classic-Pixels-v1":
        from craftax.craftax_classic.envs.craftax_pixels_env import (
            CraftaxClassicPixelsEnv,
        )
        from craftax.craftax_classic.constants import Action

        env = CraftaxClassicPixelsEnv(CraftaxClassicPixelsEnv.default_static_params())
        network = ActorCriticConv(len(Action), config["LAYER_SIZE"])
        is_classic = True
    else:
        raise ValueError(f"Unknown env: {config['ENV_NAME']}")

    env = AutoResetEnvWrapper(env)
    env_params = env.default_params

    init_x = jnp.zeros((config["NUM_ENVS"], *env.observation_space(env_params).shape))

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng, __rng = jax.random.split(rng, 3)
    network_params = network.init(_rng, init_x)

    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    train_state = checkpoint_manager.restore(
        0, items=train_state
    )

    obs, env_state = env.reset(key=__rng)
    done = 0

    if is_classic:
        from craftax.craftax_classic.play_craftax_classic import CraftaxRenderer
        from craftax.craftax_classic.constants import Achievement
    else:
        from craftax.craftax.play_craftax import CraftaxRenderer
        from craftax.craftax.constants import Achievement
    frames = []
    frames.append(render_craftax_pixels(env_state, 16))
    # import pdb;pdb.set_trace()
    while not done:
        obs = jnp.expand_dims(obs, axis=0)
        pi, value = network.apply(train_state.params, obs)
        rng, _rng = jax.random.split(rng)
        # import pdb;pdb.set_trace()
        action = pi.sample(seed=_rng)[0]

        if action is not None:
            rng, _rng = jax.random.split(rng)
            old_achievements = env_state.achievements
            obs, env_state, reward, done, info = env.step(
                _rng, env_state, action, env_params
            )
            new_achievements = env_state.achievements
            frames.append(render_craftax_pixels(env_state, 16))
    # import pdb;pdb.set_trace()
    os.makedirs(os.path.join(path, "videos"), exist_ok=True)
    save_path = os.path.join(path, "videos")
    save_name = os.path.join(save_path, "agent_visual.gif") 
    imageio.mimsave(save_name, jnp.array(frames[:-1]).astype(jnp.uint8)) 
    return save_name


def visualize_agent_rnn(path, args=None):
    with open(os.path.join(path, "config.yaml")) as f:
        raw_config = yaml.load(f, Loader=yaml.Loader)

        config = {}
        for key, value in raw_config.items():
            if isinstance(value, dict) and "value" in value:
                config[f"{key}"] = value["value"]
                if isinstance(config[f"{key}"], dict):
                    config[f"{key}"] = 0

    config["NUM_ENVS"] = 1
    # import pdb;pdb.set_trace()

    orbax_checkpointer = PyTreeCheckpointer()
    options = CheckpointManagerOptions(max_to_keep=1, create=True)
    # import pdb;pdb.set_trace()
    checkpoint_manager = CheckpointManager(os.path.join(path, "policies"), orbax_checkpointer, options)

    is_classic = False

    if config["ENV_NAME"] == "Craftax-Symbolic-v1":
        from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
        from craftax.craftax.constants import Action

        env = CraftaxSymbolicEnv(CraftaxSymbolicEnv.default_static_params())
        network = ActorCritic(len(Action), config["LAYER_SIZE"])
    elif config["ENV_NAME"] == "Craftax-Pixels-v1":
        from craftax.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv
        from craftax.craftax.constants import Action

        env = CraftaxPixelsEnv(CraftaxPixelsEnv.default_static_params())
        network = ActorCriticConv(len(Action), config["LAYER_SIZE"])
    elif config["ENV_NAME"] == "Craftax-Classic-Symbolic-v1":
        from craftax.craftax_classic.envs.craftax_symbolic_env import (
            CraftaxClassicSymbolicEnv,
        )
        from craftax.craftax_classic.constants import Action

        env = CraftaxClassicSymbolicEnv(
            CraftaxClassicSymbolicEnv.default_static_params()
        )
        # import pdb;pdb.set_trace()
        network = ActorCriticRNN(env.action_space(env.default_params).n, config=config)
        is_classic = True
    elif config["ENV_NAME"] == "Craftax-Classic-Pixels-v1":
        from craftax.craftax_classic.envs.craftax_pixels_env import (
            CraftaxClassicPixelsEnv,
        )
        from craftax.craftax_classic.constants import Action

        env = CraftaxClassicPixelsEnv(CraftaxClassicPixelsEnv.default_static_params())
        network = ActorCriticConv(len(Action), config["LAYER_SIZE"])
        is_classic = True
    else:
        raise ValueError(f"Unknown env: {config['ENV_NAME']}")

    env = AutoResetEnvWrapper(env)
    env_params = env.default_params
    network = ActorCriticRNN(env.action_space(env.default_params).n, config=config)

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng = jax.random.split(rng)
    init_x = (
        jnp.zeros(
            (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
        ),
        jnp.zeros((1, config["NUM_ENVS"])),
    )
    init_hstate = ScannedRNN.initialize_carry(
        config["NUM_ENVS"], config["LAYER_SIZE"]
    )
    
    network_params = network.init(_rng, init_hstate, init_x)

    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    train_state = checkpoint_manager.restore(
        0, items=train_state
    )

    obs, env_state = env.reset(key=_rng)
    done = 0
    # import pdb;pdb.set_trace()

    if is_classic:
        from craftax.craftax_classic.play_craftax_classic import CraftaxRenderer
        from craftax.craftax_classic.constants import Achievement
    else:
        from craftax.craftax.play_craftax import CraftaxRenderer
        from craftax.craftax.constants import Achievement
    frames = []
    frames.append(render_craftax_pixels(env_state, 16))
    # import pdb;pdb.set_trace()
    hstate = init_hstate
    while not done:
        obs = jnp.expand_dims(obs, axis=0)[None, :]
        done = jnp.array([done])[None, :]
        ac_in = (obs, done)
        # import pdb;pdb.set_trace()  
        hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
        rng, _rng = jax.random.split(rng)
        # import pdb;pdb.set_trace()
        action = pi.sample(seed=_rng)[0]

        if action is not None:
            rng, _rng = jax.random.split(rng)
            old_achievements = env_state.achievements
            obs, env_state, reward, done, info = env.step(_rng, env_state, action.item(), env_params)
            new_achievements = env_state.achievements
            frames.append(render_craftax_pixels(env_state, 16))
    # import pdb;pdb.set_trace()
    if args:
        os.makedirs(os.path.join(args.save_path, "videos"), exist_ok=True)
        save_path = os.path.join(args.save_path, "videos")
        save_name = os.path.join(save_path, args.save_name) 
        print(f"saveing to : {save_name}")
        imageio.mimsave(save_name, jnp.array(frames[:-1]).astype(jnp.uint8)) 
        return save_name
    else:
        os.makedirs(os.path.join(path, "videos"), exist_ok=True)
        save_path = os.path.join(path, "videos")
        save_name = os.path.join(save_path, "agent_visual.gif") 
        imageio.mimsave(save_name, jnp.array(frames[:-1]).astype(jnp.uint8)) 
        return save_name



def wandb_bar_chart(labels, values, name=None):
    # labels = ["A", "B", "C", "D"]
    # values = [10, 20, 15, 25]

    # Create a W&B table
    table = wandb.Table(data=[[label, value] for label, value in zip(labels, values)], columns=["Achievements", "success_rate"])

    # Log the bar plot
    wandb.log({"bar_chart": wandb.plot.bar(table, "Achievements", "success_rate", title="Craftax_achievements")})

import jax
import jax.numpy as jnp

from functools import partial

def init_state():
    """ Initialize correlation state """
    return {
        "n": jnp.array(0),
        "mean_x": jnp.array(0.0),
        "mean_y": jnp.array(0.0),
        "S_xx": jnp.array(0.0),
        "S_yy": jnp.array(0.0),
        "S_xy": jnp.array(0.0),
    }

@jit
def update_corr_state(state, x, y):
    """ Update state incrementally with new (x, y) """
    n = state["n"] + 1
    delta_x = x - state["mean_x"]
    delta_y = y - state["mean_y"]
    mean_x = state["mean_x"] + delta_x / n
    mean_y = state["mean_y"] + delta_y / n
    S_xx = state["S_xx"] + delta_x * (x - mean_x)
    S_yy = state["S_yy"] + delta_y * (y - mean_y)
    S_xy = state["S_xy"] + delta_x * (y - mean_y)

    return {
        "n": n,
        "mean_x": mean_x,
        "mean_y": mean_y,
        "S_xx": S_xx,
        "S_yy": S_yy,
        "S_xy": S_xy,
    }

@jit
def compute_correlation(state):
    """ Compute Pearson correlation coefficient """
    # if state["n"] < 2:
    #     return jnp.nan  # Not enough data
    return state["S_xy"] / jnp.sqrt(state["S_xx"] * state["S_yy"])


## functions to compute the mean and std from a stream of data
@jax.jit
def update_rms(state, x):
    count, mean, M2 = state
    count_new = count + 1.0
    delta = x - mean
    mean_new = mean + delta / count_new
    delta2 = x - mean_new
    M2_new = M2 + delta * delta2
    std_new = jnp.sqrt(M2_new / count_new)
    return (count_new, mean_new, M2_new), (mean_new, std_new)

# Function to compute incremental mean and std over a 1D stream of data.
def incremental_mean_std(data):
    # Initialize state: count=0, mean=0, M2=0.
    # Using data[0] to create a zero of the same shape as a sample.
    init_state = (0.0, jnp.zeros_like(data[0]), jnp.zeros_like(data[0]))
    # Use lax.scan to perform the updates over the data stream.
    final_state, (means, stds) = lax.scan(update_rms, init_state, data)
    return means, stds


def save_args(args_dict, path):
    # convert to a dictionary 
    for k in args_dict:
        if isinstance(args_dict[k], jax.Array):
            args_dict[k] = args_dict[k].tolist()
    # save the file 
    file_path = os.path.join(path, 'args.json') 
    with open(file_path, 'w') as f:
        json.dump(args_dict, f)