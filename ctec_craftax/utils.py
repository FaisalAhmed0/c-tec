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
from models.contrastive_model import ContrastiveModel, EmpowermentModel
from craftax.craftax_classic.renderer import render_craftax_pixels
import imageio
import jax.lax as lax
import csv
from envs.ant_maze import AntMaze
from typing import Sequence, NamedTuple, Dict
from craftax.craftax_classic.constants import Achievement, Action

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    # reward: jnp.ndarray
    obs: jnp.ndarray
    # info: jnp.ndarray



similarity_methods = {
        "l2": lambda sa_repr, g_repr: -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1)),
        "l2_no_sqrt":  lambda sa_repr, g_repr: -jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1),
        "l1":  lambda sa_repr, g_repr: -jnp.sum(jnp.abs(sa_repr[:, None, :] - g_repr[None, :, :]), axis=-1),
        "dot": lambda sa_repr, g_repr: jnp.einsum("ik,jk->ij", sa_repr, g_repr), # if the vectors are normalized then this the cosine 
    } # for the contrastive loss

similarity_methods_for_rwd = {
        "l2": lambda sa_repr, g_repr: -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1)),
        "l2_no_sqrt": lambda sa_repr, g_repr: -(jnp.sum((sa_repr - g_repr) ** 2, axis=-1)),
        "l1":  lambda sa_repr, g_repr: -jnp.sum(jnp.abs(sa_repr - g_repr), axis=-1),
        "dot": lambda sa_repr, g_repr: jnp.einsum("ik,jk->i", sa_repr, g_repr), # if the vectors are normalized then this the cosine 
    } # for computing the c-tec reward
    

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


def visualize_agent_rnn(path, config=None, args=None, log_to_wandb=False):
    if config is None:
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
    checkpoint_crl_manager = CheckpointManager(os.path.join(path, "crl"), orbax_checkpointer, options)

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
        int(config["TOTAL_TIMESTEPS"]), items=train_state
    )

    contrastive_network = ContrastiveModel(config)
    obs_shape = env.observation_space(env_params).shape[0]
    action_shape = env.action_space(env_params).n
    dummy_obs = jnp.zeros((1, obs_shape))
    dummy_future_obs = jnp.zeros((1, obs_shape))
    dummy_action = jnp.zeros((1, action_shape))
    crl_params = contrastive_network.init(_rng, dummy_obs, dummy_action, dummy_future_obs, jnp.zeros((1, config["NUM_ENVS"])),  init_hstate)

    tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),optax.adam(config["LR"], eps=1e-5),)
    crl_train_state = TrainState.create(apply_fn=contrastive_network.apply,params=crl_params,tx=tx,)
    crl_train_state = checkpoint_crl_manager.restore(int(config["TOTAL_TIMESTEPS"]), items=crl_train_state)
    similarity_method_for_rwd = similarity_methods_for_rwd[config["SIMILARITY_MEASURE"]]

    def mc_crl_reward(trans_batch, action, gamma):
        trans_batch = trans_batch
        
        state = trans_batch.obs
        # action = trans_batch.action
        dones = trans_batch.done 
        T_DELTA = config["NUM_STEPS"]
        T_total, N, D = state.shape
        deltas_desc = jnp.arange(T_DELTA-1, 0, -1)
        def one_time(_, t):
            s_t = lax.dynamic_index_in_dim(state, t, axis=0, keepdims=False)
            a_t = action
            a_t = jax.nn.one_hot(a_t, num_classes=action_shape)
            
            done = lax.dynamic_index_in_dim(dones, t, axis=0, keepdims=False)
            def accumulate(r, delta):
                k, valid = t + delta, ((t + delta) < T_total)
                s_k = lax.dynamic_index_in_dim(state, jnp.minimum(k, T_total-1),
                                            axis=0, keepdims=False)
                # import pdb;pdb.set_trace()
                obs_action_rep, future_obs_rep, log_temp, init_hidden = contrastive_network.apply(crl_train_state.params, s_t, a_t, s_k, None, None)
                d2  = jax.lax.stop_gradient(similarity_method_for_rwd(obs_action_rep, future_obs_rep))    # (N,))

                d2 = jnp.where(~done, d2*valid, 0.0)
                return d2 + gamma * r, None
            r_t, _ = lax.scan(accumulate, jnp.zeros((N,)), deltas_desc)
            norm = (1.0 - gamma ** (T_DELTA - t)) / (1.0 - gamma) if config["USE_NORM_CONSTANT"] else 1
            return _, norm*r_t
        _, reward_rev = lax.scan(one_time, None, jnp.arange(T_total-1, -1, -1))
        return reward_rev[::-1]

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
    obs_stack = []
    action_stack = []
    done_stack = []
    reward_stack = []
    achievements = set()
    achievements_timesteps_pairs = []
    t = 0
    while not done:
        last_obs = obs = jnp.expand_dims(obs, axis=0)[None, :]
        done = jnp.array([done])[None, :]
        ac_in = (obs, done)
        # import pdb;pdb.set_trace()  
        hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
        rng, _rng = jax.random.split(rng)
        # import pdb;pdb.set_trace()
        action = pi.sample(seed=_rng)[0]

        if action is not None:
            rng, _rng = jax.random.split(rng)
            # old_achievements = env_state.achievements
            obs, env_state, reward, done, info = env.step(_rng, env_state, action.item(), env_params)
            new_achievements = env_state.achievements
            if reward.item() > 0:
                for i in range(len(new_achievements)):
                    ach = new_achievements[i].item()
                    if ach:
                        if Achievement(i).name not in achievements:
                            achievements.add(Achievement(i).name)
                            achievements_timesteps_pairs.append((Achievement(i).name, t))
            frames.append(render_craftax_pixels(env_state, 16))
            transition = Transition(
                done, action, last_obs
            )
            obs_stack.append(last_obs)
            action_stack.append(action)
            done_stack.append(done)
            reward_stack.append(reward)
            last_obs = obs
        t += 1
    # import pdb;pdb.set_trace()
    achievements_timesteps = [ach_time[1] for ach_time in achievements_timesteps_pairs]
    tt = Transition(done=jnp.stack([d for d in done_stack]),action=jnp.stack([a for a in action_stack]),obs=jnp.stack([o.squeeze()[None, :] for o in obs_stack]))
    num_actions = env.action_space(env_params).n
    action_ctec_reward_per_step = jnp.zeros(shape=(len(action_stack), num_actions))
    for i in range(num_actions):
        action_ctec_reward_per_step = action_ctec_reward_per_step.at[:, i].set(-1 * mc_crl_reward(tt, jnp.array([i]), config["GAMMA_CL_REWARD"]).squeeze())
    import matplotlib.pyplot as plt
    bar_plots = []
    action_labels = [f"{i}" for i in range(num_actions)]
    ctec_return = 0
    craftex_return = 0
    ctec_returns_so_far = []
    craftex_returns_so_far = []
    time_steps_track = 0
    for i in range(action_ctec_reward_per_step.shape[0]):
        current_frame = frames[i]
        ctec_reward_per_action = action_ctec_reward_per_step[i]
        colors = ["blue"] * num_actions
        colors[action_stack[i].item()] = "red"
        ctec_return = ctec_reward_per_action[action_stack[i].item()]
        craftex_return += reward_stack[i]
        ctec_returns_so_far.append(ctec_return)
        craftex_returns_so_far.append(craftex_return)

        # Prepare data for return plots
        timesteps = list(range(i + 1))

        # Create a figure with four subplots: frame, bar plot, ctec_return, craftex_return
        fig, axs = plt.subplots(
            1, 4, figsize=(25, 4), 
            gridspec_kw={'width_ratios': [1, 2, 1, 1]}
        )
        ax_frame, ax_bar, ax_ctec_return, ax_craftex_return = axs

        # Show the frame in the left subplot
        ax_frame.imshow(current_frame.astype(np.int64))
        ax_frame.axis('off')

        # Plot the action reward bar plot in the second subplot
        ctec_reward_per_action_normazlied = (ctec_reward_per_action - ctec_reward_per_action.mean())
        ax_bar.bar(action_labels, ctec_reward_per_action_normazlied, color=colors)
        # ax_bar.set_ylim(0, 1.0)
        ax_bar.text(
            1.0, 1.0, f"timestep: {i}", 
            transform=ax_bar.transAxes,
            fontsize=12,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
        if i in achievements_timesteps:
            ax_bar.text(
                0.5, 1.05, 
                f"Achievement: {achievements_timesteps_pairs[time_steps_track][0]}", 
                transform=ax_bar.transAxes,
                fontsize=12,
                color="black",
                ha='center',
                va='bottom',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none')
            )
            time_steps_track += 1
        for spine in ["top", "right"]:
            ax_bar.spines[spine].set_visible(False)

        # Plot ctec_return up to this step in the third subplot
        ax_ctec_return.plot(timesteps, ctec_returns_so_far, color='blue', label='CTEC Return')
        ax_ctec_return.scatter([i], [ctec_returns_so_far[-1]], color='red')
        ax_ctec_return.set_xlabel("Timestep")
        ax_ctec_return.set_ylabel("CTEC reward")
        ax_ctec_return.set_title("CTEC reward")
        for spine in ["top", "right"]:
            ax_ctec_return.spines[spine].set_visible(False)
        ax_ctec_return.grid(alpha=0.3, linestyle='--', linewidth=0.5)

        # Plot craftax_return up to this step in the fourth subplot
        ax_craftex_return.plot(timesteps, craftex_returns_so_far, color='green', label='Craftax Return')
        ax_craftex_return.scatter([i], [craftex_returns_so_far[-1]], color='red')
        ax_craftex_return.set_xlabel("Timestep")
        ax_craftex_return.set_ylabel("Craftax Return")
        ax_craftex_return.set_title("Craftax Return")
        for spine in ["top", "right"]:
            ax_craftex_return.spines[spine].set_visible(False)
        ax_craftex_return.grid(alpha=0.3, linestyle='--', linewidth=0.5)

        fig.tight_layout()
        fig.canvas.draw()
        # Convert plot to numpy array
        width, height = fig.canvas.get_width_height()
        np_fig_combined = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((height, width, 3))
        plt.close(fig)
        bar_plots.append(np_fig_combined)

    print(f"ctec return: {ctec_return}")
    os.makedirs(os.path.join(path, "videos"), exist_ok=True)
    save_path = os.path.join(path, "videos")
    # imageio.mimsave("./ctec_rewd_action_dis.gif", np.stack(bar_plots, axis=0),)
    # save_name = os.path.join(save_path, "ctec_rewd_action_dis.gif") 
    save_name_mp4 = os.path.join(save_path, "ctec_rewd_action_dis.mp4") 
    # imageio.mimsave(save_name, np.stack(bar_plots, axis=0),)
    imageio.mimsave(save_name_mp4, np.stack(bar_plots, axis=0),)
    if log_to_wandb:
            # wandb.log({"ctec_rewd_action_dis": wandb.Image(save_name)})
            wandb.log({"ctec_rewd_action_dis_mp4":  wandb.Video(save_name_mp4)})
    
        
        
    if args:
        os.makedirs(os.path.join(args.save_path, "videos"), exist_ok=True)
        save_path = os.path.join(args.save_path, "videos")
        save_name = os.path.join(save_path, args.save_name) 
        print(f"saveing to : {save_name}")
        imageio.mimsave(save_name, jnp.array(frames[:-1]).astype(jnp.uint8)) 
        if log_to_wandb:
            wandb.log({"Agent Video": wandb.Image(save_name, fps=10, format="gif")})
        return save_name
    else:
        os.makedirs(os.path.join(path, "videos"), exist_ok=True)
        save_path = os.path.join(path, "videos")
        save_name = os.path.join(save_path, "agent_visual.gif") 
        imageio.mimsave(save_name, jnp.array(frames[:-1]).astype(jnp.uint8)) 
        if log_to_wandb:
            wandb.log({"Agent Video": wandb.Image(save_name)})
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



# def create_brax_env(args: argparse.Namespace) -> object:
#     env_name = "ant_hardest_maze"
#     env = AntMaze(backend= "spring", maze_layout_name=env_name[4:], include_goal_in_obs=False)
#     return env


from collections import Counter
import os
import numpy as np
from collections import defaultdict

class DiscretizedDensity:
    def __init__(self, axes=None, bin_width=1.0, goal_dim=2, run_folder=None):
        self._axes = np.array(axes, dtype=np.int64) if axes is not None else None
        self._bin_width = float(bin_width)
        self.goal_dim = goal_dim

        # use dict[int] instead of Counter for lower overhead
        self.counter = defaultdict(int)
        self.total_count = 0  # cache total count

        self.run_folder = run_folder
        if run_folder:
            self.visual_path = os.path.join(run_folder, "visuals/state_coverage")
            self.visited_states_path = os.path.join(run_folder, "visited_states")
            os.makedirs(self.visited_states_path, exist_ok=True)
            os.makedirs(self.visual_path, exist_ok=True)
        else:
            self.visual_path = None
            self.visited_states_path = None

    def discretize(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if self._axes is not None:
            obs = obs[self._axes]
        obs = np.floor(obs / self._bin_width).astype(np.int64)
        return tuple(obs) if obs.ndim > 0 else int(obs)

    def update_count(self, batch_obs, env_step=0):
        batch_obs = np.asarray(batch_obs, dtype=np.float32)

        if self.visited_states_path:
            np.savez_compressed(f"{self.visited_states_path}/{env_step}", data=batch_obs)

        if self._axes is not None:
            batch_obs = batch_obs[:, self._axes]

        # discretize in bulk
        batch_obs = np.floor(batch_obs / self._bin_width).astype(np.int64)

        # update counter more efficiently
        for obs in map(tuple, batch_obs):
            self.counter[obs] += 1
        self.total_count += len(batch_obs)

    def compute_log_prob(self, obs):
        obs_d = self.discretize(obs)
        count = self.counter.get(obs_d, 1)
        if self.total_count == 0:
            return np.log(1e-8)
        prob = count / self.total_count
        return np.log(prob + 1e-8)

    def entropy(self):
        if self.total_count == 0:
            return 0.0
        counts = np.fromiter(self.counter.values(), dtype=np.int64)
        prob = counts / self.total_count
        return -np.sum(prob * np.log(prob + 1e-8))

    def num_states(self):
        return len(self.counter)
