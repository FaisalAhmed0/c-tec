import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
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
from utils import visualize_agent_rnn
import cv2
import pickle
import json

def load_configs(path):
    with open(os.path.join(path, 'args.json'), 'r') as f:
        configs = json.load(f)
    return configs

def main(args):

    craftax_classic_achievements =  {
    0: 'COLLECT_WOOD',
    1: 'PLACE_TABLE',
    2: 'EAT_COW',
    3: 'COLLECT_SAPLING',
    4: 'COLLECT_DRINK',
    5: 'MAKE_WOOD_PICKAXE',
    6: 'MAKE_WOOD_SWORD',
    7: 'PLACE_PLANT',
    8: 'DEFEAT_ZOMBIE',
    9: 'COLLECT_STONE',
    10: 'PLACE_STONE',
    11: 'EAT_PLANT',
    12: 'DEFEAT_SKELETON',
    13: 'MAKE_STONE_PICKAXE',
    14: 'MAKE_STONE_SWORD',
    15: 'WAKE_UP',
    16: 'PLACE_FURNACE',
    17: 'COLLECT_COAL',
    18: 'COLLECT_IRON',
    19: 'COLLECT_DIAMOND',
    20: 'MAKE_IRON_PICKAXE',
    21: 'MAKE_IRON_SWORD'
}


    # with open(os.path.join(args.path, "config.yaml")) as f:
    #     raw_config = yaml.load(f, Loader=yaml.Loader)

    #     config = {}
    #     for key, value in raw_config.items():
    #         if isinstance(value, dict) and "value" in value:
    #             config[key] = value["value"]
    raw_config = load_configs(args.path)
    config = raw_config

    config["NUM_ENVS"] = 1

    orbax_checkpointer = PyTreeCheckpointer()
    options = CheckpointManagerOptions(max_to_keep=1, create=True)
    # import pdb;pdb.set_trace()
    checkpoint_manager = CheckpointManager(os.path.join(args.path, "policies"), orbax_checkpointer, options)

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
        100000, items=train_state
    )

    obs, env_state = env.reset(key=__rng)
    done = 0

    if is_classic:
        from craftax.craftax_classic.play_craftax_classic import CraftaxRenderer
        from craftax.craftax_classic.constants import Achievement
    else:
        from craftax.craftax.play_craftax import CraftaxRenderer
        from craftax.craftax.constants import Achievement
    achievements_to_screenshot = {}
    frames = []
    frames.append(render_craftax_pixels(env_state, 16))
    # import pdb;pdb.set_trace()
    time_step = 0
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
            for j, achieved in enumerate(old_achievements):
                if achieved:
                    # cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
                    achievements_to_screenshot[(time_step, craftax_classic_achievements[j])] =  cv2.resize(np.array(render_craftax_pixels(env_state, 16)).astype(np.float32), 
                                                                                                           (600, 600), interpolation=cv2.INTER_LINEAR)
            new_achievements = env_state.achievements
            frames.append(render_craftax_pixels(env_state, 16))
        time_step += 1
    # import pdb;pdb.set_trace()
    with open("testing_craftax_screenshots.pkl", "wb") as f: pickle.dump(achievements_to_screenshot, f)
    os.makedirs(os.path.join(args.save_path, "videos"), exist_ok=True)
    save_path = os.path.join(args.save_path, "videos")
    save_name = os.path.join(save_path, args.save_name) 
    print(f"saveing to : {save_name}")
    imageio.mimsave(save_name, jnp.array(frames[:-1]).astype(jnp.uint8)) 
    return save_name



def print_new_achievements(achievements_cls, old_achievements, new_achievements):
    for i in range(len(old_achievements)):
        if old_achievements[i] == 0 and new_achievements[i] == 1:
            print(f"{achievements_cls(i).name} ({new_achievements.sum()}/{22})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--rnn", action="store_true")
    parser.add_argument("--save_path", type=str, default="agent_visuals")
    parser.add_argument("--save_name", type=str, default="agent_visual.gif")
    parser.add_argument("--ckpt_num", type=int, default=100000)



    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")
    
    if args.rnn:
        visualize_agent_rnn(args.path, args)
        exit(0)
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)



    if args.debug:
        with jax.disable_jit():
            main(args)
    else:
        main(args)