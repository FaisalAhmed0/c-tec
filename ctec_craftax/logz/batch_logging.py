import time

import jax.numpy as jnp
import numpy as np
import wandb

batch_logs = {}
log_times = []

def compute_max_return_percentage(agent_return, env_name):
    # taken from https://github.com/MichaelTMatthews/Craftax_Baselines/issues/1
    max_env_return = {
        "Craftax-Symbolic-v1": 226,
        "Craftax-Pixels-v1": 226,
        "Craftax-Classic-Symbolic-v1": 22, 
        "Craftax-Classic-Pixels-v1": 22
    }
    return (agent_return / max_env_return[env_name]) * 100


def create_log_dict(info, config):
    to_log = {
        "episode_return": info["returned_episode_returns"],
        "episode_length": info["returned_episode_lengths"],
        # "crl_loss": info["crl_loss"]
    }
    if "crl_loss" in info:
        to_log["crl_loss"] = info["crl_loss"]
    if "task_reward" in info:
        to_log["task_reward"] = info["task_reward"] 
    if "crl_reward" in info:
        to_log["crl_reward"] = info["crl_reward"]
    if "task_intrinisc_correlation" in info:
        to_log["task_intrinisc_correlation"] = info["task_intrinisc_correlation"]
    if "relative_scale" in info:
        to_log["relative_scale"] = info["relative_scale"] 
    if "gamma_cl" in info:
        to_log["gamma_cl"] = info["gamma_cl"]

    sum_achievements = 0
    for k, v in info.items():
        if "achievements" in k.lower():
            to_log[k] = v
            sum_achievements += v / 100.0

    to_log["achievements"] = sum_achievements

    if config.get("TRAIN_RND") or config.get("USE_RND"):
        to_log["intrinsic_reward"] = info["reward_i"]
        to_log["extrinsic_reward"] = info["reward_e"]
    elif config.get("USE_RND"):
        to_log["rnd_loss"] = info["rnd_loss"]

    if config.get("TRAIN_ICM") or config.get("USE_ICM"):
        to_log["icm_inverse_loss"] = info["icm_inverse_loss"]
        to_log["icm_forward_loss"] = info["icm_forward_loss"]
        to_log["intrinsic_reward"] = info["reward_i"]
        to_log["extrinsic_reward"] = info["reward_e"]

    

    return to_log


def batch_log(update_step, log, config):
    update_step = int(update_step)
    if update_step not in batch_logs:
        batch_logs[update_step] = []

    batch_logs[update_step].append(log)

    if len(batch_logs[update_step]) == config["NUM_REPEATS"]:
        agg_logs = {}
        for key in batch_logs[update_step][0]:
            agg = []
            if key in ["goal_heatmap"]:
                agg = [batch_logs[update_step][0][key]]
            else:
                for i in range(config["NUM_REPEATS"]):
                    val = batch_logs[update_step][i][key]
                    if not jnp.isnan(val):
                        agg.append(val)

            if len(agg) > 0:
                if key in [
                    "episode_length",
                    "episode_return",
                    "exploration_bonus",
                    "e_mean",
                    "e_std",
                    "rnd_loss",
                ]:
                    agg_logs[key] = np.mean(agg)
                else:
                    agg_logs[key] = np.array(agg)
                if "return" in key:
                    agg_logs["max_return_percentage"] = compute_max_return_percentage(agg_logs[key].item(), config["ENV_NAME"])

        log_times.append(time.time())

        if config["DEBUG"]:
            if len(log_times) == 1:
                print("Started logging")
            elif len(log_times) > 1:
                dt = log_times[-1] - log_times[-2]
                steps_between_updates = (
                    config["NUM_STEPS"] * config["NUM_ENVS"] * config["NUM_REPEATS"]
                )
                sps = steps_between_updates / dt
                agg_logs["sps"] = sps

        wandb.log(agg_logs)
        return agg_logs
