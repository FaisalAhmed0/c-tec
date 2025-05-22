import argparse
import numpy as np


def ctec_rnn_args(sys):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Classic-Symbolic-v1")
    parser.add_argument("--model", type=str, default="crl_ppo_rnn")
    parser.add_argument("--run_name_suffix", type=str, default="")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1024,
    )
    parser.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=1e9)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.8)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument(
        "--anneal_lr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=np.random.randint(2**31))
    parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save_policy", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)
    # Contrastive
    parser.add_argument("--contrastive_hidden_dim", type=int, default=1024)
    parser.add_argument("--contrastive_number_hiddens", type=int, default=2)
    parser.add_argument("--repr_dim", type=int, default=64)
    parser.add_argument("--gamma_cl", type=float, default=0.3)
    parser.add_argument("--gamma_cl_reward", type=float, default=0.3)
    parser.add_argument("--temp_value", type=float, default=1.)
    parser.add_argument("--crl_lr", type=float, default=3e-4)
    parser.add_argument("--task_reward_coef", type=float, default=0)
    parser.add_argument("--crl_reward_coef", type=float, default=1)
    parser.add_argument("--update_proportion", type=float, default=1)
    parser.add_argument("--activation_crl", type=str, default="nn.relu")
    parser.add_argument("--similarity_measure", type=str, default="l2")
    parser.add_argument("--contrastive_loss", type=str, default="infonce")
    parser.add_argument("--logsumexp_penalty_coeff", type=float, default=0.0)
    parser.add_argument("--use_single_sample", type=int, default=1)
    parser.add_argument("--use_norm_constant", type=int, default=0)
    parser.add_argument("--use_action_in_cl", type=int, default=1)
    parser.add_argument("--rwd_rms", type=int, default=0)
    parser.add_argument("--use_relative_scale", type=int, default=0)
    parser.add_argument("--life_reward", type=float, default=0)
    parser.add_argument("--use_rnn", type=int, default=0)
    ## gamma schedule configs
    parser.add_argument("--gamma_schedule", type=str, default=None)
    parser.add_argument("--gamma_schedule_start", type=float, default=0.1)
    parser.add_argument("--gamma_schedule_end", type=float, default=1.02)
    parser.add_argument(
        "--use_layer_norm", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--use_crl_deep_model", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--use_normalize_repr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--fix_temp", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--use_empowerment", type=int, default=0
    )
    parser.add_argument("--geom_trunc", type=int, default=1024)
    parser.add_argument("--relative_scale", type=float, default=0.01)
    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    return args, rest_args



def intr_baselines_rnn_args(sys):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Classic-Symbolic-v1")
    parser.add_argument("--run_name_suffix", type=str, default="")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1024,
    )
    parser.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=1e9)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.8)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument(
        "--anneal_lr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=np.random.randint(2**31))
    parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save_policy", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)
    # EXPLORATION
    parser.add_argument("--exploration_update_epochs", type=int, default=1)
    # RND
    parser.add_argument(
        "--use_rnd", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--rnd_layer_size", type=int, default=256)
    parser.add_argument("--rnd_output_size", type=int, default=512)
    parser.add_argument("--rnd_lr", type=float, default=3e-4)
    parser.add_argument("--rnd_reward_coeff", type=float, default=1.0)
    parser.add_argument("--task_reward_coef", type=float, default=0.0)
    parser.add_argument("--rnd_loss_coeff", type=float, default=0.01)
    parser.add_argument("--rnd_gae_coeff", type=float, default=0.01)
    parser.add_argument("--task_gae_coeff", type=float, default=0)
    parser.add_argument("--agent", type=str, default="rnd_ppo")
    parser.add_argument("--rwd_rms", type=int, default=0)
    parser.add_argument(
        "--rnd_is_episodic", action=argparse.BooleanOptionalAction, default=False
    )
    # ICM
    parser.add_argument(
        "--use_icm", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--icm_reward_coeff", type=float, default=1.0)
    parser.add_argument("--icm_lr", type=float, default=3e-4)
    parser.add_argument("--icm_forward_loss_coef", type=float, default=1.0)
    parser.add_argument("--icm_inverse_loss_coef", type=float, default=1.0)
    parser.add_argument("--icm_layer_size", type=int, default=256)
    parser.add_argument("--icm_latent_size", type=int, default=32)
    # E3B
    parser.add_argument(
        "--use_e3b", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--e3b_reward_coeff", type=float, default=1.0)
    parser.add_argument("--e3b_lambda", type=float, default=0.1)
    parser.add_argument("--life_reward", type=float, default=0)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    return args, rest_args


def vanilla_ppo_rnn_args(sys):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1024,
    )
    parser.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=1e9)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.8)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument(
        "--anneal_lr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=np.random.randint(2**31))
    parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save_policy", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--agent", type=str, default="ppo_rnn")
    parser.add_argument(
        "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    return args, rest_args
