import os 
import numpy as np
from dataclasses import dataclass 

@dataclass
class CTEC_args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = np.random.randint(2**31)
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "exploration"
    wandb_entity: str = None
    wandb_mode: str = 'online'
    wandb_dir: str = '.'
    wandb_group: str = '.'
    capture_video: bool = False
    checkpoint: bool = True
    run_name_suffix: str = ""
    num_videos: int = 30

    #environment specific arguments
    env_name: str = "ant_hardest_maze"
    episode_length: int = 1000
    # to be filled in runtime
    obs_dim: int = 0
    goal_start_idx: int = 0
    goal_end_idx: int = 0

    # Algorithm specific arguments
    num_timesteps: int = 10_000_000
    num_epochs: int = 50
    num_envs: int = 512
    num_eval_envs: int = 5
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    discounting: float = 0.99
    use_dense_reward: bool = False
    tau = 0.005
    logsumexp_penalty_coeff: float = 0.1
    entropy_reg: bool = True

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    agent_number_hiddens: int = 2
    agent_hidden_dim: int = 256
    
    unroll_length: int  = 62
    reward_scaling: float = 1.0
    use_her: bool = False
    """Use hindsight experince replay"""
    multiplier_num_sgd_steps: int = 1
    deterministic_eval: bool = False
    action_repeat: int = 1
    num_evals: int = 50
    backend: str = None
    eval_env: str = None
    render_agent: bool = False
    include_goal_in_obs: bool = True
    layer_norm: bool = False
    agent_activation: str = "nn.relu"
    activation: str = "nn.relu"
    # to be filled in runtime
    env_steps_per_actor_step : int = 0
    """number of env steps per actor step (computed in runtime)"""
    num_prefill_env_steps : int = 0
    """number of env steps to fill the buffer before starting training (computed in runtime)"""
    num_prefill_actor_steps : int = 0
    """number of actor steps to fill the buffer before starting training (computed in runtime)"""
    num_training_steps_per_epoch : int = 0
    """the number of training steps per epoch(computed in runtime)"""
    render_freq: int = 12*5

    ## CRL related params
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
    future_state_rwd_sampling: str = "geometric"
    gamma_schedule: str = None
    gamma_schedule_start: float = 0.1
    gamma_schedule_end: float = 1.0
    save_all_crl_ckpts: bool = False
    ema: float=0.999
    save_replay_data: bool = False



@dataclass
class APT_args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = np.random.randint(2**31)
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "exploration"
    wandb_entity: str = None
    wandb_mode: str = 'online'
    wandb_dir: str = '.'
    wandb_group: str = '.'
    capture_video: bool = False
    checkpoint: bool = True
    run_name_suffix: str = ""
    num_videos: int = 30

    #environment specific arguments
    env_name: str = "ant_hardest_maze"
    episode_length: int = 1000
    # to be filled in runtime
    obs_dim: int = 0
    goal_start_idx: int = 0
    goal_end_idx: int = 0

    # Algorithm specific arguments
    num_timesteps: int = 10_000_000
    num_epochs: int = 50
    num_envs: int = 512
    num_eval_envs: int = 5
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    discounting: float = 0.99
    use_dense_reward: bool = False
    tau = 0.005
    logsumexp_penalty_coeff: float = 0.1
    entropy_reg: bool = True

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    agent_number_hiddens: int = 2
    agent_hidden_dim: int = 256
    
    unroll_length: int  = 62
    reward_scaling: float = 1.0
    use_her: bool = False
    """Use hindsight experince replay"""
    multiplier_num_sgd_steps: int = 1
    deterministic_eval: bool = False
    action_repeat: int = 1
    num_evals: int = 50
    backend: str = None
    eval_env: str = None
    render_agent: bool = False
    include_goal_in_obs: bool = True
    layer_norm: bool = False
    agent_activation: str = "nn.relu"
    activation: str = "nn.relu"
    # to be filled in runtime
    env_steps_per_actor_step : int = 0
    """number of env steps per actor step (computed in runtime)"""
    num_prefill_env_steps : int = 0
    """number of env steps to fill the buffer before starting training (computed in runtime)"""
    num_prefill_actor_steps : int = 0
    """number of actor steps to fill the buffer before starting training (computed in runtime)"""
    num_training_steps_per_epoch : int = 0
    """the number of training steps per epoch(computed in runtime)"""
    render_freq: int = 12*5

    ## CRL related params
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
    model: str = "apt_sac"
    contrastive_number_hiddens: int = 2
    contrastive_hidden_dim: int = 256
    use_deep_encoder: bool = False
    discounting_cl: float = 0.99
    layer_norm_crl: bool = False
    future_state_rwd_sampling: str = "geometric"
    gamma_schedule: str = None
    gamma_schedule_start: float = 0.1
    gamma_schedule_end: float = 1.0
    save_all_crl_ckpts: bool = False
    include_action_in_crl: bool = False
    ema: float = 0.999