##### Pong
# seed: int = 0
# num_steps: int = 64
# total_timesteps: float = 500e6
# gamma: float = 0.99
# gae_lambda: float = 0.8
# max_grad_norm: float = 1.0
# activation: str = "tanh"
# env_name: str = "Pong-v5"
# similarity_measure: str = "l2"
# use_action_in_cl: bool = True
# contrastive_hidden_dim: int = 2048
# contrastive_number_hiddens: int = 4
# repr_dim: int = 64
# activation_crl: str = "nn.relu"
# use_normalize_repr: bool = True
# gamma_cl: float = 0.99
# gamma_cl_reward: float = 0.99
# contrastive_loss: str = "infonce"


#### Discount of 0.99
# With representation dim of 64
sbatch train_ctec_atari ppo_jax.py --repr_dim=64 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.99 --gamma_cl_reward=0.99 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=256 --contrastive_number_hiddens=4 --similarity_measure="l2"
sbatch train_ctec_atari ppo_jax.py --repr_dim=64 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.99 --gamma_cl_reward=0.99 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=512 --contrastive_number_hiddens=4 --similarity_measure="l2"
sbatch train_ctec_atari ppo_jax.py --repr_dim=64 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.99 --gamma_cl_reward=0.99 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=1024 --contrastive_number_hiddens=4 --similarity_measure="l2"
# With representation dim of 16
sbatch train_ctec_atari ppo_jax.py --repr_dim=16 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.99 --gamma_cl_reward=0.99 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=256 --contrastive_number_hiddens=4 --similarity_measure="l2"
sbatch train_ctec_atari ppo_jax.py --repr_dim=16 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.99 --gamma_cl_reward=0.99 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=512 --contrastive_number_hiddens=4 --similarity_measure="l2"
sbatch train_ctec_atari ppo_jax.py --repr_dim=16 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.99 --gamma_cl_reward=0.99 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=1024 --contrastive_number_hiddens=4 --similarity_measure="l2"


# With representation dim of 64 and with l1 energy function
sbatch train_ctec_atari ppo_jax.py --repr_dim=64 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.99 --gamma_cl_reward=0.99 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=256 --contrastive_number_hiddens=4 --similarity_measure="l1"
sbatch train_ctec_atari ppo_jax.py --repr_dim=64 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.99 --gamma_cl_reward=0.99 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=512 --contrastive_number_hiddens=4 --similarity_measure="l1"
sbatch train_ctec_atari ppo_jax.py --repr_dim=64 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.99 --gamma_cl_reward=0.99 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=1024 --contrastive_number_hiddens=4 --similarity_measure="l1"
# With representation dim of 16 and with l1 energy function
sbatch train_ctec_atari ppo_jax.py --repr_dim=16 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.99 --gamma_cl_reward=0.99 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=256 --contrastive_number_hiddens=4 --similarity_measure="l1"
sbatch train_ctec_atari ppo_jax.py --repr_dim=16 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.99 --gamma_cl_reward=0.99 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=512 --contrastive_number_hiddens=4 --similarity_measure="l1"
sbatch train_ctec_atari ppo_jax.py --repr_dim=16 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.99 --gamma_cl_reward=0.99 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=1024 --contrastive_number_hiddens=4 --similarity_measure="l1"



#### Discount of 0.5
# With representation dim of 64
sbatch train_ctec_atari ppo_jax.py --repr_dim=64 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.5 --gamma_cl_reward=0.5 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=256 --contrastive_number_hiddens=4 --similarity_measure="l2"
sbatch train_ctec_atari ppo_jax.py --repr_dim=64 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.5 --gamma_cl_reward=0.5 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=512 --contrastive_number_hiddens=4 --similarity_measure="l2"
sbatch train_ctec_atari ppo_jax.py --repr_dim=64 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.5 --gamma_cl_reward=0.5 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=1024 --contrastive_number_hiddens=4 --similarity_measure="l2"
# With representation dim of 16
sbatch train_ctec_atari ppo_jax.py --repr_dim=16 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.5 --gamma_cl_reward=0.5 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=256 --contrastive_number_hiddens=4 --similarity_measure="l2"
sbatch train_ctec_atari ppo_jax.py --repr_dim=16 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.5 --gamma_cl_reward=0.5 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=512 --contrastive_number_hiddens=4 --similarity_measure="l2"
sbatch train_ctec_atari ppo_jax.py --repr_dim=16 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.5 --gamma_cl_reward=0.5 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=1024 --contrastive_number_hiddens=4 --similarity_measure="l2"


# With representation dim of 64 and with l1 energy function
sbatch train_ctec_atari ppo_jax.py --repr_dim=64 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.5 --gamma_cl_reward=0.5 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=256 --contrastive_number_hiddens=4 --similarity_measure="l1"
sbatch train_ctec_atari ppo_jax.py --repr_dim=64 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.5 --gamma_cl_reward=0.5 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=512 --contrastive_number_hiddens=4 --similarity_measure="l1"
sbatch train_ctec_atari ppo_jax.py --repr_dim=64 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.5 --gamma_cl_reward=0.5 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=1024 --contrastive_number_hiddens=4 --similarity_measure="l1"
# With representation dim of 16 and with l1 energy function
sbatch train_ctec_atari ppo_jax.py --repr_dim=16 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.5 --gamma_cl_reward=0.5 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=256 --contrastive_number_hiddens=4 --similarity_measure="l1"
sbatch train_ctec_atari ppo_jax.py --repr_dim=16 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.5 --gamma_cl_reward=0.5 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=512 --contrastive_number_hiddens=4 --similarity_measure="l1"
sbatch train_ctec_atari ppo_jax.py --repr_dim=16 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.5 --gamma_cl_reward=0.5 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=1024 --contrastive_number_hiddens=4 --similarity_measure="l1"



#### Discount of 0.3
# With representation dim of 64
sbatch train_ctec_atari ppo_jax.py --repr_dim=64 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.3 --gamma_cl_reward=0.3 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=256 --contrastive_number_hiddens=4 --similarity_measure="l2"
sbatch train_ctec_atari ppo_jax.py --repr_dim=64 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.3 --gamma_cl_reward=0.3 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=512 --contrastive_number_hiddens=4 --similarity_measure="l2"
sbatch train_ctec_atari ppo_jax.py --repr_dim=64 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.3 --gamma_cl_reward=0.3 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=1024 --contrastive_number_hiddens=4 --similarity_measure="l2"
# With representation dim of 16
sbatch train_ctec_atari ppo_jax.py --repr_dim=16 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.3 --gamma_cl_reward=0.3 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=256 --contrastive_number_hiddens=4 --similarity_measure="l2"
sbatch train_ctec_atari ppo_jax.py --repr_dim=16 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.3 --gamma_cl_reward=0.3 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=512 --contrastive_number_hiddens=4 --similarity_measure="l2"
sbatch train_ctec_atari ppo_jax.py --repr_dim=16 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.3 --gamma_cl_reward=0.3 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=1024 --contrastive_number_hiddens=4 --similarity_measure="l2"


# With representation dim of 64 and with l1 energy function
sbatch train_ctec_atari ppo_jax.py --repr_dim=64 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.3 --gamma_cl_reward=0.3 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=256 --contrastive_number_hiddens=4 --similarity_measure="l1"
sbatch train_ctec_atari ppo_jax.py --repr_dim=64 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.3 --gamma_cl_reward=0.3 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=512 --contrastive_number_hiddens=4 --similarity_measure="l1"
sbatch train_ctec_atari ppo_jax.py --repr_dim=64 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.3 --gamma_cl_reward=0.3 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=1024 --contrastive_number_hiddens=4 --similarity_measure="l1"
# With representation dim of 16 and with l1 energy function
sbatch train_ctec_atari ppo_jax.py --repr_dim=16 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.3 --gamma_cl_reward=0.3 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=256 --contrastive_number_hiddens=4 --similarity_measure="l1"
sbatch train_ctec_atari ppo_jax.py --repr_dim=16 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.3 --gamma_cl_reward=0.3 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=512 --contrastive_number_hiddens=4 --similarity_measure="l1"
sbatch train_ctec_atari ppo_jax.py --repr_dim=16 --env_name="Pong-v5" --use_normalize_repr --total_timesteps=500000000 --num_steps=128 --gamma_cl=0.3 --gamma_cl_reward=0.3 --num_envs=256 --wandb_project="ctec_atari" --contrastive_hidden_dim=1024 --contrastive_number_hiddens=4 --similarity_measure="l1"

