#!/bin/bash

# ======================================

# Global Configurations

# ======================================

wandb_project_name="ctec_atari"
env_name="Pong-v5"
total_timesteps=500000000
num_envs=256
num_steps=128
normalize_reprs="--use_normalize_repr"
script_name="train_ctec_atari ppo_jax.py"
run_name_suffix="ctec_atari"

# ======================================

# Parameter Lists

# ======================================

repr_dim_list=(16 64)
similarity_measure_list=("l1" "l2")
contrastive_hiddens=(256 512 1024)
contrastive_nums_hiddens=(4)
gammas_cl=(0.99 0.5 0.3)
gammas_cl_reward=(0.99 0.5 0.3)
runs=(1 2 3)

# ======================================

# Experiment Loops

# ======================================

for gamma_cl in "${gammas_cl[@]}"; do
for gamma_cl_reward in "${gammas_cl_reward[@]}"; do
for repr_dim in "${repr_dim_list[@]}"; do
for contrastive_hidden in "${contrastive_hiddens[@]}"; do
for contrastive_num_hidden in "${contrastive_nums_hiddens[@]}"; do
for similarity_measure in "${similarity_measure_list[@]}"; do
for run in "${runs[@]}"; do

```
echo "Submitting: repr_dim=${repr_dim}, sim=${similarity_measure}, hidden_dim=${contrastive_hidden}, gamma=${gamma_cl}, run=${run}"

sbatch $script_name \
    --env_name=$env_name \
    --repr_dim=$repr_dim \
    ${normalize_reprs} \
    --total_timesteps=$total_timesteps \
    --num_steps=$num_steps \
    --num_envs=$num_envs \
    --wandb_project=$wandb_project_name \
    --gamma_cl=$gamma_cl \
    --gamma_cl_reward=$gamma_cl_reward \
    --contrastive_hidden_dim=$contrastive_hidden \
    --contrastive_number_hiddens=$contrastive_num_hidden \
    --similarity_measure=$similarity_measure \
    --run_name_suffix=$run_name_suffix
```

done
done
done
done
done
done
done
