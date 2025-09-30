#!/bin/bash

# Define common parameters (fixed values)
use_wandb=1
int_rew_coef=1
ext_rew_coef=0
num_processes=16
use_model_rnn=0
policy_cnn_type=-1
model_cnn_type=-1
env_source="brax"
total_steps=30000000
int_rew_source="TDD"

# Define parameter arrays (all these can have multiple values)
game_names=("ant_hardest_maze" "humanoid_u_maze" "arm_binpick_hard")
run_ids=(58947 594793759 25 342 1)
int_rew_sources=("RND" "NGU" "E3B" "NovelD")

# Nested loops to construct and run commands for all combinations
for game_name in "${game_names[@]}"; do
  for run_id in "${run_ids[@]}"; do
  for int_rew_source in "${int_rew_sources[@]}"; do
          CMD="sbatch train src/train.py \
            --use_wandb=${use_wandb} \
            --int_rew_coef=${int_rew_coef} \
            --ext_rew_coef=${ext_rew_coef} \
            --num_processes=${num_processes} \
            --use_model_rnn=${use_model_rnn} \
            --policy_cnn_type=${policy_cnn_type} \
            --model_cnn_type=${model_cnn_type} \
            --env_source=${env_source} \
            --game_name=${game_name} \
            --run_id=${run_id} \
            --total_steps=${total_steps} \
            --int_rew_source=${int_rew_source} \
            --exp_name=\"${int_rew_source}_env_${game_name}_run_id_${run_id}\""
            eval ${CMD}
        done
      done
    done
