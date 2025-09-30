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
int_rew_source="CTEC"
n_steps=128

# Define parameter arrays (all these can have multiple values)
game_names=("ant_hardest_maze" "humanoid_u_maze" "arm_binpick_hard")
run_ids=(1 2 3)
ctec_loss_fns=("infonce")
ctec_energy_fns=("l1" "l2")
normalize_reprs=(0)
discounts=(0.99)
use_etd_rwds=(0 1)

# Nested loops to construct and run commands for all combinations
for game_name in "${game_names[@]}"; do
  for run_id in "${run_ids[@]}"; do
    for ctec_loss_fn in "${ctec_loss_fns[@]}"; do
      for ctec_energy_fn in "${ctec_energy_fns[@]}"; do
        for normalize_repr in "${normalize_reprs[@]}"; do
        for discount in "${discounts[@]}"; do
        for use_etd_rwd in "${use_etd_rwds[@]}"; do
          CMD="sbatch train_tdd src/train.py \
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
            --n_steps=${n_steps} \
            --ctec_loss_fn=${ctec_loss_fn} \
            --ctec_energy_fn=${ctec_energy_fn} \
            --normalize_repr=${normalize_repr} \
            --total_steps=${total_steps} \
            --use_etd_rwd=${use_etd_rwd} \
            --int_rew_source=${int_rew_source} \
            --discount=${discount} \
            --exp_name=\"CTEC_env_${game_name}\""
            eval ${CMD}
          done
          done
        done
      done
    done
done
done
