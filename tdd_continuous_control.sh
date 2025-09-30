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
n_steps=128

# Define parameter arrays (all these can have multiple values)
game_names=("ant_hardest_maze" "humanoid_u_maze" "arm_binpick_hard")
run_ids=(1 2 3)
tdd_loss_fns=("infonce_symmetric")
tdd_energy_fns=("mrn_pot")
use_ctec_rwds=(0 1)

# Nested loops to construct and run commands for all combinations
for game_name in "${game_names[@]}"; do
  for run_id in "${run_ids[@]}"; do
    for tdd_loss_fn in "${tdd_loss_fns[@]}"; do
      for tdd_energy_fn in "${tdd_energy_fns[@]}"; do
      for use_ctec_rwd in "${use_ctec_rwds[@]}"; do
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
            --use_ctec_rwd=${use_ctec_rwd} \
            --n_steps=${n_steps} \
            --tdd_loss_fn=${tdd_loss_fn} \
            --tdd_energy_fn=${tdd_energy_fn} \
            --total_steps=${total_steps} \
            --int_rew_source=${int_rew_source} \
            --exp_name=\"TDD_env_${game_name}\""
            eval ${CMD}
        done
      done
    done
  done
done