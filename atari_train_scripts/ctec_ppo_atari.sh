#!/bin/bash

# ================================

# Configurable Atari PPO Runner

# ================================

# Base configuration
TOTAL_TIMESTEPS=500000000
NUM_STEPS=128
NUM_ENVS=256
WANDB_PROJECT="ctec_atari_2"
USE_NORMALIZE_REPR="--use_normalize_repr"
SCRIPT="train_ctec_atari ppo_jax.py"
# Sweep parameters
ENV_NAMES=("Breakout-v5")
REPR_DIMS=(16 64)
GAMMAS=(0.3 0.5 0.99)
HIDDEN_DIMS=(1024)
NUMBERS_UNITS=(2 4)
SIM_MEASURES=("l2" "l1" "dot" "l2_no_sqrt")
FRAME_STACKS=(1 4)

# ================================

# Run combinations

# ================================
for env_name in "${ENV_NAMES[@]}"; do
for gamma in "${GAMMAS[@]}"; do
for repr_dim in "${REPR_DIMS[@]}"; do
for sim in "${SIM_MEASURES[@]}"; do
for hidden_dim in "${HIDDEN_DIMS[@]}"; do
for frame_stack in "${FRAME_STACKS[@]}"; do
for number_units in "${NUMBERS_UNITS[@]}"; do
    CMD="sbatch $SCRIPT \
      --repr_dim=${repr_dim} \
      --env_name=${env_name} \
      ${USE_NORMALIZE_REPR} \
      --total_timesteps=${TOTAL_TIMESTEPS} \
      --num_steps=${NUM_STEPS} \
      --num_envs=${NUM_ENVS} \
      --wandb_project=${WANDB_PROJECT} \
      --gamma_cl=${gamma} \
      --gamma_cl_reward=${gamma} \
      --contrastive_hidden_dim=${hidden_dim} \
      --contrastive_number_hiddens=${number_units} \
      --frame_stack=${frame_stack} \
      --similarity_measure=${sim}"
    eval ${CMD}
  done
done
done
done
done
done
done