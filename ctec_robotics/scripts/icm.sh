#!/bin/bash

# Define common parameters (fixed values)
TRACK="--track"
WANDB_PROJECT_NAME="icm"
RENDER_AGENT="--render_agent"
activation="nn.relu"
SGD_STEPS_FACTR=1
run_name_suffix="icm"

# Define parameter arrays (all these can have multiple values)
#### For humanoid_u_maze, use the following values
# ENV_NAMES=("humanoid_u_maze")
# BATCH_SIZES=(256)                           
# NUM_ENVS_VALUES=(256)                     
ENV_NAMES=("ant_hardest_maze" "arm_binpick_hard")
BATCH_SIZES=(1024)                           
NUM_ENVS_VALUES=(1024)                      
NUM_EPOCHS_VALUES=(1000)                   
NUM_TIMESTEPS_VALUES=(500000000) 
NUM_EVALS_VALUES=(2000)                    
runs=(1 2 3 4 5)
USE_COMPLETE_FUTURE_STATE_VALUES=("--no-use_complete_future_state")
EPISODE_LENGTHS=(1000)

# Nested loops to construct and run commands for all combinations
for EPISODE_LENGTH in "${EPISODE_LENGTHS[@]}"; do
for USE_COMPLETE_FUTURE_STATE in  "${USE_COMPLETE_FUTURE_STATE_VALUES[@]}"; do
    for run in "${runs[@]}"; do
        for ENV_NAME in "${ENV_NAMES[@]}"; do
            for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
                for NUM_ENVS in "${NUM_ENVS_VALUES[@]}"; do
                    for NUM_EPOCHS in "${NUM_EPOCHS_VALUES[@]}"; do
                        for NUM_TIMESTEPS in "${NUM_TIMESTEPS_VALUES[@]}"; do
                            for NUM_EVALS in "${NUM_EVALS_VALUES[@]}"; do
                                # Construct the sbatch command
                                CMD="python icm.py \
                                    --env_name=${ENV_NAME} \
                                    ${TRACK} \
                                    ${USE_COMPLETE_FUTURE_STATE} \
                                    --wandb_project_name=\"${WANDB_PROJECT_NAME}\" \
                                    --batch_size=${BATCH_SIZE} \
                                    --multiplier_num_sgd_steps=${SGD_STEPS_FACTR} \
                                    --num_envs=${NUM_ENVS} \
                                    --num_epochs=${NUM_EPOCHS} \
                                    ${RENDER_AGENT} \
                                    --run_name_suffix=${run_name_suffix} \
                                    --num_timesteps=${NUM_TIMESTEPS} \
                                    --episode_length=${EPISODE_LENGTH} \
                                    --num_evals=${NUM_EVALS} \
                                    --activation=${activation}"
                                # Print and execute the command
                                eval ${CMD}
                            done
                        done
                    done
                done
            done
        done
    done
    done
    done