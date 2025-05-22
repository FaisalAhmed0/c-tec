
#!/bin/bash
# Define common parameters (fixed values)
TRACK="--track"
WANDB_PROJECT_NAME="apt"
RENDER_AGENT="--render_agent"
contrastive_hidden_dim=1024
activation="nn.relu"
NORMALIZE_REP="--normalize_repr"
LR=0.0003
SGD_STEPS_FACTR=1
ENTROPY_REG="--entropy_reg"
run_name_suffix="apt"
checkpoint="--no-checkpoint"
logsumexp_penalty_coeff=0.1


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
runs=(1 2 3 4 5) # number of seeds, each seed is chosen randomly (results might slightly differ from the paper resutls)
REPS_DIMS=(64)
USE_COMPLETE_FUTURE_STATE_VALUES=("--no-use_complete_future_state")
CONTR_LOSSES=("infonce")
EPISODE_LENGTHS=(1000) 
energy_fns=("l1") # contrastive critic function
contrastive_number_hiddenss=(2)
LAYER_NORMS=("--no-layer_norm_crl")


# Nested loops to construct and run commands for all combinations
for LAYER_NORM in "${LAYER_NORMS[@]}"; do
for energy_fn in "${energy_fns[@]}"; do
for contrastive_number_hiddens in "${contrastive_number_hiddenss[@]}"; do
for EPISODE_LENGTH in "${EPISODE_LENGTHS[@]}"; do
for CONT_LOSS in "${CONTR_LOSSES[@]}"; do
for USE_COMPLETE_FUTURE_STATE in  "${USE_COMPLETE_FUTURE_STATE_VALUES[@]}"; do
    for REP_DIM in "${REPS_DIMS[@]}"; do
        for run in "${runs[@]}"; do
            for ENV_NAME in "${ENV_NAMES[@]}"; do
                for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
                    for NUM_ENVS in "${NUM_ENVS_VALUES[@]}"; do
                        for NUM_EPOCHS in "${NUM_EPOCHS_VALUES[@]}"; do
                            for NUM_TIMESTEPS in "${NUM_TIMESTEPS_VALUES[@]}"; do
                                for NUM_EVALS in "${NUM_EVALS_VALUES[@]}"; do 
                                    # Construct the sbatch command
                                    CMD="python apt.py \
                                        --env_name=${ENV_NAME} \
                                        ${TRACK} \
                                        ${NORMALIZE_REP} \
                                        ${USE_COMPLETE_FUTURE_STATE} \
                                        ${LAYER_NORM} \
                                        ${ENTROPY_REG} \
                                        --multiplier_num_sgd_steps=${SGD_STEPS_FACTR} \
                                        --wandb_project_name=\"${WANDB_PROJECT_NAME}\" \
                                        --batch_size=${BATCH_SIZE} \
                                        --num_envs=${NUM_ENVS} \
                                        --num_epochs=${NUM_EPOCHS} \
                                        --logsumexp_penalty_coeff=${logsumexp_penalty_coeff} \
                                        ${RENDER_AGENT} \
                                        --num_timesteps=${NUM_TIMESTEPS} \
                                        --num_evals=${NUM_EVALS} \
                                        --energy_fn=${energy_fn} \
                                        --contrastive_number_hiddens=${contrastive_number_hiddens} \
                                        --contrastive_hidden_dim=${contrastive_hidden_dim} \
                                        --episode_length=${EPISODE_LENGTH} \
                                        --activation=${activation} \
                                        --repr_dim=${REP_DIM} \
                                        --contr_loss=${CONT_LOSS} \
                                        --actor_lr=${LR} \
                                        --critic_lr=${LR} \
                                        --run_name_suffix=${run_name_suffix} \
                                        --alpha_lr=${LR}"
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
done
done
done
done
done