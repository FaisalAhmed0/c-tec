
#!/bin/bash
# Define common parameters (fixed values)
TRACK="--track"
WANDB_PROJECT_NAME="ctec_50M_monolithic_critic"
RENDER_AGENT="--render_agent"
contrastive_hidden_dim=1024
activation="nn.relu"
NORMALIZE_REP="--normalize_repr"
LR=0.0003
SGD_STEPS_FACTR=1
ENTROPY_REG="--entropy_reg"
run_name_suffix="ctec"
checkpoint="--no-checkpoint"
use_mono_ciritc="--use_monolithic_critic"



#### For humanoid_u_maze, use the following values
# ENV_NAMES=("humanoid_u_maze")
# BATCH_SIZES=(256)                           
# NUM_ENVS_VALUES=(256)                      
ENV_NAMES=("ant_hardest_maze")
BATCH_SIZES=(1024)                           
NUM_ENVS_VALUES=(1024)                      
NUM_EPOCHS_VALUES=(1000)                    
NUM_TIMESTEPS_VALUES=(50000000) 
NUM_EVALS_VALUES=(500)  # 250 -> 4 updates per epoch, 2000 -
UNROLL_LENGTHS=(31)        
runs=(1 2 3) # number of seeds, each seed is chosen randomly (results might slightly differ from the paper resutls)
REPS_DIMS=(64)
USE_COMPLETE_FUTURE_STATE_VALUES=("--no-use_complete_future_state")
CONTR_LOSSES=("infonce")
EPISODE_LENGTHS=(1000) 
NORMALIZE_REPS=("--normalize_repr" "--no-normalize_repr")
energy_fns=("l1") # contrastive critic function
logsumexp_penalty_coeffs=(0.0)
contrastive_number_hiddenss=(2)
discountings_crl=(0.99)
LAYER_NORMS=("--no-layer_norm_crl")
FUTURE_RWD_SAMPLERS=("geometric")

# Run counter
run_count=0

# Nested loops to construct and run commands for all combinations
for future_state_rwd_sampling in "${FUTURE_RWD_SAMPLERS[@]}"; do
for LAYER_NORM in "${LAYER_NORMS[@]}"; do
for discounting_crl in "${discountings_crl[@]}"; do
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
                                for UNROLL_LENGTH in "${UNROLL_LENGTHS[@]}"; do 
                                for logsumexp_penalty_coeff in "${logsumexp_penalty_coeffs[@]}"; do 
                                    # Construct the sbatch command
                                    CMD="sbatch scripts/train_ctec ctec.py \
                                        --env_name=${ENV_NAME} \
                                        ${TRACK} \
                                        ${NORMALIZE_REP} \
                                        ${USE_COMPLETE_FUTURE_STATE} \
                                        ${LAYER_NORM} \
                                        ${ENTROPY_REG} \
                                        ${use_mono_ciritc} \
                                        --multiplier_num_sgd_steps=${SGD_STEPS_FACTR} \
                                        --wandb_project_name=\"${WANDB_PROJECT_NAME}\" \
                                        --batch_size=${BATCH_SIZE} \
                                        --num_envs=${NUM_ENVS} \
                                        --num_epochs=${NUM_EPOCHS} \
                                        --unroll_length=${UNROLL_LENGTH} \
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
                                        --discounting_cl=${discounting_crl} \
                                        --actor_lr=${LR} \
                                        --critic_lr=${LR} \
                                        --future_state_rwd_sampling=${future_state_rwd_sampling} \
                                        --run_name_suffix=${run_name_suffix} \
                                        --alpha_lr=${LR}"
                                    # Print and execute the command
                                    # Increment counter
                                    run_count=$((run_count + 1))
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
done
done
done
done

echo "====================================="
echo "All runs finished. Total: $run_count"
echo "====================================="