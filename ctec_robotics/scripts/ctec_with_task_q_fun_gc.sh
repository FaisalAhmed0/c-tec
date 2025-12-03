
#!/bin/bash
# Define common parameters (fixed values)
TRACK="--track"
WANDB_PROJECT_NAME="ctec_with_task_q_fucn_gc_5"
RENDER_AGENT="--render_agent"
contrastive_hidden_dim=1024
activation="nn.relu"
NORMALIZE_REP="--normalize_repr"
LR=0.0003
SGD_STEPS_FACTR=1
ENTROPY_REG="--entropy_reg"
run_name_suffix="ctec"
checkpoint="--no-checkpoint"
logsumexp_penalty_coeff=0.0
anneal_ctec_rwd="--anneal_ctec_rwd"
zero_target_entropy="--no-use_target_entropy_zero"
use_exp_task_rwd="--no-use_exp_task_rwd"
usu_future_rwd="--no-usu_future_rwd"
future_rwd_temp=0.1
rwd_rms="--no-rwd_rms"
fix_alpha="--no-fix_alpha"
alpha="--alpha=0.01"


#### For humanoid_u_maze, use the following values
# ENV_NAMES=("humanoid_u_maze_single_goal")
# BATCH_SIZES=(256)                           
# NUM_ENVS_VALUES=(256)                      
# ENV_NAMES=("ant_hardest_maze_single_goal" "arm_binpick_hard")
ENV_NAMES=("ant_hardest_maze_hard_goals")
BATCH_SIZES=(1024)                           
NUM_ENVS_VALUES=(1024)                      
NUM_EPOCHS_VALUES=(1000)                    
NUM_TIMESTEPS_VALUES=(500000000) 
NUM_EVALS_VALUES=(2000)                    
runs=(1 2 3) # number of seeds, each seed is chosen randomly (results might slightly differ from the paper resutls)
REPS_DIMS=(64)
USE_COMPLETE_FUTURE_STATE_VALUES=("--no-use_complete_future_state")
CONTR_LOSSES=("infonce")
EPISODE_LENGTHS=(1000) 
energy_fns=("l1") # contrastive critic function
contrastive_number_hiddenss=(2)
discountings_crl=(0.99)
LAYER_NORMS=("--no-layer_norm_crl")
FUTURE_RWD_SAMPLERS=("geometric")
TASK_RWD_SCALES=(0.1 2 5 10)
CTEC_RWD_SCALES=(2 1 0.1 0.01)
ANNEAL_RATIO_VALUES=(0.15 0.25 0.5)
# TASK_RWD_SCALES=(1)
# CTEC_RWD_SCALES=(0)

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
                                for task_rwd_scale in "${TASK_RWD_SCALES[@]}"; do 
                                for CTEC_RWD_SCALE in "${CTEC_RWD_SCALES[@]}"; do 
                                    for ANNEAL_RATIO in "${ANNEAL_RATIO_VALUES[@]}"; do
                                        # Construct the sbatch command
                                        CMD="sbatch scripts/train_ctec ctec_gc.py \
                                        --anneal_ratio=${ANNEAL_RATIO} \
                                        --env_name=${ENV_NAME} \
                                        ${TRACK} \
                                        ${NORMALIZE_REP} \
                                        ${USE_COMPLETE_FUTURE_STATE} \
                                        ${LAYER_NORM} \
                                        ${fix_alpha} \
                                        ${alpha} \
                                        ${ENTROPY_REG} \
                                        ${anneal_ctec_rwd} \
                                        ${use_exp_task_rwd} \
                                        ${usu_future_rwd} \
                                        ${rwd_rms} \
                                        --exp_rwd_temp=${future_rwd_temp} \
                                        --multiplier_num_sgd_steps=${SGD_STEPS_FACTR} \
                                        --wandb_project_name=\"${WANDB_PROJECT_NAME}\" \
                                        --batch_size=${BATCH_SIZE} \
                                        --task_rwd_scale=${task_rwd_scale} \
                                        --ctec_rwd_scale=${CTEC_RWD_SCALE} \
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
                                        --discounting_cl=${discounting_crl} \
                                        --actor_lr=${LR} \
                                        --critic_lr=${LR} \
                                        --future_state_rwd_sampling=${future_state_rwd_sampling} \
                                        --run_name_suffix=${run_name_suffix} \
                                        --alpha_lr=${LR}"
                                    # Print and execute the command
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
done


echo "====================================="
echo "All runs submitted. Total: $run_count"
echo "====================================="