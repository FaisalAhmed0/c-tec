#!/bin/bash

# set global config
use_empowerment=0
wandb_project_name="ctec_craftax"
env_name="Craftax-Classic-Symbolic-v1"
task_reward_coef=0.0
use_single_sample=0
use_action_in_cl=1
num_envs=1024
rwd_rms=0
run_name_suffix="ctec_craftax"
use_rnn_in_cl=0
normalize_reprs="--use_normalize_repr"





# Define parameter lists
repr_dim_list=(64)
activation_crl_list=("nn.relu")
similarity_measure_list=("l2")
contrastive_loss_list=("infonce")
contrastive_hiddens=(1024)
contrastive_nums_hiddens=(3)
gammas_crl=(0.3)
gammas_crl_reward=(0.3)
cl_lrs=(0.0003)
nums_steps=(64)
use_norm_constants=(0)
runs=(1 2 3 4 5)
update_proportions=(1)
geometric_truncs=(2048)
crl_rewards_coef=(1)
relative_scales=(0.0)
logsumexp_penalty_coeffs=(0.0)


# Loop over all parameter combinations
for logsumexp_penalty_coeff in "${logsumexp_penalty_coeffs[@]}"; do
for relative_scale in "${relative_scales[@]}"; do
for geometric_trunc in "${geometric_truncs[@]}"; do
for update_proportion in "${update_proportions[@]}"; do
for use_norm_constant in "${use_norm_constants[@]}"; do
for gamma_crl_reward in "${gammas_crl_reward[@]}"; do
for crl_reward_coef in "${crl_rewards_coef[@]}"; do
for num_steps in "${nums_steps[@]}"; do
for run in "${runs[@]}"; do
for cl_lr in "${cl_lrs[@]}"; do
for gamma_crl in "${gammas_crl[@]}"; do
for contrastive_num_hidden in "${contrastive_nums_hiddens[@]}"; do
for contrastive_hidden in "${contrastive_hiddens[@]}"; do
for repr_dim in "${repr_dim_list[@]}"; do
    for activation_crl in "${activation_crl_list[@]}"; do
        for similarity_measure in "${similarity_measure_list[@]}"; do
            for contrastive_loss in "${contrastive_loss_list[@]}"; do
                python  ctec_ppo_rnn.py \
                    --env_name=$env_name \
                    --repr_dim=$repr_dim \
                    --activation_crl=$activation_crl \
                    --similarity_measure=$similarity_measure \
                    --contrastive_loss=$contrastive_loss \
                    --contrastive_hidden_dim=$contrastive_hidden \
                    --contrastive_number_hiddens=$contrastive_num_hidden \
                    --gamma_cl=$gamma_crl \
                    --crl_lr=$cl_lr \
                    --wandb_project=$wandb_project_name \
                    --use_empowerment=$use_empowerment \
                    --num_steps=$num_steps \
                    --task_reward_coef=$task_reward_coef \
                    --crl_reward_coef=$crl_reward_coef \
                    --use_single_sample=$use_single_sample \
                    --use_norm_constant=$use_norm_constant \
                    --gamma_cl_reward=$gamma_crl_reward \
                    --use_action_in_cl=$use_action_in_cl \
                    --num_envs=$num_envs \
                    --update_proportion=$update_proportion \
                    --geom_trunc=$geometric_trunc \
                    --logsumexp_penalty_coeff=$logsumexp_penalty_coeff \
                    --run_name_suffix=$run_name_suffix \
                    --use_rnn=$use_rnn_in_cl \
                    ${normalize_reprs} \
                    --save_policy
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