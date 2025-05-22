#!/bin/bash
# normalize_reprs="--use_normalize_repr"
wandb_project_name="craftax_rnd"
env_name="Craftax-Classic-Symbolic-v1"
task_reward_coef=0.0

# Define parameter lists
runs=(1 2 3 4 5)
nums_steps=(64)
rnd_reward_coefs=(1)
intr_rewards=("--use_rnd")



# Loop over all parameter combinations
for intr_reward in "${intr_rewards[@]}"; do
for rnd_reward_coef in "${rnd_reward_coefs[@]}"; do
for num_steps in "${nums_steps[@]}"; do
for run in "${runs[@]}"; do
                # echo "Submitting job with repr_dim=$repr_dim, activation_crl=$activation_crl, similarity_measure=$similarity_measure, contrastive_loss=$contrastive_loss contrastive_hidden=$contrastive_hidden contrastive_number_hiddens=$contrastive_num_hidden gamma_cl=$gamma_crl crl_lr=$cl_lr"
                python ppo_rnn_intr_baselines.py \
                    --wandb_project=$wandb_project_name \
                    --env_name=$env_name \
                    --num_steps=$num_steps \
                    --task_reward_coef=$task_reward_coef \
                    --rnd_reward_coeff=$rnd_reward_coef \
                    ${intr_reward} \
                    --save_policy
done
done
done
done