# Curiosity-Driven Exploration via Temporal Contrastive Learning 


## Abstract
Effective exploration in reinforcement learning requires keeping track not just of where the agent has been, but also of how the agent thinks about and represents the world: an agent should explore states that enable it to learn powerful representations.Temporal representations can include the information required to solve any potential task while avoiding the computational cost of reconstruction. In this paper, we propose an exploration method that uses temporal representations to drive exploration, maximizing coverage *as seen through the lens of these temporal representations.* We demonstrate complex exploration behaviors in locomotion, manipulation, and embodied-AI tasks, revealing previously unknown capabilities and behaviors once achievable only via extrinsic rewards. Videos of the agent's behavior are on the [project website](https://sites.google.com/view/ctec-anonymous-submission).


## Installation
- Create the conda environment
```
conda env create -f environment.yaml
conda activate ctec
```
install jax and jaxlib with cuda12 support
```
python -m pip install "jax[cuda12]=0.4.25"
```
## Experiments and hyperparameters
To track the experiments in Weights & Biases (wandb), make sure you have set up your wandb account. For more details, check https://docs.wandb.ai/quickstart

Please check the ```ctec_craftax/scripts``` folder, which contains bash scripts to run the epaper's xperiments.

### Robotics experiments
#### C-TeC
```
bash scripts/ctec_craftax.sh
```
#### E3B
```
bash scripts/e3b_craftax.sh
```
#### ICM
```
bash scripts/icm_craftax.sh
```
#### RND
```
bash scripts/rnd_craftax.sh
```