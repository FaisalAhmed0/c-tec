# Curiosity-Driven Exploration via Temporal Contrastive Learning 


## Abstract
Exploration remains a key challenge in reinforcement learning (RL), especially in long-horizon tasks and environments with high-dimensional observations. A common strategy for effective exploration is to promote state coverage or novelty, which often involves estimating the agent's state visitation distribution. In this paper, we propose **Curiosity-Driven Exploration via Temporal Contrastive Learning (C-TeC)**, an exploration method based on temporal contrastive learning that rewards agents for reaching states with unexpected futures. This incentivizes uncovering meaningful, less-visited states. **C-TeC** is simple and does not require explicit density or uncertainty estimation, while learning representations aligned with the RL objective. It consistently outperforms standard baselines in complex mazes using different embodiments (Ant and Humanoid) and robotic manipulation tasks, while also yielding more diverse behaviors in Craftax without requiring task-specific information. Videos of the agent's behavior are on the [project website](https://sites.google.com/view/ctec-anonymous-submission).


## Installation
- Create the conda environment
```
conda env create -f environment.yaml
conda activate jax_expl
```
## Experiments and hyperparameters
...
## Logging
...
## Environment
...