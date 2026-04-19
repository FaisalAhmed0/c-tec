# Temporal Representations for Exploration: Learning Complex Exploratory Behavior without Extrinsic Rewards
Project website with videos: [temp-contrastive-explr.github.io](https://temp-contrastive-explr.github.io/).
---

## Abstract

Effective exploration in reinforcement learning requires tracking not only where the agent has been, but how it represents the world: an agent should explore states that support learning useful representations. Temporal representations can carry the information needed for many downstream tasks without paying the full cost of reconstruction. This work proposes an exploration method driven by such temporal representations, maximizing coverage *as seen through those representations*. We report complex exploratory behavior in locomotion, manipulation, and embodied-AI settings—including behaviors that previously often required extrinsic rewards.

## Repository layout

High-level overview of this codebase (see subfolders for full file lists):

```
c-tec/
├── environment.yml              # Conda environment specification
├── wrappers.py                  # Shared JAX / Brax / Gymnax-style wrappers (root helpers)
├── train_ctec_atari             # Bash launcher for Atari-style training jobs
├── ctec_robotics/               # Robotics-related runs and env stubs
│   ├── envs/manipulation/
│   └── runs/
└── ctec_craftax/                # Main JAX training code, envs, and Craftax integration
    ├── args.py                  # CLI / training configuration
    ├── losses.py                # Auxiliary and contrastive losses
    ├── utils.py
    ├── jax_wrappers.py
    ├── wrappers.py, wrappers_v2.py
    ├── ctec_ppo_*.py            # C-TeC PPO entry points (continuous action, RNN, …)
    ├── ppo_rnn.py, ppo_rnn_intr_baselines.py
    ├── etd_ppo_*.py             # Elliptical / ETD-style PPO variants
    ├── view_ppo_agent.py
    ├── models/                  # Actor–critic, contrastive encoder, ICM, RND, ETD heads
    │   ├── actor_critic.py
    │   ├── contrastive_model.py
    │   ├── etd_models.py
    │   ├── icm.py
    │   └── rnd.py
    ├── envs/                    # Continuous control & manipulation (MuJoCo assets under assets/)
    ├── craftax/                 # Craftax pixel / symbolic environments (vendored)
    ├── logz/                    # Batched logging helpers
    ├── scripts/                 # Short bash launchers for baseline comparisons
    └── train_scripts/           # Larger experiment sweep scripts
```

## Installation

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate ctec
```

Install JAX and JAXlib with CUDA 12 support (adjust the version pin to match your stack):

```bash
python -m pip install "jax[cuda12]==0.4.25"
```

## Experiments and hyperparameters

To track experiments in [Weights & Biases](https://wandb.ai), configure your W&B account ([quickstart](https://docs.wandb.ai/quickstart)).

The directory `ctec_craftax/scripts` contains bash scripts used to reproduce paper experiments. From the repository root:

### Robotics experiments

**C-TeC**

```bash
bash ctec_craftax/scripts/ctec_craftax.sh
```

**E3B**

```bash
bash ctec_craftax/scripts/e3b_craftax.sh
```

**ICM**

```bash
bash ctec_craftax/scripts/icm_craftax.sh
```

**RND**

```bash
bash ctec_craftax/scripts/rnd_craftax.sh
```

## Citation

If you use this code or method, please cite the paper. Replace `author`, `year`, and `note` with the published venue and identifier when available:

```bibtex
@misc{temporal_exploration_ctec,
  title={{Temporal Representations for Exploration: Learning Complex Exploratory Behavior without Extrinsic Rewards}},
  author={Anonymous},
  year={2026},
  url={https://sites.google.com/view/ctec-anonymous-submission},
  note={Example anonymous project-page entry; substitute the official author list and bibliographic fields after publication.}
}
```
