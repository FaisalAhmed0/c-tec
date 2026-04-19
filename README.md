# Temporal Representations for Exploration: Learning Complex Exploratory Behavior without Extrinsic Rewards
Project website with videos: [temp-contrastive-explr.github.io](https://temp-contrastive-explr.github.io/).
---

## Abstract

Effective exploration in reinforcement learning requires tracking not only where the agent has been, but how it represents the world: an agent should explore states that support learning useful representations. Temporal representations can carry the information needed for many downstream tasks without paying the full cost of reconstruction. This work proposes an exploration method driven by such temporal representations, maximizing coverage *as seen through those representations*. We report complex exploratory behavior in locomotion, manipulation, and embodied-AI settings—including behaviors that previously often required extrinsic rewards.

---

## Repository structure

Robotics training and baselines live under **`ctec_robotics/`** (run jobs from that directory). The repo root also holds the Conda spec and an optional Craftax-related trainer.

```
.
├── README.md
├── environment.yaml          # Conda environment (pinned JAX, etc.)
├── .gitignore
├── ctec_craftax/
│   └── ctec_ppo_rnn.py      # Craftax / PPO–RNN experiment entry (expects companion modules on PYTHONPATH)
└── ctec_robotics/
    ├── args.py                # CLI flags and hyperparameter wiring
    ├── apt.py                 # APT baseline trainer
    ├── rnd.py                 # RND baseline trainer
    ├── icm.py                 # ICM baseline trainer
    ├── ctec.py                # C-TeC trainer (main robotics entry)
    ├── ctec_with_rept_negatives.py  # C-TeC variant (repetition / negatives)
    ├── intrinsic_rewards.py   # Intrinsic reward definitions (e.g. CRL-style)
    ├── models.py              # Networks / encoders
    ├── model_utils.py
    ├── losses.py
    ├── buffers.py
    ├── buffers_with_repetition_factor.py
    ├── evaluator.py
    ├── utils.py
    ├── envs/                  # Brax / MuJoCo MJX environments
    │   ├── __init__.py
    │   ├── ant.py, ant_maze.py, ant_ball.py, ant_push.py
    │   ├── humanoid.py, humanoid_maze.py
    │   ├── half_cheetah.py, reacher.py, pusher.py, pusher2.py, simple_maze.py
    │   ├── manipulation/    # Franka arm tasks (reach, push, grasp, bin-pick, …)
    │   └── assets/            # MuJoCo XML and meshes (incl. franka_emika_panda/)
    └── scripts/               # Bash launchers (often wrapped with sbatch on clusters)
        ├── ctec.sh
        ├── ctec_50M.sh
        ├── ctec_50M_mono_critic.sh
        ├── ctec_discounts_ablation.sh
        ├── ctec_repeated_traj.sh
        ├── apt.sh
        ├── rnd.sh
        ├── icm.sh
        └── mbrl.sh
```

---

## Installation

Create and activate the Conda environment:

```bash
conda env create -f environment.yaml
conda activate ctec
```

The environment file pins **`jax==0.4.25`**. For **GPU** training, install the CUDA 12–compatible JAX/JAXLIB build that matches your driver (see the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html)). For example:

```bash
python -m pip install "jax[cuda12]==0.4.25"
```

Adjust the CUDA variant if your stack differs.

---

## Experiments and logging

Experiment launchers live under **`ctec_robotics/scripts/`**. They reproduce the paper’s robotics setups (C-TeC and baselines).

**Weights & Biases:** To log runs with W&B, configure your account per [Weights & Biases quickstart](https://docs.wandb.ai/quickstart).

**Running jobs:** Example scripts are written for **`sbatch`** and embed **cluster-specific paths**—edit them for your setup. For local runs, activate the environment, **`cd ctec_robotics`**, and invoke the trainer directly, e.g.:

```bash
python ctec.py --env_name=humanoid_u_maze --track ...
```

Use **`--help`** on **`ctec.py`** (or baseline entry points) for the full CLI.

### Robotics experiments

From **`ctec_robotics/`** (adapt if your shell assumes another working directory):

```bash
bash scripts/ctec.sh    # C-TeC
bash scripts/apt.sh     # APT
bash scripts/rnd.sh     # RND
bash scripts/icm.sh     # ICM
```

---

## Citation
```
@inproceedings{
        mohamed2026temporal,
        title={Temporal Representations for Exploration: Learning Complex Exploratory Behavior without Extrinsic Rewards},
        author={Faisal Mohamed and Catherine Ji and Benjamin Eysenbach and Glen Berseth},
        booktitle={The Fourteenth International Conference on Learning Representations},
        year={2026},
        url={https://openreview.net/forum?id=KjYpHySlb0}
        }
```

