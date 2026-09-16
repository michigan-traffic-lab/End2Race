# End2Race: Efficient End-to-End Imitation Learning for Real-Time F1Tenth Racing

## Introduction

End2Race learns recurrent steering and desired-speed control from LiDAR and measured speed. This repository provides simulation workflows for expert demonstration collection, behavioral cloning (BC), PPO fine-tuning, and racing evaluation.

[Paper](https://arxiv.org/abs/2509.16894)

https://github.com/user-attachments/assets/5369f5ea-13fa-44c3-a6aa-5b3c2b59b10c

## Table of Contents

- [Code Structure](#code-structure)
- [Setup](#setup)
- [Collecting demonstrations](#collecting-demonstrations)
- [Training](#training)
- [Evaluation](#evaluation)
- [Raceline generation](#raceline-generation)
- [License and citation](#license-and-citation)

## Code Structure

```text
End2Race/
├── config.yaml                          # Training, collection, and evaluation configuration
├── install.sh                           # Install dependencies and the bundled simulator
├── checkpoint/                          # Pretrained policy weights
│   ├── bc.pt                            # Policy trained from expert demonstrations
│   └── ppo.pt                           # Policy fine-tuned with reinforcement learning
├── dataset/                             # Expert demonstrations used for behavioral cloning
├── f1tenth_sim/                         # F1TENTH racing simulation and track resources
│   ├── config.yaml                      # Vehicle dynamics, simulation timing, and track generation
│   ├── utils.py                         # Load racelines and shared simulator settings
│   ├── f1tenth_gym/                     # Vehicle physics, LiDAR, collisions, and rendering
│   └── f1tenth_racetracks/              # Track maps and pre-generated lanes and racelines
│       └── generate_raceline.py         # Generate lanes and raceline speed profiles
├── expert/                              # Lattice-planner expert and demonstration collection
│   ├── lattice_planner.py               # Generate and select obstacle-aware racing trajectories
│   ├── controllers.py                   # Pure Pursuit tracking and raceline-following opponent
│   ├── collect.py                       # Collect scenario batches and save demonstration CSVs
│   └── utils.py                         # Scenario generation, geometry, rendering, and summaries
├── imitation/                           # Learn a driving policy from expert demonstrations
│   ├── model.py                         # GRU mapping LiDAR and speed to steering and desired speed
│   └── train.py                         # Train the policy on demonstration sequences
├── reinforcement/                       # Fine-tune the learned policy through racing interaction
│   ├── run_ppo.py                       # Run training epochs, evaluate policies, and save checkpoints
│   ├── train_ppo.py                     # Collect rollouts and optimize PPO policy and value losses
│   ├── eval_ppo.py                      # Measure safety and overtaking after each training epoch
│   ├── policy.py                        # Extend the driving policy with a value head and action sampling
│   └── env.py                           # Racing episodes, rewards, and parallel simulation workers
└── evaluation/                          # Evaluate saved policies across tracks
    ├── eval_single.py                   # Measure lap completion, lap times, and driving metrics
    └── eval_multi.py                    # Run opponent scenario batches and report racing outcomes
```

## Setup

Requires Linux and Python 3.11. Training requires CUDA.

### Clone the repository

```bash
git clone https://github.com/michigan-traffic-lab/End2Race.git
cd End2Race
```

### Create and activate the environment

```bash
conda create -y -n end2race python=3.11
conda activate end2race
```

### Install dependencies

```bash
bash install.sh
```

Configuration: [`config.yaml`](config.yaml) for workflows; [`f1tenth_sim/config.yaml`](f1tenth_sim/config.yaml) for the simulator.

## Collecting demonstrations

```bash
python expert/collect.py
```

Collection requires an empty destination. Collision-free episodes produce training CSVs; collisions produce metadata. `summary.json` records the batch outcomes.

## Training

### Behavioral cloning

```bash
python imitation/train.py
```

Saves the final policy weights after training.

### PPO fine-tuning

```bash
python reinforcement/run_ppo.py
```

The runner alternates training and deterministic evaluation until stopped, saving policies that meet the screening criteria. The output directory must be absent or empty at launch.

## Evaluation

### Single-agent lap completion

```bash
python evaluation/eval_single.py
```

Metrics are saved to `<output_dir>/<checkpoint name>/single.json`.

### Multi-agent racing

Evaluate against a non-reactive raceline follower:

```bash
python evaluation/eval_multi.py
```

Per-map summaries are saved to `<output_dir>/<checkpoint name>/<map>/results.json`. `success_percent` includes collision-free following and overtaking.

## Raceline generation

Generate racelines from map images and YAML metadata in `f1tenth_sim/f1tenth_racetracks/<map_name>/`:

```bash
python f1tenth_sim/f1tenth_racetracks/generate_raceline.py
```

The tool generates smoothed lanes and speed profiles, overwriting existing numbered racelines. Single-agent evaluation also requires `<map_name>_raceline.csv`; copy your chosen generated raceline to that filename.

## License and citation

The project license is [Apache 2.0](LICENSE).

```bibtex
@misc{qiao2025end2race,
      title={End2Race: Efficient End-to-End Imitation Learning for Real-Time F1Tenth Racing},
      author={Zhijie Qiao and Haowei Li and Zhong Cao and Henry X. Liu},
      year={2025},
      eprint={2509.16894},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.16894},
}
```
