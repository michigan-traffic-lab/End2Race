# End2Race: An End-to-End Learning Framework for Multi-Vehicle Autonomous Racing

[![arXiv](https://img.shields.io/badge/arXiv-2509.16894-red.svg)](https://arxiv.org/abs/2509.16894)
[![F1TENTH](https://img.shields.io/badge/platform-F1TENTH-green.svg)](https://roboracer.ai/)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging_Face-yellow.svg)](https://huggingface.co/zhijieq/End2Race)

## Introduction

**End2Race** is an end-to-end learning framework for multi-vehicle autonomous racing on [F1TENTH](https://roboracer.ai/). It maps 2D LiDAR scans and vehicle speed directly to steering and speed commands in real time, providing an integrated pipeline for scenario generation, policy training, and closed-loop benchmarking. For a quick start and interactive demo of the simulation environment and pretrained policy, you can use [Hugging Face](https://huggingface.co/zhijieq/End2Race).

### Highlights

🔄 **Scalable Scenario Generation**: Automatically generates diverse overtaking scenarios for policy training and evaluation.

⚡ **Sub-Millisecond Policy**: Runs in <1 ms on NVIDIA Jetson Xavier NX, enabling real-time control at high racing speeds.

🌐 **Zero-Shot Generalization**: Transfers seamlessly to unseen tracks and opponent behaviors with robust safety and adaptive overtaking.

<!-- https://github.com/user-attachments/assets/5369f5ea-13fa-44c3-a6aa-5b3c2b59b10c -->

## Table of Contents

- [Code Structure](#code-structure)
- [Setup](#setup)
- [Configuration](#configuration)
- [Expert Demonstrations](#expert-demonstrations)
- [Training](#training)
- [Evaluation](#evaluation)
- [Raceline Generation (Optional)](#raceline-generation-optional)
- [License and Citation](#license-and-citation)
- [Contributing and Feedback](#contributing-and-feedback)

## Code Structure

```text
End2Race/
├── checkpoint/
│   ├── bc.pt                        # BC policy checkpoint
│   └── ppo.pt                       # PPO policy checkpoint
├── dataset/
│   ├── collision/                   # Collision metadata
│   ├── success/                     # Demonstration trajectories
│   └── summary.json                 # Collection summary
├── evaluation/
│   ├── eval_multi.py                # Head-to-head racing evaluation
│   └── eval_single.py               # Single-vehicle timed trials
├── expert/
│   ├── collect.py                   # Demonstration collection
│   ├── controllers.py               # Opponent and tracking controllers
│   ├── lattice_planner.py           # Frenet lattice planner
│   └── utils.py                     # Planner utilities and metrics
├── f1tenth_sim/
│   ├── config.yaml                  # Simulator configuration
│   ├── f1tenth_gym/                 # F1TENTH Gym environment
│   ├── f1tenth_racetracks/          # Track maps and racelines
│   └── utils.py                     # Simulator utilities
├── ftg/
│   └── controller.py                # Follow-the-Gap controller
├── imitation/
│   ├── model.py                     # End2Race policy network
│   └── train.py                     # Behavioral cloning (BC) training
├── reinforcement/
│   ├── env.py                       # Simulation environment and workers
│   ├── eval_ppo.py                  # PPO evaluation routine
│   ├── policy.py                    # Recurrent actor-critic network
│   ├── run_ppo.py                   # PPO training entry point
│   └── train_ppo.py                 # Rollout collection and PPO update
├── config.yaml                      # Main experiment configuration
├── install.sh                       # Installation script
├── LICENSE                          # Apache 2.0 license
└── README.md                        # Documentation
```

## Setup

### Tested Environment

* **Hardware**: 4-core CPU, 8 GB RAM (GPU recommended for training and inference)
* **System**: Windows or Linux
* **Python**: 3.11 (Conda or native installation)

### Clone Repository

```bash
git clone https://github.com/michigan-traffic-lab/End2Race.git
cd End2Race
```

### Set Up Virtual Environment

```bash
conda create --name end2race python=3.11 -y
conda activate end2race
```

### Install Dependencies

```bash
bash install.sh
```

## Configuration

- [`config.yaml`](config.yaml): Controls the overall pipeline, including dataset paths, compute resources (device and workers), and hyperparameters for demonstration collection, BC pre-training, PPO fine-tuning, and evaluation. Modify this file to tailor experiments to your setup.
- [`f1tenth_sim/config.yaml`](f1tenth_sim/config.yaml): Defines simulator-level parameters, including update frequencies, vehicle dynamic limits, and raceline generation settings. Default values follow standard physical F1TENTH specifications and typically do not require modification.

## Expert Demonstrations

Pre-collected expert demonstrations are provided in `dataset/success/` for immediate training. Running data collection is only necessary if you modify expert planner parameters or wish to generate custom scenarios. During collection, successful runs are saved as CSV trajectories in `dataset/success/`, while collision events and metadata are logged to `dataset/collision/`.

```bash
python expert/collect.py
```

## Training

### Behavioral Cloning

Trains the policy on expert demonstrations via behavioral cloning, saving checkpoints to `checkpoint/bc.pt` for direct evaluation or subsequent PPO fine-tuning.

```bash
python imitation/train.py
```

### PPO Fine-Tuning

Fine-tunes the BC policy through closed-loop multi-vehicle racing interactions.

**Single GPU:**
```bash
python reinforcement/run_ppo.py
```

**Multi-GPU:**
```bash
torchrun --nproc_per_node=<num_gpus> reinforcement/run_ppo.py
```

## Evaluation

Policies are evaluated across four benchmark tracks: **Austin**, **Hockenheim**, **Moscow Raceway**, and **Nürburgring**. Target maps and evaluation parameters can be customized in [`config.yaml`](config.yaml).

### Single-Vehicle Timed Trials

Benchmarks lap completion rates, average speeds, and driving stability. Select the policy type (`expert`, `bc`, or `ppo`) via `eval_single.method`.

```bash
python evaluation/eval_single.py
```

### Head-to-Head Racing

Benchmarks safety rates and overtaking success against an opponent raceline follower. Select the policy type (`expert`, `bc`, or `ppo`) via `eval_multi.method`.

```bash
python evaluation/eval_multi.py
```

## Raceline Generation (Optional)

Pre-computed lanes and racelines are provided for all bundled tracks in `f1tenth_sim/f1tenth_racetracks/`. Running raceline generation is optional and only necessary when introducing new track maps or modifying speed profiles. Generated CSV files are saved directly in each track's directory.

```bash
python f1tenth_sim/f1tenth_racetracks/generate_raceline.py
```

## License and Citation

This project is licensed under the [Apache 2.0 License](LICENSE).

```bibtex
@misc{qiao2025end2race,
      title={End2Race: An End-to-End Learning Framework for Multi-Vehicle Autonomous Racing},
      author={Zhijie Qiao and Haowei Li and Zhong Cao and Henry X. Liu},
      year={2025},
      eprint={2509.16894},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.16894},
}
```

## Contributing and Feedback

Contributions, discussions, and feedback are welcome! If you encounter any issues, have questions, or would like to suggest improvements or new features, please feel free to open an issue or submit a pull request.
