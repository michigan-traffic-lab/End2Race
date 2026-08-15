# End2Race: Efficient End-to-End Imitation Learning for Real-Time F1Tenth Racing


## Introduction

End2Race is an end-to-end imitation learning framework for autonomous racing on the [F1Tenth platform](https://roboracer.ai/build). By learning from expert demonstrations generated with the established PythonRobotics Frenet Optimal Trajectory (FOT) algorithm, the system captures temporal dependencies in racing dynamics to enable real-time control in competitive scenarios. End2Race addresses key challenges in autonomous racing—strategic planning, reactive control, and safe overtaking—through a unified neural network approach.

https://github.com/user-attachments/assets/5369f5ea-13fa-44c3-a6aa-5b3c2b59b10c

## Table of Contents
- [Code Structure](#code-structure)
- [Configuration](#configuration)
- [Environment Setup](#environment-setup)
- [Evaluation](#evaluation)
- [Data Collection](#data-collection)
- [Training](#training)
- [Model Architecture](#model-architecture)
- [Raceline Generation (Optional)](#raceline-generation-optional)

## Code Structure
```
end2race/
├── f1tenth_gym/               # F1Tenth simulator environment
├── f1tenth_racetracks/        # Track data with pre-generated lanes and racelines
│   ├── config.yaml            # Track and vehicle configuration
│   └── generate_raceline.py   # Raceline generation tool
├── latticeplanner/
│   ├── lattice_config.yaml    # Expert-planner configuration
│   └── lattice_planner.py     # PythonRobotics FOT expert and trajectory tracker
├── expert.py                  # One multi-agent expert collection scenario
├── model.py                   # GRU network architecture
├── train.py                   # Training script
├── collect.sh                 # Parallel collection orchestrator and dataset summary
├── eval_single.py             # One single-agent lap evaluation
├── eval_multi.py              # One multi-agent racing evaluation
├── eval_multi.sh              # Parallel multi-agent orchestrator
└── utils.py                   # Shared utilities and configuration loading
```

## Configuration

Two YAML files hold planner and track-tool settings. Workflow settings stay with the scripts that own their execution:

- `latticeplanner/lattice_config.yaml`: the `expert` section of planner and tracker settings
- `f1tenth_racetracks/config.yaml`: a `track` section for raceline generation and a
  `vehicle` section shared by the expert, the evaluators, and the raceline tool

The model dimensions and preprocessing constants live with the model definition
in `model.py`. Physics runs at 120 Hz, LiDAR-driven FOT replanning runs at
10 Hz, and data collection runs at 40 Hz (25 ms).

## Environment Setup

### Base Requirements
* **Hardware**: 4-core CPU, 8GB RAM (GPU recommended for training and inference)
* **System**: Windows or Linux
* **Python**: 3.11 in the `end2race` Conda environment

### Clone Repository
```bash
git clone https://github.com/li1164733168/end2race.git
cd end2race
```

### Setup Virtual Environment
```bash
conda env create --file environment.yml
conda activate end2race
```

If the environment already exists, synchronize it with:

```bash
conda env update --name end2race --file environment.yml
conda activate end2race
```

### Install Dependencies
```bash
bash install.sh
```

## Evaluation

The evaluation is conducted using the [F1Tenth Gym simulator](https://github.com/f1tenth/f1tenth_gym), a high-fidelity racing environment for autonomous vehicle research. 

### Single-Agent Evaluation
Evaluates the model's lap completion ability across different track configurations, testing its robustness to varying track layouts and racing line complexities without opponent interaction.

```bash
python eval_single.py Austin checkpoint/checkpoint.pt --render
```

The track and checkpoint path are required arguments. Add `--render` to save a video, or omit it for evaluation without rendering. All other evaluation settings are internal constants.

### Multi-Agent Evaluation

Evaluates the model in competitive racing scenarios against an expert opponent. The framework provides 4 pre-configured tracks with raceline files for testing: [Austin](f1tenth_racetracks/Austin/Austin_map.png), [Hockenheim](f1tenth_racetracks/Hockenheim/Hockenheim_map.png), [MoscowRaceway](f1tenth_racetracks/MoscowRaceway/MoscowRaceway_map.png), and [Nuerburgring](f1tenth_racetracks/Nuerburgring/Nuerburgring_map.png), each with 3 raceline options (`raceline0`, `raceline1`, `raceline2`). For tracks without pre-generated racelines, use the [Raceline Generation](#raceline-generation-optional) section to create them first.


```bash
python eval_multi.py Austin checkpoint/checkpoint.pt 0 raceline1 0.8 8.0 0.0 42 false
```

The required inputs are track, checkpoint, ego waypoint index, opponent raceline, opponent speed scale, evaluation duration, LiDAR noise ratio, seed, and rendering flag. `eval_multi.sh` owns these evaluation settings for batch runs.

### Multi-Agent Parallel Evaluation (Optional)

The batch evaluation runs hundreds of scenarios in parallel to comprehensively assess the model's performance across different starting positions, opponent strategies, and difficulty levels:

```bash
bash eval_multi.sh
```


## Data Collection

Collect one explicit competitive-racing scenario with:

```bash
python expert.py --map_name Austin --dataset_dir dataset --ego_idx 0 \
  --interval_idx 15 --opponent_raceline raceline1 \
  --opponent_speed_scale 0.8 --sim_duration 8.0 --render
```

These are the defaults except for `--render`, so the arguments may be omitted.

Collection runs 120 Hz physics, replans the FOT trajectory every 12th physics step (10 Hz), tracks the held trajectory at every physics step, and saves one live observation and aligned tracker action every third physics step (40 Hz). This produces exact 25 ms intervals without interpolation. To run the complete parallel multi-agent collection matrix and wait for every scenario:

```bash
bash collect.sh
```

`collect.sh` owns the output dataset directory and all batch collection settings. The current Austin batch runs 80 ego starting points against three opponent racelines at waypoint interval `15` and speed scales `0.4`, `0.6`, and `0.8`, with 8 seconds per scenario: 720 scenarios total.

After all collection workers finish, `collect.sh` prints the following, overtaking, collision, and worker-failure counts and writes `dataset/summary.json`. The summary snapshots the collection, vehicle, and expert configuration; reports collision-free, collision, overtaking, and following outcomes with their rates; counts training rows and saved videos; and provides a breakdown by opponent raceline and speed scale. It is generated from the saved CSV and collision metadata, so a partial collection also retains a summary before `collect.sh` reports a worker failure.

Each training row stores the measured ego speed, expert steering and desired-speed targets, and 180 LiDAR values from the same decision instant. Training keeps every 40 Hz row: the first row uses its measured speed as the initial speed input, and later rows use the measured speed from the preceding 25 ms step. The ego FOT expert replans every 12th physics step from the corresponding LiDAR scan, projects that scan into occupied points, and selects among dynamically feasible trajectories using mean velocity cost and worst-point clearance cost; its tracker supplies the 40 Hz labels while following that held trajectory. The non-reactive opponent tracks its assigned raceline. Every velocity choice produces a physically distinct trajectory over the candidate horizon, and generated speeds are bounded by 7.5 m/s.

Every collection and evaluation scenario initializes the ego at 50% of its 7.5 m/s maximum speed (3.75 m/s). A multi-agent opponent starts at its local raceline speed multiplied by the scenario's opponent speed scale. Subsequent acceleration and braking are determined by each controller.

## Training
Trains the End2Race model using imitation learning on collected demonstrations.

```bash
python train.py
```

The main training parameters are command-line options:

```bash
python train.py \
  --model_path checkpoint/checkpoint.pt \
  --batch_size 1024 \
  --learning_rate 0.001 \
  --num_epochs 5000 \
  --speed_loss_weight 0.05 \
  --gradient_clip_norm 1.0
```

These are the defaults, so the arguments may be omitted. Training reads successful demonstrations from `dataset/success/`. It trains one model for up to `--num_epochs`, overwriting `--model_path` with the current model every 500 epochs and at the final epoch. Each saved model must complete one collision-free lap on Austin, Hockenheim, MoscowRaceway, and Nuerburgring in that order, then evaluate Austin's 720 multi-agent scenarios. That evaluation always completes at least 100 scenarios and stops early only if its aggregate collision rate then exceeds 20%. The first failure stops that model's remaining evaluation. A model that never passes is removed after the final epoch; training tries at most 10 models. The model and `training_state.pt` remain together without nested directories; by default both are directly under `checkpoint/`. Rerunning `train.py` with an existing `training_state.pt` resumes that model at its recorded epoch.

## Model Architecture

The policy downsamples each full-circle simulator scan to 180 LiDAR values and applies one shared, fixed sigmoid normalization coefficient. The normalized LiDAR vector is concatenated with a 30-dimensional speed embedding, producing a 210-dimensional recurrent input. A single-layer GRU with 420 hidden units feeds an action head with dimensions `420 -> 128 -> 2`, which predicts steering and desired speed. During training, the speed embedding is replaced by a learned dummy embedding at 20% of timesteps. All of these dimensions live as class attributes on `End2Race` in `model.py`.

## Raceline Generation (Optional)

Generate optimized racing lines for new tracks. First, upload the track map files to `f1tenth_racetracks/{map_name}/` including `{map_name}_map.png` (binary image: white=drivable, black=walls) and `{map_name}_map.yaml` (map metadata). Then run:

```bash
python -m f1tenth_racetracks.generate_raceline
```

Edit `f1tenth_racetracks/config.yaml` for track-generation and shared vehicle settings. The expert and evaluation workflows reuse its `vehicle` section.

The FOT implementation is adapted under the MIT license from [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics), pinned to commit `b38c510e083d69a5755d98d0680bd50f3d9a91fa`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use End2Race in your research, please consider citing:

```bibtex
@article{end2race,
      title={End2Race: Efficient End-to-End Imitation Learning for Real-Time F1Tenth Racing}, 
      author={Zhijie Qiao and Haowei Li and Zhong Cao and Henry X. Liu},
      year={2025},
      eprint={2505.00284},
      url={https://arxiv.org/abs/2509.16894}, 
}
```
