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
- [PPO Fine-Tuning](#ppo-fine-tuning)
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
│   ├── lattice_config.yaml    # Expert-planner and simulation timing configuration
│   └── lattice_planner.py     # PythonRobotics FOT expert and trajectory tracker
├── install.sh                 # Dependency install into the activated environment
├── expert.py                  # One multi-agent expert collection scenario
├── model.py                   # GRU network architecture
├── train.py                   # Training script
├── ppo/                       # PPO environment and actor-critic wrappers
├── run_ppo.py                 # PPO training/evaluation orchestrator
├── train_ppo.py               # PPO rollout and optimizer updates
├── eval_ppo.py                # PPO checkpoint evaluation
├── collect.sh                 # Parallel collection orchestrator and dataset summary
├── eval_single.py             # One single-agent lap evaluation
├── eval_multi.py              # One multi-agent racing evaluation
├── eval_multi.sh              # Parallel multi-agent orchestrator
└── utils.py                   # Shared utilities and configuration loading
```

## Configuration

Two YAML files hold planner and track-tool settings. Workflow settings stay with the scripts that own their execution:

- `latticeplanner/lattice_config.yaml`: an `expert` section of planner and tracker
  settings, and a `simulation` section holding the timing contract and the ego's
  initial speed fraction
- `f1tenth_racetracks/config.yaml`: a `track` section for raceline generation and a
  `vehicle` section shared by the expert, the evaluators, and the raceline tool

The model dimensions and preprocessing constants live with the model definition
in `model.py`. The `simulation` section sets 120 Hz physics, 10 Hz LiDAR-driven FOT
replanning, and a 40 Hz (25 ms) control rate that carries the expert labels during
collection and the policy's decisions during evaluation. `expert.tracker_steps` must
equal the resulting 12 physics steps per plan; collection fails fast when it does not.

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

### Install
```bash
conda create -y -n end2race python=3.11
conda activate end2race
bash install.sh
```

`install.sh` installs every dependency into the currently activated environment, so
activate `end2race` first and run the script from the repository root.

## Evaluation

The evaluation is conducted using the [F1Tenth Gym simulator](https://github.com/f1tenth/f1tenth_gym), a high-fidelity racing environment for autonomous vehicle research. 

### Single-Agent Evaluation
Evaluates the model's lap completion ability across different track configurations, testing its robustness to varying track layouts and racing line complexities without opponent interaction.

```bash
python eval_single.py \
  --map_name Austin \
  --checkpoint_path checkpoint/epoch_00500.pt \
  --output_dir eval_results/epoch_00500 \
  --render
```

`--checkpoint_path` is required and `--map_name` defaults to `Austin`. `--output_dir`, `--noise`, `--seed`, `--lap_num`, `--start_idx`, and `--minimum_lap_time` default to `eval_results`, `0.0`, `42`, `1`, `0`, and `10.0`. The run exits 0 when the checkpoint completes every lap without a collision and 1 otherwise, and prints `PASSED`, `COLLISION`, `LAPS_COMPLETED`, `LAP_PROGRESS`, `LAP_TIME`, `MEAN_LAP_TIME`, `AVG_SPEED`, `SPEED_VARIANCE`, and `TOTAL_DISTANCE` as `KEY=VALUE` lines. It writes no results file.

With `--render` the video lands directly in `--output_dir`, and its name carries the outcome: `[c_]<map>_lap<progress>[_noiseNN].mp4`, where `c_` marks a collision, the progress is the completed laps plus the fraction of the current lap to one decimal with `.` written as `_`, and the noise suffix appears only for a nonzero `--noise` as the percentage. `Austin_lap1_0.mp4` completed the target lap; `c_Austin_lap0_5.mp4` collided halfway around; `c_Austin_lap0_5_noise10.mp4` did the same at `--noise 0.1`.

### Multi-Agent Evaluation

Evaluates the model in competitive racing scenarios against an expert opponent. The framework provides 4 pre-configured tracks with raceline files for testing: [Austin](f1tenth_racetracks/Austin/Austin_map.png), [Hockenheim](f1tenth_racetracks/Hockenheim/Hockenheim_map.png), [MoscowRaceway](f1tenth_racetracks/MoscowRaceway/MoscowRaceway_map.png), and [Nuerburgring](f1tenth_racetracks/Nuerburgring/Nuerburgring_map.png), each with 3 raceline options (`raceline0`, `raceline1`, `raceline2`). For tracks without pre-generated racelines, use the [Raceline Generation](#raceline-generation-optional) section to create them first.


```bash
python eval_multi.py \
  --map_name Austin \
  --checkpoint_path checkpoint/epoch_00500.pt \
  --output_dir eval_results/epoch_00500/Austin \
  --ego_idx 0 \
  --opponent_raceline raceline1 \
  --opponent_speed_scale 0.8
```

`--checkpoint_path` is required; `--map_name`, `--output_dir`, `--ego_raceline`, `--ego_idx`, `--opponent_raceline`, `--opponent_speed_scale`, `--interval_idx`, `--sim_duration`, `--noise`, and `--seed` default to `Austin`, `eval_results`, `raceline1`, `0`, `raceline1`, `0.8`, `15`, `8.0`, `0.0`, and `42`. The run prints `STATE`, `AVG_SPEED`, `SPEED_VARIANCE`, and `TOTAL_DISTANCE` as `KEY=VALUE` lines, where `STATE` is 1 for following, 2 for overtaking, and 3 for a collision. It writes no results file; `eval_multi.sh` owns the summary for a whole map.

With `--render` the video lands directly in `--output_dir` as `<c|f|o>_ol<opponent raceline>_e<ego index>_o<opponent index>_s<speed scale>[_noiseNN].mp4`, with the noise suffix present only for a nonzero `--noise`. The name carries the scenario alone, so the checkpoint and map belong in `--output_dir`.

### Multi-Agent Parallel Evaluation (Optional)

The batch evaluation runs hundreds of scenarios in parallel to comprehensively assess the model's performance across different starting positions, opponent strategies, and difficulty levels:

```bash
bash eval_multi.sh checkpoint/epoch_00500.pt eval_results
```

The checkpoint is required and the output root defaults to `eval_results`. The batch runs 80 start points against 3 opponent racelines and 3 opponent speed scales on Austin, 720 scenarios across 12 workers, and lays every artifact under `<output root>/<checkpoint stem>/<map>/`. It always completes at least 100 scenarios and stops early only if its aggregate collision rate then exceeds 20%.

The batch renders no video, so `results.json` is its only artifact: the batch configuration, the following, overtaking, collision, and error counts, and their percentages. A collision-guard stop, a `Ctrl-C`, and a `SIGTERM` each still write it, so `planned_scenarios`, `completed_scenarios`, `complete`, and `stop_reason` say how much of the batch the numbers cover; percentages always use `completed_scenarios` as their denominator. The batch exits 0 when every scenario ran, 1 on worker errors, 2 on a collision-guard stop, and 130 or 143 when interrupted.

### Checkpoint Qualification

Training has no hyperparameter sweep and performs no evaluation. Monitor the pipeline externally and evaluate each `epoch_00500.pt` checkpoint in this order:

1. Run `eval_single.py` for Austin, Hockenheim, MoscowRaceway, and Nuerburgring, stopping at the first failure. Enforce a 90-second wall-clock timeout externally for each map; exceeding it is a failed single-agent gate.
2. After all four maps pass, run `eval_multi.sh` for Austin's 720 scenarios.
3. Qualify the model only when all 720 scenarios complete without worker errors and `success_percent` in `results.json` is strictly greater than `80.0`.

Retain a qualified checkpoint and record its four single-agent metric blocks plus the complete multi-agent `results.json` beside it. Delete a checkpoint that fails a single-agent map or the multi-agent safety requirement. An interrupted or errored evaluation is not a model result; resolve the runtime failure before deciding whether to retain the checkpoint.

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

`collect.sh` owns the output dataset directory and all batch collection settings. It refuses to start when the dataset directory already holds collected episodes, so every collection begins from an empty one, and it stops the remaining scenarios once the collision rate exceeds 20% after 25 completed scenarios. The current Austin batch runs 80 ego starting points against three opponent racelines at waypoint interval `15` and speed scales `0.4`, `0.6`, and `0.8`, with 8 seconds per scenario: 720 scenarios total.

When the workers finish, `collect.sh` prints the following, overtaking, collision, and worker-failure counts and writes `summary.json` into the dataset directory. It exits 0 when every scenario ran, 1 on worker failures, and 2 on a collision-guard stop. The summary snapshots the collection, vehicle, and expert configuration; reports collision-free, collision, overtaking, and following outcomes with their rates; counts training rows and saved videos; and provides a breakdown by opponent raceline and speed scale. It is generated from the saved CSV and collision metadata, so a collection cut short by the collision guard or by a worker failure still leaves a summary of what it collected.

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
  --dataset_dir dataset \
  --output_dir checkpoint \
  --speed_loss_weight 0.05 \
  --gradient_clip_norm 1.0
```

The training schedule is fixed: 500 epochs, a checkpoint at epoch 500, batch size 1024, and learning rate `1e-4`. Consequently, each run writes one weight-only `epoch_00500.pt` checkpoint to `--output_dir`. It contains only `model.state_dict()`—no optimizer state, training state, or resume metadata. These values are constants in `train.py`, not command-line options, and there is no hyperparameter sweep. Training reads collision-free episodes from `<--dataset_dir>/success/`, performs no evaluation, and keeps no resume state. Follow [Checkpoint Qualification](#checkpoint-qualification) externally after training.

## PPO Fine-Tuning

Fine-tune a checkpoint produced by `train.py` with recurrent PPO:

```bash
python run_ppo.py \
  --checkpoint_path checkpoint/epoch_00500.pt
```

`run_ppo.py` externally sequences the separate training and evaluation modules. Every epoch trains once through all 720 Austin scenarios. The full pool is randomly shuffled into 45 batches of 16 different scenarios, and the 16 workers collect one stochastic trajectory per scenario in parallel before four optimizer passes over the completed rollout batch. Evaluation runs after every tenth epoch: the updated policy is screened deterministically on all 720 scenarios and then runs one single-vehicle lap on Austin, Hockenheim, MoscowRaceway, and Nuerburgring. A single-vehicle failure stops the four-map gate, then the next epoch begins. Training continues epoch by epoch. The policy learns steering and speed log standard deviations initialized to `0.05` and `0.50`. A shared recurrent actor-value model carries over the IL policy and initializes its scalar value head. An evaluation epoch overwrites `checkpoint/ppo/ppo.pt` when all four single-vehicle laps pass and deterministic full-pool safety is strictly higher than the previous best. PPO artifacts live inside `checkpoint/ppo/`. See [ppo/README.md](ppo/README.md) for the reward, exploration behavior, and artifacts.

## Model Architecture

The policy downsamples each full-circle simulator scan to 180 LiDAR values and applies one shared, fixed sigmoid normalization coefficient. The normalized LiDAR vector is concatenated with a 30-dimensional speed embedding, producing a 210-dimensional recurrent input. A single-layer GRU with 420 hidden units feeds an action head with dimensions `420 -> 128 -> 2`, which predicts steering and desired speed. During supervised training, the speed embedding is replaced by a learned dummy embedding at 20% of timesteps. PPO uses the measured previous speed at every timestep. All of these dimensions live as class attributes on `End2Race` in `model.py`.

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
