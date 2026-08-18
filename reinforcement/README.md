# PPO Fine-Tuning

This package trains an End2Race policy with recurrent PPO. `train_ppo.py` contains training only, `eval_ppo.py` contains evaluation only, and `run_ppo.py` sequences them.

## Pipeline

- One shared recurrent actor-value model loads either an IL checkpoint from `imitation/train.py` or a full PPO checkpoint from an earlier run. An IL checkpoint initializes the policy weights and starts a new value head. A PPO checkpoint restores both policy and value-head weights. The optimizer, epoch counter, recurrent hidden state, and rollout state always start fresh.
- The default pool contains 720 scenarios: 80 ego starts, 3 opponent racelines, and 3 opponent speed scales.
- Launch with `torchrun --nproc_per_node=N` to give each rank one GPU and `--num_envs` environment workers. A plain Python launch remains the single-device form. Every epoch trains on the complete 720-scenario pool: one shuffled order is synchronized across ranks, divided evenly, and collected concurrently while the policy remains fixed. Every scenario contributes exactly one trajectory per epoch.
- A launch trains one model for an unlimited number of epochs until the process is stopped. Stochastic collection uses fixed steering and speed standard deviations of `0.05` and `0.50`.
- Each trajectory receives per-step GAE. After all 720 trajectories are collected, one PPO-Clip and value-regression update uses advantage statistics reduced across every rank. Gradients accumulate across local worker-sized rollout chunks, are summed across ranks, and are globally clipped before one identical optimizer step on every replica.
- Evaluation runs after every update. Deterministic screening of all 720 scenarios is divided across ranks and gathered by rank 0.

The simulator runs at 120 Hz and holds each actor action for three physics steps, matching the 40 Hz End2Race control rate. PPO bounds the actor's desired speed to `[0, 20]`. The expert `RacelineFollower` controls the opponent with a 120 Hz tracker and 10 Hz replanning.

The environment reward is:

```text
0.025 * ego_progress_delta
- 1.0 on ego collision
```

An overtake is classified when the ego center reaches at least one full vehicle length (`0.58 m`) ahead of the opponent center in wrapped Frenet progress. The policy observes 180 ego LiDAR values and the previous ego speed. A trajectory ends on ego collision or at the configured time limit. Opponent ground-truth poses support outcome classification.

For IL inputs, value-head initialization uses fresh process randomness. Learned-policy action samples and scenario ordering are also fresh on every launch.

The actor and value head use one Adam optimizer with a constant learning rate of `1e-5`.

## Run

Pass either an IL checkpoint produced by `imitation/train.py` or a PPO checkpoint produced by an earlier run:

```bash
python -m reinforcement.run_ppo \
  --checkpoint_path checkpoint/epoch_00500.pt
```

For four GPUs:

```bash
torchrun --standalone --nproc_per_node=4 --module reinforcement.run_ppo \
  --checkpoint_path checkpoint/epoch_00500.pt
```

PPO saves `config.json`, `episodes.jsonl`, `metrics.jsonl`, and qualifying weight-only actor-critic checkpoints inside `checkpoint/ppo/`. Every deterministic full-pool evaluation with safety above 90% and an overtake rate above 60% saves another checkpoint in order: `ppo_001.pt`, `ppo_002.pt`, and so on. Optimizer and runtime training state are never saved. `episodes.jsonl` and `metrics.jsonl` identify every record by epoch, and each metrics record names the checkpoint saved for that epoch. Each launch starts with a clean `checkpoint/ppo/` directory.

The terminal reports the learning rate once at startup and one global start and completion line for each deterministic screening. Each completed stochastic training batch reports collision, following, and overtaking counts; policy steering and speed mean and standard deviation; mean value estimate, episode return, and elapsed phase time. Each PPO update reports value loss, approximate KL divergence, clip fraction, and gradient norms before and after clipping. Repeated Gym maintenance notices and the expected RK4 integrator warning are suppressed for PPO workers.
