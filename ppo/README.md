# PPO Fine-Tuning

This module fine-tunes a behavior-cloned End2Race checkpoint with recurrent PPO. The entry point is `train_ppo.py` in the repository root.

## Pipeline

- The actor loads the complete End2Race checkpoint and uses its steering and speed outputs as the mean of a fixed-standard-deviation Gaussian policy.
- The critic loads an independent copy of the same BC backbone and replaces the action head with a scalar value head.
- The default pool contains 720 scenarios: 80 ego starts, 3 opponent racelines, and 3 opponent speed scales.
- Each scenario is broadcast to 16 environments, producing 16 complete trajectories from one policy with independent action noise.
- Each trajectory receives per-step GAE. One epoch collects every scenario once, shuffles the resulting groups, and trains once through them with PPO-Clip and value regression.
- `batch_size` counts scenarios, not transitions. The default 32 scenarios contain up to 512 trajectories.

The simulator runs at 120 Hz and holds each actor action for three physics steps, matching the 40 Hz End2Race control rate. The latticeplanner `RacelineFollower` controls the opponent with a 120 Hz tracker and 10 Hz replanning.

The environment reward is:

```text
0.01 * ego_progress_delta
+ 0.02 * (ego_progress_delta - opponent_progress_delta)
- 2.0 * ego_collision
```

The actor observes 180 ego LiDAR values and the previous ego speed. Opponent ground-truth poses are available only to the environment for progress-based training rewards.

## Run

Pass a checkpoint produced by `train.py`:

```bash
python train_ppo.py \
  --checkpoint_path checkpoint/epoch_05000.pt \
  --output_dir runs/ppo
```

The output directory contains the deployable `checkpoint.pt`, resolved `config.json`, per-trajectory `episodes.jsonl`, and per-epoch `metrics.jsonl`. Training metrics describe the stochastic rollout collected before each optimization pass; evaluate `checkpoint.pt` separately before comparing safety or overtaking rates.
