# PPO Fine-Tuning

This module fine-tunes a behavior-cloned End2Race checkpoint with recurrent PPO. `train_ppo.py` contains training only, `eval_ppo.py` contains evaluation only, and the external `run_ppo.py` entry point sequences them.

## Pipeline

- One shared recurrent actor-value model loads the complete End2Race checkpoint. The IL action head remains the policy mean, and a scalar value head is initialized from scratch over the same GRU features.
- The default pool contains 720 scenarios: 80 ego starts, 3 opponent racelines, and 3 opponent speed scales.
- Every epoch trains on the complete 720-scenario pool. The pool is randomly shuffled and split into 45 batches of 16 different scenarios. The 16 workers collect one stochastic trajectory per scenario in parallel, so every scenario contributes exactly one trajectory per epoch.
- Training continues epoch by epoch. The policy owns trainable steering and speed log-standard-deviation parameters initialized to standard deviations `0.05` and `0.50`, and PPO learns both through the clipped policy objective.
- Each trajectory receives per-step GAE. Each completed batch of up to 16 scenario trajectories immediately receives four PPO-Clip and value-regression update passes before the next batch is collected. Old rollout log-probabilities remain fixed across all four passes. The worker count is the training batch size, and the final batch may be smaller.
- Evaluation runs after every tenth epoch. The updated policy is deterministically screened on all 720 scenarios and then runs one single-vehicle lap on Austin, Hockenheim, MoscowRaceway, and Nuerburgring. The single-vehicle gate stops at its first failed map, then the next epoch begins.

The simulator runs at 120 Hz and holds each actor action for three physics steps, matching the 40 Hz End2Race control rate. The latticeplanner `RacelineFollower` controls the opponent with a 120 Hz tracker and 10 Hz replanning.

The environment reward is:

```text
0.02 * ego_progress_delta
- 1.0 on ego collision
```

An overtake is classified when the ego center reaches at least one full vehicle length (`0.58 m`) ahead of the opponent center in wrapped Frenet progress. The policy observes 180 ego LiDAR values and the previous ego speed. A trajectory ends on ego collision or at the configured time limit. Opponent ground-truth poses support outcome classification.

Value-head initialization, learned-policy action samples, and scenario ordering use fresh process randomness on every launch.

The unified policy, value head, and learned action standard deviations use one Adam optimizer with learning rate `2e-5`.

## Run

Pass a checkpoint produced by `train.py`:

```bash
python run_ppo.py \
  --checkpoint_path checkpoint/epoch_00500.pt
```

PPO saves `ppo.pt`, `config.json`, `episodes.jsonl`, and `metrics.jsonl` inside `checkpoint/ppo/`. An evaluation epoch overwrites `ppo.pt` when all four single-vehicle laps pass and deterministic full-pool safety is strictly higher than the previous best. `episodes.jsonl` contains stochastic training records every epoch and deterministic screening records every tenth epoch. `metrics.jsonl` contains training metrics every epoch and evaluation metrics every tenth epoch. Each launch starts with a clean `checkpoint/ppo/` directory.

The terminal reports live deterministic-screening counts after every worker batch. Each completed stochastic training batch reports collision, following, and overtaking counts; policy steering and speed mean and standard deviation; mean value estimate and episode return; current learned action standard deviations; and elapsed phase time. Repeated Gym maintenance notices and the expected RK4 integrator warning are suppressed for PPO workers.
