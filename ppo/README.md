# PPO Fine-Tuning

This module fine-tunes a behavior-cloned End2Race checkpoint with recurrent PPO. `train_ppo.py` contains training only, `eval_ppo.py` contains evaluation only, and the external `run_ppo.py` entry point sequences them.

## Pipeline

- One shared recurrent actor-value model loads the complete End2Race checkpoint. The IL action head remains the policy mean, and a scalar value head is initialized from scratch over the same GRU features.
- The default pool contains 720 scenarios: 80 ego starts, 3 opponent racelines, and 3 opponent speed scales.
- Launch with `torchrun --nproc_per_node=N` to give each rank one GPU and `--num_envs` environment workers. A plain Python launch remains the single-device form. Every epoch trains on the complete 720-scenario pool: one shuffled order is synchronized across ranks, divided evenly, and collected concurrently while the policy remains fixed. Every scenario contributes exactly one trajectory per epoch.
- Training runs for at most 100 epochs. Stochastic collection uses fixed steering and speed standard deviations of `0.05` and `0.50`.
- Each trajectory receives per-step GAE. After all 720 trajectories are collected, one PPO-Clip and value-regression update uses advantage statistics reduced across every rank. Gradients accumulate across local worker-sized rollout chunks, are summed across ranks, and are globally clipped before one identical optimizer step on every replica.
- Evaluation runs after every update. Deterministic screening of all 720 scenarios is divided across ranks and gathered by rank 0.

The simulator runs at 120 Hz and holds each actor action for three physics steps, matching the 40 Hz End2Race control rate. PPO bounds the actor's desired speed to `[0, 20]`. The latticeplanner `RacelineFollower` controls the opponent with a 120 Hz tracker and 10 Hz replanning.

The environment reward is:

```text
0.02 * ego_progress_delta
- 1.0 on ego collision
```

An overtake is classified when the ego center reaches at least one full vehicle length (`0.58 m`) ahead of the opponent center in wrapped Frenet progress. The policy observes 180 ego LiDAR values and the previous ego speed. A trajectory ends on ego collision or at the configured time limit. Opponent ground-truth poses support outcome classification.

Value-head initialization, learned-policy action samples, and scenario ordering use fresh process randomness on every launch.

The unified policy and value head use one Adam optimizer. Its learning rate starts at `1e-6`, increases by `1e-6` each epoch, reaches `1e-5` at epoch 10, and remains capped at `1e-5`.

## Run

Pass a checkpoint produced by `train.py`:

```bash
python run_ppo.py \
  --checkpoint_path checkpoint/epoch_00500.pt
```

For four GPUs:

```bash
torchrun --standalone --nproc_per_node=4 run_ppo.py \
  --checkpoint_path checkpoint/epoch_00500.pt
```

PPO saves `config.json`, `checkpoints.json`, `episodes.jsonl`, `metrics.jsonl`, and qualifying policy checkpoints inside `checkpoint/ppo/`. Every deterministic full-pool evaluation with safety above 90% and an overtake rate above 60% saves another checkpoint in qualification order: `ppo_001.pt`, `ppo_002.pt`, and so on. `checkpoints.json` is a JSON array updated after each save with the checkpoint's epoch, learning rate, screening results, rollout metrics, and PPO diagnostics. `episodes.jsonl` contains stochastic training and deterministic screening records after every update. `metrics.jsonl` contains training and evaluation metrics after every update. Each launch starts with a clean `checkpoint/ppo/` directory.

The terminal reports live deterministic-screening counts after every worker batch. Each completed stochastic training batch reports collision, following, and overtaking counts; policy steering and speed mean and standard deviation; mean value estimate, episode return, and elapsed phase time. Each PPO update reports its transition count, learning rate, value loss, approximate KL divergence, clip fraction, and gradient norms before and after clipping. Repeated Gym maintenance notices and the expected RK4 integrator warning are suppressed for PPO workers.
