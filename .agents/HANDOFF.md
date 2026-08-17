# End2Race Handoff

## Supervised Training

- `train.py` trains for 500 epochs with batch size 1024 and Adam learning rate `1e-4`.
- Epoch 500 writes the weight-only checkpoint `epoch_00500.pt`.
- `eval_single.py` evaluates one lap, while `eval_multi.py` and `eval_multi.sh` evaluate competitive scenarios.
- Checkpoint qualification runs Austin, Hockenheim, MoscowRaceway, and Nuerburgring in that order with a 90-second external timeout per map, followed by all 720 Austin competitive scenarios.
- A checkpoint qualifies when all four laps pass, all 720 scenarios complete without worker errors, and `success_percent` is greater than `80.0`.

## PPO

- `run_ppo.py` sequences training, deterministic evaluation, and checkpoint promotion. `train_ppo.py` owns PPO updates, and `eval_ppo.py` owns the 720-scenario screen.
- Each unbounded epoch shuffles all 720 scenarios into 45 batches of 16. Every worker collects one stochastic trajectory, and each batch produces one optimizer update.
- The recurrent actor-value model loads the IL policy, adds a value head over the shared GRU output, and learns steering and speed standard deviations initialized to `0.05` and `0.50`.
- PPO uses Adam learning rate `2e-5`, discount `0.999`, GAE factor `0.95`, clip range `0.2`, value weight `0.5`, and gradient norm limit `0.5`.
- The reward is `0.01 * ego_progress_delta - 1.0 * ego_collision`. An overtake is classified when the ego center is at least one vehicle length (`0.58 m`) ahead in wrapped Frenet progress.
- Episodes run for eight seconds by default or until ego collision. Time-limit trajectories bootstrap from the final value estimate.
- Evaluation runs after epochs 10, 20, 30, and so on. It screens all 720 scenarios, then evaluates one lap on each of the four maps.
- `checkpoint/ppo/ppo.pt` is promoted when all four laps pass and deterministic safety improves.
- `checkpoint/ppo/` also contains `config.json`, `episodes.jsonl`, and `metrics.jsonl`.

## Verification

- PPO Python modules compile successfully.
- The scenario pool contains 720 unique scenarios and each epoch produces 45 batches of 16.
- Evaluation cadence, current PPO parameters, metric summaries, and repository diff formatting pass focused checks.
