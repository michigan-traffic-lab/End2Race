# Non-PPO Handoff

Scope: non-PPO changes accumulated after `13def98`; `ppo/` is intentionally excluded.

## Changes

- Centralized simulation timing in `lattice_config.yaml`; simplified FOT, expert collection, path loading, scenario generation, and JSON summaries. FOT equations, feasibility checks, and trajectory costs are unchanged, while candidate generation now uses the configured four workers instead of the former environment-variable default of one.
- Decoupled supervised training from evaluation and restart/resume policy. `train.py` now trains one model and saves `epoch_<epoch>.pt`; `sweep.sh` owns the hyperparameter grid, four-map single-agent screen, and 720-scenario multi-agent gate.
- Converted both evaluators to flag-based CLIs, explicit evaluation mode, stable machine-readable metrics, scoped video paths, argument validation, and partial/final JSON summaries.
- Simplified collection/install scripts, moved repository coding rules into `.agents/skills/`, removed the obsolete root skill and Conda environment file, and updated README/current ignore rules.

## Bugs fixed

- Single-agent screening no longer runs inside training mode and randomly masks the speed embedding; the new CLIs reject zero-lap and invalid multi-agent scenarios.
- Multi-agent early stopping exits every scheduling loop, terminates child Python processes, checks the final worker batch, preserves partial results, and cannot report success for an incomplete or errored sweep.
- RK4 and legacy Gym notice noise was removed without changing simulator integration.

## Validation

Python compilation, shell syntax, diff checks, summary fixtures, and stubbed sweep/multi-agent orchestration passed. No full training or real 720-scenario simulator sweep was run for this handoff.
