---
name: clean-code
description: Keep End2Race code and artifacts direct, minimal, and current. Use when implementing, refactoring, reviewing, or cleaning code; changing configuration or CLI surfaces; removing obsolete behavior; or saving checkpoints, training state, metrics, and configuration.
---

# Clean Code

Scope: what should exist, and where its output lands. Necessity, lifecycle, artifacts, change size.

## Keep the Implementation Current

- Write the simplest implementation that expresses the current design.
- Remove stale code, documentation, configuration keys, metrics, and scripts in the same change that makes them obsolete.
- Preserve backward compatibility only when the user explicitly requests it.
- Do not add compatibility aliases, migration branches, deprecated names, or fallback paths for removed behavior.
- Do not add dummy wrappers or pass-through helper functions. Inline one-use helpers unless the helper names a real domain step.
- Create a class for real state, model, dataset, or configuration contracts, not to namespace functions.
- Validate real external inputs and fail fast on broken invariants. Do not guard against impossible internal states.
- Do not add broad argparse surfaces. Prefer the existing configuration path or a constant when there is a single intended workflow.
- Rename variables, configuration keys, metrics, files, and documentation when their meaning changes.
- Keep edits scoped. Do not refactor unrelated code, but remove dead paths directly related to the change.
- Move one module at a time. Update its imports and delete the old entry point in the same change.
- Keep style changes and behavior changes in separate diffs. A move must not alter a value, default, or formula.

## Keep Training and Evaluation Separate

- Keep training entry points responsible only for training and saving training artifacts. Never import, invoke, embed, or otherwise mix evaluation logic into training code.
- Keep the single-agent evaluator responsible only for single-agent evaluation.
- Keep the multi-agent evaluator responsible only for multi-agent evaluation.
- Keep pipeline sequencing, checkpoint screening, retries, process supervision, and monitoring in an external orchestrator rather than in training or evaluation code.
- Preserve these boundaries under all circumstances; do not add convenience paths that combine training and evaluation in one program.

## Name and Save Artifacts Simply

Every training entry point saves the same way. The `ppo/run_ppo.py` orchestrator owns PPO artifact persistence.

- Keep the model, training state, metrics, and resolved configuration together in one flat artifact directory.
- Keep PPO artifacts flat inside the fixed `checkpoint/ppo/` directory. Do not expose another PPO artifact-directory path.
- Use role-based names: `ppo.pt` for the deployable PPO model and `config.json` for the run's resolved arguments and configuration.
- Reserve `.jsonl` for append-only streams such as `metrics.jsonl` and `episodes.jsonl`. Write a record produced once to `.json`.
- Overwrite `ppo.pt` in place. Do not keep per-epoch or per-update history, numbered snapshots, or duplicate model aliases.
- Do not save optimizer, critic, training, or resume state for PPO.
- Do not repeat the experiment, model, map, or method name in artifact filenames when the directory or saved configuration already records it.
- Do not create nested run, model, checkpoint, or metrics directories beneath the artifact directory.

## Review the Change

- Search for old names and removed concepts with `rg`.
- Verify removed configuration keys and CLI options no longer exist and have no compatibility aliases.
- Confirm documentation describes only the current behavior.
- Confirm artifact paths and filenames follow the flat layout.
- Run focused compile, tests, or smoke checks for touched paths.
- Confirm a style-only diff carries no behavior change.
- Run `git diff --check`.
