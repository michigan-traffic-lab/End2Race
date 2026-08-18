# Code Guidelines

Keep End2Race code, configuration, documentation, and artifacts direct, minimal,
current, and consistent with the repository's concise script-oriented style.

## Keep the Implementation Current

- Write the simplest implementation that expresses the current design.
- Remove stale code, documentation, configuration keys, metrics, and scripts in the same change that makes them obsolete.
- Preserve backward compatibility only when explicitly requested. Do not add compatibility aliases, migration branches, deprecated names, or fallback paths for removed behavior.
- Do not add dummy wrappers or pass-through helpers. Inline one-use helpers unless the helper names a real domain step.
- Create a class for real state, model, dataset, or configuration contracts, not to namespace functions.
- Validate real external inputs and fail fast on broken invariants. Do not guard against impossible internal states.
- Do not add broad argument-parser surfaces. Prefer the existing configuration path or a constant when there is a single intended workflow.
- Rename variables, configuration keys, metrics, files, and documentation when their meaning changes.
- Keep edits scoped. Do not refactor unrelated code, but remove dead paths directly related to the change.
- Move one module at a time. Update its imports and delete the old entry point in the same change.
- Keep style and behavior changes separate. A move must not alter a value, default, or formula.

## Structure Python Clearly

- Prefer top-level imports, focused functions or classes, and a short executable entry point.
- Put script orchestration in `main()` and call it under `if __name__ == "__main__":`.

## Write Explicit Imports

- Put imports at the top of the file unless a local import prevents a real cycle or is required for process initialization.
- Group standard-library, third-party, and local imports with one blank line between groups.
- Import names explicitly. Do not add wildcard imports.
- Use established aliases such as `np`, `nn`, and `optim`.
- Do not add `from __future__ import annotations`. Python 3.11 evaluates the notation used by this project.
- Do not reorder untouched imports solely for style.

## Format Command-Line Arguments Consistently

- Use `parse_arguments()` when a script has a dedicated parser and return parsed arguments directly.
- Keep an `add_argument` call on one line when readable; otherwise wrap it by semantic group, not at a character count.
- Give optional value arguments an appropriate `type` and `default`.
- Use actions for boolean flags and do not invent defaults for required positional arguments.
- Do not add `help`. The argument name and its group carry the contract.
- Group related arguments with blank lines or a short section comment.

## Use Comments, Documentation, and Types Sparingly

- Let names and structure explain ordinary operations.
- Comment non-obvious reasons, invariants, units, array shapes, indexing, and numerical constraints.
- Use short section comments only when they make a long script easier to scan.
- Do not add a module docstring. File responsibilities belong in `doc/`.
- Preserve module docstrings that carry vendored licenses, copyright, or upstream attribution.
- Add a concise function or class docstring only when its contract is unclear from its name and signature.
- Put long design explanations in project documentation.
- Keep existing comments when moving files; do not convert them to docstrings in passing.
- Add annotations at data boundaries, reusable APIs, dataset and model contracts, and complex return structures when they improve clarity. Do not annotate every local helper mechanically.
- Follow the notation in the touched file. The project targets Python 3.11, so `X | None` and built-in generics need no compatibility import.

## Format for Meaning

- Use `snake_case` and complete words.
- Prefix private helpers with `_`.
- Keep established domain abbreviations such as `ego`, `opp`, `idx`, `lidar`, and `obs`.
- Keep an expression on one line while it remains readable. Wrap by semantic group, not by character count.
- Do not introduce temporary variables solely to shorten a line.
- Preserve the touched file's quote style unless changing a string for another reason.

## Handle Data and Errors Explicitly

- Use an explicit NumPy dtype when the dtype is part of the data contract.
- Convert NumPy and tensor scalars to `float`, `int`, or `bool` before JSON serialization.
- Raise a specific exception with an actionable message. Use `SystemExit` for command-line misuse.
- Do not catch an exception only to re-report it; preserve the traceback.
- Use f-strings for human-readable output and preserve stable machine-readable output consumed by other scripts.

## Keep Training and Evaluation Separate

- Keep training entry points responsible only for training and saving training artifacts. Do not import, invoke, or embed evaluation logic in training code.
- Keep the single-agent evaluator responsible only for single-agent evaluation.
- Keep the multi-agent evaluator responsible only for multi-agent evaluation.
- Keep pipeline sequencing, checkpoint screening, retries, process supervision, and monitoring in an external orchestrator.

## Name and Save Artifacts Simply

The `reinforcement/run_ppo.py` orchestrator owns PPO artifact persistence.

- Keep the model, training state, metrics, and resolved configuration together in one flat artifact directory.
- Keep PPO artifacts flat inside `checkpoint/ppo/`. Do not expose another PPO artifact-directory path.
- Use role-based names: `ppo.pt` for the deployable PPO model and `config.json` for the resolved configuration.
- Reserve `.jsonl` for append-only streams such as `metrics.jsonl` and `episodes.jsonl`; write a record produced once to `.json`.
- Overwrite `ppo.pt` in place. Do not keep per-epoch or per-update history, numbered snapshots, or duplicate model aliases.
- Do not save optimizer, critic, training, or resume state for PPO.
- Do not repeat experiment, model, map, or method names in filenames when the directory or saved configuration already records them.
- Do not create nested run, model, checkpoint, or metrics directories beneath the artifact directory.

## Review Changes

- Search for old names and removed concepts with `rg`.
- Verify removed configuration keys and CLI options no longer exist and have no compatibility aliases.
- Confirm documentation describes only current behavior.
- Confirm artifact paths and filenames follow the flat layout.
- Run focused compile, tests, or smoke checks for touched paths.
- Run `git diff --check`.
