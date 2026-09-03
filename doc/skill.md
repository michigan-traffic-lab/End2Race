# Code Guidelines

Keep End2Race code, configuration, documentation, and artifacts direct, minimal,
current, and consistent with the repository's concise script-oriented style.

## Keep the Implementation Current

- Write the simplest implementation that expresses the current design.
- Remove stale code, documentation, configuration keys, metrics, and scripts in the same change that makes them obsolete.
- Preserve backward compatibility only when explicitly requested. Do not add compatibility aliases, migration branches, deprecated names, or fallback paths for removed behavior.
- Do not stack names for document, classes, saved file. A bad example would be model_A_B_B_C_E_F.pt.
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

## Build Clean Imitation-Learning Ablations

Treat the baseline implementation as the control. Keep it unchanged and create
standalone sibling files for each ablation. An ablation changes one requested
factor; parameter tuning and additional architectural features are separate
experiments.

### Modify the Model at One Boundary

- Start by copying `imitation/model.py` to `imitation/model_<ablation>.py`. Preserve unchanged code, ordering, formatting, constants, preprocessing, masking, initialization, and output layers.
- Do not import, subclass, or wrap the baseline model to implement the ablation. The sibling model must remain readable and runnable on its own.
- Replace only the module under study and the state handling that replacement requires. Keep the input ordering and output contract unchanged unless they are the explicit ablation factor.
- For a temporal-backbone ablation, preserve the complete path through LiDAR preprocessing, previous-speed embedding, masking, and concatenation into `[B, T, 210]`; replace the GRU block at that point and continue to produce `[B, T, 420]` for the unchanged action head.
- When capacity matching is requested, report both exact trainable-parameter counts and the difference. Do not add unrelated layers merely to claim architectural equivalence.
- Prefer maintained PyTorch modules over handwritten attention, normalization, or recurrent primitives. For a Transformer replacement, use `nn.TransformerEncoderLayer` and `nn.TransformerEncoder` unless the requested factor cannot be represented by them.
- Do not add RoPE, KV caching, residual scaling, dropout, new normalization, auxiliary losses, or other Transformer features unless one is explicitly the factor being tested.
- Preserve the baseline initialization policy. Add only the minimal initialization needed to prevent cloned Transformer layers from starting with identical parameters.
- Keep streaming state explicit in `encode()` and `forward()`. A context window or cache must have one documented length and the evaluator must pass back exactly the state returned by the model.

### Copy the Training Pipeline Before Replacing the Model

- Start by copying `imitation/train.py` to `imitation/train_<ablation>.py`. Keep `SequenceDataset` and `train_epoch` local; do not import them from the baseline training module.
- Change only the model import and class, the ablation checkpoint directory, model-specific constants, and the forward-state unpacking required by the new model.
- Preserve dataset discovery, column order, sequence construction, previous-speed alignment, DataLoader behavior, initialization, optimizer, loss terms and weights, gradient clipping, epoch count, checkpoint interval, and all default arguments.
- If the baseline does not set a training seed or a fixed DataLoader generator, do not add either to the ablation. Keep an evaluator seed separate from training randomness.
- Do not add a scheduler, warmup, dropout, resume state, optimizer checkpoint, mixed precision, or broader CLI solely because the replacement architecture commonly uses it.
- A hyperparameter sweep is not the clean ablation. Give it a separate script and artifact name, and do not parameterize the canonical ablation trainer unless the user explicitly requests that permanent interface.
- Save the canonical checkpoint under `checkpoint/ckp_ablation_<ablation>/epoch_00500.pt`. When repeated tuned runs share that directory, encode only the changed values and repetition index, for example `epoch500_bs128_lr1e-4_1.pt`; do not label an unseeded run with a seed.

### Copy Evaluators and Preserve the Protocol

- Leave `evaluation/eval_single.py`, `evaluation/eval_multi.py`, and `evaluation/eval_multi.sh` unchanged. Create `evalsingle_ablation_<ablation>.py`, `evalmulti_ablation_<ablation>.py`, and `evalmulti_ablation_<ablation>.sh` as standalone copies.
- In the copied evaluators, replace only the model import and class, default checkpoint and output paths, recurrent-state initialization, model call, and shell module name required by the ablation.
- Preserve maps, scenario grids, start points, opponent policies and speeds, simulation duration, control rate, noise, evaluation seed, rendering default, worker count, stopping rules, metrics, exit status, and `results.json` schema.
- Load checkpoints strictly into the ablation class. Run Python evaluators from the repository root with `python -m evaluation.<module>` so repository imports resolve consistently.
- Keep result roots separate, such as `eval_results/<ablation>_ablation/<checkpoint_stem>/`, so different models and baseline results cannot overwrite one another.

### Verify the Single Changed Factor

- Diff each ablation file against its baseline and enumerate every difference. Any unexplained difference invalidates a clean single-factor claim.
- Verify preprocessing and tensors immediately before the replaced module are equal for controlled inputs. Check output shapes, parameter counts, finite loss and gradients, and a real optimizer update for every parameter group.
- For causal sequence models, test that future-token changes do not affect past outputs, full-sequence and streaming outputs agree over the trained window, and behavior remains finite after the context window begins sliding.
- Compile each Python entry point, run `bash -n` on each shell script, strictly load the intended checkpoint, and run at least one single-agent and one multi-agent smoke scenario before a full panel.
- Qualify a full multi-agent panel only when `planned_scenarios`, `completed_scenarios`, `complete`, and `errors` prove that the requested scenario set finished. Do not infer completion from videos, directory size, or process exit alone.
- Report steering and speed losses separately from their weighted sum. Keep training fit, closed-loop performance, and causal explanations distinct; a lower supervised loss does not by itself establish a better controller.

## Name and Save Artifacts Simply

The `reinforcement/run_ppo.py` orchestrator owns PPO artifact persistence.

- Keep the model, training state, metrics, and resolved configuration together in one flat artifact directory.
- Keep PPO artifacts flat inside `checkpoint/ppo/`. Do not expose another PPO artifact-directory path.
- Use `ppo_NNN.pt` for deployable PPO models, numbered consecutively from `ppo_001.pt` for each qualifying evaluation, and `config.json` for the resolved configuration.
- Reserve `.jsonl` for append-only streams such as `metrics.jsonl` and `episodes.jsonl`; write a record produced once to `.json`.
- Save every qualifying PPO model under its next consecutive number without overwriting an earlier qualifying model. Do not save models for non-qualifying epochs or create duplicate model aliases.
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
