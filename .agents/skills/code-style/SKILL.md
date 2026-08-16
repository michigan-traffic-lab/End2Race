---
name: code-style
description: Generate, edit, refactor, or review Python code in the End2Race repository using its concise script-oriented syntax and layout. Use for End2Race training, evaluation, collection, model, configuration, and utility code.
---

# End2Race Code Style

Scope: how existing Python source is written. Layout, imports, naming, formatting.

## Structure Python Files Clearly

- Prefer top-level imports, focused functions or classes, and a short executable entry point.
- Put a script's orchestration in `main()` and call it under `if __name__ == "__main__":`.

## Write Explicit Imports

- Put imports at the top of the file unless a local import prevents a real cycle or is required for process initialization.
- Group standard-library, third-party, and local imports with one blank line between groups.
- Import names explicitly. Do not add wildcard imports.
- Use established aliases such as `np`, `nn`, and `optim`.
- Do not add `from __future__ import annotations`. Python 3.11 evaluates the notation this project uses.
- Do not reorder untouched imports solely for style.

## Format Command-Line Arguments Consistently

- Use `parse_arguments()` when a script has a dedicated parser and return parsed arguments directly.
- Keep an `add_argument` call on one line. Wrap it only when the wrapped form reads better, never at a character count.
- Give optional value arguments an appropriate `type` and `default`.
- Use actions for boolean flags and do not invent defaults for required positional arguments.
- Do not add `help`. The argument name and its group carry the contract, and help text forces the call onto several lines.
- Group related arguments with blank lines or a short section comment.

## Use Comments and Documentation Sparingly

- Let names and structure explain ordinary operations.
- Comment non-obvious reasons, invariants, units, array shapes, indexing, and numerical constraints.
- Use short section comments only when they make a long script easier to scan. Do not target a fixed comment density.
- Do not add a module docstring. File responsibilities belong in `.agents/` documentation.
- Keep the module docstring of vendored code when it carries a license, copyright, or upstream attribution. Never delete or rewrite that notice.
- Add a concise function or class docstring only when its contract is not clear from its name and signature.
- Put long design explanations in project documentation, not source comments.
- Keep the existing comments when merging files. Do not convert them to docstrings in passing.

## Add Types Where They Clarify Contracts

- Do not annotate every local helper mechanically.
- Add annotations at data boundaries, reusable APIs, dataset and model contracts, and complex return structures when they improve understanding or checking.
- Follow the notation already used in the touched file. The project targets Python 3.11, so `X | None` and built-in generics are available without an import.

## Format for Meaning

- Use `snake_case` and complete words.
- Prefix private helpers with `_`.
- Keep established domain abbreviations such as `ego`, `opp`, `idx`, `lidar`, and `obs`.
- Keep an expression on one line while it stays readable. Never wrap because of a character count.
- Wrap by semantic group when wrapping reads better. One item per line is fine for a dictionary or a long keyword list.
- Do not introduce temporary variables only to satisfy line width.
- Preserve the touched file's quote style unless changing a string for another reason.

## Handle Data and Errors Explicitly

- Use an explicit NumPy dtype when the dtype is part of the data contract.
- Convert NumPy and tensor scalars to `float`, `int`, or `bool` before JSON serialization.
- Raise a specific exception with an actionable message. Use `SystemExit` for command-line misuse.
- Do not catch an exception in order to re-report it. Let every traceback survive.
- Use f-strings for human-readable output and preserve stable machine-readable output consumed by another script.

## Check the Style

- Confirm imports are explicit and grouped.
- Confirm comments and docstrings add information rather than repeat the code.
- Confirm line breaks follow semantic groups.
- Confirm style-only edits do not reformat unrelated code.
