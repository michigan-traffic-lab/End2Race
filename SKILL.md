---
name: clean-code
description: Use when editing, reviewing, or cleaning code. Enforces concise direct implementations, no defensive clutter, no stale code, no backward compatibility shims, no excessive argparse, and no dummy wrappers.
---

# Clean Code

When changing code, keep the implementation direct and minimal.

## Rules

- Write the simplest code that expresses the current design.
- Remove stale code, stale docs, stale config keys, stale metrics, and unused scripts in the same change that makes them obsolete.
- Do not preserve backward compatibility unless the user explicitly asks for it.
- Do not add compatibility aliases, migration branches, deprecated names, or fallback paths for removed behavior.
- Do not add dummy wrappers or pass-through helper functions. Inline one-use helpers unless the helper names a real domain step.
- Do not add defensive programming for impossible states. Validate real external inputs and fail fast on broken invariants.
- Do not add broad argparse surfaces. Prefer existing configuration patterns or constants when there is a single intended workflow.
- Keep names current. If behavior changes, rename variables, config keys, metrics, files, and docs to match the new behavior.
- Keep comments rare and useful. Explain non-obvious reasoning, not what the next line does.
- Keep edits scoped. Do not refactor unrelated code, but do remove directly related dead paths.
- All checkpoints, metrics, model name, shall have the simplest name such as checkpoint_xxxxx.pt, instead of A_B_C_checkpoint_xxxxx.pt
- All checkpoint, metrics, and saved config shall be directly put under checkpoint folder without nesting.

## Review Checklist

Before finishing a change:

- Search for old names and removed concepts with `rg`.
- Verify config validation rejects removed keys instead of accepting aliases.
- Confirm docs describe the current behavior only.
- Run focused compile/tests or smoke checks for touched paths.
- Check `git diff --check`.
