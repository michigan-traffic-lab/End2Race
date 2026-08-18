# Fixed Training and Qualification Procedure

Train one model at a time with the fixed settings in `imitation/train.py`. Training and
evaluation remain separate; monitor and sequence the pipeline externally.

## Prerequisite

Use the complete Austin collection before training. It contains 720 scenarios:
80 start points, three opponent racelines, and three opponent speed scales.
Training uses only collision-free CSV demonstrations in `dataset/success/`.

## Fixed Training Settings

Every model uses exactly:

- 500 epochs;
- one checkpoint at epoch 500, producing `epoch_00500.pt`;
- batch size 1024;
- learning rate `1e-4`.

These settings are constants, not sweep dimensions. Do not run a hyperparameter
sweep and do not add evaluation or qualification logic to `imitation/train.py`.

## Qualification Order

1. Train one model and save its checkpoint at epoch 500.
2. Evaluate one collision-free lap in this order: Austin, Hockenheim,
   MoscowRaceway, then Nuerburgring. Apply a 90-second wall-clock timeout to
   each map externally; a timeout fails that map.
3. Stop the checkpoint evaluation at the first failed single-agent lap; do not
   run later maps or the multi-agent suite.
4. If all four single-agent laps pass, evaluate the checkpoint on Austin's 720
   multi-agent scenarios.
5. Qualify the model only when all 720 scenarios complete without worker errors
   and overall safety (`success_percent`) is strictly greater than 80%.

A single-agent pass means completing one full loop without a collision.

## Artifact Policy

Retain only models that pass all four single-agent maps and the complete
720-scenario multi-agent safety gate. Save each qualified checkpoint and a
metrics record containing all four single-agent results and the complete
multi-agent `results.json`. Discard checkpoints that fail either gate.

Every checkpoint is weight-only and contains `model.state_dict()`. Do not save
optimizer state, training state, or resume metadata.

An interrupted evaluation or worker failure is not a model result. Resolve the
runtime problem before retaining or discarding that checkpoint.

Report experiment progress conversationally. Do not create or maintain a
separate training-status document.
