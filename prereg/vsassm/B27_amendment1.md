# B27 amendment 1: standardised input to the fine-tune's classification head

Date: 2026-11-17, about 10:55 UTC. Written after 5-6 epochs of three stage-1 LoRA runs on arena A (training and dev
splits only) and before any LoRA selection, any LoRA run on arenas B or C, and any LoRA test run. No test number of
any B27 arm had been read when this was written (the only test numbers in existence are gate G0 reproductions of
published arms).

## What was seen (train and dev only)

- Stage 1 on arena A (rank 16; lr 1e-3, 3e-4, 1e-4; 5-6 of 12 epochs each): every run predicted one class for the
  whole training split at every epoch. Training accuracy alternated between 0.4915 and 0.5085 (the two class rates)
  and the running training loss stayed at 0.70-0.78. These are the same symptoms as the published fine-tune's final
  training accuracy of 0.50-0.53.
- Diagnostic (b27_diag_probe.py, CPU, cached frozen states, train and dev only; logs/diag_probe.txt on the box):
  - The frozen final state of the last non-pad token is nearly the same vector for every input. The mean pairwise
    cosine over the training split is 0.976 (A), 0.968 (B) and 0.954 (C), at a norm of about 50-62.
  - A linear head on these raw states (Adam, lr 1e-2, 2,000 full-batch steps) reaches training accuracy 0.735 on A
    (0.995 on C, 1.000 on B).
  - The same head on the states standardised per dimension (training-split mean and std) fits every training split
    (1.000 on A, B and C). Its dev accuracy is 0.502 (A), 0.518 (C) and 0.922 (B).
- Reading: the plateau comes from the readout geometry, not the task. A randomly initialised linear head on a state
  dominated by one shared direction sees almost no input-dependent signal at these learning rates. A fine-tune that
  cannot fit its training split for this reason does not answer the referee, who asked for one that fits.

## Change

- The classification head reads the last-token state standardised per dimension: (h - mu) / (sd + 1e-4).
- mu and sd are computed once per run, before training, over the training split, from the initial model. At
  initialisation LoRA's B matrices are zero, so this is the frozen base model. They are fixed buffers and not trained.
- Nothing else changes: the grid, stages, selection and stage-3 rule, final runs, bars, readings and predictions all
  stay as registered.
- All stage-1 configurations are rerun with this head (out/lora_tune_a1/). The three stopped runs are reported as the
  pre-amendment record (logs/lora1{a,b,c}.lane1.log) and are not used for any selection. Their 1.2 GPU-hours count
  towards the cap.

## Side observation, recorded now (no bar depends on it)

- The standardised linear probe on B's frozen last-token state reaches 0.922 on B dev. The paper's floor probe on B
  (0.815 on test, raw state, 2 epochs) is therefore a weak floor.
- The verdict reports this beside the B results. The registered bars keep the banked floor.
