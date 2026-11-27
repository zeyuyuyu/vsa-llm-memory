# B27 amendment 4: the learning-rate schedule becomes a tuned factor of the fine-tune

Date: 2026-11-17, about 12:25 UTC. Written before any LoRA selection and before any LoRA test run (the only test
numbers read are the G0 gate reproductions, all exact).

## What was seen (training and dev splits only)

- **Registered stage 1 on B** (amendment 3; bf16, all linear layers, batch 8, 6% warm-up then cosine over 12 epochs):
  one class at lr 3e-4 and 1e-4 after 6 epochs, and at lr 1e-3 after 5.
- **Constant-rate diagnostics on B** (full training split, seed 42, lr 2e-4, no warm-up):

  | precision | layers | batch | weight decay | after epoch 1 (train / dev) | after 2 epochs |
  |---|---|---|---|---|---|
  | bf16 | all linear | 8 | 0 | 0.957 / 0.962 | 0.978 / 0.958 |
  | fp32 | attention + dense | 2 | 0.01 | 0.988 / 0.980 | (published v15 configuration on the B27 split) |
  | bf16 | attention + dense | 2 | 0.01 | one class | one class |

- **Reading:** leaving the one-class plateau depends on the schedule and partly on chance. This fits the published
  fine-tune's spread across seeds on B (0.980, 0.895, 0.510). The constant schedule of the published fine-tune
  escaped in 2 of 3 diagnostic runs; the registered warm-up + cosine schedule escaped in none of 3. A tuning grid
  without the published schedule would not be a fair tuning of the published baseline.

## Change (Part 2 only)

- **Stage 1** is the registered lr grid {3e-5, 1e-4, 3e-4, 1e-3} at rank 16 and 12 epochs, crossed with schedule in
  {cosine (registered: 6% warm-up, cosine to 0), constant (the published fine-tune: no warm-up, constant rate)}.
  That is 8 configurations per arena. The cosine half is the amendment-3 queue, unchanged.
- **Selection:** the registered fit, selection and tie rules over all 8 configurations. Ties go to rank 16, then the
  smaller rate, then the cosine schedule.
- **Later stages** (1b edge rule, 2 rank, 3 extension) apply within the selected schedule, as registered.
- **Budget:** stage 1 now projects to about 10 GPU-hours. If the cap is approached, the registered cut order applies
  (stage 2's rank 4 first, then the non-primary label-free finals), then stage 2's rank 64.
- Everything else is unchanged.
