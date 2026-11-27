# B27 amendment 3: back to the registered fine-tune protocol (withdraws amendment 2)

Date: 2026-11-17, about 12:15 UTC. Written before any LoRA selection and before any LoRA test run. The only test
numbers read so far are the G0 gate reproductions of published arms (all 27 cells exact so far).

## What was seen (training and dev splits only)

- **Amendment 2's tuning runs** (raw head, rank 16, cosine over 30 epochs) were stopped after 8-14 epochs. Every one
  still predicted a single class:
  - A at lr 1e-3 (11 epochs) and at 1e-4 (8 epochs);
  - B at 1e-3 (13 epochs);
  - C at 1e-3 (14 epochs).
- **Diagnostic on B** (out/smoke/diag_B_*; full 1,000-instance training split, 2 epochs, seed 42, constant rate):
  - bf16, all linear layers, batch 8, lr 2e-4: training accuracy 0.957 and dev 0.962 after one epoch (0.978 and
    0.958 after two). This is the registered design, apart from the schedule, at a rate inside the registered grid.
  - bf16, attention only, batch 2, lr 2e-4, weight decay 0.01: one class after 2 epochs.
  - The two fp32 controls were still running when this was written; they are reported in the verdict.
- **Reading:** the fine-tune can learn B quickly. What failed was the rate: 1e-3 is too high for this setup, and
  amendment 2 dropped exactly the two rates of the registered grid (3e-4 and 3e-5) that sit nearest the working one.
  Amendment 2 is withdrawn.
- On A the registered design (raw head) sat at one class for 5-6 epochs at lr 1e-3, 3e-4 and 1e-4 (the stopped
  pre-amendment runs). Whether A, or C, can be fitted is what the registered stages are for.

## Change

- **The fine-tune protocol is the registered one** (pre-registration, Part 2):
  - raw last-token head; all linear layers; bf16; AdamW (weight decay 0); 6% warm-up, then cosine;
  - stage 1: rank 16, lr in {3e-5, 1e-4, 3e-4, 1e-3}, 12 epochs;
  - stage 1b (edge rule), stage 2 (ranks 4 and 64 at the selected rate), stage 3 (30 epochs on the configuration
    closest to fitting, if none fits);
  - the registered fit, selection and tie rules (b27_select.py lora).
- **The only difference:** the training split is scored every 3 epochs and at the last epoch, to save GPU time (no
  effect on training). The fit criterion uses the scored epochs.
- All stage-1 runs start from scratch. Nothing from amendments 1 and 2 enters a selection.
- **Budget check:** about 8.5 GPU-hours have been spent so far, and the label-free tuning and finals need about 8.5
  more. The registered LoRA stages are projected at 9-17 GPU-hours, so the total is 26-34, under the cap of 40. If
  the projection passes the cap, the registered cut order applies: first stage 2's rank 4, then the non-primary
  label-free families' finals.
