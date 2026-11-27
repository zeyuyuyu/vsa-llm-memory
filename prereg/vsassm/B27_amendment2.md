# B27 amendment 2: the fine-tune's schedule (withdraws amendment 1)

Date: 2026-11-17, about 11:35 UTC. Written before any LoRA selection, any LoRA run on arenas B or C, and any LoRA test
run. The only test numbers read so far are the G0 gate reproductions of published arms (all exact).

## What was seen (training and dev splits only)

- **Amendment 1 (fixed standardisation of the head input) did not remove the plateau.** On A, after 4 epochs at lr
  1e-3, 3e-4 and 1e-4, training accuracy still sat at the class rates (0.4915 / 0.5085), and the running loss rose to
  0.90-1.90 (from 0.70-0.78 with the raw head). The likely reason is that LoRA moves the shared component, so the
  init-time standardisation becomes stale. Amendment 1 is withdrawn, and its runs are not used for any selection.
- **Readout smokes on 512 A training instances (8 epochs, rank 16; out/smoke/lora4_*, lora5_*):** none rose clearly
  above the class rate.
  - Batch-statistics normalisation, lr 1e-4: training accuracy 0.50-0.54. With head lr 1e-2: 0.48-0.57.
  - Batch norm at lr 1e-3 with head lr 1e-2: 0.49-0.53.
  - LP-FT (head first fitted on the standardised frozen states, then LoRA at 1e-4): the fitted head was lost after
    one epoch (loss 6-12, training accuracy 0.51-0.59).
  - Answer-token readout through the pretrained LM head (" no" / " yes"), lr 1e-4 and 1e-3: 0.49-0.55 after 6 epochs.
  - (Frozen-LoRA head-only and lr 1e-5 controls still running when this was written; reported in the verdict.)
- **Earlier smoke (pre-registration):** with the raw head, 32 C instances were fitted (0.97) at lr 1e-3 after about
  16 presentations per instance, and 64 instances were not fitted after 10.
- **Reading:** these labels are not linearly accessible in the initial states (dev at chance for every readout), so a
  fine-tune can fit them only by memorising. Memorising is slow for every readout tried (Zhang et al. 2017 report the
  same for random labels). 12 epochs do not fit A, and the registered grid of 12-epoch runs followed by a 30-epoch
  stage would exceed the 40 GPU-hour cap (A takes about 200 s per epoch on this box).

## Change (Part 2 only; Part 1 is unchanged)

- **Readout:** the registered one (raw final state of the last non-pad token, random linear head), as in the
  pre-registration.
- **Tuning (seed 42, train + dev):** rank 16; lr in {1e-3, 1e-4}, the two ends of the range that fitted in the smokes;
  30 epochs (cosine over 30); dev scored every epoch; training split scored every 3 epochs and at the last epoch.
  - Dropped for budget: lr 3e-4 and 3e-5, the rank stage (4 and 64) and stage 3, which is subsumed by the 30-epoch runs.
  - The paper will state that rank was not tuned.
- **Selection:** the registered rule (b27_select.py lora_a2). Among the configurations that fit (training accuracy
  >= 0.95 at a scored epoch), the highest dev accuracy; otherwise the highest dev accuracy, and the arena's
  fine-tune is labelled "did not fit its training split". Ties within 0.004 go to the smaller rate.
- **Final runs:** the selected configuration, seeds 42, 123 and 456, 30 epochs; test at each seed's dev-best epoch.
- **Bars, readings and predictions:** unchanged. The prediction "fits in all three arenas" now refers to 30 epochs.
- **Budget:** about 7.7 GPU-hours for tuning and 11.6 for the finals. Spent before this amendment: about 5.3
  GPU-hours (label-free tuning running on two GPUs, stopped LoRA runs, smokes).
