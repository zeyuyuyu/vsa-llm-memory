# B31 exploratory run 1 (2026-11-18 ~07:20 UTC; registered as EXPLORATORY before it runs; no registered bar applies)

**Why.** The registered naming head reads Qwen3-8B's FINAL layer (the protocol kept from Phi-2). The registered test
outputs of the naming-head finals (read after their dev-selected epochs were fixed; nothing was changed because of
them) show that on Qwen3-8B the head misnames many decimal fillers (arena A: 42% of context fillers; the candidate:
52% in A, 23% in B), mostly as nearby numbers (222 -> 226, 105 -> 10, 26 -> 226; b31_namediag.py, hand-read in
out/handread_qwen8_*_pre.txt). Qwen's tokenizer splits every number into single digits, whereas Phi-2 has one token for
most numbers; a final layer tuned for next-token prediction may not keep exact digit identity. This run asks whether an
intermediate layer names the symbols better. It is reported beside the registered result, never instead of it.

**Design (fixed now).**
- Features: span means of `hidden_states[L]` (the output of block L of 36) for L in {9, 18, 27}, train / dev / test of
  all three arenas, same spans, same instances (`b31_prep.py feats_layers`).
- Arm: the registered naming head exactly as in the finals (perm, named, every mention, 4 epochs, AdamW 3e-4), seeds
  42 / 123 / 456 (`b31_arms.py --enc_model qwen8L<L> --arm name --mode final`).
- Layer selection: per arena, the layer with the highest mean dev accuracy (perm/named) over the three seeds at each
  seed's dev-selected epoch; ties to the later layer. Test is reported for the selected layer (all three layers' test
  numbers are logged and listed in the verdict as exploratory).
- Readings, if the selected layer's naming head reaches >= 0.95 in arenas A and C: the Qwen3-8B shortfall of the
  registered head is a read-out choice (which layer is read), and the paper may say so as an exploratory observation
  with the layer chosen on dev. Otherwise the shortfall stands as measured.
- Compute: one GPU, about 0.3 GPU-hours, inside the B31 cap.
