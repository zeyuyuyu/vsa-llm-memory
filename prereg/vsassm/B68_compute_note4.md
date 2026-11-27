# B68 compute note 4 (2026-11-23 08:12 UTC; correction of compute note 3 and a third cut; no bar, rule or prediction changed)

**Correction.** Compute note 3 called phi2L2_B and phi2L4_B "the next two cells in the registered cut order"; they were
not. The registered cut order is the reverse of the priority order, so after phi2L2_C, phi2L4_C, phi2L0_C, qwen8L2_C and
qwen8L4_C (compute note 2) the next cell was qwen8L0_C, and only then phi2L2_B, phi2L4_B. Compute note 2 had kept
qwen8L0_C on purpose (to test the C-by-construction reading at one source), which already departed from the order.

**Cut: qwen8L0_C**, which restores the registered order (arena C cut entirely, then Phi-2 B L2 and L4). Its 64 dev lines
were removed from q/tune at 08:11:33 UTC (q/tune.before_cut4 keeps the original); none had started (0 dev and 0 G1
outputs). The decision was made on throughput alone (at 08:11 UTC 169 dev lines still queued, qwen8L4_B 50 of 64 done);
no test output of any cell had been read.

**Final scope:** arena A, all six cells (Qwen3-8B and Phi-2, L0, L2, L4); arena B, four cells (Qwen3-8B L0, L2, L4;
Phi-2 L0). NOT RUN: all six arena-C cells and phi2L2_B, phi2L4_B. P6 (arena C) is therefore not tested; the
C-by-construction reading rests on the training-row feature hash (0.52 at L0 on both models, prereg) only.
