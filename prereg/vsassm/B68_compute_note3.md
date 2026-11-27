# B68 compute note 3 (2026-11-23 07:35 UTC; second scope cut by the registered cut order; no bar, rule or prediction changed)

At 07:33 UTC 194 of the dev runs had finished; the arena-A key-value bottleneck runs (dkvb, 300-550 s each in B42)
lowered the rate to about 6.5-7 runs per minute. Projection with compute note 2's scope: arena-A dev done near 08:00
UTC, the last test run near 08:50 UTC, the second scorer and verdict near 09:00 UTC, leaving under 45 minutes for the
sample reading and the notes inside the time box.

**Cut: phi2L2_B and phi2L4_B**, the next two cells in the registered cut order (after the five arena-C cells of compute
note 2). Their 128 dev lines were removed from q/tune at 07:34:50 UTC (q/tune.before_cut3 keeps the original); none had
started (0 dev outputs). They are reported as "not run". Arena B has no headroom by construction (accuracies only),
and qwen8L0_B, qwen8L2_B, qwen8L4_B and phi2L0_B still run. Prediction P10 ("every arena-A and arena-B cell completes")
therefore fails; the other predictions are unchanged.

Remaining scope: all six arena-A cells, four arena-B cells (Qwen3-8B L0 / L2 / L4, Phi-2 L0), one arena-C cell
(Qwen3-8B L0). No test output of any cell exists at this time (no finals released).
