# B42 compute note 1 (2026-11-18 about 22:30 UTC; no design change, no bar or prediction changed)

1. **Tuning of arena C ran on GPUs** (A800 GPUs 1, 2 and 5, 22:05-22:21 UTC): 120 tuning runs and the 4 G1 runs, all
   complete, no failed run; selections made by `b42_select.py` as registered (out/select.json).
2. **No GPU was idle afterwards.** At 22:22 UTC every usable GPU (0-2, 4-7) was held by the B40 and B46 workers (their
   locks in the shared claim directory), and B42 uses idle GPUs only. The two B42 GPU workers waiting for the arena-C
   finals (fC1, fC2; still in their claim loop, holding no GPU and no lock, no job started) were stopped.
3. **The arena-C finals run on the box CPU instead** (`b42_cpuworker.sh`: 8 lanes x 6 threads, the same queue file and
   flock, `--device cpu`; `b42_arms.py` now records "cpu (N threads)" instead of a GPU name). Same code, arguments and
   seeds; only the device differs. Consequences, stated before any final output is read: (a) the finals are not
   bit-identical to what a GPU would give; the dev-selected configurations are unchanged, and each final seed still
   picks its epoch on its own dev; (b) the rerun binding-loss map (bind, which also carries the post-hoc readings)
   will differ slightly from B31's GPU numbers, so its plain reading is compared with B31 as a consistency check, not
   as an exact reproduction. CPU occupancy goes to logs/cpu_ledger.txt and is reported as core-hours beside the
   GPU-hours.
4. **Arenas A and B:** the tuning queue (252 runs, including G0 and G1 for A and B) is served by three GPU workers that
   wait for idle GPUs. If no GPU has freed by the time the arena-C finals end, the A/B queue is also served by the CPU
   worker. The registered cut-off for A and B (2026-11-19 14:00 UTC) is unchanged.
