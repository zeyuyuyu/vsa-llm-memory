# V-157 AMENDMENT 1 — scheduling only (2026-10-15, after seed 0, before seeds 1–2 finish)

**What changed.** The prereg ran three seeds in sequence on GPU 6. Seed 0 took 191 min, 135 of them on the N128_H4
cell, so seeds 1 and 2 would have taken about 6 more hours. The user asked to speed things up. GPU 7 had become free.
- The sequential launcher (PID 3822028) and the seed-1 process (PID 4096420, about 22 min in, cells N4/16/64_H1
  done) were stopped.
- Seeds 1 and 2 now run split by cell as six processes: N128_H4 alone; N4/16/64 at H4; all N at H1–H2. They are
  spread over GPUs 6, 7, 1 and 4 (`run_v157_split.sh`).

**What did not change.** Nothing that can move a number:
- b3.py (md5 9d32c789), the seeds, per_cell 50, batch size 8, the grid, the bars and the frozen scorer.
- Each cell's instances come from their own RNG, `random.Random(7000003*seed + 1009*N + H + 17)`, so the split
  produces exactly the instances the sequential run would have.
- v157_score.py pools by cell across files and asserts 150 per cell. It is run on
  v157_s0.json + the six v157_s{1,2}_*.json files.

**Bonus determinism check.** The stopped seed-1 run had finished N4_H1, N16_H1 and N64_H1 on GPU 6. That partial file
is kept as `b_logs/v157_s1_PARTIAL_gpu6.json` and will be compared, instance by instance, with the same cells
recomputed on GPU 6. It is not used in scoring.

**What was seen before the amendment.** Seed 0's full output, including its N128_H4 cell: CoT 12/50, shard and dict
22/50. That has no bearing on a scheduling change, and no bar, arm or analysis was altered.
