# B68 compute note 5 (2026-11-23 08:31 UTC; test runs finished; written before any test output is read)

1. **Runs.** Dev: 640 commands (10 cells x 64), 07:05:50-08:24 UTC. Test: 180 per-seed finals (10 cells x 6 families x
   3 seeds) and 10 naming-head processes (3 seeds each), started 07:43:15-08:26:04 UTC, last finished by 08:28:20 UTC.
   Every command exited with rc 0 (b68_out/compute_summary.txt). Each cell's finals were queued only after its dev
   choices were committed (prereg/B68_dev_choices.md: batches A1 07:43, A2 07:57, A3 07:58, A4 08:14, B1-B4 08:15-08:26
   UTC); each final ran once.
2. **Scope** as compute notes 2-4: arena A, six cells; arena B, four cells (qwen8L0/L2/L4_B, phi2L0_B); not run:
   arena C (six cells), phi2L2_B, phi2L4_B.
3. **Two early reads of test outputs by programs, not by me:** at 07:58 UTC `b68_queues.py merge` and the second scorer
   (b42_indep.py) were started on the qwen8L0_A and qwen8L4_A finals to save time; their output files were not opened.
   b68_post.sh reruns both on every cell.
4. **Workers stopped** at 08:28:59 UTC (lane subshells killed by PID after q/final and q/tune were empty and no job ran;
   each worker then wrote its ledger end line and removed its own claim lock; the claim directory is empty, GPUs 1, 2, 4,
   5, 6, 7 back at 1 MiB). GPU 0 and GPU 3 were never used.
5. **Compute:** 8.77 A800 GPU-hours (logs/gpu_ledger.txt, copied to b68_out/gpu_ledger.txt: 0.28 feature extraction,
   0.21 checks and self-tests, 8.28 dev and test workers, 6 GPUs x 1.38 h); 55.2 lane-hours of commands (dev 43.7, test
   11.1, naming heads 0.4). No PACE, no paid API.
