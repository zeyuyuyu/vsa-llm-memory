# B68 compute note 1 (2026-11-23 about 07:08 UTC; launch; no design change, no bar or prediction changed)

1. **Pre-registration** committed (eef5596) and OpenTimestamps-stamped (a654a98) at 07:05 UTC, before any dev run.
2. **Dev phase launched 07:05:50 UTC** on the A800 box: q/tune holds the 1,152 registered dev lines (18 cells x 64, in
   the registered priority order: A, then B, then C; within an arena Qwen3-8B L0, L4, L2, then Phi-2 L0, L4, L2). Six
   workers (`b68_worker.sh 7 2`), one per GPU, claimed GPUs 1, 2, 4, 5, 6 and 7 at 07:06:10-07:06:16 UTC, each idle at
   claim (memory.used 1 MiB, no compute process; GPU 0 held 18.6 GB of another user's job and GPU 3 4.5 GB; neither is
   ever used). 7 lanes x 2 threads per GPU; q/final (empty until a batch's dev choices are committed) is served first.
3. Before registration the six feature workers (06:53:59-06:58:09 UTC) and four check workers (06:57:54-07:03:10 UTC)
   held GPUs; they are in logs/gpu_ledger.txt and count toward the GPU-hours.
4. Progress is followed with `b68_progress.py` (counts of finished files and wall times only; no accuracy is printed).
