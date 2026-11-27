# B31 compute note 4 (2026-11-18 ~11:05 UTC; no design change)

1. **The three PACE jobs of compute note 2 ran** despite the `--test-only` estimates: backfilled on one
   A100-SXM4-80GB node (atl1-1-01-002-28-0), 07:57-08:47, 08:02-08:52 and 08:48-09:38 UTC, exit code 0, 50 min each
   (2.52 GPU-hours, free tier gts-nimam6, about USD 1.06). Their outputs (qwen8_A_id0_s{42,123,456}.json; same code,
   arguments and model snapshot as the A800 runs) were copied to the box; logs in b31_out/pace/. They never ran on the
   A800 (their lines had been removed from its queue before submission).
2. **Because PACE backfills short jobs quickly, the remaining Qwen3-8B phase-3 jobs go to PACE:** the scheduler
   starts no new Qwen worker on the A800 (B31_MAXHEAVY=0; the two running Qwen workers and the ex1 worker finish what
   they hold), and `b31_pace_bridge.sh` (local) moves each new line of q/p3_heavy to PACE (one sbatch per line, no
   retry; a failed submission is logged for a manual `squeue` check) and copies finished PACE outputs back to the box.
   The first two (Qwen arena-A LoRA + identity finals, seeds 123 and 456) were submitted at 11:00 UTC (jobs
   15813765-15813766). Phi-2 jobs stay on the A800 (the model is not cached on PACE).
3. The A800 footprint falls as the Qwen jobs drain (at most the two Phi-2 workers remain), which is what the main
   session asked for. GPU time on PACE is added to the ledger in the verdict.
