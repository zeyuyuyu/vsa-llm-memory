# B31 compute note 2 (2026-11-18 ~07:10 UTC; no design change, no bar or prediction changed)

1. **Request.** At about 07:00 UTC the main session relayed that PACE is reachable again and asked B31 to offload part
   of its queue so that the A800 box can free GPUs for other work (B35).
2. **PACE start estimates** (`sbatch --test-only`, 1 GPU, 3 h, 06:59 UTC): H200 2026-12-04, H100 2026-12-09, A100
   2026-11-23, RTX Pro Blackwell 2026-11-19 19:56 (not usable: the PACE torch build, 2.5.1+cu121, has no Blackwell
   kernels). Every PACE start is one to sixteen days away, later than the A800 would finish the whole queue.
3. **What was done.**
   - The three Qwen3-8B arena-A equalised-LoRA finals (seeds 42, 123, 456; the only Qwen jobs nothing else waits on)
     were taken off the A800 queue and submitted to PACE (free tier gts-nimam6, jobs 15810112-15810114, H200/H100/
     A100-80GB, 3 h each) with the same code, arguments and model snapshot (Qwen/Qwen3-8B b968826d). Rule: if they have
     not started when the A800 Qwen queues run dry, they are cancelled (`squeue` first, `scancel` of these job ids only)
     and put back on the A800 queue, so a job never runs twice.
   - The A800 footprint is capped: the scheduler (b31_orch.sh v2) keeps at most 4 B31 workers on the box, at most 2 of
     them on Qwen3-8B queues; two Qwen arena-C tuning jobs were moved behind that cap. The six GPUs B31 held at 07:00
     drop to four as the running tuning jobs finish (expected from about 07:30 UTC).
   - GPU time on PACE, if any, is added to the ledger.
