# B47 compute note 1 (2026-11-19; no design change, no bar or prediction changed)

1. **Registration and timestamp.** The pre-registration and run code were committed in a1bf128 at 05:26 UTC (the file's
   header says "about 05:55", written before the commit; the commit time is the binding one) and stamped with
   OpenTimestamps (prereg/B47_..._prereg.md.ots, SHA-256 82d27d38...a0a98; committed in 6e2afec) before any B47 test
   run. The analysis code (verdict, second scorer, bootstrap, Fig. 2b,c assembly, hand-reading) was committed in d4d60f9
   at 05:37 UTC, before any B47 test output was read.
2. **Free tier exhausted -> paid tier.** The first submissions (05:27 UTC, account gts-nimam6) did not start: the CPU
   sweep jobs were held with reason AssocGrpBillingMinute (the free tier's monthly billing limit) and the GPU jobs stayed
   in the queue. Three consistency-sweep parts had run on the free tier before the limit. At 05:33 UTC every pending
   B47 job was cancelled (by job id, squeue-checked) and resubmitted unchanged under gts-nimam6-paid (the GPU smoke job
   was not resubmitted; the tuning runs exercise the same code). Spend is reported in the verdict's compute section.
3. **Wider GPU constraint refused.** An attempt to let the LoRA jobs (gen, scratch; ~30 GB) also run on L40S / A100-40GB
   (scontrol update of the pending jobs) was refused by Slurm (time limit invalid for that partition); nothing changed.
4. **8,000 full fine-tune grid queued early.** The fullgen grid at 8,000 (lr 3e-6 / 1e-5 / 3e-5, seed 42) does not
   depend on any 800 selection (rank plays no role in a full fine-tune), so it was queued at 05:37 UTC (3 segments of
   <= 1.5 h each). The gen grid at 8,000 uses the rank selected at 800, as registered (see item 7).
5. **Naming head at 8,000 (secondary) queued** on any GPU (frozen fp16 features; b44_feats.py and b44_name.py unchanged):
   job b47_name8k.
6. **A800 box.** At 05:10 and 05:36 UTC every usable GPU of the box held a process (four claimed by B44 / B50 workers,
   three by other users' processes); B47 uses no box GPU. The box served only as the source of B31's saved outputs
   (read-only copy to b47_out/src_b31, *.pt excluded).
7. **8,000 gen grid queued at both ranks (05:47 UTC).** To avoid a second queue wait after the 800 selection, the gen grid
   at 8,000 (lr 3e-5 / 1e-4 / 3e-4) was queued at rank 8 and at rank 64. Only the runs at the rank selected at 800 (the
   registered rule) enter the selection and the finals; the other three are reported as run and not used, and no
   test split is scored for them.
8. **Queue (05:45 UTC).** PACE GPU queues were long (about 900 pending jobs on gpu-h100 / gpu-h200, 580 on gpu-a100);
   no B47 GPU job had started. The time limit of the pending 800-instance jobs was lowered from 1:30 to 0:50 (scontrol;
   backfill), which those runs need at most (a LoRA run checkpoints at every evaluation point and resumes in its second
   segment; a full fine-tune at 800 takes well under 0.5 h on these GPUs). Adding the RTX PRO 6000 Blackwell partition
   was refused by Slurm. Nothing about the runs changes.
