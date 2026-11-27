# B47 compute note 4 (2026-11-19 ~18:05 UTC; no design change)

1. **Box GPUs became free (17:54 / 17:58 UTC)** and the two B47 workers claimed GPU 6 (m1) and GPU 0 (m2). Before any
   command was queued on the box, the PACE final jobs that were to move were cancelled while still PENDING (squeue-
   checked by job name, both segments): fullgen 8,000 s123, gen 8,000 s123 / s456, gen 800 s123, scratch 800 s123 /
   s456. gen 800 s456 had started on PACE and stays there; fullgen 8,000 s456, fullgen 800 s123 / s456, fullcls 800
   s123 / s456 and the two exploratory controls stay on PACE. No configuration runs in both places.
2. **GPU 6 lost to an unclaimed process.** Seconds after m1's claim, a process outside the claim directory (another
   user's, 68.8 GB) started on GPU 6; the four B47 runs m1 popped died of CUDA OOM within the first training steps
   (rc 1, no output, no checkpoint, nothing read). m1 was stopped (its own process group) and its claim removed; the
   four commands were re-queued unchanged for m2 (GPU 0), ordered so that at most one full fine-tune shares the GPU
   with one LoRA run.
3. **Micro-batches on the box:** LoRA runs use --mbs 4 and the full fine-tune --mbs 8 (batch 8 in every case; the
   registered recipe), so that one full fine-tune and one LoRA run fit in 80 GB. Seeds 42 (PACE) used --mbs 8; the
   difference is gradient accumulation order only.
4. **Moved back (18:3x UTC).** The PACE queue was then starting jobs within minutes while the shared GPU 0 ran at
   about a third of an H200's speed, so the three box commands that had not started (fullgen 8,000 s123, scratch 800
   s456, gen 8,000 s456) were taken off the box queue (verified not started) and submitted to PACE. The box keeps gen
   800 s123 and gen 8,000 s123 (running); scratch 800 s123 finished there.
