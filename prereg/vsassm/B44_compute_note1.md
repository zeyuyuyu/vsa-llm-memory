# B44 compute note 1 (2026-11-18 ~22:50 UTC; no design change, no bar or prediction changed)

1. **No A800 GPU is free.** At 22:15 and 22:40 UTC all seven usable GPUs of the box were claimed by B40 / B42 workers
   (GPU 3 is never used). B44 therefore runs every GPU job on PACE (free tier gts-nimam6) and uses the box only for
   CPU work: the data self-test, the tokenisation cache, the binding-loss map at initialisation (b44_xtalk.py init, CPU)
   and the tokenisation checksums. The imposed sweep (b44_xtalk.py sweep) ran on the local machine's CPU.
2. **Tokenisation on PACE.** Copying the 274 MB cache from the box ran at ~80 kB/s, so PACE built its own cache with the
   same b44_tok.py (job 15837356, CPU). Content checksums (b44_tokmd5.py: token ids, spans, labels, targets, text md5)
   are identical on the box and on PACE for every file (std / ho train, dev, test; k-sweep splits).
3. **GPU type.** PACE ignored the partition list of the inference job script and placed the first feature job on a
   V100 (no bf16). It was cancelled after 46 s, before any feature was written, and every inference job now carries
   `-C H200|H100|A100|L40S` (bf16-capable). The LoRA jobs carry `-C H200|H100|A100-80GB` as in B31. Jobs on frozen
   features (naming head, exact rows, probe) and the Phi-2 k-sweep (fp16, as B27 / B31) may run on any GPU.
4. **Smoke dependency released.** The tuning runs were first made to wait for the smoke job; since every PACE job
   waits in the same queue, the dependency was removed (the 800-instance runs exercise the same code path end to end
   within ~30 minutes). The 80,000-instance runs checkpoint at every evaluation point, so a late bug costs at most one
   segment.
5. **Phi-2 on PACE.** The box's snapshot (microsoft/phi-2 810d3678) was downloaded into the PACE cache for the Fig. 2c
   sweep (refs/main set to that revision).
6. **Shorter PACE jobs (backfill).** PACE's start estimates for 3-hour jobs were days away (A100 11-26, H200 12-04,
   H100 12-08), while jobs cut to 1 h started within minutes. The pending 8,000- and 80,000-instance chains were
   cancelled before starting and resubmitted with --max_hours 1.3 and a 1:30 time limit per segment (3 segments for
   8,000, 5 for 80,000; a segment that finds its run finished exits). The 800-instance runs, the smoke and the short
   inference jobs had their time limits lowered to 1:00-1:30 in place. The 80,000-instance feature sets are computed
   in four contiguous chunks per split (b44_feats.py --chunk / --nchunks; b44_name reads the chunks in order), one
   short job each. Nothing about the runs themselves changes.
7. **CoT test runs submitted before the dev selection finished.** The prereg scores all four prompting arms on the full
   std test regardless of the dev selection (which only decides the arm also run on the ho test), so the four std-test
   runs were queued at 23:45 UTC as resumable chains (b44_cot.py keeps finished batches in <out>.part). The dev100 runs
   and the selection are unchanged.
8. **800-instance selections** (seed 42, dev only; b44_out/select.json): lora std lr 3e-4 (dev 0.716 vs 0.516); lora ho
   lr 1e-4 (0.526 vs 0.526, tie -> smaller rate); scratch std lr 1e-4 (0.998 vs 1.000, within 0.004 -> smaller rate;
   both runs stopped early at saturation). Finals (seeds 123 / 456; scratch ho seeds 42 / 123 / 456 at the scratch-std
   rate) run with --final, which scores the test split once from each run's dev-best state right after training; the
   seed-42 tuning runs of the selected configurations are scored once by b44_scoretest.sh.
9. **A800 GPUs became free (23:50 UTC; B42 finished, only B40's GPU 0 claimed).** Four b44_worker.sh workers claimed
   GPUs 1, 2, 4 and 5 by the shared claim protocol. Before they started, the PACE jobs they replace were cancelled
   while still pending (none had started): the feature chunks and frozen-feature arms (now on the box: features of
   std / ho in one piece, then exact rows, naming head and probe), the nine 800-instance LoRA finals, and the four CoT
   std-test chains. The box GPU-hours are in /path/to/work/b44/logs/gpu_ledger.txt and are added to the PACE ledger.
   The seed-42 test scorings of the 800-instance tuning runs stay on PACE (their saved states are there).
10. **Two more box workers (GPUs 6, 7; 23:55 UTC)** took the 8,000-instance tuning runs of lora ho and scratch std
    (both rates, seed 42); their four PACE chains were cancelled while pending. PACE keeps the lora std tuning runs
    (8,000 running, 80,000 lr 1e-4 running, 80,000 lr 3e-4 pending). The box workers release their GPUs when their
    queues are empty.
11. **8,000-instance selections** (dev only): lora std lr 1e-4 (0.800 vs 0.506; at 3e-4 the run stays at chance);
    lora ho lr 1e-4 (0.804 vs 0.804, tie -> smaller rate), so lora ho at 80,000 runs at 1e-4 (rule). CoT dev100:
    zs-nothink 0.92, fs-nothink 1.00, zs-think 0.99, fs-think 1.00 -> fs-nothink (ties: thinking off first); it is
    the arm run on the ho test. The PACE dev100 job hit its time limit during fs-think and was resumed from its .part
    file (14 min) before the selection was written; the fs-nothink ho-test run had been queued at 01:00 UTC because no
    fs-think dev result could exceed fs-nothink's 1.00.
12. **Placement at 01:45 UTC.** PACE: lora std 80,000 tuning (both rates, resumed segments), lora std 8,000 finals,
    lora ho 8,000 finals, lora ho 80,000 seeds 42 / 123. Box: lora ho 80,000 seed 456 (its PACE chain cancelled while
    pending; the box worker waits for a free GPU: B40 re-claimed GPU 2 when B44's ho feature worker released it).
13. **80,000 lora std selection** (dev): lr 1e-4 (0.804 vs 0.506; at 3e-4 the run never leaves chance), finals queued
    on PACE. **8,000 scratch std selection:** lr 1e-4 (best dev 1.000 at both rates; tie -> smaller rate; fixed once the
    1e-4 run reached 1.000, so the dependent runs were queued before that run's early stop): scratch std 80,000 and
    scratch ho 80,000 (seeds 42 / 123 / 456, lr 1e-4) on PACE; scratch std 8,000 finals and scratch ho 8,000 on the box.
14. **Probe at 80,000 (cut 2) is run**, since the projected total (about 70 GPU-hours at 03:20 UTC) stays below the cap;
    it runs on the box after the lora ho 8,000 seed-42 test scoring.
15. **Session restart (about 04:50 UTC).** The parent session restarted; the running jobs survived. At 05:00 UTC the
    next PACE segments of lora std 80,000 seed 123 (2,500 of 10,000 steps done) and lora ho 80,000 seed 123 (5,000
    done) had been pending for over an hour. Their chains were cancelled while pending (squeue-checked) to restart
    them on the A800, but by then another user's jobs (not using the claim directory) occupied GPUs 0, 2, 4 and 6, so
    the two box workers never claimed a GPU; they were stopped (by PID, before claiming), and the two runs were
    resubmitted to PACE as new chains (15848676-680, 15848681-684) that resume from their own checkpoints. No run
    executed twice.
16. **Second session restart; PACE unreachable (2026-11-19, 17:35 UTC).** The parent session was down for about 12 h.
    On resuming, PACE could not be reached (GlobalProtect VPN down), so the PACE outputs written after the last pull
    (05:00 UTC) cannot be read: the scratch ho 80,000 runs (seeds 42 / 123 / 456; finished on PACE 05:07-05:23 UTC, logged
    test 1.000 / 0.998 / 1.000 in the job logs read at the time) and whatever the pending PACE segments of lora std
    80,000 (seeds 123, 456), lora ho 80,000 (seeds 42, 123) and scratch std 80,000 (seed 42) did afterwards. Runs whose
    PACE twin is known to have finished are not rerun (scratch ho 80,000). The five runs whose PACE state is unknown were
    queued on the box as fresh runs (workers r80A-E, one run each); at 17:40 UTC only one GPU was free (r80A: scratch std
    80,000 seed 42); B43 and B48 hold the others. Rule: a lora 80,000 run that has not started by 18:30 UTC cannot finish
    before the report deadline, so its worker is stopped (by PID) and the cell is reported as incomplete; if PACE comes
    back, a finished PACE twin is used instead of a box run. The A800 box holds: lora ho 80,000 seed 456 (finished
    06:21 UTC, test 0.810), the CoT test runs (all finished).
17. **PACE reachable again (about 17:55 UTC).** sacct shows scratch std 80,000 seed 42 finished on PACE at 07:53 UTC
    (job 15843010), so the box run of the same configuration (r80A, started 17:37 UTC) was a duplicate: it was stopped
    after 16 minutes, before its first evaluation point, and nothing from it is used. The four waiting box workers
    (lora std 80,000 seeds 123 / 456, lora ho 80,000 seeds 42 / 123) were stopped before claiming a GPU, because their
    PACE chains are alive (checkpoints at 5,000 / 8,750 / 7,500 / 5,000 of 10,000 steps; next segments pending).
    The finished PACE outputs (scratch std and ho at 80,000) were copied back.
18. **Free-tier billing limit.** The pending PACE segments of the four unfinished lora 80,000 runs were blocked with
    reason AssocGrpBilling (the free-tier account gts-nimam6 reached its billing limit, as B47 recorded). Changing their
    account in place failed (time-limit validation), so the chains were cancelled while pending and resubmitted on the
    paid account gts-nimam6-paid (b44_pace_paid.sbatch; 15875401-413), resuming from the same checkpoints (std seed 123
    at 5,000 / 10,000 steps, std seed 456 at 8,750, ho seed 42 at 7,500, ho seed 123 at 5,000). Spend is reported from
    sacct in the verdict. Leftover chain segments of finished runs were cancelled. If they do not finish by about 22:30 UTC
    the 80,000 lora cells are reported as incomplete, as registered.
19. **Paid chains cut to 1 h segments (18:00 UTC)** for backfill. The runs keep --max_hours 1.3, so a segment may be
    stopped by Slurm before its graceful exit; it then resumes from its last checkpoint (written every 1,250 steps),
    losing at most the steps since that checkpoint. Nothing else changes.
20. **All registered cells finished (checked 2026-11-20 02:40 UTC).** The four paid PACE chains completed (lora std
    80,000 seeds 123 / 456, lora ho 80,000 seeds 42 / 123); every lora and scratch cell at 800 / 8,000 / 80,000, std and
    ho, has three seeds. Total about 67.5 GPU-hours (PACE 34.6: free 28.1 + paid 6.5; A800 32.9), below the cap of 80.
