# B48 compute note 1 (2026-11-19; no design change, no bar or prediction changed)

1. **Time of registration.** The header of prereg/B48_one_form_and_alias_naming_prereg.md says "about 06:30 UTC"; the
   file was in fact committed at about 05:35 UTC (e88a6a8) and time-stamped with OpenTimestamps at about 05:36 UTC
   (668ee59). No B48 job had started on PACE at that time and no test feature, test output or test score existed.
2. **Self-test (A800 box CPU, 05:38-05:40 UTC; tokenizers only, no GPU, no model).** `b48_prep.py selftest` on both
   tokenizers (out/selftest_{phi2,qwen8}.json on the box): in every split C1 has the same chain, label and mention
   structure as arena C in 800 / 500 / 500 of 800 / 500 / 500 instances, and the dictionary answer equals the label in
   all of them. **C1: 0 of 46,511 (train) / 29,087 (dev) / 29,024 (test) instance-symbols are written with more than one
   token sequence, under the Phi-2 and the Qwen3-8B tokenizer; arena C: 24,481 / 15,385 / 15,412 (53%)**, i.e. in C
   about half of the symbols in every instance appear as two different token sequences, in C1 none. No truncation (C1
   longest 457 Phi-2 / 513 Qwen3-8B tokens). Stories under the Qwen3-8B tokenizer: every mention span decodes to text
   containing the mention (W 74 and P 86 mentions per story; longest W 544, P 289 tokens). (Under the Phi-2 tokenizer,
   which part B does not use, 20 dev and 5 test W mentions do not decode cleanly.)
3. **PACE accounts.** The free tier had no billing minutes left; the six first submissions (15849365-71) were cancelled
   before they started and resubmitted on gts-nimam6-paid (15849510-15). `scontrol update Account=` was refused for
   pending jobs ("time limit invalid").
4. **LM test runs queued behind the dev runs.** The LM-written-names runs on the test split (15849684 W, 15849685 P)
   start only after the dev runs end successfully (afterok). If the dev outputs show a parse problem, they are cancelled
   before any test output is read, and the fix is recorded in an amendment before a test run.

# B48 compute note 2 (2026-11-19 17:40 UTC = 11-20 01:40 box time; no design change, no bar or prediction changed)

5. **Session restart; all B48 work moves to the A800 box.** The working session was interrupted for about 12 hours.
   At 05:50 UTC every B48 PACE job (feature job 15849510, LM dev 15849511-12, LM test 15849684-85 behind the dev jobs,
   tuning 15849513-15 behind the feature job) was still pending (PACE: ~1,180 pending GPU jobs). At 17:35 UTC PACE is
   unreachable (GT VPN down), so whether any of them has since started is unknown. **No PACE output of B48 will be read**:
   every B48 number comes from runs on the A800 box (/path/to/work/b48), which had GPUs 0, 1, 2, 4, 5 and 7 idle (GPU 6
   held by another experiment's claim; GPU 3 never used). When PACE is reachable again, any B48 job still pending is
   cancelled; any output already written there stays unread (if an LM test job ran there, its output is a test output
   that was never read, and the A800 test run is the only one used). Code on the box is md5-identical to the committed
   code (b48_*.py from e88a6a8). GPU occupancy: /path/to/work/b48/logs/gpu_ledger.txt (b48_worker.sh, the shared claim
   protocol of B42).
6. **Deadline (coordinator, 17:35 UTC):** report within 8 hours; anything not finished by then is reported as not run.

# B48 compute note 3 (2026-11-19 ~17:50 UTC; no design change, no bar or prediction changed)

7. **Features and LM dev runs on the A800 (17:37-17:47 UTC).** GPUs 0, 1, 2, 4, 5 claimed by the shared protocol: C1
   features (Phi-2, Qwen3-8B; train / dev / test) and story features (Qwen3-8B layers 9 / 18 / 27 / 36; W, P) were done
   by 17:40 UTC; the LM-written-names dev runs: P 212 s, W ~6 min. **P dev parse check:** 499 of 500 stories have all 28
   lines (one stopped after 16; its 12 missing statements give their mentions keys of their own, as registered), mean 191
   new tokens; no prompt change. The test runs (P, then W) were queued on the same GPU-0 worker after W dev.
8. **Tuning on the box CPU.** After 17:41 UTC GPUs 1, 2, 4, 5, 7 were taken by B43 / B44 jobs. As in B42 (compute note
   1 there), the tuning queues are served by `b48_cpuworker.sh` (copy of b42_cpuworker.sh: the same queue file and flock
   as the GPU worker, so no run executes twice; each command run with `--device cpu`; lanes x threads in
   logs/cpu_ledger.txt). One GPU worker (w_tune1) keeps waiting for an idle GPU and takes lines from the same queues.
   For `b31_arms.py` on the CPU, `b48_run.py` replaces `torch.cuda.get_device_name` (which b31_arms calls only to record
   the device) by a function returning "cpu (N threads)"; no numerical change. CPU and GPU runs are not bit-identical;
   every run records its device, and selection uses whatever each tuning run produced.

# B48 compute note 4 (2026-11-19 ~17:55 UTC; no design change, no bar or prediction changed)

9. **Part-B selection run too early, corrected before any test output existed.** At 17:50 UTC I ran
   `b48_alias.py select` as soon as the part-B tuning queue was empty, but four P tuning runs (layers 27 and 36) were still
   running; the selection therefore saw 12 of 16 runs (all 8 W, 4 of 8 P), and the two head finals were launched from it.
   The P final (PID 3804020) was killed after 2 minutes, before it had written any output (outputs are written only after
   all three seeds); the W final was left running, since its 8-run grid was complete. Rerun on all 16 runs
   (out/b/select_full.json, now out/b/select.json; the partial one kept as out/b/select_partial_grid.json), the selection
   is unchanged: W layer 18, lr 3e-4 (best dev chain-naming 0.895, epoch 9); P layer 18, lr 3e-4 (0.886, epoch 7). The P
   final was re-queued with the same arguments. No test number was read at any point.

# B48 compute note 5 (2026-11-19 ~18:05 UTC; no design change, no bar or prediction changed)

10. **LM test runs done on the A800 GPU 0** (P test ~4 min, W test ~8 min; finished 17:58 UTC), after the dev runs, with
    the registered prompt. **Dev check of the scoring path (dev split only):** oracle 1.000; hash W 0.534-0.558, P
    0.562-0.576; lm P 0.998 (pronoun naming 0.998), lm W 0.588-0.608 with **W alias naming 0.121**. Read on the first 200 W
    dev stories (2,400 alias mentions), the LM mostly **copies the alias as written** (1,862, 78%), writes the right
    listed name in 305 (13%), another listed name in 226 (9%), another string in 7. This is not a scoring error (the
    written strings were compared by hand with the listed names); it is what the registered prompt elicits, and the test
    run had already been made with it. No prompt was changed. The verdict adds this breakdown for the test split as a
    descriptive line (not a bar). A prompt that asks for a choice among the listed names, or thinking mode, was not run.
11. **CPU lanes.** A second CPU worker (6 lanes x 6 threads) was added on the same queue at 18:01 UTC; the box CPU is shared
    with another user's jobs (nice 10). GPU 0, freed at 17:58, was claimed by B47 before B48's GPU worker rescanned.

# B48 compute note 6 (2026-11-19 ~18:15 UTC; part B complete; no design change, no bar or prediction changed)

12. **Part B ran to completion** (head finals on the box CPU, 17:50-18:01 UTC; LM test runs on GPU 0; all arms scored on
    test once by `b48_alias.py score`). Outputs in b48_out/b/ (gzipped), interim part-B verdict in
    b48_out/b/b48_verdict_partB_interim.txt (generated by b48_verdict.py while part A was still tuning; the final
    verdict regenerates it unchanged).
13. **A bug in the second scorer, found by the second scorer.** Its first run disagreed with the main scorer for the head
    only (agreement 0.69-0.76, the same accuracy 0.508 / 0.516 in every seed). Cause: in `b48_indep.py` the head branch
    appended a placeholder key for every statement before appending the real keys, so the chain read the placeholders.
    Fixed (the placeholder line removed); rerun: agreement 1.000 for every arm, dataset and seed, accuracies equal. The
    main scorer was not changed. For lm on P the largest score difference is 0.013 with identical decisions: the one
    story whose LM output stopped after 16 lines gives its 12 unwritten mentions keys of their own, and the two scorers
    name those keys differently (both are distinct random codes).

# B48 compute note 7 (2026-11-19 ~18:15 UTC; no design change, no bar or prediction changed)

14. **Part-A tuning on the shared CPU takes ~100 s per epoch** (Phi-2 VQ), i.e. ~3 h for the remaining runs. To make a
    partial finish balanced across encoders, the 105 lines still queued were reordered once under the queue lock
    (b48_reorder_once.py; the queue before the change is kept in logs/tuneA_before_reorder.txt): first the Qwen3-8B
    reference / gate runs (G0, binding-loss rates, G1, probe grid, naming-head dev run), then each family's grid on Phi-2
    and Qwen3-8B in turn (vq, dkvb, gumbel, st, cw); B42's within-grid order is kept. Three finals whose configuration is
    fixed by the pre-registration and needs no tuning (exact symbols; the naming head on each encoder) were put at the
    front of the same queue; the final queue made later lists them again, and b31_arms skips a run whose output exists.
    `b48_orch_A.sh` makes the selection and launches the remaining finals as soon as all 134 tuning outputs exist.
15. A third CPU worker (6 lanes x 4 threads) joined the same tuning queue at 18:41 UTC (20 lanes in all), because the
    box CPU was saturated (load ~335; 43% of it other jobs at nice 10) and Qwen3-8B tuning runs take ~25 min each.

# B48 compute note 8 (2026-11-20 ~02:50 UTC; part A complete; no design change, no bar or prediction changed)

16. **The session was interrupted again (~19:00 UTC 11-19 to ~02:39 UTC 11-20).** The box-side orchestrator
    (b48_orch_A.sh) ran unattended: all 134 tuning outputs present, selection and the final queue made, and the 17 part-A
    finals launched at 19:43 UTC on two CPU workers; all finished without a failed run (the three fixed-configuration
    finals had run earlier from the tuning queue and were skipped by the final queue). The second scorer, verdict and
    hand-reading ran at ~02:40 UTC on the box CPU.
17. **Recomputed vs logged accuracies (ties at the threshold, as B42 note 1).** In 4 part-A arms (Phi-2 dkvb, gumbel,
    st; Qwen3-8B dkvb, gumbel) the accuracy recomputed from the saved 6-decimal scores differs from the logged one by at
    most 0.008 in a seed; the verdict reports the recomputed numbers (flagged); every flagged arm FAILS under either.
    The second scorer (float64 chain from the saved assignments) agrees with the main runs on every instance for every
    discrete final (agreement 1.000, max score difference 0.0).
18. **Compute.** A800: 0.58 GPU-hours (features, LM-written-names runs) and 181 CPU core-hours (tuning and finals of
    parts A and B; lanes x threads x hours). PACE: the B48 jobs submitted on 11-19 (paid account) could not be checked
    after 05:50 UTC 11-19 (PACE unreachable); if any started, its cost is on gts-nimam6-paid and its output is unread.
    No paid API.
