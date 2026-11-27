# B49 compute note 1 (placement; no design change)

Date: 2026-11-20, 01:50 box clock (Asia/Shanghai). Written before any B49 dev result has been read and before any test
store has been scored by a B49 memory.

What happened: the B49 dev phase was started on the local CPU at 13:43 on 11-19 (b49_dev.sh for Llama and bge).
Memory filled (swap 8 / 8 GB, together with other sessions' jobs), and the local machine rebooted before any dev step
had finished (21 of 216 VSA dev lines per encoder; no choice, no test). The session restarted at 01:35 on 11-20 with a
hard deadline of 8 hours. At that time 12 of the local machine's 16 cores were taken by another session's job
(fixx_opt.py), which B49 must not touch, so the registered compute plan ("memory runs: local CPU") could not finish the
panel, the new competitors and the mechanism runs in time.

Change (placement only): every B49 memory run (dev and test) runs on the A800 box's CPU (128 cores, 1 TB RAM; load 1.3
when checked), under nice 10, in a separate working copy /path/to/work/b49 of the same code (files copied from the
repository; md5 of every code file recorded in b49_out/box_code_md5.txt), with a Python environment matching the local
one (Python 3.10, numpy 2.2.6, scipy 1.15.3, faiss-cpu 1.16.0, torch 2.11.0 CPU build; ~/envs/b49). No GPU is used, no
other process is touched. The codes on the box have the same md5 as the local files (b49_out/logs/box_setup.log).
B36's own precedent allowed the box's CPU when idle (prereg/B36, 'Compute').

Consequences: fixed random components are drawn from the same seeds with the same numpy version, so they are identical;
faiss k-means fits (PQ / OPQ) and torch float32 matmuls may differ in the last bits from a local run (different BLAS /
thread counts). Nothing about grids, rules, seeds, accounting, choices, bars or predictions changes. Choices are copied
back, committed in the repository, and copied to the box before any test run; test outputs are copied back and merged,
scored, checked and hand-read locally.

Execution order on the box (b49_box/phase1.sh, b49_box/mech_dev.sh): all fits that depend only on the training split
first (so no two processes write one cache file), then the dev grids in parallel. b49_dev.sh was reordered accordingly
(every PQ / OPQ size a budget can reach is fitted before the grids: b49_run.py fitlist).

Code fixes made after the pre-registration commit and before any dev result was read (b49_common.py): (1) the LAM's
one-hot targets are centred (e_v - 1/V) with value 0's score minus the sum of the stored columns, as the pre-registration
specifies (the first draft stored contrast columns with a zero column for value 0, which biased reads against value 0);
(2) the Gram matrices of the feature prefixes are accumulated segment by segment (the same numbers, computed once).

## Addendum (2026-11-20 02:35 box clock; before any dev result is read)

- The box is shared: from about 02:00 other sessions' GPU jobs used about 80 of its 128 cores (not niced), and every
  GPU except GPU 3 was taken, so no GPU is used by B49.
- B32's VSA dev grid of the four new encoders ran at about one store-pass per minute in one process; it was stopped
  at 02:32 and split into six parallel processes per encoder, one per (preprocessing, feature map) part
  (b49_box/vsa_split.py; the part's lines already written are copied first, so no store runs twice; each process runs
  b32_run.dev unchanged with GRID['prep'] / GRID['fmap'] set to its part). B32's choice reads the six part files and the
  index file (b49_box/finalize_choose.py, b32_run.choose and b36_run.choose unchanged); b49_dev.sh's own b32choose,
  which read the stopped single-process file, is superseded.
- The mechanism runs start in the registered order (qwen8 T1, qwen8 T3, e5 T2; b49_box/mech_primary.sh); the
  replications (b46 T1 / T3, bge T2) start only if the primary runs finish with time to spare.

## Addendum 2 (2026-11-20 11:35; after the panel and primary mechanism tests)

- Session break 04:45-10:39 (usage limit); nothing was started or read in between. All dev phases had finished by
  04:43; choices were committed at 5b6f03f (panel and new families) and 31f5968 (primary mechanism runs) before their
  test runs (10:46-11:07); replications b46 T3 and bge T2 at 20b9298, b46 T1 after its dev phase.
- The B36 choice step of b49_dev.sh (b36_run.choose) failed by design after the VSA split (it read the stopped file);
  b49_box/finalize_choose.py made B32's and B36's choices from the part files (CHOICES_DONE in each steps.log).
- b49_fam.choose crashed once (B36's grid not loaded without a Setup; fixed in 904d45a before any test).
- b49_predcheck.py's P4 first counted the mechanism encoders' learned maps; restricted to the seven panel encoders as
  registered (the prediction text is unchanged).
- Compute: A800 box CPU only (nice 10), about 01:47-04:45 and 10:46-12:30; peak about 100 of my threads beside other
  sessions' jobs; 0 GPU-hours; local CPU only for choices, commits and summaries; no paid API.
- 12:00: replication b46 T1 stopped before its index fits finished (deadline); reported as not run. b49_mech.py fits gained --only (one OPQ size per process).
