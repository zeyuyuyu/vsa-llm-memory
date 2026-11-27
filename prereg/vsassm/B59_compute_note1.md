# B59 compute note 1 (02:22 machine clock; during development; no B59 test run yet, no test output exists)

- **Part 1 on the box.** The box's copies of the encoders' codes were taken from the box's own B49 directories
  (/path/to/work/b49/b49_out, the files B49 used there) after the copy over the network ran at about 0.4 MB/s; every file
  was checked against the local copy by MD5 (all 7 code matrices, entities.json and emb_strings.json identical). Code
  files on the box: MD5-identical to the committed ones. Box environment: numpy 1.26.4, torch 2.4.1, scipy 1.13.1
  (local: numpy 2.2.6, torch 2.11, scipy 1.15.3); the registered cross-machine check (one bge MF part and one Granite
  engine job run on both machines) measures what this changes. dev1 started 02:13 with 10 jobs x 3 threads.
- **Part 2 P-LDPC dev pass 2** (registered rule: every design at the quarter steps within 3 grid steps of each
  condition's pass-1 N*; pass-1 N* as B58: 91 / 91 / 91 / 128 / 256 / 1,024 / 4,096; 448 (design, budget) entries, 700
  evaluations per dev store, 912 jobs) is split into four shards of every 4th job (b59_store.py --shard, added and
  committed before the run; no change to any evaluation). Shards 1-3 run on the box (6 workers each) since 02:15. Shard
  0 started locally (6 workers) at 02:13 but ran at less than half the box's speed under other sessions' load (7 of 228
  jobs in 7 minutes, against 16-17 per box shard; free memory 1-3 GB); it was stopped at 02:20 and restarts on the box (6 workers) as soon as part
  1 dev1 ends; part 1 dev2 and test then run with 8 jobs x 3 threads, so B59 stays within 48 box cores.
- **Staged test start for part 2.** The designs of two families do not depend on P-LDPC pass 2 and are already
  committed: U (one design; B38's dev results) and P-BCH (its choice is B58's committed results/b58_choice.json, from
  the same pass-1 and pass-2 files b59_store.choose uses). Their full-grid test plan (results/b59_plan_u_bch.json: U
  98 (condition, budget) cells missing from B38's test, 4,900 store evaluations; P-BCH T2 2,150, T3 1,550, T4 4,300
  store evaluations; 1,041 B58 evaluations reused) may start before the P-LDPC dev pass 2 ends. P-LDPC's test and the
  dev union P-dev follow only after their choices are committed. No test output is read before every dev choice of
  both parts is committed. The final plan for U and P-BCH is checked to be identical to this one.
- Nothing about designs, grids, seeds, bars or predictions changed.
