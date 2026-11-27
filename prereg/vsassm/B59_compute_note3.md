# B59 compute note 3 (02:48 machine clock; part 2 dev choices written; no part 2 test output has been read)

- **P-LDPC dev pass 2 finished** (four shards, 912 jobs, 8,400 evaluations = 700 x 12 dev stores; shard 0 on the box
  02:28-02:46). Merged into results/b59_dev2_ldpc.json. **Dev choices** (results/b59_choice_store.json, b59_store.choose:
  B58's rule per code family over pass 1 + pass 2; U from B38's dev; the dev union P-dev). Dev N* (cells per pair,
  k1 / k2 / k4 / e05 / k8 / k16 / k32): P-LDPC 91 / 91 / 91 / 108 / 152 / 861 / 2,896 (B58, octave resolution: 91 / 91
  / 91 / 128 / 256 / 1,024 / 4,096); P-BCH 91 / 91 / 91 / 108 / 181 / 1,722 / not reached (= B58); U 76 / 76 / 76 / 91 /
  128 / 431 / 1,722; P-dev = U's at every k.
- **Full test plan** results/b59_plan_full.json: U's 98 missing (condition, budget) cells (4,900 store evaluations)
  and, per code family and condition, every budget 90.5 ... 4,096 with the family's design there; tiers T2 4,000, T3
  2,550, T4 8,800 store evaluations, 1,594 B58 test evaluations reused. Its U and P-BCH parts are identical to
  results/b59_plan_u_bch.json (compute note 1), checked entry by entry.
- **Already running from the committed U / P-BCH plan** (compute note 1): U (T1) finished locally (02:23-02:42, 300
  jobs); P-BCH T2 on the box (12 workers, from 02:42); P-BCH T3 locally (4 workers, from 02:42). No output of these runs
  has been read (only job counts in the logs).
- **Now**: P-LDPC T2 on the box; P-LDPC T3 locally after P-BCH T3; T4 (both families) on the box once part 1's test
  frees its 24 cores. B59 stays within 48 box cores.
- **Cut decided now (registered time rule):** the damage conditions (T5) are **not run**: T1-T4 cannot finish before
  03:40, the registered condition for T5. Whatever of T4 (then T3) has not finished by 04:00 is cut by a further note
  written before the part 2 verdict is computed; N* in regions that were not evaluated are then reported as bounds.
- Nothing about designs, grids, seeds, bars or predictions changed.
