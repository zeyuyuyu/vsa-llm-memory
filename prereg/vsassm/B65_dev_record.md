# B65 dev record (dev split only; committed before any test run)

Written 2026-11-23 after both positions' seed-0 grids and edge extensions finished (times in the commit). Every number
is on the 800-row dev split; every choice is b65_plan.py's registered rule, applied mechanically (out/choices.json).
Raw files: b65_out/dev/ (every reader.json, train.json, choices.json, the plan logs and the queues). The seed-1 / 2 dev
curves (best step only, no choice) are appended in a later commit, also before any test output is read.

## Layer scan (operand positions; VSA encoder, lr 1e-3, 100 epochs; dev read exact, published read)

| layer | 8 | 12 | 14 | 16 | 17 | 18 | 20 | 24 |
|---|---|---|---|---|---|---|---|---|
| read exact | 0.8788 | 0.8550 | 0.9575 | 0.9800 | 0.9850 | 0.9812 | 0.9850 | 0.9837 |

Choice: OPS layer 17; LAST layer 20 (B40's).

## Readers (seed 0; dev read exact under each read rule)

| position | encoder | lr | epochs | published | best match / disc |
|---|---|---|---|---|---|
| last | disc | 0.001 | 30 | - | 0.8900 |
| last | disc | 0.001 | 100 | - | 0.9287 |
| last | disc | 0.001 | 300 | - | 0.9463 |
| last | disc | 0.0001 | 30 | - | 0.5587 |
| last | disc | 0.0001 | 100 | - | 0.7412 |
| last | disc | 0.0001 | 300 | - | 0.8200 |
| last | vsa | 0.001 | 100 | 0.8650 | 0.9600 |
| last | vsa | 0.001 | 300 | 0.8938 | 0.9762 |
| last | vsa | 0.001 | 1000 | 0.8963 | 0.9712 |
| last | vsa | 0.0001 | 100 | 0.8338 | 0.9613 |
| last | vsa | 0.0001 | 300 | 0.8650 | 0.9712 |
| last | vsa | 0.0001 | 1000 | 0.8700 | 0.9725 |
| ops | disc | 0.001 | 30 | - | 1.0000 |
| ops | disc | 0.001 | 100 | - | 1.0000 |
| ops | disc | 0.001 | 300 | - | 1.0000 |
| ops | disc | 0.0001 | 30 | - | 0.9962 |
| ops | disc | 0.0001 | 100 | - | 1.0000 |
| ops | disc | 0.0001 | 300 | - | 1.0000 |
| ops | vsa | 0.001 | 100 | 0.9850 | 1.0000 |
| ops | vsa | 0.001 | 300 | 0.9900 | 1.0000 |
| ops | vsa | 0.001 | 1000 | 0.9912 | 1.0000 |
| ops | vsa | 0.0001 | 100 | 0.9775 | 0.9975 |
| ops | vsa | 0.0001 | 300 | 0.9875 | 1.0000 |
| ops | vsa | 0.0001 | 1000 | 0.9887 | 1.0000 |

Reader choices: last_disc lr 1e-3 e300 (0.9463); last_vsabm lr 1e-3 e300 (0.9762); last_vsapub lr 1e-3 e1000 (0.8963); ops_disc lr 1e-3 e30 (1.0000); ops_vsabm lr 1e-3 e100 (1.0000); ops_vsapub lr 1e-3 e300 (0.9900)

## Write-back (seed 0; dev first-token argmax at steps 0, 250, 500, 750, 1000; best over 250-1000)

| position | arm | lr | curve | best (step) |
|---|---|---|---|---|
| last | disc | 0.0003 | 0.1825 0.2562 0.3125 0.3488 0.4113 | 0.4113 (1000) |
| last | disc | 0.001 | 0.1825 0.5525 0.9012 0.9225 0.9413 | 0.9413 (1000) |
| last | disc | 0.003 | 0.1825 0.9425 0.9475 0.9487 0.9500 | 0.9500 (1000) |
| last | disc | 0.01 | 0.1825 0.9513 0.9475 0.9513 0.9513 | 0.9513 (250) |
| last | disc | 0.03 | 0.1825 0.9513 0.9513 0.9513 0.9513 | 0.9513 (250) |
| last | vsabm | 0.0003 | 0.1713 0.9650 0.9688 0.9725 0.9725 | 0.9725 (750) |
| last | vsabm | 0.001 | 0.1713 0.9725 0.9738 0.9738 0.9738 | 0.9738 (500) |
| last | vsabm | 0.003 | 0.1713 0.9738 0.9738 0.9738 0.9738 | 0.9738 (250) |
| last | vsabm | 0.01 | 0.1713 0.9725 0.9725 0.9738 0.9725 | 0.9738 (750) |
| last | vsapub | 0.0003 | 0.1725 0.8962 0.9075 0.9075 0.9075 | 0.9075 (500) |
| last | vsapub | 0.001 | 0.1725 0.9113 0.9062 0.9125 0.9075 | 0.9125 (750) |
| last | vsapub | 0.003 | 0.1725 0.9087 0.9087 0.9100 0.9100 | 0.9100 (750) |
| last | vsapub | 0.01 | 0.1725 0.9062 0.9062 0.9113 0.9100 | 0.9113 (750) |
| ops | disc | 0.0003 | 0.1787 0.2450 0.3063 0.3575 0.4325 | 0.4325 (1000) |
| ops | disc | 0.001 | 0.1787 0.5887 0.9700 0.9888 0.9900 | 0.9900 (1000) |
| ops | disc | 0.003 | 0.1787 0.9988 0.9975 0.9988 0.9975 | 0.9988 (250) |
| ops | disc | 0.01 | 0.1787 1.0000 1.0000 1.0000 1.0000 | 1.0000 (250) |
| ops | disc | 0.03 | 0.1787 1.0000 1.0000 1.0000 1.0000 | 1.0000 (250) |
| ops | vsabm | 0.0003 | 0.1688 0.9912 0.9938 0.9975 0.9975 | 0.9975 (750) |
| ops | vsabm | 0.001 | 0.1688 0.9975 0.9975 0.9975 0.9975 | 0.9975 (250) |
| ops | vsabm | 0.003 | 0.1688 0.9975 0.9975 0.9975 0.9975 | 0.9975 (250) |
| ops | vsabm | 0.01 | 0.1688 0.9975 0.9975 0.9975 0.9975 | 0.9975 (250) |
| ops | vsapub | 0.0003 | 0.1613 0.9812 0.9888 0.9875 0.9875 | 0.9888 (500) |
| ops | vsapub | 0.001 | 0.1613 0.9850 0.9888 0.9888 0.9888 | 0.9888 (500) |
| ops | vsapub | 0.003 | 0.1613 0.9888 0.9888 0.9888 0.9888 | 0.9888 (250) |
| ops | vsapub | 0.01 | 0.1613 0.9888 0.9875 0.9888 0.9888 | 0.9888 (250) |

Write-back choices (rule: best dev; ties -> lr nearest 1e-3, then smaller; strict edge optimum -> extend):
- last_disc: lr 1e-2 (interior); grid ['3e-4', '1e-3', '3e-3', '1e-2', '3e-2']; dev {'1e-2': 0.95125, '1e-3': 0.94125, '3e-2': 0.95125, '3e-3': 0.95, '3e-4': 0.41125}
- last_vsabm: lr 1e-3 (interior); grid ['3e-4', '1e-3', '3e-3', '1e-2']; dev {'1e-2': 0.97375, '1e-3': 0.97375, '3e-3': 0.97375, '3e-4': 0.9725}
- last_vsapub: lr 1e-3 (interior); grid ['3e-4', '1e-3', '3e-3', '1e-2']; dev {'1e-2': 0.91125, '1e-3': 0.9125, '3e-3': 0.91, '3e-4': 0.9075}
- ops_disc: lr 1e-2 (interior); grid ['3e-4', '1e-3', '3e-3', '1e-2', '3e-2']; dev {'1e-2': 1.0, '1e-3': 0.99, '3e-2': 1.0, '3e-3': 0.99875, '3e-4': 0.4325}
- ops_vsabm: lr 1e-3 (interior); grid ['3e-4', '1e-3', '3e-3', '1e-2']; dev {'1e-2': 0.9975, '1e-3': 0.9975, '3e-3': 0.9975, '3e-4': 0.9975}
- ops_vsapub: lr 1e-3 (interior); grid ['3e-4', '1e-3', '3e-3', '1e-2']; dev {'1e-2': 0.98875, '1e-3': 0.98875, '3e-3': 0.98875, '3e-4': 0.98875}

## What the dev numbers already say (the test decides)

- Reproduction (checks, prereg Q3): at LAST, the lr-1e-3 reader snapshots reproduce B40's separately trained readers'
  dev reads exactly (VSA published 0.8650 / 0.8938 / 0.8963 at 100 / 300 / 1000 epochs, DISC 0.8900 / 0.9287 /
  0.9463), and the LAST write-back curves at lr 1e-3 and 3e-4 reproduce B40's dev record to the printed digit (VSA-PUB
  .1725 .9113 .9062 .9125 .9075 and .1725 .8962 .9075 .9075 .9075; DISC .1825 .5525 .9012 .9225 .9413 and .1825
  .2562 .3125 .3488 .4113). Reader lr 1e-4 is worse than 1e-3 for every reader (LAST DISC by 0.13 at 300 epochs).
- The referee's point (1) holds on dev: DISC at LAST was under-tuned in B40. Its dev best rises from 0.9413 (lr 1e-3,
  B40's choice) to 0.9500 (3e-3) and 0.9513 (1e-2 and 3e-2, tied; the tie rule picks 1e-2, interior after one
  extension). The VSA write-backs are flat in their lr (LAST VSA-PUB 0.9075-0.9125, VSA-BM 0.9725-0.9738).
- At LAST the dev ordering is VSA-BM 0.974 > DISC 0.951 > VSA-PUB 0.913 (B40's dev: DISC 0.941, VSA-PUB 0.9125).
- At OPS every module reads almost perfectly on dev (read exact: DISC 1.000 at 30 epochs, VSA-BM 1.000 at 100,
  VSA-PUB 0.990 at 300; the OPS layer is 17: 0.985, tied within 0.005 with 16, 18, 20, 24) and the write-back dev
  accuracies are VSA-PUB 0.9888 (flat over every lr), VSA-BM 0.9975 (flat), DISC 1.000 at lr 1e-2 and 3e-2 (tied; the
  tie rule picks 1e-2, interior after one extension); DISC at B40's lr 1e-3 would have been 0.990.
- Prediction status on dev (scored in the note after the test): Q1 held (DISC's chosen lr is 1e-2 at both positions);
  Q2 held (VSA spans 0.005 / 0.0013 at LAST, 0 / 0 at OPS; no VSA extension); Q3 held (OPS layer 17; OPS dev reads
  >= 0.99 for all three; LAST reproduction exact; lr 1e-4 never better). The dev-optimum-interior check passes for all
  six (arm, position) cells; for DISC at both positions the optimum is a plateau that reaches the extended edge
  (1e-2 = 3e-2), resolved by the registered tie rule, at the read ceiling (LAST DISC reads 0.946 exact on dev).
- GPU and procedure: compute notes 2 and 3 (six OPS VSA runs re-run once after OOM at their first step; dependent
  phases chained onto the workers' live queues).

# B65 dev record, part 2: seeds 1 and 2 (dev split; appended before any test output is read)

Chosen settings of part 1; each seed keeps its own best dev step (B40's rule); nothing is chosen here.


| run | curve | best (step) |
|---|---|---|
| last_disc_s1 | 0.1825 0.9500 0.9513 0.9513 0.9500 | 0.9513 (500) |
| last_disc_s2 | 0.1825 0.9513 0.9513 0.9513 0.9513 | 0.9513 (250) |
| last_vsabm_s1 | 0.1737 0.9712 0.9700 0.9712 0.9712 | 0.9712 (250) |
| last_vsabm_s2 | 0.1787 0.9650 0.9700 0.9700 0.9700 | 0.9700 (500) |
| last_vsapub_s1 | 0.1775 0.9062 0.9100 0.9100 0.9113 | 0.9113 (1000) |
| last_vsapub_s2 | 0.1825 0.9050 0.9075 0.9087 0.9113 | 0.9113 (1000) |
| ops_disc_s1 | 0.1787 1.0000 1.0000 1.0000 1.0000 | 1.0000 (250) |
| ops_disc_s2 | 0.1787 0.9975 0.9962 1.0000 1.0000 | 1.0000 (750) |
| ops_vsabm_s1 | 0.1650 0.9962 0.9962 0.9962 0.9962 | 0.9962 (250) |
| ops_vsabm_s2 | 0.1713 0.9950 0.9950 0.9950 0.9950 | 0.9950 (250) |
| ops_vsapub_s1 | 0.1700 0.9888 0.9888 0.9888 0.9888 | 0.9888 (250) |
| ops_vsapub_s2 | 0.1700 0.9900 0.9888 0.9900 0.9900 | 0.9900 (250) |
- reader last_disc_lr1e-3_s1: e30 disc 0.8913, e100 disc 0.9300, e300 disc 0.9463
- reader last_disc_lr1e-3_s2: e30 disc 0.8838, e100 disc 0.9312, e300 disc 0.9463
- reader last_vsa_lr1e-3_s1: e100 pub 0.8662 bm 0.9650, e300 pub 0.8925 bm 0.9725, e1000 pub 0.8950 bm 0.9725
- reader last_vsa_lr1e-3_s2: e100 pub 0.8600 bm 0.9650, e300 pub 0.8913 bm 0.9725, e1000 pub 0.8963 bm 0.9737
- reader ops_disc_lr1e-3_s1: e30 disc 1.0000, e100 disc 1.0000, e300 disc 1.0000
- reader ops_disc_lr1e-3_s2: e30 disc 1.0000, e100 disc 1.0000, e300 disc 1.0000
- reader ops_vsa_lr1e-3_s1: e100 pub 0.9800 bm 0.9987, e300 pub 0.9900 bm 1.0000, e1000 pub 0.9912 bm 1.0000
- reader ops_vsa_lr1e-3_s2: e100 pub 0.9800 bm 1.0000, e300 pub 0.9912 bm 1.0000, e1000 pub 0.9912 bm 1.0000
