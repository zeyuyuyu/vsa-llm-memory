# B50 dev record (dev split only; committed before any gated DISC test run)

Written 2026-11-19, about 13:45 box time. Every number below is on the task-N dev split (800 items, 100 per trained
type; no held-out type), from live runs on A800 GPU 6 (b50_run.py, outputs /path/to/work/b50/preds/dev/). The rule is
the registered matching rule (prereg/B50_ood_gates_cot_budget_prereg.md); it is mechanical and saw no test number.
Thresholds: b50_out/out/thresholds.json (copy of /path/to/work/b50/out/thresholds.json). Running already, as
registered (no dev choice exists for them): COT2048 on the task-N and task-R test splits, and the layer-20 test-state
cache (no output is read from either before the gated runs are queued).

## VSA gate on dev (B40's VSA seeds, unchanged)

| seed | VSA gate open (of 800) | VSA dev accuracy |
|---|---|---|
| 0 | 797 | 0.9000 |
| 1 | 797 | 0.8962 |
| 2 | 797 | 0.9050 |

So k_s = 797 (open rate 0.99625) in every seed; P1 (k_s >= 796) holds on dev.

## DISC forced open on dev and the matched thresholds (open iff score >= tau)

DISC forced-open dev accuracy 0.9237 / 0.9187 / 0.9287 (seeds 0-2); B40's 0.8 softmax gate would open on 800 of 800 dev
items in every seed. tau = the 797th largest of the 800 dev scores; each gate opens on exactly 797 dev items (no
ties).

| score | seed 0 tau | seed 1 tau | seed 2 tau | dev min | dev 1% | dev median | dev max (seed 0) |
|---|---|---|---|---|---|---|---|
| MSP (log-odds of p_max) | +7.172 | +7.170 | +7.248 | +5.92 | +8.08 | +12.15 | +16.74 |
| ENERGY (logsumexp) | +9.068 | +9.019 | +8.948 | +7.63 | +9.94 | +13.00 | +16.52 |
| MAHA (-d^2, pinv) | -19,727 | -19,727 | -19,727 | -23,444 | -14,426 | -5,369 | -1,643 |
| MAHA-LW (-d^2, Ledoit-Wolf) | -15,355 | -15,355 | -15,355 | -18,410 | -11,387 | -4,260 | -1,355 |
| MAXLOGIT (secondary) | +9.065 | +9.017 | +8.948 | +7.63 | +9.94 | +13.00 | +16.52 |

(MSP log-odds 7.17 is p_max = 0.99923: the matched MSP gate is far stricter than B40's 0.8.) MAHA and MAHA-LW do not
depend on the DISC seed, so their thresholds coincide across seeds. Sensitivity thresholds (dev open rates 0.99 and
0.95) are in thresholds.json.

## Mahalanobis fit (16,000 training states, layer 20)

Tied covariance eigenvalues 3.9e-5 to 1.76 (median 5.1e-4); none below the pseudo-inverse cut-off; Ledoit-Wolf
shrinkage 0.0065. Dev squared distances (median 5,369) run above the in-sample ones seen in the smoke (median about
4,860 on 64 training rows), as expected for out-of-sample points in 4,096 dimensions; the dev tail is long (1% at
14,426, max 23,444).

## Independent score check on dev (b50_check_scores.py; sklearn EmpiricalCovariance / LedoitWolf classes, numpy heads)

Scores recomputed offline from B40's dev cache (a separate batched forward pass) against the live-logged scores:
Spearman 0.9990-0.9991 (MSP), 0.9994-0.9995 (ENERGY, MAXLOGIT), 0.9997 (MAHA, MAHA-LW) in every seed; median relative
difference 7e-4 to 2e-3; gate decision at tau differs on 0 or 1 of 800 items per gate and seed (the item at the
threshold boundary; cache vs live numerics). P14 (Spearman >= 0.999) holds on dev, narrowly for MSP.
