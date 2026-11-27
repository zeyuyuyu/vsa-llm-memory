# B50 smoke record (training rows only; no dev or test number of any arm seen)

Written 2026-11-19, about 13:30 box time, after the pre-registration commit (c2816c8; OpenTimestamps proof c6ff6de)
and before any B50 dev or test run.

Box: A800 GPU 6, claimed by b50_worker.sh under the shared lock rule at 05:23Z (GPU 4, idle when the worker started,
was taken by another user's job before the claim; the worker waited until GPU 6 was idle). Code on the box
md5-identical to git (b40_core.py, b40_data.py, b40_run.py, b40_verdict.py, b40_check.py and every b50_* file).

## Commands (training rows of task N / R; outputs /path/to/work/b50/out/smoke/)

- fitmaha on B40's 16,000 task-N training states at layer 20: 22 s. Tied covariance eigenvalues 3.9e-5 to 1.76
  (median 5.1e-4; condition number about 4.5e4), none below the pseudo-inverse cut-off, so MAHA's pinv is an ordinary
  inverse here. Ledoit-Wolf shrinkage coefficient 0.0065.
- DISC seed 0 with gate open / p08 / B40's own DiscN, 64 training rows (multiplication): gate p08 equals B40's DiscN on
  64 of 64 rows (prediction, gate and module answer); forced-open predictions also equal (every row is open at 0.8).
- Scores on these 64 training rows (in the Mahalanobis fitting set): MSP log-odds 8.8 to 13.5, energy 10.0 to 13.7,
  MAHA -8,890 to -3,196, MAHA-LW -7,839 to -2,697, max-logit 10.0 to 13.7.
- Gate ENERGY with threshold 1e30 (closes on every row): 0 of 64 open and every prediction equals BASE's on the same
  rows, so a closed gate leaves the forward pass untouched (the premise of the composition check).
- VSA seed 0 and BASE on 64 training rows; COT2048 on 8 task-N and 8 task-R training rows (204 and 163 decoding steps;
  every item ended with an end token; n_gen and hit_limit logged; batch timing written).

## Failure found and fixed (no effect on any registered rule)

- A negative threshold passed as `--thr -1e30` is read by argparse as a new option and the run exits (rc 2) before
  loading anything. MAHA and MAHA-LW thresholds are negative, so b50_queues.py now writes `--thr=<value>`. The same
  smoke command re-run with `--thr=-1e30`: see below. Nothing else changed.
- Re-run with `--thr=-1e30` (MAHA, opens on every row): 64 of 64 open, every prediction equals the forced-open run's,
  MAHA scores identical to the forced-open run's (max difference 0.0).

Smoke complete; every B50 code path ran. Next: the dev runs of the matching rule and the COT2048 test runs (no dev
choice exists for COT2048), then the dev record, then the gated DISC test runs.
