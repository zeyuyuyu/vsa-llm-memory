# B40 amendment 1 (dev only; before any test run): two more read-noise levels

Date: 2026-11-19, about 01:05 box time. No test split has been scored by any arm.

Dev calibration (b40_noisecal.py; the chosen seed-0 readers on the cached DEV states; read accuracy only, no LLM
forward, no write-back; same noise as the Injector: isotropic Gaussian with norm rho x the state's norm, per
position):

| reader | rho 0 | 0.25 | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|---|
| N VSA (type + both operands exact) | 0.896 | 0.543 | 0.080 | 0.004 | 0.000 | 0.000 | 0.000 |
| N DISC | 0.946 | 0.923 | 0.805 | 0.407 | 0.038 | 0.002 | 0.000 |
| R VSA (entity per mention) | 1.000 | 1.000 | 1.000 | 1.000 | 0.851 | 0.309 | 0.099 |
| R DISC | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.975 | 0.558 |

The registered grid rho in {0.25, 0.5, 1.0, 2.0} spans the range where task-N reads degrade, but on task R it reaches
only the onset for the VSA reads and none for DISC. Added: rho in {4.0, 8.0} for both tasks (evaluation-only runs of
the trained seeds, VSA, DISC and, on task R, VSA-soft). Bars, readings and predictions unchanged (P13 now covers six
levels). Nothing else changes.
