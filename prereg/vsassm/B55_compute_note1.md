# B55 compute note 1 (2026-11-20; procedure only)

The prereg (commit 907154b, OpenTimestamps proof 3ed3d5b, 05:51Z) was written while all seven usable A800 GPUs were
running another user's jobs, so it registered a PACE fallback beside the A800 worker. Within a minute of the commit
those jobs ended: the B55 worker (b55_worker.sh, started 05:52Z) claimed A800 GPU 0 at 05:52:26Z under the shared lock
rule, the smoke on 16 training rows per task passed at 05:52:40Z (rc 0; files under /path/to/work/b55/smoke, never
the test split), and the 60 registered A800-plan test jobs started at 05:52:40Z in two lanes on that GPU.

No PACE job was ever submitted: the code, B40's checkpoints and B50's thresholds were copied to PACE scratch
(/path/to/pace-scratch/b55) and the job plan was generated there with `sbatch --test-only` only, before it was
clear the box would free up; no sbatch was issued, so there is no PACE test output and 0 PACE GPU-hours. The A800 plan
is therefore the one that runs (same GPU type as B40, comparison arms = B40's and B50's files, same-noise checks as
registered).

Afterwards: the 60 test jobs finished at 06:32:36Z (worker log), rc 0, no tracebacks; logs/gpu_hours.txt has 65 lines (60
test + 5 smoke), 1.305 lane-hours in all, on NVIDIA A800 80GB PCIe GPU 0 (two lanes sharing one GPU, 0.67 h of GPU
wall time, 05:52:26-06:32:36Z). The `sbatch --test-only` estimate on PACE (paid account) was a start on 2026-11-27, a
week out, so the fallback would not have helped. 0 PACE GPU-hours; nothing was billed. The PACE scratch copy (code,
B40 checkpoints, thresholds) is left in place, unused.
