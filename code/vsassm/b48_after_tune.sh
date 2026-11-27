#!/bin/bash
# B48 (prereg/B48_one_form_and_alias_naming_prereg.md): after the dev tuning runs, on the PACE login node (seconds):
# part A selection (b48_select.py), part B selection (b48_alias.py select), the final queues, and the final jobs.
# No retry: a failed submission is reported and checked with squeue before anything is resubmitted.
cd /path/to/pace-scratch/b48 || exit 1
/usr/bin/python3 b48_select.py --root out --out out/select.json || exit 1
/usr/bin/python3 b48_alias.py select --root out/b --out out/b/select.json || exit 1
/usr/bin/python3 b48_queues.py final || exit 1
echo fA1=$(sbatch --parsable -J b48_finalA1 b48_pace.sbatch queue q/finalA 6)
echo fA2=$(sbatch --parsable -J b48_finalA2 b48_pace.sbatch queue q/finalA 6)
echo fB=$(sbatch --parsable -J b48_finalB b48_pace.sbatch queue q/finalB 2)
