#!/bin/bash
# B47 sweep part on PACE CPU (prereg/B47_phi2_finetunes_hierboot_consistency_prereg.md; no GPU, nothing trained).
# usage: sbatch -J b47_sw_<what>_<part> b47_sweep_cpu.sbatch <what> <part> <nparts>
#SBATCH -A gts-nimam6
#SBATCH -p cpu-small
#SBATCH -c 16
#SBATCH --mem=40G
#SBATCH -t 3:00:00
#SBATCH -o /path/to/pace-scratch/b47/logs/%x_%j.out
cd /path/to/pace-scratch/b47 || exit 1
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
echo "B47-CPU start $(date -u +%FT%TZ) host $(hostname) job $SLURM_JOB_ID"
/usr/bin/python3 -u b47_sweep.py --what $1 --part $2 --nparts $3 --threads 16 --out out/sweep/$1_p$2of$3.json
rc=$?
echo "B47-CPU end $(date -u +%FT%TZ) rc=$rc"
exit $rc
