#!/bin/bash
# B47 job on PACE Phoenix, any GPU (frozen-feature work) (prereg/B47_phi2_finetunes_hierboot_consistency_prereg.md): runs ONE b47 command (the
# script's arguments) on one >= 80 GB GPU (H200, H100 or A100-80GB), Phi-2 snapshot microsoft/phi-2 810d3678 (the box's,
# copied by B44 compute note 1 item 5). A run that hits its --max_hours budget exits 3 after writing a checkpoint; the
# next job of its chain resumes it (a job whose run is already finished exits at once: "exists, skip").
# usage: sbatch -J b47_<name> b47_pace.sbatch b47_ft.py ... --max_hours 1.2 --out out/ft/<file>.json
#SBATCH -A gts-nimam6
#SBATCH -p gpu-v100,gpu-rtx6000,gpu-a100,gpu-h100,gpu-h200,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 1:30:00
#SBATCH -o /path/to/pace-scratch/b47/logs/%x_%j.out
cd /path/to/pace-scratch/b47 || exit 1
export HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "B47-PACE start $(date -u +%FT%TZ) host $(hostname) job $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case "$1" in *.sh) bash "$@";; *) /usr/bin/python3 -u "$@";; esac
rc=$?
echo "B47-PACE end $(date -u +%FT%TZ) rc=$rc"
exit $rc
