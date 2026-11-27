#!/bin/bash
# B44 job on PACE Phoenix (prereg/B44_data_scaling_heldout_cot_crosstalk_prereg.md): runs ONE b44 command (the
# script's arguments) on any GPU (frozen-feature arms: the naming head, exact rows and the probe need little GPU memory but up to ~200 GB of RAM for the 80,000-instance fp16 features). Same code, same Qwen/Qwen3-8B snapshot (b968826d)
# and the same tokenised data (md5-checked) as on the A800 box. A LoRA run that hits its --max_hours budget exits 3
# after writing a checkpoint; the next job of its dependency chain resumes it (a job whose run is already finished
# exits at once: "exists, skip").
# usage: sbatch -J b44_<name> b44_pace.sbatch b44_lora.py ... --max_hours 2.6 --out out/lora/<file>.json
#SBATCH -A gts-nimam6
#SBATCH -p gpu-v100,gpu-rtx6000,gpu-a100,gpu-h100,gpu-h200,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=220G
#SBATCH -t 3:00:00
#SBATCH -o /path/to/pace-scratch/b44/logs/%x_%j.out
cd /path/to/pace-scratch/b44 || exit 1
export HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export B31_QWEN8=/path/to/pace-scratch/hf/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
echo "B44-PACE start $(date -u +%FT%TZ) host $(hostname) job $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash "$@"
rc=$?
echo "B44-PACE end $(date -u +%FT%TZ) rc=$rc"
exit $rc
