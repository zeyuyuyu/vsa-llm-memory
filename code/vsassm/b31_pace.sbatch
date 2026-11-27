#!/bin/bash
# B31 job on PACE Phoenix (prereg/B31_encoder_equalised_baselines_prereg.md; compute note 2): runs ONE b31 command
# (given as the script's arguments) on one H200 or H100 (>= 80 GB; the Qwen3-8B LoRA needs 60-67 GB at batch 8).
# Same code, same model snapshot (Qwen/Qwen3-8B b968826d), same arguments as on the A800 box.
# usage: sbatch -J b31_<name> b31_pace.sbatch b31_lora.py --enc qwen8 ... --out out/lora_final/<file>.json
#SBATCH -A gts-nimam6
#SBATCH -p gpu-h200,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH -C H200|H100
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 3:00:00
#SBATCH -o /path/to/pace-scratch/b31/logs/%x_%j.out
cd /path/to/pace-scratch/b31 || exit 1
export HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export B31_QWEN8=/path/to/pace-scratch/hf/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
echo "B31-PACE start $(date -u +%FT%TZ) host $(hostname) job $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/usr/bin/python3 -u "$@"
rc=$?
echo "B31-PACE end $(date -u +%FT%TZ) rc=$rc"
