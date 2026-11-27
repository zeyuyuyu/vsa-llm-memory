#!/bin/bash
#SBATCH -J nsr1_force
#SBATCH --account=gts-nimam6
#SBATCH -p gpu-h200 --gres=gpu:h200:1 -t 2:00:00
#SBATCH -N1 --ntasks-per-node=4 --mem=96G
#SBATCH -o /path/to/pace-scratch/logs/nsr1_force_%j.out
# NSR1 budget forcing for runs that hit the 8,192-token limit (apps/real/force_answer.py as committed at 707351b).
# $1 = model, $2 = problems file, $3 = output dir of that arm's runs.
cd /path/to/pace-scratch/vsassm
export HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case $1 in
  qwen3-32b) SNAP=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-32B/snapshots/* | head -1) ;;
  qwen3-8b) SNAP=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-8B/snapshots/* | head -1) ;;
esac
python3 -u nsr/vendor_707351b/force_answer.py --model $SNAP --name $1 --problems $2 --runs "$3/$1_NSR1_n*.jsonl" \
  --out $3/forced_$1.jsonl --bs 4
