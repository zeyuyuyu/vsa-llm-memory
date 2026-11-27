#!/bin/bash
#SBATCH -J real_force
#SBATCH --account=gts-nimam6-paid
#SBATCH -N1 --ntasks-per-node=4 --mem=96G
#SBATCH -o /path/to/pace-scratch/logs/real_force_%j.out
# Budget forcing for the LLM reference runs that hit the reasoning limit (apps/real/force_answer.py).
# $1 = model (qwen3-1.7b | qwen3-8b | qwen3-32b), $2 = directory with that model's test2 runs, $3 = problems glob.
cd /path/to/pace-scratch/vsassm
export HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case $1 in
  qwen3-32b) SNAP=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-32B/snapshots/* | head -1) ;;
  qwen3-8b) SNAP=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-8B/snapshots/* | head -1) ;;
  qwen3-1.7b) SNAP=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-1.7B/snapshots/* | head -1) ;;
esac
python3 -u apps/real/force_answer.py --model $SNAP --name $1 --problems ${3:-apps/real/llm_problems_test2_assets_*.jsonl} \
  --runs "$2/$1_REAL_n*.jsonl" --out $2/forced_$1.jsonl --bs 4
