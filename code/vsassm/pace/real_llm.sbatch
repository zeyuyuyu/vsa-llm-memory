#!/bin/bash
#SBATCH -J real_llm
#SBATCH --account=gts-nimam6-paid
#SBATCH -N1 --ntasks-per-node=4 --mem=96G
#SBATCH -o /path/to/pace-scratch/logs/real_llm_%j.out
# Realistic scenarios: an LLM reads the whole site log and answers with its reasoning mode on (greedy, 8,192 new
# tokens). $1 = model (qwen3-1.7b | qwen3-8b | qwen3-32b), $2 = problems file, $3 = problem numbers of this shard.
# $4 = output directory (default $OUT, else apps/real/out), $5 = batch size (default $BS, else 8). Positional,
# because the environment does not reach the job on this cluster.
cd /path/to/pace-scratch/vsassm
export HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case $1 in
  qwen3-32b) SNAP=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-32B/snapshots/* | head -1) ;;
  qwen3-8b) SNAP=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-8B/snapshots/* | head -1) ;;
  qwen3-1.7b) SNAP=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-1.7B/snapshots/* | head -1) ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -u b6_qwen.py --model $SNAP --name $1 --task REAL --problems $2 --out ${4:-${OUT:-apps/real/out}} --bs ${5:-${BS:-8}} --max_new 8192 --ns $3
