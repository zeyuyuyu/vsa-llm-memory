#!/bin/bash
#SBATCH -J nsr1_llm
#SBATCH --account=gts-nimam6
#SBATCH -p gpu-h200 --gres=gpu:h200:1 -t 8:00:00
#SBATCH -N1 --ntasks-per-node=4 --mem=96G
#SBATCH -o /path/to/pace-scratch/logs/nsr1_llm_%j.out
# NSR1 LLM arms, apps/real protocol (b6_qwen.py: thinking on, greedy, 8,192 new tokens).
# $1 = model (qwen3-32b | qwen3-8b), $2 = problems file, $3 = output dir, $4 = problem numbers (n) of this shard, $5 = bs.
cd /path/to/pace-scratch/vsassm
export HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case $1 in
  qwen3-32b) SNAP=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-32B/snapshots/* | head -1) ;;
  qwen3-8b) SNAP=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-8B/snapshots/* | head -1) ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -u b6_qwen.py --model $SNAP --name $1 --task NSR1 --problems $2 --out $3 --bs ${5:-8} --max_new 8192 --ns $4
