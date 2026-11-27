#!/bin/bash
#SBATCH -J b24dev
#SBATCH --account=gts-nimam6-paid
#SBATCH -p gpu-h200,gpu-h100,gpu-a100,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --constraint=H200|H100|A100|L40S
#SBATCH -N1 --ntasks-per-node=8 --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /path/to/pace-scratch/logs/b24dev_%j.out
# B24 dev pipeline check (seed-7 dev set, 4k and 16k, 20 samples per task): B21's reader + store (ruler_vsa.py,
# unchanged) writes B21-format rows, then arm F reads their pairs and arm R retrieves over the same samples.
cd /path/to/pace-scratch/vsassm
export HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S8=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-8B/snapshots/* | head -1)
T=niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt
D=/path/to/pace-scratch/bench/ruler_dev
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; hostname; date
mkdir -p b24_dev/reader b24_dev/arms
for L in 4096 16384; do
  [ -f b24_dev/reader/${L}_done ] || { rm -f b24_dev/reader/${L}_niah_single_1_0.jsonl; python3 -u ruler_vsa.py run b24_dev/reader/${L}_niah_single_1_0.jsonl --model $S8 \
    --root $D --length $L --tasks $T --n 20 && touch b24_dev/reader/${L}_done; } || exit 1
done
for L in 4096 16384; do
  for M in armf armr; do
    python3 -u ruler_b24.py $M b24_dev/arms/${M}_${L}.jsonl --model $S8 --root $D --length $L --tasks $T \
      --b21 b24_dev/reader --n 20 || exit 1
  done
done
date; echo B24 DEV DONE
