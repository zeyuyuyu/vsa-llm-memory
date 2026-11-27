#!/bin/bash
# B44: tokenisation cache on PACE (CPU only), the same b44_tok.py as on the box; checksums compared with the box's.
#SBATCH -A gts-nimam6
#SBATCH -p cpu-small
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH -o /path/to/pace-scratch/b44/logs/%x_%j.out
cd /path/to/pace-scratch/b44 || exit 1
export HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export B31_QWEN8=/path/to/pace-scratch/hf/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
KS=""; for k in 2 4 8 16 24 32; do for s in train dev test; do KS="$KS k$k:$s"; done; done
/usr/bin/python3 -u b44_tok.py --enc qwen8 --items std:train std:dev std:test ho:train ho:dev ho:test $KS
/usr/bin/python3 -u b44_tokmd5.py data/qwen8_*.pt
