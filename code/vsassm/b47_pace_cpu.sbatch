#!/bin/bash
# B47: Phi-2 tokenisation cache on PACE (CPU only; b47_tok.py = b44_tok's code), then the content checksums.
#SBATCH -A gts-nimam6
#SBATCH -p cpu-small
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH -o /path/to/pace-scratch/b47/logs/%x_%j.out
cd /path/to/pace-scratch/b47 || exit 1
export HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
/usr/bin/python3 -u b47_tok.py --root data
/usr/bin/python3 -u b44_tokmd5.py data/phi2_*.pt
