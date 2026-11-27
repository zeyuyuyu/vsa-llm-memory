#!/bin/bash
# B35 on PACE Phoenix (prereg/B35_soft_token_vsa_prereg.md): run the b35_run.py commands of one list file on one GPU,
# P lanes in parallel (command k goes to lane k mod P). A run whose summary.json already exists is skipped by
# b35_run.py itself. GPU occupancy goes to logs/gpu_ledger_pace.txt.
# usage: sbatch [-p ... --gres=gpu:1] b35_pace.sbatch <list_file> <P>
#SBATCH -J b35
#SBATCH --account=gts-nimam6
#SBATCH -N1 --ntasks=1 -c 8 --mem=96G
#SBATCH -t 3:00:00
#SBATCH -o /path/to/pace-scratch/b35/logs/slurm_%j.out
LIST=$1; P=${2:-2}
cd /path/to/pace-scratch/b35 || exit 1
export HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
export B35_MODEL=/path/to/pace-scratch/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
G=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "pace $SLURM_JOB_ID $(hostname) $G $LIST start $(date -u +%FT%TZ)" >> logs/gpu_ledger_pace.txt
mapfile -t CMDS < "$LIST"
run_lane() {
  local i=$1 k c
  for ((k=i; k<${#CMDS[@]}; k+=P)); do
    c=${CMDS[$k]}
    echo "[$(date -u +%FT%TZ)] job$SLURM_JOB_ID lane$i $c" >> logs/pace.cmds
    python3 -u $c >> logs/pace_${SLURM_JOB_ID}_lane$i.log 2>&1
    echo "[$(date -u +%FT%TZ)] job$SLURM_JOB_ID lane$i rc=$? $c" >> logs/pace.cmds
  done
}
for i in $(seq 0 $((P-1))); do run_lane $i & done
wait
echo "pace $SLURM_JOB_ID $(hostname) $G $LIST end $(date -u +%FT%TZ)" >> logs/gpu_ledger_pace.txt
