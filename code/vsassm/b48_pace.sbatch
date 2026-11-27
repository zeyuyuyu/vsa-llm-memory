#!/bin/bash
# B48 job on PACE Phoenix (prereg/B48_one_form_and_alias_naming_prereg.md). Two forms:
#   sbatch -J b48_<name> b48_pace.sbatch cmd <python script and args...>     one command
#   sbatch -J b48_<name> b48_pace.sbatch sh <shell script>                    one shell script
#   sbatch -J b48_<name> b48_pace.sbatch queue <queue file> <lanes>          pop lines of a queue file (flock), <lanes>
#                                                                            at a time on this one GPU
# Same Qwen/Qwen3-8B snapshot (b968826d) and microsoft/phi-2 snapshot as B31 / B44. No retry inside the job.
#SBATCH -A gts-nimam6-paid
#SBATCH -p gpu-h200,gpu-h100,gpu-a100,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH -C H200|H100|A100|L40S
#SBATCH -c 8
#SBATCH --mem=150G
#SBATCH -t 1:00:00
#SBATCH -o /path/to/pace-scratch/b48/logs/%x_%j.out
cd /path/to/pace-scratch/b48 || exit 1
export HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export B31_QWEN8=/path/to/pace-scratch/hf/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
echo "B48-PACE start $(date -u +%FT%TZ) host $(hostname) job $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MODE=$1; shift
if [ "$MODE" = cmd ]; then
  /usr/bin/python3 -u "$@"; rc=$?
elif [ "$MODE" = sh ]; then
  bash "$@"; rc=$?
else
  Q=$1; P=${2:-1}; mkdir -p logs
  pop() { flock "$Q.lock" -c "head -n1 '$Q'; sed -i '1d' '$Q'"; }
  lane() {
    while true; do
      c=$(pop); [ -z "$c" ] && break
      echo "[$(date -u +%FT%TZ)] job$SLURM_JOB_ID lane$1 $c" >> logs/cmds.log
      /usr/bin/python3 -u $c >> logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.lane$1.log 2>&1
      r=$?
      echo "[$(date -u +%FT%TZ)] job$SLURM_JOB_ID lane$1 rc=$r $c" >> logs/cmds.log
      [ $r -ne 0 ] && echo "$c" >> "$Q.failed"
    done
  }
  for i in $(seq 1 $P); do lane $i & done
  wait; rc=0
fi
echo "B48-PACE end $(date -u +%FT%TZ) rc=$rc"
exit $rc
