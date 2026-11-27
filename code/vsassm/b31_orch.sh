#!/bin/bash
# B31 phase scheduler on the A800 box (prereg/B31_encoder_equalised_baselines_prereg.md). Every 2 minutes: re-run the
# registered selection (b31_select.py, train + dev outputs only), append newly unlocked jobs to the live queues
# (b31_queues.py p2a once, p2b and p3 incrementally), and start a b31_worker.sh for a non-empty queue when fewer
# workers serve it than it has jobs (a worker itself waits for an idle GPU; never GPU 3, never another process).
# Stops when every final output exists. Launches only; it never kills anything.
# v2 (compute notes 2, 3): at most 5 B31 workers on the box at a time (at most 3 of them on Qwen3-8B queues), so that GPUs
# are left for other work; heavy workers chain p2_heavy -> p3_heavy, medium workers p2_medium -> p3_medium.
B=/path/to/work/b31; cd $B || exit 1
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
export HF_HOME=/path/to/work/hf_home HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
nall() { ps -eo ppid=,cmd= | awk '$1==1 && $2=="/bin/bash" && $3=="./b31_worker.sh"' | wc -l; }
nheavy() { ps -eo ppid=,cmd= | awk '$1==1 && $2=="/bin/bash" && $3=="./b31_worker.sh" && $0 ~ /_heavy/' | wc -l; }
MAXALL=5; MAXHEAVY=${B31_MAXHEAVY:-3}
launch() {  # kind(heavy|medium) queue lanes next_queue
  local kind=$1 q=$2 lanes=$3 nq=$4
  [ -s $q ] || return
  [ $(nall) -lt $MAXALL ] || return
  if [ $kind = heavy ] && [ $(nheavy) -ge $MAXHEAVY ]; then return; fi
  (setsid nohup ./b31_worker.sh $q $lanes o$(date +%H%M%S) $nq $lanes > /dev/null 2>&1 < /dev/null &)
  echo "$(date -u +%FT%TZ) launched worker on $q (then $nq)" >> logs/orch.log
  sleep 3
}
while true; do
  $PY b31_select.py > logs/select.out 2>&1
  na=$(ls out/arms_tune/*.json 2>/dev/null | wc -l); np=$(ls out/probe_tune/*.json 2>/dev/null | wc -l)
  if [ ! -f q/done_p2a ] && [ $na -ge 48 ] && [ $np -ge 6 ]; then
    $PY b31_queues.py p2a >> logs/orch.log 2>&1 && touch q/done_p2a
  fi
  $PY b31_queues.py p2b >> logs/orch.log 2>&1
  $PY b31_queues.py p3 >> logs/orch.log 2>&1
  launch heavy q/p2_heavy 1 q/p3_heavy
  launch heavy q/p3_heavy 1 q/p2_heavy
  launch medium q/p2_medium 2 q/p3_medium
  launch medium q/p3_medium 2 q/p2_medium
  nf=$(ls out/lora_final/*_s*.json out/arms_final/*.json out/probe_final/*.json 2>/dev/null | wc -l)
  if [ $nf -ge $((36 + 3 + 30 + 6)) ]; then echo "$(date -u +%FT%TZ) all finals present" >> logs/orch.log; break; fi
  sleep 120
done
