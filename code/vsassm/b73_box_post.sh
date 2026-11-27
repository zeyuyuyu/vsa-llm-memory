#!/bin/bash
# B73 (compute note 5): the steps after the language-model test, run in parallel instead of b73_box_all.sh's sequential
# tail (same commands, same outputs; the chain's shell was stopped by its PID while its test runner kept running).
set -u
W=$HOME/b73_run; cd $W || exit 1
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
export B73_ROOT=$W/fresh B73_OUT=$W/run OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
until grep -q "^finished" logs/runplan_llm.log 2>/dev/null; do sleep 10; done
tail -1 logs/runplan_llm.log
echo "tested $(date +%T)"
$PY b73_run.py assemble qwen8 llama mistral b46 > logs/assemble.log 2>&1
for e in qwen8 llama mistral b46; do $PY b73_verdict.py part --asm run/asm --encs $e > logs/verdict_part_$e.log 2>&1 & done
$PY b73_check.py bits run qwen8 llama mistral b46 > logs/check_bits.log 2>&1 &
( printf 'LAMC qwen8 1024 16 0-4\nNLMSC llama 256 2 0-4\nSTORE mistral 1024 2\nVSAS qwen8 256 16\nVSAPC llama 1024 16\nSDMT mistral 256 16 5-9\nVSAR b46 256 2\nLAMC b46 256 16 10-14\n' > plan_det.txt
  B73_OUT=$W/run_det B73_DEADLINE=inf nice -n 5 $PY b73_run.py runplan plan_det.txt --workers 8 > logs/runplan_det.log 2>&1
  $PY b73_check.py same run run_det > logs/check_same.log 2>&1 ) &
$PY b73_sample.py qwen8 > logs/sample_qwen8.txt 2>&1 &
wait
echo "parts $(date +%T)"
$PY b73_verdict.py --encs qwen8,llama,mistral,b46 > logs/verdict_panel_llm.log 2>&1
$PY b73_check.py score run qwen8 llama mistral b46 > logs/check_score.log 2>&1
tar czf llm_results.tgz results/b73_part_*.json results/b73_check_*.json results/b73_verdict.* logs/*.log logs/*.txt plan_llm.txt plan_det.txt
md5sum llm_results.tgz
echo "ALL_DONE $(date +%T)"
