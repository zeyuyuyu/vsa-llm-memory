#!/bin/bash
# B73 language-model chain on the A800 box (compute note 4): encode the four language models (b73_box.sh embed, one
# idle GPU each among 1 2 4 5 6 7), then the test (b73_run.py runplan, 48 workers, no unit started after B73_DEADLINE),
# then assembly, the verdict's per-encoder parts, the checks, a determinism sample and the Qwen3-8B sample reading.
# One launch; every step runs once. usage (on the box): nohup bash b73_box_all.sh DEADLINE_EPOCH > logs/all.log 2>&1 &
set -u
W=$HOME/b73_run; cd $W || exit 1
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
mkdir -p logs run
echo "start $(date +%T)"
for m in qwen8 llama mistral b46; do
  bash b73_box.sh embed $m > logs/embed_$m.log 2>&1 &
  sleep 25
done
wait
grep -h "EMBED_DONE\|NO_IDLE_GPU" logs/embed_*.log
md5sum fresh/b32_data/emb_*_f16.npy | tee logs/codes_md5.txt
echo "encoded $(date +%T)"
export B73_ROOT=$W/fresh B73_OUT=$W/run B73_DEADLINE=$1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ENCS=""
for e in qwen8 llama mistral b46; do
  case $e in qwen8) f=emb_qwen8_mean_L9_f16;; llama) f=emb_llama31i_mean_L24_f16;; mistral) f=emb_mistral7i_eol_L24_f16;;
    b46) f=emb_b46_granite33i_mean_L30_f16;; esac
  [ -f fresh/b32_data/$f.npy ] && ENCS="$ENCS,$e"
done
ENCS=${ENCS#,}
echo "encoders with codes: $ENCS"
$PY b73_run.py plan --encs $ENCS > plan_llm.txt
nice -n 5 $PY b73_run.py runplan plan_llm.txt --workers 48 > logs/runplan_llm.log 2>&1
tail -1 logs/runplan_llm.log
echo "tested $(date +%T)"
$PY b73_run.py assemble ${ENCS//,/ } > logs/assemble.log 2>&1
$PY b73_verdict.py part --asm run/asm --encs $ENCS > logs/verdict_part.log 2>&1
$PY b73_verdict.py --encs $ENCS > logs/verdict_panel_llm.log 2>&1
$PY b73_check.py score run ${ENCS//,/ } > logs/check_score.log 2>&1
$PY b73_check.py bits run ${ENCS//,/ } > logs/check_bits.log 2>&1
printf 'LAMC qwen8 1024 16 0-4\nNLMSC llama 256 2 0-4\nSTORE mistral 1024 2\nVSAS qwen8 256 16\nVSAPC llama 1024 16\nSDMT mistral 256 16 5-9\n' > plan_det.txt
B73_OUT=$W/run_det B73_DEADLINE=inf nice -n 5 $PY b73_run.py runplan plan_det.txt --workers 6 > logs/runplan_det.log 2>&1
$PY b73_check.py same run run_det > logs/check_same.log 2>&1
$PY b73_sample.py qwen8 > logs/sample_qwen8.txt 2>&1
tar czf llm_results.tgz results/b73_part_*.json results/b73_check_*.json results/b73_verdict.* logs/*.log logs/*.txt plan_llm.txt
md5sum llm_results.tgz
echo "ALL_DONE $(date +%T)"
