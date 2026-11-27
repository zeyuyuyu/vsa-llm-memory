#!/bin/bash
# B49 phase 1 on the A800 box's CPU (prereg/B49_compute_note1.md): dev split only, no test store is built.
# 1. panel dev (B32 + B36 unchanged) for llama, mistral, bge, minilm; 2. learned-map choice (B32's criterion, no memory)
# for b46, llama, mistral, bge, minilm; 3. the new families' dev grid on all seven encoders; 4. mechanism codes (T1, T2,
# T3) and their R1 dev runs. CPU only (nice 10); no GPU; no other process touched.
export B49_ROOT=/path/to/work/b49 B49_PY=/path/to/HOME/envs/b49/bin/python B49_NICE=10
export PYTHONDONTWRITEBYTECODE=1
cd $B49_ROOT || exit 1
PY=$B49_PY; N="nice -n 10"
L=b49_out/logs; mkdir -p $L
ts() { date '+%F %T'; }
echo "$(ts) phase1 start" >> $L/phase1.log
declare -A TAG=([llama]=llama31i_mean_L24_f16 [mistral]=mistral7i_eol_L24_f16 [bge]=bge_small [minilm]=minilm)
for e in llama mistral bge minilm; do
  until [ -f b49_out/b32_data/emb_${TAG[$e]}.npy.ok ]; do sleep 30; done
  $PY b49_run.py $e setup b49_out/b32_data/emb_${TAG[$e]}.npy >> $L/phase1.log 2>&1
  nohup bash b49_dev.sh $e 8 > b49_out/enc_$e/logs/driver.log 2>&1 &
done
# learned map (one process per encoder)
for kv in b46=b46_granite33i_mean_L30_f16 llama=llama31i_mean_L24_f16 mistral=mistral7i_eol_L24_f16 bge=bge_small minilm=minilm; do
  e=${kv%%=*}
  ( cd b49_out && OMP_NUM_THREADS=8 $N $PY ../b49_lhchoose.py lh_choice_$e.json $kv > logs/lhchoose_$e.log 2>&1
    $PY -c "import json,os; d=json.load(open('lh_choice_$e.json'))['$e']; os.makedirs('fam/$e',exist_ok=True); json.dump(d['choice'],open('fam/$e/lh_choice.json','w')); print('$e',d['choice'],d['base_criterion'])" >> logs/phase1.log 2>&1 ) &
done
# mechanism codes (T2 needs e5 / bge codes)
( for bt in qwen8:T1 qwen8:T3 b46:T1 b46:T3 e5:T2 bge:T2; do
    b=${bt%%:*}; t=${bt##*:}
    [ $b = e5 ] && until [ -f b49_out/b32_data/emb_e5_base.npy.ok ]; do sleep 30; done
    [ $b = bge ] && until [ -f b49_out/b32_data/emb_bge_small.npy.ok ]; do sleep 30; done
    OMP_NUM_THREADS=8 $N $PY b49_mech.py $b $t make > $L/mech_make_${b}_$t.log 2>&1
    echo "$(ts) mech make $b $t rc=$?" >> $L/phase1.log
    nohup bash b49_box/mech_dev.sh $b $t > $L/mech_dev_${b}_$t.log 2>&1 &
  done ) &
# new families' dev grid, all seven encoders (after each learned map is chosen)
for e in qwen8 e5 b46 llama mistral bge minilm; do
  ( until [ -f b49_out/fam/$e/lh_choice.json ]; do sleep 30; done
    for t in bge_small minilm; do [ $e = ${t%_small} ] && until [ -f b49_out/b32_data/emb_$t.npy.ok ]; do sleep 30; done; done
    [ $e = e5 ] && until [ -f b49_out/b32_data/emb_e5_base.npy.ok ]; do sleep 30; done
    OMP_NUM_THREADS=8 $N $PY b49_fam.py $e prefit > $L/fam_prefit_$e.log 2>&1
    OMP_NUM_THREADS=2 $N $PY b49_fam.py $e dev --workers 6 > $L/fam_dev_$e.log 2>&1
    echo "$(ts) fam dev $e rc=$?" >> $L/phase1.log ) &
done
wait
echo "$(ts) phase1 waiters done" >> $L/phase1.log
