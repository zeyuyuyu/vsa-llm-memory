#!/bin/bash
# B49 mechanism run, dev part (R1 only; prereg/B49 section 6): fits, B32 VSA / index grids, B36 families, choices
# (B32 / B36), learned map, new families' dev grid. Dev split only.
B=$1; T=$2
cd ${B49_ROOT:-/path/to/work/b49} || exit 1
PY=${B49_PY:-/path/to/HOME/envs/b49/bin/python}; N="nice -n 10"
W=b49_out/mech_${B}_$T; L=$W/logs; mkdir -p $L
ts() { date '+%F %T'; }
echo "$(ts) mech dev start $B $T" >> $L/steps.log
for prep in center whiten; do for fam in pq4 pq8 opq8; do
  B32_FAISS_THREADS=2 OMP_NUM_THREADS=2 $N $PY b49_mech.py $B $T fits $prep $fam > $L/fit_${prep}_$fam.log 2>&1 &
done; done
OMP_NUM_THREADS=4 $N $PY b49_mech.py $B $T pqbank > $L/pqbank.log 2>&1
echo "$(ts) pqbank rc=$?" >> $L/steps.log
OMP_NUM_THREADS=2 $N $PY b49_mech.py $B $T dev36 --workers 4 > $L/dev36.log 2>&1 &
P1=$!
OMP_NUM_THREADS=4 $N $PY b49_mech.py $B $T dev32vsa > $L/dev32vsa.log 2>&1 &
P2=$!
( cd b49_out && OMP_NUM_THREADS=8 $N $PY ../b49_lhchoose.py lh_choice_${B}_$T.json ${B}_$T=${B}_$T > logs/lhchoose_${B}_$T.log 2>&1
  $PY -c "import json,os; d=json.load(open('lh_choice_${B}_$T.json'))['${B}_$T']; os.makedirs('fam/${B}_$T',exist_ok=True); json.dump(d['choice'],open('fam/${B}_$T/lh_choice.json','w'))" )
B49_GRID_V=2,16 B49_GRID_BPF=4,8,16 OMP_NUM_THREADS=8 $N $PY b49_fam.py ${B}_$T prefit > $L/fam_prefit.log 2>&1
B49_GRID_V=2,16 B49_GRID_BPF=4,8,16 OMP_NUM_THREADS=2 $N $PY b49_fam.py ${B}_$T dev --workers 4 > $L/fam_dev.log 2>&1 &
P3=$!
while pgrep -u $(whoami) -f "b49_mech.py $B $T fits" > /dev/null; do sleep 20; done
echo "$(ts) fits done" >> $L/steps.log
OMP_NUM_THREADS=4 $N $PY b49_mech.py $B $T dev32idx > $L/dev32idx.log 2>&1
echo "$(ts) dev32idx rc=$?" >> $L/steps.log
wait $P1; echo "$(ts) dev36 rc=$?" >> $L/steps.log
wait $P2; echo "$(ts) dev32vsa rc=$?" >> $L/steps.log
wait $P3; echo "$(ts) fam dev rc=$?" >> $L/steps.log
$PY b49_mech.py $B $T choose > $L/choose.log 2>&1; echo "$(ts) choose rc=$?" >> $L/steps.log
echo "$(ts) MECH_DEV_DONE" >> $L/steps.log
