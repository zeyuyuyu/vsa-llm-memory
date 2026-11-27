#!/bin/bash
# B43 test-phase submissions on PACE (prereg/B43_natural_text_and_chunked_baseline_prereg.md). Run on the PACE login
# node from /path/to/pace-scratch/b43, only after the prereg (with its dev record) is committed.
# usage: bash b43_submit.sh ruler128 | ruler64 | bab | propara
cd /path/to/pace-scratch/b43 || exit 1
mkdir -p out logs
Q8=/path/to/pace-scratch/hf/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
R=/path/to/pace-scratch/bench/ruler
NV="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt"
case $1 in
  ruler128) for t in $NV; do sbatch -J b43_r128_$t -t 6:00:00 b43_pace.sbatch b43_coa.py ruler out/coa_ruler_131072_$t.jsonl \
              --model $Q8 --root $R --length 131072 --tasks $t --n 100 --bs 16; done ;;
  ruler64)  for t in $NV; do sbatch -J b43_r64_$t -t 4:00:00 b43_pace.sbatch b43_coa.py ruler out/coa_ruler_65536_$t.jsonl \
              --model $Q8 --root $R --length 65536 --tasks $t --n 100 --bs 16; done ;;
  bab)      for t in qa1 qa2 qa3 qa4 qa5; do sbatch -J b43_bab_$t -t 6:00:00 b43_pace.sbatch b43_coa.py babilong \
              out/coa_bab_128k_$t.jsonl --model $Q8 --length 128k --tasks $t --n 100 --bs 16; done ;;
  propara)  sbatch -J b43_pptest -t 2:00:00 b43_pace.sbatch b43_propara.py gen direct_think,direct_nothink,extract test \
              out/pp_test --model $Q8 ;;
esac
squeue -u <user> -o "%i %j %T %M %R"
