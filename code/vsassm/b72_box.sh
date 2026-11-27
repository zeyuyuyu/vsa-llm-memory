#!/bin/bash
# B72 on the A800 box (prereg/B72_format_side_info_prereg.md, 'Compute'): fresh directory ~/b72_run, code copied from the
# repository (md5 in ~/b72_run/code_md5.txt), panel codes copied from ~/b71_run's B49 copies after the md5 check of
# b71_box.sh. One process per job, OMP / BLAS threads 1, nice 5, at most 48 workers. Plans are written locally
# (b72_run.py plan) and copied in; the committed choice files go to results/.
# usage (on the box): bash b72_box.sh setup
#                     bash b72_box.sh run PLAN.jsonl WORKERS
set -u
W=$HOME/b72_run
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
export B60_DATA=$W/b49_out B72_OUT=$W/b72_out B72_RES=$W/results B64_DATA=$W/b49_out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
case $1 in
setup)
  mkdir -p $W/b49_out/b32_data $W/b72_out/logs $W/results
  S=$HOME/b71_run/b49_out/b32_data
  declare -A MD5=([emb_qwen8_mean_L9_f16.npy]=1e4f5a91b57bd8987c899b770ff0c915 [emb_e5_base.npy]=836cff40465c5aa326052b7c36ef344d
    [emb_bge_small.npy]=5cda7b452f00669817796baa13d4cebe [emb_minilm.npy]=9d2d9bcccee31d61bc05ea306d255c54
    [emb_b46_granite33i_mean_L30_f16.npy]=c474cf933dc90f0d0f3eaff1b95e26a3
    [emb_llama31i_mean_L24_f16.npy]=a55892a0ae069b160cb2dfaedce914a4 [emb_mistral7i_eol_L24_f16.npy]=52207e3edaddb67145252805c30954be
    [entities.json]=f369b096f9edd953214193bc97c4ee32 [emb_strings.json]=1db6556d2d3a0b1d8d904d7cf7961b1e)
  for f in "${!MD5[@]}"; do
    dst=$W/b49_out/b32_data/$f
    [ -f $dst ] || cp $S/$f $dst
    if [ "$(md5sum < $dst | cut -c1-32)" = "${MD5[$f]}" ]; then echo "ok $f"; else echo "MISSING $f"; fi
  done
  (cd $W && md5sum b32_common.py b32_run.py b36_common.py b49_common.py b53_run.py b56_run.py b60_run.py b62_run.py \
     b64_run.py b66_lsfc.py b71_run.py b72_run.py > code_md5.txt; cat code_md5.txt) ;;
run)
  cd $W || exit 1
  nohup nice -n 5 $PY b72_run.py runplan $W/b72_out/$2 --workers $3 > $W/b72_out/logs/run_${2%.jsonl}.log 2>&1 &
  echo "started $2 pid $!" ;;
choose)  # the registered dev rule on the box's dev rows (compute note 1); choice files -> choices_out.tgz
  cd $W || exit 1
  for e in qwen8 b46 llama mistral; do $PY b72_run.py choose $e ${2:-}; done
  tar czf choices_out.tgz results/b72_choice_qwen8.json results/b72_choice_b46.json results/b72_choice_llama.json \
    results/b72_choice_mistral.json; ls -la choices_out.tgz ;;
plantest)  # after the committed choices are back in results/ (md5 checked by the caller)
  cd $W || exit 1
  $PY b72_run.py plan test --encs qwen8,b46,llama,mistral --arms $2 --out $W/b72_out/plan_test_box_$3.jsonl ;;
after)  # (B) verdict + bits audit on the box, pack (A) / (D) test outputs (compute note 2)
  cd $W || exit 1
  for e in qwen8 b46 llama mistral; do $PY b72_bverdict.py $e & done; wait
  $PY b72_bverdict.py bits qwen8 b46 llama mistral
  $PY b72_pack.py pack $W/ad_test.npz _OPT _R_test
  tar czf after_out.tgz results/b72_bpart_*.json results/b72_check_bits_box.json ad_test.npz; ls -la after_out.tgz ;;
*) echo "unknown $1" ;;
esac
