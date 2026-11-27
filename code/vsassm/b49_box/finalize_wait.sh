#!/bin/bash
# wait for every dev part of the given panel encoders, then make their choices (b49_box/finalize_choose.py)
cd /path/to/work/b49 || exit 1
for e in "$@"; do
  ( W=b49_out/enc_$e
    until grep -q DEV_DONE $W/logs/steps.log && [ $(grep -l "part done" $W/logs/vsa_split_*.log 2>/dev/null | wc -l) -eq 6 ]; do sleep 30; done
    /path/to/HOME/envs/b49/bin/python b49_box/finalize_choose.py $e > $W/logs/finalize_choose.log 2>&1 ) &
done
wait
