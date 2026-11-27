#!/bin/bash
# B44 Fig. 2c sweep for one encoder (run on PACE or the box): features of the k-sweep splits, then for every k the
# exact rows, the naming head (3 seeds) and the equalised probe. Usage: bash b44_kseq.sh <enc> [python]
ENC=$1; PY=${2:-/usr/bin/python3}
ITEMS=""; for k in 2 4 8 16 24 32; do for s in train dev test; do ITEMS="$ITEMS k$k:$s"; done; done
$PY -u b44_tok.py --enc $ENC --items $ITEMS || exit 1
$PY -u b44_feats.py --enc $ENC --items $ITEMS || exit 1
for k in 2 4 8 16 24 32; do
  for arm in exact name probe; do
    $PY -u b44_name.py --cell k$k --enc $ENC --arm $arm --out out/k/${ENC}_k${k}_${arm}.json || echo "FAILED k$k $arm"
  done
done
