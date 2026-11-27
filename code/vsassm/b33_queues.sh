#!/bin/bash
# B33 queues (prereg/B33_fresh_task2_prereg.md), written on the box in /path/to/work/b33 and consumed by b33_worker.sh.
#   bash b33_queues.sh smoke   -> q_smoke  (training rows only; run first, after registration)
#   bash b33_queues.sh main    -> q_main (prompting, both memory runs) and q_lora (12 fine-tune runs, one per seed)
# Workers: bash b33_worker.sh q_smoke 1 smoke ; bash b33_worker.sh q_main 1 main ; bash b33_worker.sh q_lora 1 loraN
cd /path/to/work/b33 || exit 1
case "$1" in
  smoke)
    cat > q_smoke <<'EOF'
b33_babi.py --smoke --patterns completed --seeds 42 --encodings comm perm --out out_smoke
b33_prompt.py --smoke --out out_smoke
b33_lora.py --smoke --rank 8 --epochs 2 --lr 1e-4 --seeds 42 --out out_smoke
EOF
    ;;
  main)
    cat > q_main <<'EOF'
b33_prompt.py --out out
b33_babi.py --patterns completed --out out
b33_babi.py --patterns published --out out
EOF
    : > q_lora
    for cfg in "16 6" "8 2"; do
      set -- $cfg
      for trn in expr all; do
        for s in 42 123 456; do
          echo "b33_lora.py --rank $1 --epochs $2 --lr 1e-4 --train $trn --seeds $s --out out" >> q_lora
        done
      done
    done
    ;;
esac
wc -l q_* 2>/dev/null
