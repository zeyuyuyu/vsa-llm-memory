#!/bin/bash
# B31 (compute note 3): move queued jobs onto queues that B31 workers already holding a GPU still serve (so that freed GPUs are not lost):
# Phi-2 jobs p2_medium/p3_medium -> p1_medium (served by l1/l2), Qwen jobs p2_heavy/p3_heavy -> q/ex1 while the ex1
# worker is alive and has fewer than 2 jobs left. Job lines are moved, never copied.
cd /path/to/work/b31
mv_lines() {  # src dst
  local src=$1 dst=$2
  [ -s $src ] || return
  flock $src.lock -c "cat $src > $src.mv; : > $src"
  flock $dst.lock -c "cat $src.mv >> $dst"
  echo "$(date -u +%FT%TZ) moved $(wc -l < $src.mv) from $src to $dst" >> logs/retain.log
  rm -f $src.mv
}
alive() { ps -eo ppid=,cmd= | awk -v n="$1" '$1==1 && $3=="./b31_worker.sh" && $0 ~ n' | grep -c . ; }
if [ $(alive " l1 | l2 ") -gt 0 ] || [ $(alive "p1_medium") -gt 0 ]; then
  for q in q/p2_medium q/p3_medium; do mv_lines $q q/p1_medium; done
fi
if [ ! -f q/NO_HEAVY_RETAIN ] && [ $(alive " ex1") -gt 0 ] && [ $(wc -l < q/ex1) -lt 2 ]; then
  for q in q/p2_heavy q/p3_heavy; do
    [ -s $q ] || continue
    flock $q.lock -c "head -n 1 $q > $q.mv; tail -n +2 $q > $q.rest; cat $q.rest > $q; rm $q.rest"
    flock q/ex1.lock -c "cat $q.mv >> q/ex1"
    echo "$(date -u +%FT%TZ) moved 1 from $q to q/ex1" >> logs/retain.log; rm -f $q.mv
    break
  done
fi
wc -l q/p1_medium q/ex1 q/p2_heavy q/p2_medium 2>/dev/null | tr "\n" " "
