#!/bin/bash
# B43: gzip the per-sample outputs for the commit (originals stay on disk, uncommitted).
cd /path/to/vsassm
for f in b43_out/dev16_coa.jsonl b43_out/coa_test/*.jsonl b43_out/pp_dev/raw_*.jsonl b43_out/pp_test/raw_*.jsonl; do
  [ -f "$f" ] && { [ -f "$f.gz" ] && [ "$f.gz" -nt "$f" ] || gzip -kf9 "$f"; }
done
du -ch b43_out/*.gz b43_out/*/*.gz | tail -1
