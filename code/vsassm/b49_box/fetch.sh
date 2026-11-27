#!/bin/bash
# B49: copy the box's dev / test outputs, choices and logs into the local b49_out/ (box = authoritative for runs);
# codes, fits and codebooks stay where they are.
cd /path/to/vsassm || exit 1
rsync -a --timeout=120 --exclude '*.npy' --exclude '*.npy.ok' --exclude 'fitcache/' --exclude 'pqcache/' \
  --exclude '*.faiss' --exclude '*.npz' cavin:/path/to/work/b49/b49_out/ b49_out/
rsync -a --timeout=120 cavin:/path/to/work/b49/results/ b49_out/box_results/
echo "fetch rc=$?"
