#!/bin/bash
# B44: copy finished outputs (JSON only) from the A800 box into b44_out/ (box runs: features, frozen-feature arms,
# 800-instance LoRA finals, CoT std test). Checkpoints and adapter states stay on the box.
cd /path/to/vsassm/b44_out || exit 1
timeout 900 rsync -a -e "ssh -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm/b44-%r@%h:%p -o ControlPersist=30m" \
  --include='*/' --include='*.json' --exclude='*' cavin:/path/to/work/b44/out/ ./ 2>&1 | tail -2
timeout 300 rsync -a -e "ssh -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm/b44-%r@%h:%p -o ControlPersist=30m" \
  --include='*.log' --include='*.cmds' --include='gpu_ledger.txt' --exclude='*' cavin:/path/to/work/b44/logs/ ./box_logs/ 2>&1 | tail -2
