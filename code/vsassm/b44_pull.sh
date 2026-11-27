#!/bin/bash
# B44: copy finished outputs (JSON only: no checkpoints, no adapter states) from PACE to b44_out/, and job logs.
PK="-i $HOME/.ssh/id_ed25519_pace -o BatchMode=yes -o ConnectTimeout=25"
cd /path/to/vsassm/b44_out || exit 1
timeout 300 rsync -a -e "ssh $PK" --include='*/' --include='*.json' --exclude='*' \
  <user>@login-phoenix.pace.gatech.edu:/path/to/pace-scratch/b44/out/ ./ 2>&1 | tail -2
timeout 300 rsync -a -e "ssh $PK" <user>@login-phoenix.pace.gatech.edu:/path/to/pace-scratch/b44/logs/ ./pace_logs/ 2>&1 | tail -2
ls lora name cot k smoke 2>/dev/null | head -80
