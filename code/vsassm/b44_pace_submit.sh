#!/bin/bash
# B44: submit ONE b44 command to PACE as a chain of NSEG segment jobs (each <= 3 h; the LoRA run checkpoints and exits
# 3 when its --max_hours budget is reached, the next segment resumes; a segment that finds the run finished exits).
# No retry: a failed submission is logged and must be checked with squeue before anything is resubmitted.
# usage: [DEP=<jobid>] [TL=<time limit, default 1:30:00>] [SB=<sbatch file>] b44_pace_submit.sh <name> <nseg> <b44 command and args...>   (DEP: the first segment waits for
# that job to end OK)
NAME=$1; NSEG=$2; shift 2
PK="-i $HOME/.ssh/id_ed25519_pace -o BatchMode=yes -o ConnectTimeout=25"
PH=<user>@login-phoenix.pace.gatech.edu
PR=/path/to/pace-scratch/b44
LOG=/path/to/vsassm/b44_out/pace_submissions.log
mkdir -p $(dirname $LOG)
D0=""; [ -n "$DEP" ] && D0="--dependency=afterok:$DEP"
TL=${TL:-1:30:00}; SB=${SB:-b44_pace.sbatch}
r=$(timeout 120 ssh $PK $PH "cd $PR && prev=; for i in \$(seq 1 $NSEG); do if [ -z \"\$prev\" ]; then o=\$(sbatch --parsable -t $TL -J b44_$NAME $D0 $SB $*); else o=\$(sbatch --parsable -t $TL -J b44_$NAME --dependency=afterany:\$prev $SB $*); fi; echo \$o; prev=\$o; done" < /dev/null 2>&1 | tr '\n' ' ')
echo "$(date -u +%FT%TZ) $NAME nseg=$NSEG jobs: $r cmd: $*" >> $LOG
echo "$NAME: $r"
