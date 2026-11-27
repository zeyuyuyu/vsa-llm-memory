#!/bin/bash
# B47: submit ONE b47 command to PACE as a chain of NSEG segment jobs (b44_pace_submit.sh with B47 paths). No retry: a
# failed submission is logged and must be checked with squeue before anything is resubmitted.
# usage: [DEP=<jobid>] [TL=<time limit, default 1:30:00>] [ACC=gts-nimam6|gts-nimam6-paid] b47_pace_submit.sh <name> <nseg> <command and args...>
NAME=$1; NSEG=$2; shift 2
PK="-i $HOME/.ssh/id_ed25519_pace -o BatchMode=yes -o ConnectTimeout=25"
PH=<user>@login-phoenix.pace.gatech.edu
PR=/path/to/pace-scratch/b47
LOG=/path/to/vsassm/b47_out/pace_submissions.log
mkdir -p $(dirname $LOG)
D0=""; [ -n "$DEP" ] && D0="--dependency=afterok:$DEP"
TL=${TL:-1:30:00}; ACC=${ACC:-gts-nimam6}
r=$(timeout 120 ssh $PK $PH "cd $PR && prev=; for i in \$(seq 1 $NSEG); do if [ -z \"\$prev\" ]; then o=\$(sbatch --parsable -A $ACC -t $TL -J b47_$NAME $D0 b47_pace.sbatch $*); else o=\$(sbatch --parsable -A $ACC -t $TL -J b47_$NAME --dependency=afterany:\$prev b47_pace.sbatch $*); fi; echo \$o; prev=\$o; done" < /dev/null 2>&1 | tr '\n' ' ')
echo "$(date -u +%FT%TZ) $NAME nseg=$NSEG acc=$ACC jobs: $r cmd: $*" >> $LOG
echo "$NAME: $r"
