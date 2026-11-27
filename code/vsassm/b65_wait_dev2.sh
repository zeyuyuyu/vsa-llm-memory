#!/bin/bash
# B65: start `b65_orch.sh dev2 <pos>` as soon as the three first-grid workers of that position have ended.
# usage: b65_wait_dev2.sh last RWl [n_workers] | ops RWo [n_workers]
B=/path/to/work/b65; cd $B || exit 1
P=$1; W=$2; N=${3:-3}
until [ "$(grep -c "^$W[123] gpu. end" logs/gpu_ledger.txt)" = "$N" ]; do sleep 20; done
bash b65_orch.sh dev2 $P
