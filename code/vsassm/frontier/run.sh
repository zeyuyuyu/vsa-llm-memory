#!/bin/bash
# $1 model, $2 provider, $3 name, $4 cap (combined over both benchmarks)
cd /path/to/vsassm
python3 -u frontier_api.py frontier/ruler128k_prompts.jsonl frontier/ruler_$3.jsonl --model $1 --provider $2 --cap $4
S=$(python3 -c "import json; print(sum(json.loads(l).get('cost', 0) or 0 for l in open('frontier/ruler_$3.jsonl')))")
python3 -u frontier_api.py frontier/lme100_prompts.jsonl frontier/lme_$3.jsonl --model $1 --provider $2 --cap $4 --spent_before $S
echo DONE_$3
