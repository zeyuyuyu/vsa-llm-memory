#!/bin/bash
#SBATCH -J bench_dl
#SBATCH --account=gts-nimam6
#SBATCH --partition=cpu-small
#SBATCH -N1 --ntasks-per-node=4 --mem=16G
#SBATCH -t 03:00:00
#SBATCH -o /path/to/pace-scratch/logs/bench_dl_%j.out
mkdir -p /path/to/pace-scratch/bench && cd /path/to/pace-scratch/bench
python3 - <<'PY'
from huggingface_hub import snapshot_download
for repo, pats in [("RMT-team/babilong-1k-samples", None), ("RMT-team/babilong", None),
                   ("RMT-team/babilong-train-5k-samples", ["data/*/0k.json", "data/*/4k.json", "data/*/16k.json", "README*"])]:
    p = snapshot_download(repo, repo_type="dataset", local_dir=repo.replace("/", "__"), allow_patterns=pats)
    print("ok", repo, p, flush=True)
PY
python3 -c "import pandas, pyarrow; print('pandas', pandas.__version__, 'pyarrow', pyarrow.__version__)"
