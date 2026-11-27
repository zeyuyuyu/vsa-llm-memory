#!/bin/bash
# B50 on PACE (fallback when the A800 box has no idle GPU): fetch the model file set B40 used. B40's shards equal the
# LFS oids of unsloth/Llama-3.1-8B-Instruct (prereg/B40_smoke_record.md); the sha256 of each shard is compared with
# results/b40_model_sha256_modelscope_etag.txt and the job fails on any mismatch.
#SBATCH -A gts-nimam6
#SBATCH -p cpu-small
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1:30:00
#SBATCH -o /path/to/pace-scratch/b50/logs/%x_%j.out
cd /path/to/pace-scratch/b50 || exit 1
export HF_HOME=/path/to/pace-scratch/hf
/usr/bin/python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("unsloth/Llama-3.1-8B-Instruct", local_dir="/path/to/pace-scratch/b50/llama31_8b_instruct",
                  allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
PY
cd llama31_8b_instruct && sha256sum model-0000*-of-00004.safetensors > ../model_sha256.txt
cd .. && python3 - <<'PY'
want = {l.split()[0]: l.split()[1] for l in open("b40_model_sha256_modelscope_etag.txt") if l.strip()}
got = {l.split()[1]: l.split()[0] for l in open("model_sha256.txt")}
bad = [k for k in want if got.get(k) != want[k]]
print("sha256 mismatches:", bad)
raise SystemExit(1 if bad else 0)
PY
