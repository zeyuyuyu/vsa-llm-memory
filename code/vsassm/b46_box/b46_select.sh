#!/bin/bash
# B46 encoder choice on the A800 box's CPU (prereg/B46_second_llm_family_prereg.md): B32's criterion (b32_lm_select.py,
# unchanged, copied from /path/to/work/b32 with md5 check) for every variant of every candidate, the registered choice
# rule (b46_choose_encoder.py), and the Qwen3-8B recipe check. No memory, no value, no test string.
B=/path/to/work/b46; cd $B || exit 1
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
export OMP_NUM_THREADS=8
cp -n /path/to/work/b32/b32_lm_select.py . ; md5sum b32_lm_select.py
for f in entities.json emb_strings.json; do cp -n /path/to/work/b32/b32_data/$f b32_data/; done; md5sum b32_data/entities.json b32_data/emb_strings.json
for t in llama31i mistral7i; do
  [ -f b32_data/emb_${t}_mean_L8.npy ] && $PY b32_lm_select.py $t mean_L8 mean_L16 mean_L24 mean_L32 eol_L8 eol_L16 eol_L24 eol_L32
done
[ -f b32_data/emb_granite33i_mean_L10.npy ] && $PY b32_lm_select.py granite33i mean_L10 mean_L20 mean_L30 mean_L40 eol_L10 eol_L20 eol_L30 eol_L40
$PY b46_choose_encoder.py llama31i,mistral7i,granite33i
# recipe check: b46_embed.py's Qwen3-8B layer-9 mean codes against B32's
if [ -f b32_data/emb_qwen8chk_mean_L9.npy ]; then
  $PY b32_lm_select.py qwen8chk mean_L9
  $PY -c "
import numpy as np
a = np.load('b32_data/emb_qwen8chk_mean_L9.npy'); b = np.load('/path/to/work/b32/b32_data/emb_qwen8_mean_L9.npy')
c = (a * b).sum(1)
print(f'qwen8chk vs B32 qwen8 mean_L9: row cosine min {c.min():.6f} median {np.median(c):.6f}; share >= 0.999 {(c >= 0.999).mean():.4f}; share >= 0.99 {(c >= 0.99).mean():.4f}')
"
fi
echo SELECT_DONE
