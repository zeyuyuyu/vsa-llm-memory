#!/bin/bash
# B46: resumable ranged download of Mistral-7B-Instruct-v0.3 from ModelScope (mistralai/Mistral-7B-Instruct-v0.3,
# the official mistralai repository mirrored by ModelScope; no proxy). Same method as b40's dl_ms2.sh. Verifies every
# shard's sha256 against ModelScope's listing (written below before the download). consolidated.safetensors (a second
# copy of the weights in Mistral's own format) is not downloaded.
D=/path/to/work/b46/mistral7b_instruct_v03; mkdir -p $D/parts; cd $D || exit 1
U="https://www.modelscope.cn/models/mistralai/Mistral-7B-Instruct-v0.3/resolve/master"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
for f in config.json generation_config.json model.safetensors.index.json special_tokens_map.json tokenizer.json \
         tokenizer.model tokenizer_config.json params.json; do
  [ -s $f ] || curl -sL --retry 5 --max-time 300 -o $f "$U/$f"; echo "$f $(stat -c %s $f)"
done
cat > expected_sha256.txt <<'X'
ce6fb6f6f4d0183f4813cbf4ece24109da629a08d4210da46f77e1d8b0bd5c19  model-00001-of-00003.safetensors
8c0e72f148366b6a3709e002a98706a33d31aec8515090c856c95b2044f92ae0  model-00002-of-00003.safetensors
905dd405363e43d95779c1c1155a2dbfd36155914ae95dbd934e12e490cfb4ca  model-00003-of-00003.safetensors
e553af6fff7d7ad76e830608b218c5c0b0822998d5a1a96099a74cd3c1cb1a49  tokenizer.json
0533dec9cfe319163801b6618d0f3ec9cfa126b6288e3df5deca6e32acb09cd2  tokenizer_config.json
affafc6478ec0fd07a32f0ca57aa2fc57743f4d17d6730f86a96ac24d1507f99  config.json
X
NP=12; declare -A SIZES=([1]=4949453792 [2]=4999819336 [3]=4546807800)
part() {  # f i a b
  local f=$1 i=$2 a=$3 b=$4 p=parts/$1.$2 got
  for t in $(seq 1 30); do
    got=$(stat -c %s $p 2>/dev/null || echo 0)
    [ "$got" -ge "$((b-a+1))" ] && return 0
    curl -sL --max-time 900 -r $((a+got))-$b "$U/$f" >> $p
  done
  echo "FAILED $p"
}
shard() {
  local s=$1 f=model-0000$1-of-00003.safetensors size=${SIZES[$1]}
  [ -s $f ] && [ "$(stat -c %s $f)" = "$size" ] && { echo "$f present"; return; }
  local chunk=$(( (size + NP - 1) / NP ))
  for i in $(seq 0 $((NP-1))); do
    a=$((i*chunk)); b=$(( (i+1)*chunk - 1 )); [ $b -ge $size ] && b=$((size-1))
    part $f $i $a $b &
  done
  wait
  cat $(for i in $(seq 0 $((NP-1))); do echo parts/$f.$i; done) > $f
  echo "$f got $(stat -c %s $f) want $size $(date +%T)"
  rm -f parts/$f.*
}
for s in 1 2 3; do shard $s & done
wait
sha256sum -c expected_sha256.txt && echo SHA_OK || echo SHA_MISMATCH
echo ALLDONE $(date +%T)
