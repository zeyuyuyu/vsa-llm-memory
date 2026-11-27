"""B44: content checksum of a b44_tok cache file (token ids, spans, labels, targets), independent of pickling, so that
caches built on different machines can be compared.  python b44_tokmd5.py data/qwen8_std_train.pt ..."""
import hashlib, sys
import torch
for p in sys.argv[1:]:
    o = torch.load(p, weights_only=False)
    h = hashlib.md5()
    for i in range(o["n"]):
        h.update(o["ids"][i].numpy().tobytes()); h.update(o["spans"][i].numpy().tobytes())
    h.update(o["label"].numpy().tobytes()); h.update("|".join(o["tgt"]).encode()); h.update(o["text_md5"].encode())
    print(p, o["n"], h.hexdigest())
