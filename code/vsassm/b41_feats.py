"""B41 mention features on frozen Qwen3-8B (box; one GPU). For every bank row the mention text is put in the carrier
sentence and the hidden states over the name's tokens (offset mapping) are averaged at layers LAYERS (hidden_states[L],
output of block L of 36). Output: <out>/<split>_L<L>.npy (fp16, rows in bank order) and <split>_meta.json.

  python b41_feats.py --split train|dev|test [--rhos 0.06 0.15 0.3] --out /path/to/work/b41/feats --gpu 7
"""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b41_names as NM                                                      # noqa: E402

QWEN8 = os.environ.get("B41_QWEN8", "/path/to/work/hf/hub/models--Qwen--Qwen3-8B/snapshots/"
                                    "b968826d9c46dd6066d109eabc6255188de91218")
LAYERS = [9, 18, 27, 36]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--rhos", type=float, nargs="*", default=None)
    ap.add_argument("--out", default="/path/to/work/b41/feats")
    ap.add_argument("--bs", type=int, default=512)
    a = ap.parse_args()
    from transformers import AutoTokenizer, AutoModel
    os.makedirs(a.out, exist_ok=True)
    rows = NM.bank(a.split, rhos=a.rhos)
    tok = AutoTokenizer.from_pretrained(QWEN8)
    tok.padding_side = "right"
    dev = "cuda:0"
    llm = AutoModel.from_pretrained(QWEN8, torch_dtype=torch.bfloat16).to(dev).eval()
    n = len(rows)
    outs = {L: np.lib.format.open_memmap(os.path.join(a.out, "%s_L%d.npy" % (a.split, L)), mode="w+",
                                         dtype=np.float16, shape=(n, llm.config.hidden_size)) for L in LAYERS}
    pre = NM.CARRIER.split("{}")[0]
    t0 = time.time()
    ntok = []
    for b in range(0, n, a.bs):
        chunk = rows[b:b + a.bs]
        texts = [NM.CARRIER.format(r[3]) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")
        masks = torch.zeros(enc["input_ids"].shape, dtype=torch.float32)
        for i, r in enumerate(chunk):
            s0, s1 = len(pre), len(pre) + len(r[3])
            for t, (u, v) in enumerate(offs[i].tolist()):
                if v > u and u < s1 and v > s0 and enc["attention_mask"][i, t] == 1:
                    masks[i, t] = 1.0
            assert masks[i].sum() > 0, texts[i]
            ntok.append(int(masks[i].sum()))
        with torch.no_grad():
            o = llm(input_ids=enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev),
                    output_hidden_states=True)
        m = masks.to(dev)[..., None]
        for L in LAYERS:
            h = o.hidden_states[L].float()
            outs[L][b:b + len(chunk)] = ((h * m).sum(1) / m.sum(1)).half().cpu().numpy()
        if (b // a.bs) % 40 == 0:
            print("%s %d/%d %.0fs" % (a.split, b, n, time.time() - t0), flush=True)
    for L in LAYERS:
        outs[L].flush()
    json.dump({"split": a.split, "rhos": a.rhos, "n": n, "digest": NM.digest(rows), "layers": LAYERS,
               "padding_side": tok.padding_side, "mean_name_tokens": float(np.mean(ntok)),
               "seconds": time.time() - t0, "model": QWEN8},
              open(os.path.join(a.out, "%s_meta.json" % a.split), "w"), indent=1)
    print("done", a.split, n, "%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
