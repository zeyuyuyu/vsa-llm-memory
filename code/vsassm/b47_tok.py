"""B47 Phi-2 tokenisation cache (prereg/B47_phi2_finetunes_hierboot_consistency_prereg.md), b44_tok's code and format
(b44_tok.build: token ids of the arena text, no truncation, no padding; every mention's token span by character
offsets, checked against b31_struct.tokenize on the first 300 instances; the scratchpad target).

  k32:train / k32:dev / k32:test  B31's arena C (= b44_data.ksplit(32, ...): train seed 42 (800), dev 777, test 4242)
  std:train                        the first 8,000 instances of B44's std training stream (seed 44042; nested)
  std:dev / std:test               B44's std dev / test (= B31's dev / test; checked identical to k32 here)

Self-test written to <root>/selftest_tok.json: text md5 of every split, k32 = B31 texts (b31_struct.build("C", ...)),
std dev/test token ids identical to k32 dev/test, dictionary walk = label on every instance.

  python b47_tok.py --root data
"""
import argparse, json, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b44_data as BD                                                      # noqa: E402
import b44_tok as BT                                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(HERE, "data"))
    a = ap.parse_args()
    import b31_prep as BP
    import b31_struct as S
    tok = BP.tokenizer("phi2")
    os.makedirs(a.root, exist_ok=True)
    rep = {}
    items = [("k32", "train"), ("k32", "dev"), ("k32", "test"), ("std", "train"), ("std", "dev"), ("std", "test")]
    for v, sp in items:
        p = BT.path(a.root, "phi2", "%s:%s" % (v, sp))
        t0 = time.time()
        if v == "k32":
            st = BD.ksplit(32, sp)
            b31 = S.build("C", sp)
            rep["k32_%s_equals_b31" % sp] = [s["text"] for s in st] == [s["text"] for s in b31]
        else:
            st = BD.split("std", sp, 8000 if sp == "train" else None)
        rep["dict_equals_label_%s_%s" % (v, sp)] = sum(int(BD.dict_answer(s["text"])["label"] == s["label"]) for s in st)
        rep["n_%s_%s" % (v, sp)] = len(st)
        if not os.path.exists(p):
            obj = BT.build(tok, st)
            obj.update(enc="phi2", item="%s:%s" % (v, sp), n=len(st))
            torch.save(obj, p + ".tmp"); os.replace(p + ".tmp", p)
        obj = torch.load(p, weights_only=False)
        rep["md5_%s_%s" % (v, sp)] = obj["text_md5"]
        rep["max_len_%s_%s" % (v, sp)] = max(len(x) for x in obj["ids"])
        print("tok phi2 %s:%s n %d max_len %d %.0fs" % (v, sp, len(st), rep["max_len_%s_%s" % (v, sp)],
                                                       time.time() - t0), flush=True)
    for sp in ("dev", "test"):
        A = torch.load(BT.path(a.root, "phi2", "k32:%s" % sp), weights_only=False)
        B = torch.load(BT.path(a.root, "phi2", "std:%s" % sp), weights_only=False)
        rep["std_%s_ids_equal_k32" % sp] = all(x.tolist() == y.tolist() for x, y in zip(A["ids"], B["ids"])) and \
            len(A["ids"]) == len(B["ids"])
    json.dump(rep, open(os.path.join(a.root, "selftest_tok.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
