"""B33 memory arms on bAbI task 2 (prereg/B33_fresh_task2_prereg.md).

Exactly B28's task-2 memory route (b28_babi.py --task 2, which is V-37b's parse + ordering rule + two unbindings, with
the naming head trained as published), scored on the B33 evaluation sets:
  --patterns completed  the corrected parse: "grabbed" added to the pick-up verbs and "discarded" to the drop verbs
                        (b28_babi.GET_COMPLETED / DROP_COMPLETED, unchanged), nothing else changed   [PRIMARY]
  --patterns published  V-37b's patterns as published (secondary: how much the fix matters on fresh data)
Arms: exact codebook rows (parser) and the naming head (probe_argmax), under B28's five encodings (comm = published
HRR is the registered one), seeds 42/123/456. The naming head is trained once per seed on the parse's expressible
training stories, exactly as in B28 (same order of random draws), and read on every set. A question the parse cannot
read gets no answer and is scored wrong. Per-question outputs: <out>/b33_mem_<patterns>.json.
--smoke scores the first 64 TRAINING rows as the only set (no evaluation set is built).
"""
import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b28_common as C                                                # noqa: E402
import b33_common as B                                                # noqa: E402
from b28_babi import GET_COMPLETED, DROP_COMPLETED, eval2, train_head, sf, index_rows   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", default="completed", choices=["published", "completed"])
    ap.add_argument("--sets", nargs="+", default=B.SETS)
    ap.add_argument("--fresh_file", default="b33_data/b33_fresh.jsonl")
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    ap.add_argument("--encodings", nargs="+", default=C.ENCODINGS)
    ap.add_argument("--head_epochs", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="out")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    M = C.load_ngcls()
    v13, v5, v37b = M["v13"], M["v5"], M["v37b"]
    if a.patterns == "completed":
        v37b.GET, v37b.DROP = GET_COMPLETED, DROP_COMPLETED
    tok = C.tokenizer()
    dev = a.device
    d, vocab, sets = B.load_sets(v5, a.fresh_file, [] if a.smoke else a.sets)
    locs = {l: i for i, l in enumerate(vocab)}
    persons, objs = {}, {}
    for text, _ in d["train"]:
        for _, k, s_, o, _, _ in v37b.events(text):
            persons.setdefault(s_, len(persons))
            if k != "move":
                objs.setdefault(o, len(objs))
    n_p, n_l, n_o = len(persons), len(locs), len(objs)
    P, L, O = 0, n_p, n_p + n_l
    TOT = n_p + n_l + n_o
    tr_all, _ = v37b.build(d["train"], tok, persons, locs, objs)
    tr = [r for r in tr_all if r["expressible"]]
    if a.smoke:
        sets = {"smoke_train64": d["train"][:64]}
    out = {"patterns": a.patterns, "smoke": a.smoke, "seeds": a.seeds, "encodings": a.encodings, "vocab": vocab,
           "n_train_parsed": len(tr_all), "n_train": len(tr), "codebook": TOT, "persons": persons, "objects": objs,
           "sets": {}}
    print("B33 memory (%s): %d persons, %d locations, %d objects; %d of %d training stories parsed, %d expressible"
          % (a.patterns, n_p, n_l, n_o, len(tr_all), len(d["train"]), len(tr)), flush=True)
    built = {}
    for name, split in sets.items():
        te, _ = v37b.build(split, tok, persons, locs, objs)
        idx = index_rows(split, te, tok, 768)
        n_long = sum(int(len(tok(t)["input_ids"]) > 768) for t, _ in split)
        built[name] = te
        out["sets"][name] = {"n_questions": len(split), "n_parsed": len(te), "n_longer_than_768_tokens": n_long,
                             "row_index": idx, "labels_all": [int(y) for _, y in split],
                             "story_id": B.story_ids(split), "texts": [t for t, _ in split],
                             "expressible": [int(r["expressible"]) for r in te],
                             "bundle_loc": [int(r["bundle_loc"]) for r in te],
                             "gold": [{"obj_ev": [(o, h) for o, h, _, _ in r["obj_ev"]],
                                       "moves": [(p, l) for p, l, _, _ in r["moves"]], "q_obj": r["q_obj"],
                                       "label": r["label"]} for r in te],
                             "runs": {}}
        print("  set %-8s %4d questions, %4d parsed, %4d expressible, %d longer than 768 tokens"
              % (name, len(split), len(te), sum(r["expressible"] for r in te), n_long), flush=True)

    from transformers import AutoModel
    llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                    torch_dtype=torch.float16).to(dev).eval()
    for p in llm.parameters():
        p.requires_grad_(False)
    Htr = v13.cache(tr, llm, dev)
    Hte = {name: v13.cache(te, llm, dev) for name, te in built.items()}
    del llm
    torch.cuda.empty_cache()
    print("  states cached (%.0fs)" % (time.time() - t0), flush=True)

    for seed in a.seeds:
        torch.manual_seed(seed)
        sym = F.normalize(torch.randn(TOT, C.VSA_DIM,
                                      generator=torch.Generator().manual_seed(seed + 11)), dim=1).to(dev)
        X, Y = [], []
        for b, r in enumerate(tr):
            H = Htr[b].to(dev)
            for oid, hid, osp, hsp in r["obj_ev"]:
                X.append(sf(H, osp)); Y.append(O + oid)
                X.append(sf(H, hsp)); Y.append(P + hid)
            for pid, lid, psp, lsp in r["moves"]:
                X.append(sf(H, psp)); Y.append(P + pid)
                X.append(sf(H, lsp)); Y.append(L + lid)
            X.append(sf(H, r["q_span"])); Y.append(O + r["q_obj"])
        X = torch.stack(X).to(dev); Y = torch.tensor(Y).to(dev)
        idh = train_head(X, Y, TOT, a.head_epochs, dev, v13.HID)
        idh.eval()
        with torch.no_grad():
            for name, te in built.items():
                cache = {}

                def named(H, sp, b):
                    k = (b, sp[0], sp[1])
                    if k not in cache:
                        cache[k] = int(idh(sf(H, sp)).argmax().item())
                    return cache[k]

                S = out["sets"][name]
                for enc_name in a.encodings:
                    enc = C.Encoding(enc_name, sym, seed)
                    for arm in ["parser", "probe_argmax"]:
                        rec = []
                        for b, r in enumerate(te):
                            H = Hte[name][b].to(dev)
                            idf = (lambda H_, sp, gid: gid) if arm == "parser" else \
                                (lambda H_, sp, gid, b=b: named(H_, sp, b))
                            o = eval2(r, H, idf, enc, P, L, O, v13)
                            o["correct"] = int(o["pred"] == r["label"])
                            rec.append(o)
                        cc = np.array([o["correct"] for o in rec])
                        acc_all = float(cc.sum() / S["n_questions"])
                        S["runs"]["%s/%s/%d" % (arm, enc_name, seed)] = {"acc_all": acc_all, "per_q": rec}
                        print("  %-8s %-12s %-5s seed %3d  all %.4f  (parsed %d: %.4f)"
                              % (name, arm, enc_name, seed, acc_all, len(cc), cc.mean() if len(cc) else float("nan")),
                              flush=True)
        os.makedirs(a.out, exist_ok=True)
        fn = os.path.join(a.out, "b33_mem_%s%s.json" % (a.patterns, "_smoke" if a.smoke else ""))
        json.dump(out, open(fn + ".tmp", "w")); os.replace(fn + ".tmp", fn)
    out["secs"] = round(time.time() - t0, 1)
    json.dump(out, open(fn + ".tmp", "w")); os.replace(fn + ".tmp", fn)
    print("B33-MEM-DONE %s (%.0fs) -> %s" % (a.patterns, time.time() - t0, fn), flush=True)


if __name__ == "__main__":
    main()
