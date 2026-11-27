"""V-101 (prereg dd06b93): does the store's capacity wall move with dimension?

No language model. The wall is a property of the store, so this measures the store: hash-identity codes,
exact two-hop composition, and a sweep of dimension against store size. The manuscript's capacity sketch --
cross-talk O(1/d), SNR falling as bindings accumulate -- predicts capacity linear in d, and every number in
the paper uses d = 2048 with nothing varying it.

Uses V-100's REAL pooled 2Wiki gold triples, not synthetic chains. A first probe with synthetic chains
(disjoint entity names) failed the gate at k=128 -- 0.9167 against V-100's 0.7417 -- because real triples
repeat entities and the resulting aliasing makes retrieval harder. That was the fourth consecutive gate
failure caused by my registering a bar against a banked number without matching the configuration that
produced it. Matching it here.
"""
import argparse, hashlib, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F


def codebook(n_rows, dim, seed, device):
    g = torch.Generator().manual_seed(seed)
    rows = []
    for i in range(0, n_rows, 20000):
        m = min(20000, n_rows - i)
        rows.append(F.normalize(torch.randn(m, dim, generator=g), dim=-1))
    return torch.cat(rows).to(device)


def row_index(s, n_rows):
    h = hashlib.blake2b(str(s).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % n_rows


def bind(a, b, d):
    return torch.fft.irfft(torch.fft.rfft(a.float()) * torch.fft.rfft(b.float()), n=d).to(a.dtype)


def involution(x):
    return torch.cat([x[..., :1], torch.flip(x[..., 1:], dims=[-1])], dim=-1)


def unbind(ab, a, d):
    return bind(involution(a), ab, d)


def one_instance_real(row, pool, dim, CB, n_rows, device, norm_str, head):
    """V-100's protocol: this instance's own gold triples plus other instances' as distractors, two hops
    through the store, scored by exact match against the gold answer."""
    triples = [(s, o) for s, _, o in pool]
    vec = lambda s: CB[row_index(norm_str(s), n_rows)]
    objs = [o for _, o in triples]
    OV = torch.stack([vec(o) for o in objs])
    store = torch.stack([bind(vec(s), vec(o), dim) for s, o in triples]).sum(0)

    def hop(cur):
        hat = F.normalize(unbind(store, vec(cur), dim), dim=-1)
        return objs[int(F.cosine_similarity(OV, hat.unsqueeze(0)).argmax().item())]

    return norm_str(hop(hop(head))) == norm_str(row["answer"])


def one_instance(k, dim, CB, n_rows, rng, device):
    """Synthetic fallback, kept for reference; not used for the scored grid."""
    tag = rng.randrange(10 ** 9)
    a, b, c = ("A%d" % tag), ("B%d" % tag), ("C%d" % tag)
    triples = [(a, b), (b, c)]
    while len(triples) < k:
        t = rng.randrange(10 ** 9)
        triples.append(("X%d" % t, "Y%d" % t))
    rng.shuffle(triples)
    vec = lambda s: CB[row_index(s, n_rows)]
    objs = [o for _, o in triples]
    OV = torch.stack([vec(o) for o in objs])
    store = torch.stack([bind(vec(s), vec(o), dim) for s, o in triples]).sum(0)

    def hop(cur):
        hat = F.normalize(unbind(store, vec(cur), dim), dim=-1)
        sims = F.cosine_similarity(OV, hat.unsqueeze(0))
        return objs[int(sims.argmax().item())]

    return int(hop(hop(a)) == c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    ap.add_argument("--ks", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_rows", type=int, default=200003)
    ap.add_argument("--out", default="v101.json")
    a = ap.parse_args()

    from v85_data import load_rows, norm
    from v100 import pooled, gold_triples, head_entity
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_rows(400, split="validation", cache_dir=os.environ.get("HF_HOME"))
    rows = [r for r in rows if gold_triples(r) and head_entity(r)]
    print("  %d usable dev instances (V-100's slice and pooling)" % len(rows), flush=True)
    print("===== V-101 dimension vs capacity (prereg dd06b93) device=%s =====" % device, flush=True)
    print("  gate: at d=2048, k=128 -> 0.7417 and k=256 -> 0.2417 within .05 (V-100's store arm)", flush=True)
    print("  P2: k*(8192) >= 4 x k*(2048) at the 0.70 accuracy level", flush=True)

    out = {"prereg": "dd06b93", "grid": {}}
    for dim in a.dims:
        row = {}
        for seed in a.seeds:
            CB = codebook(a.n_rows, dim, 1000 + seed, device)
            for k in a.ks:
                rng = random.Random(7000 + seed)
                hits = []
                for i in range(min(a.n, len(rows))):
                    pool, own = pooled(rows, i, k, rng)
                    if not own:
                        continue
                    hits.append(one_instance_real(rows[i], pool, dim, CB, a.n_rows, device,
                                                  norm, head_entity(rows[i])))
                row.setdefault(k, []).append(float(np.mean(hits)))
            del CB
            torch.cuda.empty_cache() if device == "cuda" else None
        out["grid"][dim] = {k: float(np.mean(v)) for k, v in row.items()}
        print("  d=%-5d " % dim + "  ".join("k%d=%.3f" % (k, out["grid"][dim][k]) for k in a.ks), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)

    # ---- scoring ----
    G = out["grid"]
    print("", flush=True)
    print("===== V-101 SCORING =====", flush=True)
    g2048 = G.get(2048, {})
    p1 = (abs(g2048.get(128, -1) - 0.7417) <= 0.05) and (abs(g2048.get(256, -1) - 0.2417) <= 0.05)
    notes = ["P1 gate: d=2048 reproduces V-100 (k128 %.4f vs .7417, k256 %.4f vs .2417): %s"
             % (g2048.get(128, float("nan")), g2048.get(256, float("nan")), p1)]
    if not p1:
        notes.append("GATE FAILS: this store differs from V-100's, so no reading is taken. Note V-100's "
                     "arm used real 2Wiki triples where this uses synthetic chains, which is a protocol "
                     "difference I should have matched before registering the bar.")
        out["verdict"] = " ; ".join(notes)
        json.dump(out, open(a.out, "w"), indent=1)
        print("  >>> %s" % out["verdict"], flush=True)
        print("V101-DONE", flush=True)
        return

    def kstar(d):
        ks = sorted(G[d])
        ok = [k for k in ks if G[d][k] >= 0.70]
        return max(ok) if ok else 0

    ks_by_d = {d: kstar(d) for d in sorted(G)}
    notes.append("k* at the .70 level: " + ", ".join("d%d=%d" % (d, k) for d, k in ks_by_d.items()))
    p2 = ks_by_d.get(8192, 0) >= 4 * max(1, ks_by_d.get(2048, 0))
    notes.append("P2 k*(8192) >= 4 x k*(2048): %s (%d vs %d)"
                 % (p2, ks_by_d.get(8192, 0), 4 * ks_by_d.get(2048, 0)))
    p3 = ks_by_d.get(512, 10 ** 9) <= max(1, ks_by_d.get(2048, 0)) / 2
    notes.append("P3 k*(512) <= k*(2048)/2: %s (%d vs %.0f)"
                 % (p3, ks_by_d.get(512, -1), ks_by_d.get(2048, 0) / 2))
    if p2 and p3:
        notes.append("THE CAPACITY WALL IS A DIMENSION CHOICE, not a limit of the algebra, exactly as the "
                     "manuscript's own O(1/d) sketch predicts. The boundary section can state the scaling "
                     "rather than only the wall.")
    elif not p2:
        notes.append("CAPACITY DOES NOT SCALE WITH DIMENSION as our own sketch predicts, so the sketch is "
                     "wrong or incomplete about this store and the theory section needs a caveat. Reported "
                     "first because it cuts against our own theory.")
    else:
        notes.append("Scaling is not clean; the wall is governed by more than d and k, and no scaling claim "
                     "is made.")
    out["verdict"] = " ; ".join(notes)
    out["kstar"] = ks_by_d
    json.dump(out, open(a.out, "w"), indent=1)
    print("  >>> V-101 VERDICT: %s" % out["verdict"], flush=True)
    print("V101-DONE", flush=True)


if __name__ == "__main__":
    main()
