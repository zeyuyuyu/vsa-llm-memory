"""V-102 (prereg cc5ae08): does sharding recover what superposition loses?

V-101's protocol -- V-100's pooled real 2Wiki gold triples, hash identity, exact two-hop composition -- with
the store split into S shards. Each hop runs against every shard and the answer comes from the shard whose
cleanup cosine is highest, so the experiment measures both halves of the trade: cleaner unbinding per shard
against the ambiguity of choosing among shards.

No language model. The manuscript asserts sharding as a remedy; this tests it.
"""
import argparse, hashlib, json, os, random, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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


def sharded_two_hop(pool, head, S, dim, CB, n_rows, norm_str):
    """Split the pool into S shards, run each hop against all of them, and take the answer from the shard
    with the highest cleanup cosine. S = 1 is the unsharded store."""
    vec = lambda s: CB[row_index(norm_str(s), n_rows)]
    triples = [(s, o) for s, _, o in pool]
    if not triples:
        return ""
    shards = [triples[i::S] for i in range(S)]
    built = []
    for sh in shards:
        if not sh:
            continue
        objs = [o for _, o in sh]
        OV = torch.stack([vec(o) for o in objs])
        store = torch.stack([bind(vec(s), vec(o), dim) for s, o in sh]).sum(0)
        built.append((store, OV, objs))

    def hop(cur):
        """Best (cosine, answer) across shards. The cosine is what selects the shard, so a shard holding
        nothing relevant has to lose on it -- that is the ambiguity sharding introduces."""
        best_c, best_a = -2.0, ""
        cv = vec(cur)
        for store, OV, objs in built:
            hat = F.normalize(unbind(store, cv, dim), dim=-1)
            sims = F.cosine_similarity(OV, hat.unsqueeze(0))
            j = int(sims.argmax().item())
            if float(sims[j]) > best_c:
                best_c, best_a = float(sims[j]), objs[j]
        return best_a

    h1 = hop(head)
    return hop(h1) if h1 else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ks", type=int, nargs="+", default=[128, 256, 512, 1024])
    ap.add_argument("--Ss", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--dims", type=int, nargs="+", default=[2048, 4096])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_rows", type=int, default=200003)
    ap.add_argument("--out", default="v102.json")
    a = ap.parse_args()

    from v85_data import load_rows, norm
    from v100 import pooled, gold_triples, head_entity
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("===== V-102 sharding (prereg cc5ae08) device=%s =====" % device, flush=True)
    print("  gate: S=1, d=2048 reproduces V-101 (k128 .728, k256 .219, k512 .039) within .05", flush=True)
    print("  P2: K=512, d=2048, S=8 reaches >= .70 against .039 unsharded", flush=True)

    rows = load_rows(400, split="validation", cache_dir=os.environ.get("HF_HOME"))
    rows = [r for r in rows if gold_triples(r) and head_entity(r)]
    print("  %d usable dev instances" % len(rows), flush=True)

    out = {"prereg": "cc5ae08", "grid": {}}
    for dim in a.dims:
        for seed_i, seed in enumerate(a.seeds):
            CB = codebook(a.n_rows, dim, 1000 + seed, device)
            for K in a.Ks:
                for S in a.Ss:
                    if S > K:
                        continue
                    rng = random.Random(7000 + seed)
                    hits = []
                    for i in range(min(a.n, len(rows))):
                        pool, own = pooled(rows, i, K, rng)
                        if not own:
                            continue
                        ans = sharded_two_hop(pool, head_entity(rows[i]), S, dim, CB, a.n_rows, norm)
                        hits.append(int(norm(ans) == norm(rows[i]["answer"])))
                    key = "d%d_K%d_S%d" % (dim, K, S)
                    out["grid"].setdefault(key, []).append(float(np.mean(hits)))
            del CB
            if device == "cuda":
                torch.cuda.empty_cache()
        for K in a.Ks:
            line = "  d=%-5d K=%-5d " % (dim, K)
            for S in a.Ss:
                k = "d%d_K%d_S%d" % (dim, K, S)
                if k in out["grid"]:
                    line += "S%d=%.3f  " % (S, float(np.mean(out["grid"][k])))
            print(line, flush=True)
        json.dump(out, open(a.out, "w"), indent=1)

    G = {k: float(np.mean(v)) for k, v in out["grid"].items()}
    print("", flush=True)
    print("===== V-102 SCORING =====", flush=True)
    ref = {"d2048_K128_S1": 0.728, "d2048_K256_S1": 0.219, "d2048_K512_S1": 0.039}
    bad = [(k, G.get(k), v) for k, v in ref.items() if abs(G.get(k, -1) - v) > 0.05]
    notes = ["P1 gate: S=1 reproduces V-101: %s%s" % (not bad, "" if not bad else " off at %s" % bad)]
    if bad:
        notes.append("GATE FAILS: the sharding wrapper changed the unsharded path, so no reading is taken.")
        out["verdict"] = " ; ".join(notes)
        json.dump(out, open(a.out, "w"), indent=1)
        print("  >>> %s" % out["verdict"], flush=True)
        print("V102-DONE", flush=True)
        return

    v = G.get("d2048_K512_S8", float("nan"))
    p2 = v >= 0.70
    notes.append("P2 K=512 d=2048 S=8 reaches %.4f >= .70 (unsharded .039): %s" % (v, p2))
    a4 = G.get("d2048_K256_S4", float("nan"))
    a8 = G.get("d2048_K512_S8", float("nan"))
    p3 = abs(a4 - a8) <= 0.05
    notes.append("P3 equal facts-per-shard flat in S: |%.4f - %.4f| = %.4f <= .05: %s"
                 % (a4, a8, abs(a4 - a8), p3))
    if p2 and p3:
        notes.append("SHARDING IS A CLEAN REMEDY and the manuscript's assertion is now measured: capacity is "
                     "per shard, and cross-shard selection is close to free at these sizes.")
    elif not p2:
        notes.append("SHARDING DOES NOT RECOVER ACCURACY, so the manuscript asserts a remedy that does not "
                     "work and the sentence must be corrected. Reported first because it cuts against us.")
    else:
        notes.append("Sharding works but selection is NOT free (%.4f cost at equal facts-per-shard), so the "
                     "honest statement is capacity per shard minus a measured selection cost."
                     % abs(a4 - a8))
    out["verdict"] = " ; ".join(notes)
    json.dump(out, open(a.out, "w"), indent=1)
    print("  >>> V-102 VERDICT: %s" % out["verdict"], flush=True)
    print("V102-DONE", flush=True)


if __name__ == "__main__":
    main()
