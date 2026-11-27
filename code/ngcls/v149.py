"""V-149: does the paper's store result belong to HRR, or to the VSA family?

Every store measurement this project has banked -- V-65, V-100/101/102, V-106, V-130, store_rebase --
uses exactly ONE model: Holographic Reduced Representations, with FFT circular convolution, the
involution inverse, sum bundling and cosine cleanup. The manuscript's claim is about "a vector symbolic
memory". That is a family-level claim resting on single-model evidence.

Kleyko, Rachkovskij, Osipov & Rahimi, "A Survey on HDC/VSA, Part I" (ACM CSUR 55(6), 2022), sec 4.1,
states the gap as an open issue: "in general it is not clear whether there is one best model, which will
dominate all others, or whether each model has its own applicability scope", and asks for systematic
experimental comparison. This sweep answers it for this task by swapping the algebra and NOTHING else.

Arms (Survey Part I sec 2.3):
  HRR  2.3.3  real, circular convolution, involution inverse, sum bundling, cosine
  MAP  2.3.7  bipolar, Hadamard product (self-inverse), sum bundling, cosine
  BSC  2.3.6  binary, XOR (self-inverse), MAJORITY-RULE bundling, 1 - normalised Hamming
  FHRR 2.3.5  unit-modulus complex, elementwise product, conjugate inverse, sum bundling, cosine

No language model, no API, no training.
"""
import argparse, hashlib, json, os, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CHUNK = 20000


def row_index(s, n_rows):
    h = hashlib.blake2b(str(s).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % n_rows


# ----------------------------------------------------------------------------- algebras
class HRR:
    name, arith, integer_only = "HRR", "float32 FFT (O(d log d))", False

    def codebook(self, n, d, seed, dev):
        g = torch.Generator().manual_seed(seed)
        out = []
        for i in range(0, n, CHUNK):
            m = min(CHUNK, n - i)
            out.append(F.normalize(torch.randn(m, d, generator=g), dim=-1))
        return torch.cat(out).to(dev)

    def bind(self, a, b, d):
        return torch.fft.irfft(torch.fft.rfft(a.float()) * torch.fft.rfft(b.float()), n=d).to(a.dtype)

    def _inv(self, x):
        return torch.cat([x[..., :1], torch.flip(x[..., 1:], dims=[-1])], dim=-1)

    def unbind(self, ab, a, d):
        return self.bind(self._inv(a), ab, d)

    def bundle(self, stack, g):
        return stack.sum(0)

    def sim(self, hat, OV):
        return F.cosine_similarity(OV, F.normalize(hat, dim=-1).unsqueeze(0))


class MAP:
    name, arith, integer_only = "MAP", "float32 multiply-add (O(d))", False

    def codebook(self, n, d, seed, dev):
        g = torch.Generator().manual_seed(seed)
        out = []
        for i in range(0, n, CHUNK):
            m = min(CHUNK, n - i)
            out.append(torch.where(torch.randn(m, d, generator=g) >= 0, 1.0, -1.0))
        return torch.cat(out).to(dev)

    def bind(self, a, b, d):
        return a * b

    def unbind(self, ab, a, d):
        return ab * a                      # Hadamard product on bipolar is self-inverse

    def bundle(self, stack, g):
        return stack.sum(0)                # MAP-I: unclipped integer-valued sum

    def sim(self, hat, OV):
        return F.cosine_similarity(OV, hat.unsqueeze(0))


class BSC:
    name, arith, integer_only = "BSC", "uint8 XOR + popcount (O(d), integer only)", True

    def codebook(self, n, d, seed, dev):
        g = torch.Generator().manual_seed(seed)
        out = []
        for i in range(0, n, CHUNK):
            m = min(CHUNK, n - i)
            out.append((torch.rand(m, d, generator=g) < 0.5).to(torch.uint8))
        return torch.cat(out).to(dev)

    def bind(self, a, b, d):
        return a ^ b

    def unbind(self, ab, a, d):
        return ab ^ a                      # XOR is self-inverse

    def bundle(self, stack, g):
        """Kanerva's majority rule. Ties (even stack) are broken by a fair coin from the run's own
        generator -- recorded as a protocol choice, not left to a library default."""
        n = stack.shape[0]
        s = stack.to(torch.int32).sum(0)
        maj = (s * 2 > n).to(torch.uint8)
        if n % 2 == 1:
            return maj
        ties = (s * 2 == n)
        coin = (torch.rand(maj.shape, generator=g) < 0.5).to(torch.uint8).to(maj.device)
        return torch.where(ties, coin, maj)

    def sim(self, hat, OV):
        return 1.0 - (OV ^ hat.unsqueeze(0)).to(torch.float32).mean(-1)


class FHRR:
    name, arith, integer_only = "FHRR", "complex64 multiply (O(d))", False

    def codebook(self, n, d, seed, dev):
        g = torch.Generator().manual_seed(seed)
        out = []
        for i in range(0, n, CHUNK):
            m = min(CHUNK, n - i)
            ph = torch.rand(m, d, generator=g) * (2.0 * float(np.pi))
            out.append(torch.polar(torch.ones_like(ph), ph))
        return torch.cat(out).to(dev)

    def bind(self, a, b, d):
        return a * b                       # phase addition

    def unbind(self, ab, a, d):
        return ab * torch.conj(a)          # phase subtraction

    def bundle(self, stack, g):
        return stack.sum(0)

    def sim(self, hat, OV):
        num = (OV * torch.conj(hat).unsqueeze(0)).sum(-1).real
        den = OV.abs().pow(2).sum(-1).sqrt() * hat.abs().pow(2).sum(-1).sqrt()
        return num / (den + 1e-12)


ALGEBRAS = {a.name: a for a in (HRR(), MAP(), BSC(), FHRR())}


# ----------------------------------------------------------------------------- readouts
def two_hop_pooled(row, pool, alg, d, CB, n_rows, norm_str, head, g):
    """V-100/V-101 protocol, algebra pluggable: this instance's gold triples padded with others' as
    distractors, two hops through one unsharded store, exact match against the gold answer."""
    triples = [(s, o) for s, _, o in pool]
    if not triples:
        return 0
    vec = lambda s: CB[row_index(norm_str(s), n_rows)]
    objs = [o for _, o in triples]
    OV = torch.stack([vec(o) for o in objs])
    store = alg.bundle(torch.stack([alg.bind(vec(s), vec(o), d) for s, o in triples]), g)

    def hop(cur):
        hat = alg.unbind(store, vec(cur), d)
        return objs[int(alg.sim(hat, OV).argmax().item())]

    return int(norm_str(hop(hop(head))) == norm_str(row["answer"]))


def sharded_two_hop(pool, head, S, alg, d, CB, n_rows, norm_str, g):
    """v102.sharded_two_hop with the algebra pluggable. Shard selection is by cleanup similarity, so a
    shard holding nothing relevant has to lose on it."""
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
        store = alg.bundle(torch.stack([alg.bind(vec(s), vec(o), d) for s, o in sh]), g)
        built.append((store, OV, objs))

    def hop(cur):
        best_c, best_a = -2.0, ""
        cv = vec(cur)
        for store, OV, objs in built:
            sims = alg.sim(alg.unbind(store, cv, d), OV)
            j = int(sims.argmax().item())
            if float(sims[j]) > best_c:
                best_c, best_a = float(sims[j]), objs[j]
        return best_a

    h1 = hop(head)
    return hop(h1) if h1 else ""


# ----------------------------------------------------------------------------- main
def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()[:8] if os.path.exists(p) else "ABSENT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["HRR", "MAP", "BSC", "FHRR"])
    ap.add_argument("--dims", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--ks", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--cap_seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--task_seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--n_rows", type=int, default=200003)
    ap.add_argument("--task_dim", type=int, default=2048)
    ap.add_argument("--shard_at", type=int, default=64)
    ap.add_argument("--out", default="v149.json")
    a = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from v85_data import load_rows, em, norm
    from v100 import pooled, gold_triples, head_entity

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"prereg": "V-149", "device": device,
           "module_md5": {m: md5(m) for m in ("v85_data.py", "v100.py", "v102.py", "v101.py")},
           "config": vars(a), "capacity": {}, "task": {}, "timing": {}}

    # ---------------- arm M1: capacity law by model (V-101 protocol) ----------------
    rows400 = load_rows(400, split="validation", cache_dir=os.environ.get("HF_HOME"))
    rows400 = [r for r in rows400 if gold_triples(r) and head_entity(r)]
    print("  M1: %d usable dev instances (V-101's slice and pooling)" % len(rows400), flush=True)

    for mname in a.models:
        alg = ALGEBRAS[mname]
        out["capacity"][mname] = {}
        for dim in a.dims:
            acc = {}
            for seed in a.cap_seeds:
                CB = alg.codebook(a.n_rows, dim, 1000 + seed, device)
                g = torch.Generator().manual_seed(90000 + seed)
                for k in a.ks:
                    rng = random.Random(7000 + seed)
                    hits = []
                    for i in range(min(a.n, len(rows400))):
                        pool, own = pooled(rows400, i, k, rng)
                        if not own:
                            continue
                        hits.append(two_hop_pooled(rows400[i], pool, alg, dim, CB, a.n_rows,
                                                   norm, head_entity(rows400[i]), g))
                    acc.setdefault(k, []).append(float(np.mean(hits)))
                del CB
                if device == "cuda":
                    torch.cuda.empty_cache()
            out["capacity"][mname][dim] = {k: float(np.mean(v)) for k, v in acc.items()}
            print("  %-5s d=%-5d " % (mname, dim)
                  + "  ".join("k%d=%.3f" % (k, out["capacity"][mname][dim][k]) for k in a.ks), flush=True)
            json.dump(out, open(a.out, "w"), indent=1)

    # ---------------- arm M2: task readout by model (store_rebase protocol) ----------------
    rows = json.load(open("rows300.json")) if os.path.exists("rows300.json") else load_rows(300, split="validation")
    assert len(rows) == 300
    print("  M2: 300 banked rows, d=%d, shard_at=%d" % (a.task_dim, a.shard_at), flush=True)

    for mname in a.models:
        alg = ALGEBRAS[mname]
        per_seed, per_row, t_tot, n_q = {}, {}, 0.0, 0
        for seed in a.task_seeds:
            CB = alg.codebook(a.n_rows, a.task_dim, seed, device)
            g = torch.Generator().manual_seed(90000 + seed)
            hits = []
            for r in rows:
                pool = gold_triples(r)
                S = max(1, (len(pool) + a.shard_at - 1) // a.shard_at)
                t0 = time.perf_counter()
                ans = sharded_two_hop(pool, head_entity(r), S, alg, a.task_dim, CB, a.n_rows, norm, g)
                t_tot += time.perf_counter() - t0
                n_q += 1
                hits.append(int(em(ans, r["answer"])))
            per_row[str(seed)] = hits
            per_seed[str(seed)] = sum(hits) / len(hits)
            del CB
            if device == "cuda":
                torch.cuda.empty_cache()
            print("    %-5s seed %-4d %d/300 = %.4f" % (mname, seed, sum(hits), per_seed[str(seed)]), flush=True)
        out["task"][mname] = {"per_seed": per_seed, "per_row_hits": per_row,
                              "mean_em": float(np.mean(list(per_seed.values()))),
                              "arith": alg.arith, "integer_only": alg.integer_only}
        out["timing"][mname] = {"sec_per_query": t_tot / max(1, n_q)}
        print("  %-5s MEAN %.4f   (%.4f s/query, %s)" % (mname, out["task"][mname]["mean_em"],
              out["timing"][mname]["sec_per_query"], alg.arith), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)

    json.dump(out, open(a.out, "w"), indent=1)
    print("V149-DONE", flush=True)


if __name__ == "__main__":
    main()
