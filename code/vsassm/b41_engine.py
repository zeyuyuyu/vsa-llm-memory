"""B41 arms and cost counts (prereg/B41_superposed_uncertain_multihop_prereg.md). One run = one (split, seed, N, rho):
every arm answers the same queries from the same per-mention naming distributions; per query and per h in {2,4,6}
the run records the answer, correctness, operations, memory traffic (32-bit words read) and state size (words).

Naming distribution of a mention: softmax of the B41 naming head's logits / T (b41_head.py) for the bank variant.

Table arms (one engine, distributions truncated to top-k by zeroing the rest, never renormalised):
  BEAM(k, B)  each mention stored as its top-k candidates; the state keeps the top-B entities (merging paths that reach
              the same entity by summing); hop: w_f = <p, ps_f> over the facts of the hop's relation, p' = sum_f w_f po_f
              ARGMAX = BEAM(1, 1); B = inf keeps everything
  ENUM(k)     the same top-k readings, every path enumerated without merging (k^h growth): accuracy = BEAM(k, inf),
              cost counted from the propagated path counts
  EXACT       full distributions everywhere (dense rows), the same sum-product: the strongest exact baseline
VSA arms: bipolar codebook PHI (E x D, seeded), code of a mention = sum_{top-kb} p(e) PHI[e]; one memory per
(context, relation): M = sum_f code(s_f) * P(code(o_f)) (P a fixed coordinate permutation, so s and o are not
interchangeable); hop: x <- P^-1(x * M_r), x /= |x|.
  VSA-END(D, kb)      clean-up only when the answer is read (argmax of PHI x)
  VSA-HOP(D, kb, c)   clean-up after every hop: scores PHI x, keep the top-c entities with weights max(score, 0),
                      re-bundle, normalise

Cost model (prereg section "Cost model"): one multiply-add, comparison, hash probe or normalisation step = 1 op.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b41_names as NM                                                      # noqa: E402
import b41_data as DA                                                       # noqa: E402

E, M_REL = NM.E, NM.M_REL
HOPS_OUT = (2, 4, 6)
KS = [1, 2, 4, 8, 16]
BS = [1, 4, 16, 64, 0]                    # 0 = inf
ENUM_K = [2, 4, 8, 16]
DS = [256, 1024, 4096, 16384, 65536]
KB = [1, 4, 16]
MODES = ["END", "HOP1", "HOP4", "HOP16"]
INF = 0


def table_configs():
    out = [("BEAM", k, b) for k in KS for b in BS]
    return out


def name_table(k, b):
    if k == 1 and b == 1:
        return "ARGMAX"
    return "BEAM_k%d_B%s" % (k, "inf" if b == 0 else b)


class Run:
    def __init__(self, split, seed, N, rho, nq, logits_path, dev):
        self.split, self.seed, self.N, self.rho, self.dev = split, seed, N, rho, dev
        d = DA.make(seed, N, nq, split)
        self.d = d
        meta = json.load(open(logits_path.replace("_logits.npy", "_logits_meta.json")))
        idx = DA.row_index(split, meta["rhos"])
        lg = np.load(logits_path, mmap_mode="r")
        def rows(ents, js):
            return np.array([idx(int(e), rho, int(j)) for e, j in zip(ents, js)], np.int64)
        def dist(r):
            x = torch.from_numpy(np.ascontiguousarray(lg[np.sort(r)]).astype(np.float32))
            # restore order
            order = np.argsort(np.argsort(r, kind="stable"), kind="stable")
            x = x[torch.from_numpy(order)]
            return torch.softmax(x.to(dev).double(), -1)
        self.rs, self.ro = rows(d["f_s"], d["f_js"]), rows(d["f_o"], d["f_jo"])
        self.r0 = rows(d["q_ents"][:, 0], d["q_j0"])
        self.Ps, self.Po, self.P0 = dist(self.rs), dist(self.ro), dist(self.r0)
        self.nf, self.nq = len(d["f_s"]), len(d["q_ctx"])
        # slots: (ctx, rel) -> facts
        key = d["f_ctx"].astype(np.int64) * M_REL + d["f_rel"]
        order = np.argsort(key, kind="stable")
        uk, start, cnt = np.unique(key[order], return_index=True, return_counts=True)
        self.slot_of_fact = np.empty(self.nf, np.int64)
        self.slot_of_fact[order] = np.repeat(np.arange(len(uk)), cnt)
        self.n_slots = len(uk)
        self.ctx_of_slot = (uk // M_REL).astype(np.int64)
        kmap = {int(k): i for i, k in enumerate(uk)}
        qkey = d["q_ctx"][:, None].astype(np.int64) * M_REL + d["q_rels"]
        self.q_slot = np.vectorize(lambda k: kmap.get(int(k), -1))(qkey)          # (nq, 6)
        self.facts_in_slot = [order[s:s + c] for s, c in zip(start, cnt)]
        self.slot_size = cnt
        self.pairs = []
        for t in range(DA.HMAX):
            pq, pf = [], []
            for q in range(self.nq):
                s = self.q_slot[q, t]
                if s >= 0:
                    f = self.facts_in_slot[s]
                    pq.append(np.full(len(f), q)); pf.append(f)
            pq = np.concatenate(pq) if pq else np.zeros(0, np.int64)
            pf = np.concatenate(pf) if pf else np.zeros(0, np.int64)
            self.pairs.append((torch.from_numpy(pq).to(dev), torch.from_numpy(pf).to(dev)))
        self.nf_hop = np.array([[self.slot_size[s] if s >= 0 else 0 for s in self.q_slot[q]] for q in range(self.nq)])
        self.answers = d["q_ents"][:, list(HOPS_OUT)]                                # (nq, 3)

    # ---------------------------------------------------------------- tables
    @staticmethod
    def trunc(P, k):
        if k >= E:
            return P
        v, i = torch.topk(P, k, -1)
        return torch.zeros_like(P).scatter_(-1, i, v)

    @staticmethod
    def trunc_state(p, B):
        if B == INF or B >= E:
            return p
        v, i = torch.topk(p, B, -1)
        return torch.zeros_like(p).scatter_(-1, i, v)

    def _hop(self, p, Ps, Po, t, count, chunk=1 << 14):
        pq, pf = self.pairs[t]
        out = torch.zeros_like(p)
        post = torch.zeros(self.nq, dtype=torch.float64, device=self.dev)
        scat = torch.zeros(self.nq, dtype=torch.float64, device=self.dev)
        for b in range(0, len(pq), chunk):
            q, f = pq[b:b + chunk], pf[b:b + chunk]
            A, pp = Ps[f], p[q]
            w = (A * pp).sum(-1)
            out.index_add_(0, q, w[:, None] * Po[f])
            if count:
                po = ((A > 0) & (pp > 0)).sum(-1).double()
                post.index_add_(0, q, po)
                scat.index_add_(0, q, (po > 0).double() * (Po[f] > 0).sum(-1).double())
        return out, post, scat

    def run_table(self, k, B):
        Ps, Po = self.trunc(self.Ps, k), self.trunc(self.Po, k)
        p = self.trunc_state(self.trunc(self.P0, k), B)
        nq = self.nq
        ops = torch.full((nq,), float(E), dtype=torch.float64, device=self.dev)      # start: selection over E
        tr = torch.full((nq,), float(E), dtype=torch.float64, device=self.dev)       # read the start distribution
        st = 2 * (p > 0).sum(-1).double()
        res = {"correct": [], "answer": [], "ops": [], "traffic": [], "state": []}
        for t in range(DA.HMAX):
            nnz = (p > 0).sum(-1).double()
            p2, post, scat = self._hop(p, Ps, Po, t, True)
            nnz2 = (p2 > 0).sum(-1).double()
            s = p2.sum(-1, keepdim=True)
            p2 = torch.where(s > 0, p2 / s.clamp_min(1e-300), p2)
            ops += nnz + post + scat + nnz2                                    # probes + MACs + normalise
            tr += nnz + nnz + 2 * post + 2 * scat                              # state read + keys + postings + objects
            if B != INF:
                ops += torch.where(nnz2 > B, nnz2, torch.zeros_like(nnz2))     # top-B selection
            p = self.trunc_state(p2, B)
            st = torch.maximum(st, 2 * (p > 0).sum(-1).double())
            if t + 1 in HOPS_OUT:
                alive = p.sum(-1) > 0
                ans = torch.where(alive, p.argmax(-1), torch.full_like(alive, -1, dtype=torch.long))
                fin = (p > 0).sum(-1).double()                                 # final argmax
                res["answer"].append(ans.cpu().numpy())
                res["ops"].append((ops + fin).cpu().numpy())
                res["traffic"].append((tr + fin).cpu().numpy())
                res["state"].append(st.cpu().numpy().copy())
        return self._pack(res)

    def run_exact(self):
        p = self.P0.clone()
        res = {"correct": [], "answer": [], "ops": [], "traffic": [], "state": []}
        ops = np.full(self.nq, float(E)); tr = np.full(self.nq, float(E))
        for t in range(DA.HMAX):
            p2, _, _ = self._hop(p, self.Ps, self.Po, t, False)
            s = p2.sum(-1, keepdim=True)
            p = torch.where(s > 0, p2 / s.clamp_min(1e-300), p2)
            nfh = self.nf_hop[:, t].astype(float)
            ops += 2 * nfh * E + E                                             # dense dot + dense scatter + normalise
            tr += 2 * nfh * E + E                                              # dense rows of ps and po + state
            if t + 1 in HOPS_OUT:
                alive = p.sum(-1) > 0
                ans = torch.where(alive, p.argmax(-1), torch.full_like(alive, -1, dtype=torch.long))
                res["answer"].append(ans.cpu().numpy())
                res["ops"].append(ops + E); res["traffic"].append(tr + E)
                res["state"].append(np.full(self.nq, float(E)))
        return self._pack(res)

    def run_enum(self, k):
        """Path counts n (no merging): ops per hop = paths (probes) + sum_f paths_matched_f * (1 + |po_f|)."""
        As, Ao = (self.trunc(self.Ps, k) > 0).double(), (self.trunc(self.Po, k) > 0).double()
        n = (self.trunc(self.P0, k) > 0).double()
        ops = np.full(self.nq, float(E)); tr = np.full(self.nq, float(E))
        st = 2 * n.sum(-1).cpu().numpy()
        res = {"ops": [], "traffic": [], "state": [], "paths": []}
        for t in range(DA.HMAX):
            pq, pf = self.pairs[t]
            n2 = torch.zeros_like(n)
            mac = torch.zeros(self.nq, dtype=torch.float64, device=self.dev)
            for b in range(0, len(pq), 1 << 14):
                q, f = pq[b:b + (1 << 14)], pf[b:b + (1 << 14)]
                pm = (As[f] * n[q]).sum(-1)
                n2.index_add_(0, q, pm[:, None] * Ao[f])
                mac.index_add_(0, q, pm * (1 + Ao[f].sum(-1)))
            paths = n.sum(-1)
            ops += (paths + mac).cpu().numpy()
            tr += (2 * paths + 2 * mac).cpu().numpy()
            n = n2
            st = np.maximum(st, 2 * n.sum(-1).cpu().numpy())
            if t + 1 in HOPS_OUT:
                fin = n.sum(-1).cpu().numpy()
                res["ops"].append(ops + fin); res["traffic"].append(tr + fin); res["state"].append(st.copy())
                res["paths"].append(fin)
        return {kk: np.stack(v, 1) for kk, v in res.items()}

    # ---------------------------------------------------------------- VSA
    def codebook(self, D):
        g = torch.Generator(device="cpu").manual_seed(41_000_000 + 1000 * self.seed + D % 997)
        PHI = (torch.randint(0, 2, (E, D), generator=g, dtype=torch.int8) * 2 - 1).float().to(self.dev)
        P = torch.randperm(D, generator=g).to(self.dev)
        return PHI, P, torch.argsort(P)

    def codes(self, W, PHI, chunk=4096):
        out = []
        for b in range(0, W.shape[0], chunk):
            out.append(W[b:b + chunk] @ PHI)
        return torch.cat(out) if out else torch.zeros(0, PHI.shape[1], device=self.dev)

    def build_memory(self, PHI, P, kb, chunk=4096):
        D = PHI.shape[1]
        Mem = torch.zeros(self.n_slots, D, device=self.dev)
        sl = torch.from_numpy(self.slot_of_fact).to(self.dev)
        for b in range(0, self.nf, chunk):
            cs = self.trunc(self.Ps[b:b + chunk], kb).float() @ PHI
            co = self.trunc(self.Po[b:b + chunk], kb).float() @ PHI
            Mem.index_add_(0, sl[b:b + chunk], cs * co[:, P])
        return Mem

    def run_vsa(self, D, kb, PHI, P, Pinv, Mem):
        """All four modes from one memory. Returns {mode: packed result}."""
        W0 = self.trunc(self.P0, kb).float()
        kb_eff = (W0 > 0).sum(-1).double().cpu().numpy()
        x0 = W0 @ PHI
        x0 = x0 / x0.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        qs = torch.from_numpy(self.q_slot).to(self.dev)
        out = {}
        for mode in MODES:
            c = 0 if mode == "END" else int(mode[3:])
            x = x0.clone()
            alive = torch.ones(self.nq, dtype=torch.bool, device=self.dev)
            ops = E + kb_eff * D + 2 * D                                     # select, bundle, normalise
            tr = E + kb_eff * D + 2 * kb_eff
            res = {"answer": [], "ops": [], "traffic": [], "state": []}
            for t in range(DA.HMAX):
                s = qs[:, t]
                ok = s >= 0
                Mr = torch.where(ok[:, None], Mem[s.clamp_min(0)], torch.zeros_like(x))
                x = (x * Mr)[:, Pinv]
                nrm = x.norm(dim=-1, keepdim=True)
                alive = alive & (nrm[:, 0] > 0)
                x = x / nrm.clamp_min(1e-30)
                ops = ops + 3 * D                                            # unbind + normalise
                tr = tr + 2 * D                                              # memory row + state
                if c:
                    sc = x @ PHI.T
                    v, i = torch.topk(sc, c, -1)
                    w = v.clamp_min(0)
                    ans = i[:, 0]
                    last = (t + 1 == DA.HMAX)
                    if not last:
                        x = torch.zeros(self.nq, E, device=self.dev).scatter_(1, i, w) @ PHI
                        nrm = x.norm(dim=-1, keepdim=True)
                        alive = alive & (nrm[:, 0] > 0)
                        x = x / nrm.clamp_min(1e-30)
                    ops = ops + E * D + E
                    tr = tr + E * D
                    if t + 1 in HOPS_OUT:
                        a_ = torch.where(alive, ans, torch.full_like(ans, -1))
                        res["answer"].append(a_.cpu().numpy())
                        res["ops"].append(np.broadcast_to(ops, (self.nq,)).astype(float).copy())
                        res["traffic"].append(np.broadcast_to(tr, (self.nq,)).astype(float).copy())
                        res["state"].append(np.full(self.nq, float(D)))
                    if not last:
                        ops = ops + c * D + 2 * D
                        tr = tr + c * D
                elif t + 1 in HOPS_OUT:
                    sc = x @ PHI.T
                    ans = sc.argmax(-1)
                    a_ = torch.where(alive, ans, torch.full_like(ans, -1))
                    res["answer"].append(a_.cpu().numpy())
                    res["ops"].append(np.broadcast_to(ops + E * D + E, (self.nq,)).astype(float).copy())
                    res["traffic"].append(np.broadcast_to(tr + E * D, (self.nq,)).astype(float).copy())
                    res["state"].append(np.full(self.nq, float(D)))
            out[mode] = self._pack(res)
        return out

    def _pack(self, res):
        r = {k: np.stack(v, 1) for k, v in res.items() if v}
        r["correct"] = r["answer"] == self.answers
        return r

    # ---------------------------------------------------------------- store costs (per context, reported apart)
    def store_costs(self):
        f_per_ctx = self.nf / self.d["n_ctx"]
        slots_per_ctx = self.n_slots / self.d["n_ctx"]
        out = {"facts_per_ctx": f_per_ctx, "slots_per_ctx": slots_per_ctx, "queries_per_ctx": self.d["Q_per_ctx"]}
        for k in KS:
            nz_s = float((self.trunc(self.Ps, k) > 0).sum()) / self.d["n_ctx"]
            nz_o = float((self.trunc(self.Po, k) > 0).sum()) / self.d["n_ctx"]
            # relation word + (id, prob) per stored candidate + one posting (fact id) per subject candidate
            out["table_k%d" % k] = {"build_ops": f_per_ctx * 2 * E, "size_words": f_per_ctx + 2 * (nz_s + nz_o) + nz_s}
        out["EXACT"] = {"build_ops": 0.0, "size_words": f_per_ctx * (1 + 2 * E)}
        for D in DS:
            for kb in KB:
                out["VSA_D%d_kb%d" % (D, kb)] = {"build_ops": f_per_ctx * (2 * E + 2 * kb * D + D),
                                                 "size_words": slots_per_ctx * D, "codebook_words": E * D}
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--logits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--Ds", type=int, nargs="*", default=DS)
    ap.add_argument("--no_vsa", action="store_true")
    ap.add_argument("--no_tables", action="store_true", help="exploratory VSA-only runs (extra D)")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    R = Run(a.split, a.seed, a.N, a.rho, a.nq, a.logits, dev)
    out = {}
    # per-mention naming diagnostics
    ms = np.concatenate([R.d["f_s"], R.d["f_o"], R.d["q_ents"][:, 0]])
    am = torch.cat([R.Ps.argmax(-1), R.Po.argmax(-1), R.P0.argmax(-1)]).cpu().numpy()
    diag = {"mention_argmax_error": float((am != ms).mean()), "n_mentions": int(len(ms)),
            "nf": R.nf, "nq": R.nq, "n_ctx": R.d["n_ctx"], "n_slots": R.n_slots,
            "mean_facts_per_hop": R.nf_hop.mean(0).tolist(), "decoy_rate": float(R.d["q_decoy"].mean()),
            "start_argmax_correct": (R.P0.argmax(-1).cpu().numpy() == R.d["q_ents"][:, 0]).tolist()}
    if not a.no_tables:
        for (_, k, b) in table_configs():
            out[name_table(k, b)] = R.run_table(k, b)
        out["EXACT"] = R.run_exact()
        for k in ENUM_K:
            e = R.run_enum(k)
            base = out["BEAM_k%d_Binf" % k]
            out["ENUM_k%d" % k] = {"answer": base["answer"], "correct": base["correct"], "ops": e["ops"],
                                   "traffic": e["traffic"], "state": e["state"], "paths": e["paths"]}
    t1 = time.time()
    if not a.no_vsa:
        for D in a.Ds:
            PHI, P, Pinv = R.codebook(D)
            for kb in KB:
                Mem = R.build_memory(PHI, P, kb)
                res = R.run_vsa(D, kb, PHI, P, Pinv, Mem)
                for mode, r in res.items():
                    out["VSA_%s_D%d_kb%d" % (mode, D, kb)] = r
                del Mem
            del PHI
            torch.cuda.empty_cache()
    flat = {}
    for name, r in out.items():
        for k, v in r.items():
            flat["%s|%s" % (name, k)] = v
    flat["_answers"] = R.answers
    flat["_decoy"] = R.d["q_decoy"]
    flat["_nf_hop"] = R.nf_hop
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, **flat)
    meta = {"split": a.split, "seed": a.seed, "N": a.N, "rho": a.rho, "nq": a.nq, "diag": diag,
            "store": R.store_costs(), "sec_tables": t1 - t0, "sec_total": time.time() - t0,
            "acc": {name: r["correct"].mean(0).round(4).tolist() for name, r in out.items()},
            "ops": {name: r["ops"].mean(0).tolist() for name, r in out.items()}}
    json.dump(meta, open(a.out.replace(".npz", ".json"), "w"))
    print("done", a.split, a.seed, a.N, a.rho, "tables %.0fs total %.0fs" % (t1 - t0, time.time() - t0), flush=True)
    for name in ["ARGMAX", "BEAM_k4_B16", "BEAM_k16_Binf", "EXACT", "VSA_END_D65536_kb16", "VSA_HOP4_D65536_kb16",
                 "VSA_HOP4_D4096_kb4"]:
        if name in out:
            print("  %-24s acc %s  ops %s" % (name, out[name]["correct"].mean(0).round(3),
                                             np.round(out[name]["ops"].mean(0)).astype(int)), flush=True)


if __name__ == "__main__":
    main()
