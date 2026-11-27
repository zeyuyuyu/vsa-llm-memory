"""B34 checks from other angles (prereg/B34_inmemory_directory_prereg.md).

python3 b34_check.py closed TEST_STORES.json.gz BER.json CHOICE.json SIM.json OUT.txt [--workers N]
    Closed-form expected share of probed keys read exactly, per arm x k x time, computed WITHOUT simulating a single
    cell: only the noiseless layouts / codes and the device model's measured cell bit-error rate p(k, t) (the device
    cells are i.i.d.), compared with the simulated totals. Exact (binomial) for the repetition-coded arms (T-iv, T-i,
    T-iih, T-v); exact up to tie handling for the CAM nearest-Hamming search (rows are independent); the T-iiib value
    decode treats candidates as independent and the VSA uses a Gaussian approximation of the clean-up similarities.
python3 b34_check.py handread TEST_STORES.json.gz CHOICE.json STORE_ID ARM K TIME [N]
    Re-simulates one store / arm / k / time with the run's own seeds and prints up to N failed keys with the cause.
"""
import collections
import json
import math
import sys

import numpy as np
from scipy import stats
from scipy.special import log_ndtr

import b25_pcm as P
import b34_inmem as M
import ruler_vsa as V

D = M.D


# ---------------------------------------------------------------- building blocks
def q_maj(r, p):
    """P(majority of r i.i.d. copies wrong), ties at random."""
    if p <= 0:
        return 0.0
    x = stats.binom(r, p)
    if r % 2:
        return float(x.sf(r // 2))
    return float(x.sf(r // 2) + 0.5 * x.pmf(r // 2))


class Soft:
    """P(the true row is the nearest in Hamming distance) for a clean query against independent noisy rows."""

    def __init__(self, W, p):
        self.W, self.p = W, p
        self.cache = {}
        if p <= 0:
            self.dmax = 0
            self.pt = np.array([1.0])
        else:
            bt = stats.binom(W, p)
            self.dmax = int(bt.isf(1e-13)) + 1
            self.pt = bt.pmf(np.arange(self.dmax + 1))
        self.normal = W > 2000

    def surv(self, delta):
        """(P(D > d), P(D >= d)) for d = 0..dmax, D = delta - X + Y, X ~ Bin(delta, p), Y ~ Bin(W - delta, p)."""
        if delta in self.cache:
            return self.cache[delta]
        d = np.arange(self.dmax + 1)
        if self.p <= 0:
            gt = (delta > d).astype(float)
            ge = (delta >= d).astype(float)
        elif self.normal:
            mu = delta * (1 - 2 * self.p) + self.W * self.p
            sd = math.sqrt(self.W * self.p * (1 - self.p))
            gt = stats.norm.sf((d + 0.5 - mu) / sd)
            ge = stats.norm.sf((d - 0.5 - mu) / sd)
        else:
            a = stats.binom.pmf(np.arange(delta + 1), delta, 1 - self.p)      # delta - X
            b = stats.binom.pmf(np.arange(self.W - delta + 1), self.W - delta, self.p)
            pm = np.convolve(a, b)
            cdf = np.cumsum(pm)
            le = cdf[np.minimum(d, len(cdf) - 1)]
            lt = np.concatenate([[0.0], cdf])[np.minimum(d, len(cdf))]
            gt = np.clip(1 - le, 0, 1)
            ge = np.clip(1 - lt, 0, 1)
        self.cache[delta] = (gt, ge)
        return gt, ge

    def win_hist(self, h):
        """h: Counter of the noiseless distances from the query to every other row."""
        lgt = np.zeros(self.dmax + 1)
        lge = np.zeros(self.dmax + 1)
        for delta, c in h.items():
            if delta > self.dmax + 40 * math.sqrt(self.W * max(self.p, 1e-9)) + 40 and not self.normal:
                continue                                      # cannot come closer than the true row
            gt, ge = self.surv(delta)
            lgt += c * np.log(np.maximum(gt, 1e-300))
            lge += c * np.log(np.maximum(ge, 1e-300))
        return float(np.sum(self.pt * 0.5 * (np.exp(lgt) + np.exp(lge))))


# ---------------------------------------------------------------- closed form per arm
def cf_store(store, sid, arm, info, p):
    """Expected number of probed keys read exactly (and the per-key probabilities)."""
    s = store
    probs = []
    if arm == "T-iv":
        r = info["r"]
        qv = q_maj(r, p)
        for k in s.keys:
            probs.append((1 - qv) ** (s.b * s.count[k]))
        return probs
    if arm == "T-v":
        r = info["r"]
        qv = q_maj(r, p)
        mp = info["mphf"]
        pb = s.lastbit + s.b
        for k in s.keys:
            rows = [s.row_of[(k, j)] for j in range(s.count[k])]
            nb = len({int(mp["h0"][i]) for i in rows})
            probs.append((1 - qv) ** (mp["db"] * nb + pb * len(rows)))
        return probs
    if arm in ("T-vhd", "T-vhdd"):
        mp = info["mphf"]
        if arm == "T-vhd":
            qd = q_maj(info["rd"], p)
        elif "_dgeom" not in info:                            # distances from each stored block code to its rivals
            hb, wb = info["hb"], info["wb"]
            dg = {}
            cand_d = [np.stack([M.hd_code("dblk%d:%d" % (c_, v), wb) for v in range(1 << hb)]).astype(np.float32)
                      for c_ in range(2)]
            for g in range(mp["B"]):
                sg = int(mp["sigma"][g])
                blk = (sg >> hb, sg & ((1 << hb) - 1))
                dg[g] = [collections.Counter(np.rint(np.delete((wb - cand_d[c_] @ cand_d[c_][blk[c_]]) / 2, blk[c_]))
                                             .astype(int).tolist()) for c_ in range(2)]
            info["_dgeom"] = dg
        if "_geom" not in info:
            wv = info["wv"]
            rows = info["fields_logical"]["valhd"].astype(np.float32)
            cand = np.stack([M.hd_code("value:" + v, wv) for v in s.cd["vocab"]]).astype(np.float32)
            if s.lastbit:
                cand = np.concatenate([cand, cand * M.hd_code("last", wv)[None, :].astype(np.float32)], 0)
            vd = {}
            for k in s.keys:
                for j in range(s.count[k]):
                    sl = int(mp["slot"][s.row_of[(k, j)]])
                    dv = (wv - cand @ rows[sl]) / 2
                    t = int(np.argmin(dv))
                    vd[(k, j)] = collections.Counter(np.rint(np.delete(dv, t)).astype(int).tolist())
            info["_geom"] = vd
        softv = Soft(info["wv"], p)
        softd = Soft(info["wb"], p) if arm == "T-vhdd" else None
        for k in s.keys:
            rows_ = [s.row_of[(k, j)] for j in range(s.count[k])]
            bks = {int(mp["h0"][i]) for i in rows_}
            if arm == "T-vhd":
                pk = (1 - qd) ** (mp["db"] * len(bks))
            else:
                pk = 1.0
                for g in bks:
                    for c_ in range(2):
                        pk *= softd.win_hist(info["_dgeom"][g][c_])
            for j in range(s.count[k]):
                pk *= softv.win_hist(info["_geom"][(k, j)])
            probs.append(pk)
        return probs
    meta = info["arrays"]["rows"][1]
    fl = info["fields_logical"]
    if arm == "T-i":
        rv = info["rv"]
        qv = q_maj(rv, p)
        for k in s.keys:
            probs.append(((1 - p) ** info["wk"] * (1 - qv) ** s.b) ** s.count[k])
        return probs
    if arm == "T-iih":
        r = info["r"]
        qv = q_maj(r, p)
        for k in s.keys:
            probs.append((1 - qv) ** ((info["wk"] + s.b) * s.count[k]))
        return probs
    # soft directory arms: noiseless distances are precomputed once per store (geom), then evaluated per p
    if "_geom" not in info:
        info["_geom"] = soft_geometry(s, arm, info)
    geom = info["_geom"]
    W = geom["W"]
    soft = Soft(W, p)
    if arm == "T-iiib":
        softv = Soft(geom["wv"], p)
    else:
        pb = s.lastbit + s.b
        qv = q_maj(info["rp"], p)
    for k in s.keys:
        pk = 1.0
        for j in range(s.count[k]):
            pk *= soft.win_hist(geom["kd"][(k, j)])
            if arm == "T-iiib":
                pk *= softv.win_hist(geom["vd"][(k, j)])
            else:
                pk *= (1 - qv) ** pb
        probs.append(pk)
    return probs


def soft_geometry(s, arm, info):
    """Histograms of the noiseless distances (in cells) from each probed (key, j)'s query to every other row, and for
    T-iiib from the stored value code to every other vocabulary candidate."""
    fl = info["fields_logical"]
    key = fl["key"].astype(np.float32)
    rd = info.get("rd", 1)
    g = {"W": key.shape[1] * rd, "kd": {}, "vd": {}}
    if arm == "T-iiib":
        vf = fl["valhd"].astype(np.float32)
        wv = vf.shape[1]
        g["wv"] = wv
        vocab = s.cd["vocab"]
        cand = np.stack([M.hd_code("value:" + v, wv) for v in vocab]).astype(np.float32)
        if s.lastbit:
            cand = np.concatenate([cand, cand * M.hd_code("last", wv)[None, :].astype(np.float32)], 0)
    for k in s.keys:
        for j in range(s.count[k]):
            i = s.row_of[(k, j)]
            dl = np.delete((key.shape[1] - key @ key[i]) / 2 * rd, i)
            g["kd"][(k, j)] = collections.Counter(np.rint(dl).astype(int).tolist())
            if arm == "T-iiib":
                dv = (wv - cand @ vf[i]) / 2
                t = int(np.argmin(dv))
                g["vd"][(k, j)] = collections.Counter(np.rint(np.delete(dv, t)).astype(int).tolist())
    return g


def vsa_geometry(s):
    """Noiseless clean-up similarities of every probed key's reads (j = 0 .. count, the last one being the stop read)."""
    st = s.st
    g = []
    for a, k in enumerate(s.keys):
        sh = st["sh"][a]
        X = st["b1"][sh] * st["rms0"][sh] * st["Kc"][st["kidx"][a]]
        reads = [(X * st["Pc"][j]) @ st["Vc"].T / D for j in range(min(s.count[k] + 1, st["maxj"]))]
        g.append((float(st["rms0"][sh]), reads))
    return g


def cf_vsa(store, p, geom, nt=False):
    """Gaussian approximation of B25's read (V-S) or of the first-value-by-argmax read (V-Snt): under i.i.d. sign
    flips at rate p each similarity is ~ N((1 - 2p) s0, rms0^2 4p(1 - p) / d), candidates treated as independent."""
    s = store
    probs = []
    th = P.THR
    for a, k in enumerate(s.keys):
        rms0, reads = geom[a]
        sig = rms0 * math.sqrt(4 * p * (1 - p) / D)
        pk = 1.0
        c = s.count[k]
        for j, s0 in enumerate(reads):
            mu = s0 * (1 - 2 * p)
            if j < c:
                t = s.want[a][j]
                thr = -np.inf if (nt and j == 0) else th
                others = np.delete(mu, t)
                if sig < 1e-12:
                    pk *= float(mu[t] >= thr and mu[t] > others.max())
                    continue
                lo = max(thr, mu[t] - 8 * sig)
                hi = mu[t] + 8 * sig
                if lo >= hi:
                    pk *= 0.0
                    continue
                sgrid = np.linspace(lo, hi, 1601)
                rel = others[others > mu[t] - 12 * sig]
                # rivals binned at sigma / 20 (the product over thousands of rivals, evaluated per bin)
                cnt, edges = np.histogram(rel, bins=max(1, int(np.ceil((rel.max() - rel.min()) / (sig / 20)))) if len(rel)
                                          else 1)
                cen = 0.5 * (edges[1:] + edges[:-1])
                keep = cnt > 0
                lp = (cnt[keep][None, :] * log_ndtr((sgrid[:, None] - cen[keep][None, :]) / sig)).sum(1) if len(rel) \
                    else np.zeros_like(sgrid)
                dens = stats.norm.pdf(sgrid, mu[t], sig) * np.exp(lp)
                pk *= float(np.trapz(dens, sgrid))
            else:                                             # the stop read: nothing reaches the threshold
                if sig < 1e-12:
                    pk *= float(mu.max() < th)
                else:
                    pk *= float(np.exp(np.sum(log_ndtr((th - mu) / sig))))
        probs.append(pk)
    return probs


def _closed_job(job):
    sid, pairs, choice, ber = job
    s = M.Store(pairs, "b26b:%s" % sid.replace("/", ":"))
    out = {}
    lam = choice.get("T-v", {}).get("lam", 4)
    infos = {}
    for arm in M.TABLES:
        if arm == "T-iv":
            from b26b_pcm_table import table_layout
            _, _, b, r, _, _ = table_layout(s.pairs, s.st, s.cd)
            infos[arm] = {"r": r}
        else:
            infos[arm] = M.layout(s, arm, s.C, share=choice.get(arm, {}).get("share"), lam=lam)
    vg = vsa_geometry(s)
    for (k, tl), p in ber.items():
        for arm in M.TABLES:
            if infos[arm] is None:
                out["%s|%s|x%g" % (arm, tl, k)] = [0.0, 0.0, len(s.keys)]
                continue
            pr = np.clip(cf_store(s, sid, arm, infos[arm], p), 0, 1)
            out["%s|%s|x%g" % (arm, tl, k)] = [float(np.sum(pr)), float(np.sum(np.array(pr) * (1 - np.array(pr)))),
                                              len(pr)]
        for arm, nt in (("V-S", False), ("V-Snt", True)):
            pr = np.clip(cf_vsa(s, p, vg, nt), 0, 1)
            out["%s|%s|x%g" % (arm, tl, k)] = [float(np.sum(pr)), float(np.sum(np.array(pr) * (1 - np.array(pr)))),
                                              len(pr)]
    print("closed form done", sid, flush=True)
    return out


def closed(stores_f, ber_f, choice_f, sim_f, out_f, workers):
    stores = M.load_store_file(stores_f)
    ber = M.read_ber(ber_f)
    choice = json.load(open(choice_f))
    sim = json.load(open(sim_f))["totals"]
    jobs = sorted([(sid, pairs, choice, ber) for sid, pairs in stores], key=lambda j: -len(j[1]))
    from multiprocessing import Pool
    with Pool(workers) as pool:
        per = list(pool.imap_unordered(_closed_job, jobs))
    tot = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for o in per:
        for kk, (e, v, n) in o.items():
            tot[kk][0] += e
            tot[kk][1] += v
            tot[kk][2] += n
    json.dump({kk: v for kk, v in tot.items()}, open(out_f.replace(".txt", ".json"), "w"), indent=1)
    lines = ["# B34 closed-form check: expected share of the probed keys read exactly from the measured cell bit-error",
             "# rate p(k, t) and the noiseless codes only (no cell simulated), against the device simulation.",
             "# cell: simulated / closed form / difference in points (flag '!' if |diff| > max(1 point, 4 se))", ""]
    arms = list(M.TABLES) + ["V-S", "V-Snt"]
    flags = 0
    worst = (0, None)
    for arm in arms:
        lines.append(arm)
        for k in M.KS:
            row = []
            for tl, _ in M.TIMES:
                kk = "%s|%s|x%g" % (arm, tl, k)
                e, v, n = tot[kk]
                cf = e / n
                sm = sim.get(kk)
                if sm is None:
                    row.append("   (no sim)        ")
                    continue
                d = (sm[2] - cf) * 100
                se = math.sqrt(max(v, 0.0)) / n * 100
                bad = abs(d) > max(1.0, 4 * se)
                flags += bad
                if abs(d) > abs(worst[0]):
                    worst = (d, kk)
                row.append("%6.2f/%6.2f %+5.2f%s" % (sm[2] * 100, cf * 100, d, "!" if bad else " "))
            lines.append("  x%-3g %s" % (k, "  ".join(row)))
    lines.append("")
    lines.append("cells flagged: %d of %d; largest |difference| %+.2f points at %s" % (
        flags, len(arms) * len(M.KS) * len(M.TIMES), worst[0], worst[1]))
    open(out_f, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    json.dump({kk: v for kk, v in tot.items()}, open(out_f.replace(".txt", ".json"), "w"), indent=1)


# ---------------------------------------------------------------- hand reading of failures
def handread(stores_f, choice_f, sid, arm, k, tl, nshow=5):
    stores = dict(M.load_store_file(stores_f))
    choice = json.load(open(choice_f))
    pairs = stores[sid]
    s = M.Store(pairs, "b26b:%s" % sid.replace("/", ":"))
    tt = dict(M.TIMES)[tl]
    vocab = s.cd["vocab"]
    print("store %s  arm %s  x%g  %s  probed keys %d  pairs %d  values %d  S %d" % (
        sid, arm, k, tl, len(s.keys), s.n, s.Nv, s.st["S"]))
    if arm in ("V-S", "V-Snt"):
        st = s.st
        out, dg = P.store_array("B1", st["mem"], st["b1"], s.seed, tt, k)
        rows = out["gdc"]
        print("store bit errors %.4f" % dg["bit_err"])
        dig = lambda Y, Vc=st["Vc"]: Y @ Vc.T / D
        rd = M.read_keys_nt if arm == "V-Snt" else P.read_keys
        got = rd(rows, st["rms0"], st["sh"], st["kidx"], st["Kc"], st["Pc"], dig, st["maxj"])
        bad = [a for a in range(len(s.keys)) if got[a] != s.want[a]]
        print("failed %d of %d" % (len(bad), len(s.keys)))
        for a in bad[:nshow]:
            sh = st["sh"][a]
            X = rows[sh] * st["rms0"][sh] * st["Kc"][st["kidx"][a]]
            X0 = st["b1"][sh] * st["rms0"][sh] * st["Kc"][st["kidx"][a]]
            print("  key %r  want %s  got %s  shard %d  flipped cells in shard %d / %d" % (
                s.keys[a], [vocab[i] for i in s.want[a]], [vocab[i] for i in got[a]], sh,
                int((rows[sh] != st["b1"][sh]).sum()), D))
            for j in range(min(len(s.want[a]) + 1, st["maxj"])):
                sims = (X * st["Pc"][j]) @ st["Vc"].T / D
                sims0 = (X0 * st["Pc"][j]) @ st["Vc"].T / D
                b = int(np.argmax(sims))
                t = s.want[a][j] if j < len(s.want[a]) else None
                print("    j=%d  true sim %s (noiseless %s)  best %r %.3f  threshold %.2f" % (
                    j, "%.3f" % sims[t] if t is not None else "-", "%.3f" % sims0[t] if t is not None else "-",
                    vocab[b], sims[b], P.THR))
        return
    lam = choice.get("T-v", {}).get("lam", 4)
    info = M.layout(s, arm, s.C, share=choice.get(arm, {}).get("share"), lam=lam)
    tag = arm
    aseed = V._seed("b34:%s:%s:%s" % (sid, tag, "C")) % (2 ** 62)
    sig = {}
    for an, (cells, _) in info["arrays"].items():
        asd = V._seed("b34:%s:%s:%s:%s" % (sid, tag, "C", an)) % (2 ** 62)
        sig[an] = M.signs_over_times(cells, asd, k, [(tl, tt)])[tl]
    rng = np.random.default_rng([aseed, 9, int(tt), int(k * 10)])
    diag = []
    got = M.read_table(s, info, sig, rng, diag)
    bad = [a for a in range(len(s.keys)) if got[a] != s.want[a]]
    pinfo = {kk: v for kk, v in info.items() if kk in ("W", "wk", "wv", "rd", "rp", "rv", "r", "h", "R", "B", "db", "Lmax")}
    print("layout %s  cells/C %.3f" % (pinfo, info["cells"] / s.C))
    for an, (cells, _) in info["arrays"].items():
        print("  array %s: %d cells, flipped %.4f" % (an, cells.size, float((sig[an] != cells).mean())))
    print("failed %d of %d" % (len(bad), len(s.keys)))
    dmap = collections.defaultdict(list)
    for a, j, row, true in diag:
        dmap[a].append((j, row, true))
    for a in bad[:nshow]:
        k_ = s.keys[a]
        print("  key %r  want %s  got %s" % (k_, [vocab[i] for i in s.want[a]], [vocab[i] if i < len(vocab) else
                                                                                 "#%d" % i for i in got[a]]))
        if arm in ("T-vhd", "T-vhdd"):
            mp = info["mphf"]
            wv = info["wv"]
            vocab_ = s.cd["vocab"]
            cand = np.stack([M.hd_code("value:" + v, wv) for v in vocab_]).astype(np.float32)
            nv = len(vocab_)
            if s.lastbit:
                cand = np.concatenate([cand, cand * M.hd_code("last", wv)[None, :].astype(np.float32)], 0)
            for j in range(min(len(got[a]), s.maxj + 1)):
                i = s.row_of.get((k_, j))
                h1, h2, g = M._mp_hash(s, mp, k_, j)
                if arm == "T-vhd":
                    dec = int(M.pm_to_int(M.majority(sig["disp"][g:g + 1], mp["db"], info["rd"],
                                                     np.random.default_rng(0)))[0])
                else:
                    hb, wb = info["hb"], info["wb"]
                    parts = []
                    for c_ in range(2):
                        cd_ = np.stack([M.hd_code("dblk%d:%d" % (c_, v), wb) for v in range(1 << hb)]).astype(np.float32)
                        parts.append(int(np.argmax(cd_ @ sig["disp"][g, c_ * wb:(c_ + 1) * wb].astype(np.float32))))
                    dec = (parts[0] << hb) | parts[1]
                sl = M.mphf_slot(h1, h2, dec, mp["R"])
                true_sl = int(mp["slot"][i]) if i is not None else None
                sims = cand @ sig["rows"][sl].astype(np.float32)
                best_c = int(np.argmax(sims))
                tc = None
                if i is not None:
                    tc = int(s.rv[i]) + (nv if (s.lastbit and s.rlast[i]) else 0)
                print("    j=%d bucket %d sigma %d decoded %d -> row %d (true row %s); in that row the best candidate is %s "
                      "(%s, last=%d) at similarity %.0f of %d; the true value+flag code at %s" % (
                          j, g, int(mp["sigma"][g]), dec, sl, true_sl, best_c, vocab_[best_c % nv], int(best_c >= nv),
                          sims[best_c], wv, "%.0f" % sims[tc] if (tc is not None and sl == true_sl) else "-"))
            continue
        if arm == "T-v":
            mp = info["mphf"]
            r = info["r"]
            for j in range(s.count[k_]):
                i = s.row_of[(k_, j)]
                g = int(mp["h0"][i])
                dsig = sig["disp"][g].reshape(mp["db"], r)
                dec = M.pm_to_int(M.majority(sig["disp"][g:g + 1], mp["db"], r, np.random.default_rng(0)))[0]
                sl = int(mp["slot"][i])
                psig = sig["rows"][sl].reshape(s.lastbit + s.b, r)
                plog = info["fields_logical"]["pay"][sl]
                print("    j=%d bucket %d sigma %d decoded %d (copies agreeing per bit: %s); row %d payload votes "
                      "for the stored bit per bit: %s" % (
                          j, g, int(mp["sigma"][g]), dec,
                          (dsig == M.bits_pm([mp["sigma"][g]], mp["db"])[0][:, None]).sum(1).tolist(), sl,
                          (psig == plog[:, None]).sum(1).tolist()))
            continue
        meta = info["arrays"]["rows"][1]
        kf, wk, rk = M._field(sig["rows"], meta, "key")
        logical = info["fields_logical"]["key"]
        for j, row, true in dmap.get(a, []):
            q = np.repeat(logical[true], rk)
            dt = int((kf[true] != q).sum())
            dc = int((kf[row] != q).sum())
            d0 = int((np.repeat(logical[row], rk) != q).sum())
            print("    j=%d chose row %d (%r, j=%d) at distance %d (noiseless %d); true row %d at distance %d "
                  "(its own flipped cells %d of %d)" % (j, row, s.rk[row], int(s.rj[row]), dc, d0, true, dt, dt,
                                                       kf.shape[1]))
        if arm in ("T-i", "T-iih"):
            for j in range(s.count[k_]):
                i = s.row_of[(k_, j)]
                if arm == "T-i":
                    nk = int((kf[i] != logical[i]).sum())
                    print("    j=%d true row %d: %d of %d key cells flipped (exact match needs 0)" % (j, i, nk, wk))
                else:
                    dec = M.majority(kf[i:i + 1], wk, rk, np.random.default_rng(0))[0]
                    print("    j=%d true row %d: %d of %d key bits wrong after the %d-copy majority" % (
                        j, i, int((dec != logical[i]).sum()), wk, rk))


if __name__ == "__main__":
    if sys.argv[1] == "closed":
        w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
        closed(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], w)
    elif sys.argv[1] == "handread":
        a = sys.argv
        handread(a[2], a[3], a[4], a[5], float(a[6]), a[7], int(a[8]) if len(a) > 8 else 5)
