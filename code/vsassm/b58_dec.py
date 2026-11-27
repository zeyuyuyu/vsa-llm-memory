"""B58 part (c) (prereg/B58_protected_store_prereg.md): the LDPC decoder discrepancy of B38.

python3 b58_dec.py c1 OUT.json      # cause: B38's 10 decoder-check cells (1,007 codewords), PKG vs OWN vs numba port,
                                    # non-finite posteriors, inputs at the clip, floored-probability mechanism test
python3 b58_dec.py c2 OUT.json      # reference: exhaustive block ML on short IRA codes (K = 16), channels A and B
python3 b58_dec.py port OUT.json    # numba port vs b38_check._sp_decode on 2,000 synthetic words

PKG = ldpc.BpDecoder (product_sum, 60 iterations, received_vector, per-bit channel probabilities), B38's registered
decoder; OWN = b38_check._sp_decode; NB = b58_bp.NbDecoder (numba port of OWN). Channel inputs exactly as B38
(b38_eq.ldpc_decode_stream / b38_check.decoders): full = copy votes for bit 1 x log((1 - e) / e), pr = clip(1 / (1 +
exp(min(|full|, 60))), 1e-15, 0.5), hard = full > 0 (ties at random), OWN's LLR of bit 0 = +-log((1 - pr) / pr).
"""
import collections
import gzip
import json
import math
import re
import sys
import time

import numpy as np

import b38_check as K
import b38_eq as E
import b58_bp as NB


def pkg_decoder(H):
    from ldpc import BpDecoder
    return BpDecoder(H, error_rate=0.1, max_iter=E.LDPC_ITERS, bp_method="product_sum",
                     input_vector_type="received_vector")


def inputs_from_full(full, rng, floor=1e-15):
    """B38's channel inputs from 'full' (LLR of bit = 1). Returns (pr, hard, llr0) with llr0 = LLR of bit = 0."""
    pr = np.clip(1.0 / (1.0 + np.exp(np.minimum(np.abs(full), 60.0))), floor, 0.5)
    hard = (full > 0).astype(np.uint8)
    ties = full == 0
    hard[ties] = rng.integers(0, 2, int(ties.sum()))
    llr0 = np.where(hard == 0, 1, -1) * np.log((1 - pr) / pr)
    return pr, hard, llr0


def pkg_decode(bpd, pr, hard):
    bpd.update_channel_probs(pr)
    d = np.asarray(bpd.decode(hard), np.uint8)
    lpr = np.asarray(bpd.log_prob_ratios, np.float64)
    return d, bool(bpd.converge), int(bpd.iter) if hasattr(bpd, "iter") else -1, lpr


# =============================================================================================== c1: B38's check cells
def c1_cells():
    """The 10 E-LDPC cells of results/b38_check_decoders_ldpc.txt (sid, B, cond, R), in file order."""
    cells = []
    for line in open("results/b38_check_decoders_ldpc.txt"):
        m = re.match(r"LDPC (\S+) B=(\S+) (\S+) R=(\S+): words", line)
        if m:
            B = min(E.GRID, key=lambda g: abs(g - float(m.group(2))))
            cells.append((m.group(1), B, m.group(3), float(m.group(4))))
    return cells


def c1(out_f):
    stores = dict(E.load_stores("b34_out/test_stores.json.gz"))
    tot = collections.Counter()
    rows = []
    t0 = time.time()
    for sid, B, cond, R in c1_cells():
        S38 = E.Store38(sid, stores[sid])
        A = E.build_arm(S38, "E-LDPC", B, R)
        lay = A["lay"]
        tag = "%s|%s" % ("E-LDPC", json.dumps(R))
        ch = E.Channel(sid, tag, B, len(A["cells"]), int(B * S38.n))
        sig = ch.apply(A["cells"], cond)
        sums = E.rep_sum(sig, lay["counts"])
        code = E.ira_code(lay["K"], lay["R"])
        c = E.parse_cond(cond)
        e_ = min(max(c["eps"], 1e-6), 0.4999)
        Le = math.log((1 - e_) / e_)
        full = np.zeros(lay["ncw"] * lay["N"])
        full[A["keep"]] = sums * Le
        full[~A["keep"]] = -60.0
        coded = np.zeros(lay["ncw"] * lay["N"], np.uint8)
        coded[A["keep"]] = A["coded"]
        bpd = pkg_decoder(code["H"])
        nb = NB.NbDecoder(code["H"], E.LDPC_ITERS)
        cell = collections.Counter()
        for floor in (1e-15, 1e-6):
            rng = np.random.default_rng(1)                       # b38_check.decoders' tie seed
            pr, hard, llr0 = inputs_from_full(full, rng, floor)
            for w in range(lay["ncw"]):
                sl = slice(w * lay["N"], (w + 1) * lay["N"])
                dp, cp, _, lpr = pkg_decode(bpd, pr[sl], hard[sl])
                do, co = K._sp_decode(code["H"], llr0[sl], E.LDPC_ITERS)
                tag_ = "" if floor == 1e-15 else "fl6_"
                tru = coded[sl]
                kk = lay["K"]
                agree = np.array_equal(dp[:kk], do[:kk])
                p_ok = np.array_equal(dp[:kk], tru[:kk])
                o_ok = np.array_equal(do[:kk], tru[:kk])
                nonfin = bool(~np.isfinite(lpr).all())
                cell[tag_ + "words"] += 1
                cell[tag_ + "agree"] += agree
                cell[tag_ + "pkg_ok"] += p_ok
                cell[tag_ + "own_ok"] += o_ok
                cell[tag_ + "pkg_conv"] += cp
                cell[tag_ + "own_conv"] += co
                cell[tag_ + "pkg_nonfinite"] += nonfin
                cell[tag_ + "own_ok_pkg_fail"] += o_ok and not p_ok
                cell[tag_ + "own_ok_pkg_fail_nonfinite"] += o_ok and not p_ok and nonfin
                cell[tag_ + "pkg_ok_own_fail"] += p_ok and not o_ok
                cell[tag_ + "disagree_nonfinite"] += (not agree) and nonfin
                cell[tag_ + "agree_nonfinite"] += agree and nonfin
                if floor == 1e-15:
                    lin = llr0[sl]
                    wrong = (hard[sl] != tru)
                    cell["inputs_at_clip"] += int((np.abs(lin) > 34.0).sum())
                    cell["inputs_conf_wrong"] += int((wrong & (np.abs(lin) >= 20)).sum())
                    cell["words_with_conf_wrong"] += bool((wrong & (np.abs(lin) >= 20)).any())
                    cell["disagree_with_conf_wrong"] += (not agree) and bool((wrong & (np.abs(lin) >= 20)).any())
                    dn, cn, _ = nb.decode_many(lin[None, :])
                    cell["nb_eq_own"] += np.array_equal(dn[0], do)
        rows.append(dict(sid=sid, B=B, cond=cond, R=R, **cell))
        tot.update(cell)
        print("%s B=%.0f %s R=%.3f %s  %.0fs" % (sid, B, cond, R, dict(cell), time.time() - t0), flush=True)
    json.dump(dict(cells=rows, totals=dict(tot)), open(out_f, "w"), indent=1)
    print("totals", dict(tot))


# =============================================================================================== c2: exhaustive ML
def all_codewords(code):
    K_ = code["K"]
    U = ((np.arange(2 ** K_)[:, None] >> np.arange(K_ - 1, -1, -1)[None, :]) & 1).astype(np.uint8)
    return E.ira_encode(code, U)


def c2(out_f, frames=400):
    rows = []
    t0 = time.time()
    for R in (0.25, 1 / 3, 0.5, 2 / 3):
        code = E.ira_code(16, R)
        C = all_codewords(code)                                  # (65536, N)
        Cpm = (1.0 - 2.0 * C).astype(np.float32)                 # bit 0 -> +1
        N = code["N"]
        bpd = pkg_decoder(code["H"])
        nb = NB.NbDecoder(code["H"], E.LDPC_ITERS)
        for chan, epss in (("A", (0.0208065, 0.12263475, 0.27600645)), ("B", (0.0208065, 0.05, 0.12263475))):
            for eps in epss:
                for rho in (4, 16, 30):
                    rng = np.random.default_rng(E.seed_of("b58c2", R, chan, eps, rho))
                    Le = math.log((1 - eps) / eps)
                    msg = rng.integers(0, 2 ** 16, frames)
                    cw = C[msg]                                  # (frames, N)
                    sgn = 1 - 2 * cw.astype(np.int64)            # +1 = bit 0
                    if chan == "A":
                        flips = (rng.random((frames, N, rho)) < eps).sum(2)
                        v = sgn * (rho - 2 * flips)
                    else:
                        stuck = rng.random((frames, N)) < 2 * eps
                        ss = rng.integers(0, 2, (frames, N)) * 2 - 1
                        v = np.where(stuck, rho * ss, rho * sgn)
                    full = -v * Le                               # LLR of bit 1
                    st = collections.Counter()
                    LLR0 = np.zeros((frames, N))
                    HARD = np.zeros((frames, N), np.uint8)
                    PR = np.zeros((frames, N))
                    for f in range(frames):
                        PR[f], HARD[f], LLR0[f] = inputs_from_full(full[f], rng)
                    score = LLR0.astype(np.float32) @ Cpm.T      # (frames, 65536): sum llr0 * (1 - 2c)
                    ml = score.argmax(1)
                    nbo, nbc, _ = nb.decode_many(LLR0)
                    for f in range(frames):
                        dp, cp, _, lpr = pkg_decode(bpd, PR[f], HARD[f])
                        do, co = K._sp_decode(code["H"], LLR0[f], E.LDPC_ITERS)
                        sent = cw[f]
                        mlw = C[ml[f]]
                        st["frames"] += 1
                        st["ml_ok"] += np.array_equal(mlw, sent)
                        st["pkg_ok"] += np.array_equal(dp, sent)
                        st["own_ok"] += np.array_equal(do, sent)
                        st["nb_ok"] += np.array_equal(nbo[f], sent)
                        st["pkg_eq_ml"] += np.array_equal(dp, mlw)
                        st["own_eq_ml"] += np.array_equal(do, mlw)
                        st["pkg_eq_own"] += np.array_equal(dp, do)
                        st["nb_eq_own"] += np.array_equal(nbo[f], do)
                        st["ml_ok_pkg_fail"] += np.array_equal(mlw, sent) and not np.array_equal(dp, sent)
                        st["ml_ok_own_fail"] += np.array_equal(mlw, sent) and not np.array_equal(do, sent)
                        st["pkg_nonfinite"] += bool(~np.isfinite(lpr).all())
                        st["pkg_conv"] += cp
                        st["own_conv"] += co
                    rows.append(dict(R=R, N=N, chan=chan, eps=eps, rho=rho, **st))
                    print("R=%.3f N=%d %s eps=%.4f rho=%2d  ML %3d  PKG %3d  OWN %3d  NB %3d  /%d  PKG nonfinite %d  %.0fs" % (
                        R, N, chan, eps, rho, st["ml_ok"], st["pkg_ok"], st["own_ok"], st["nb_ok"], frames,
                        st["pkg_nonfinite"], time.time() - t0), flush=True)
    json.dump(rows, open(out_f, "w"), indent=1)


# =============================================================================================== port check
def port(out_f, words=2000):
    rng = np.random.default_rng(58)
    tot = collections.Counter()
    per = []
    for R in E.RATES:
        code = E.ira_code(1024, R)
        nb = NB.NbDecoder(code["H"], E.LDPC_ITERS)
        n = words // len(E.RATES)
        for chan in ("A", "B"):
            eps = float(rng.choice([0.0208065, 0.05, 0.12263475, 0.27600645]))
            rho = int(rng.choice([1, 2, 4, 8, 16]))
            U = rng.integers(0, 2, (n // 2, code["K"])).astype(np.uint8)
            cw = E.ira_encode(code, U)
            sgn = 1 - 2 * cw.astype(np.int64)
            if chan == "A":
                v = sgn * (rho - 2 * (rng.random(cw.shape + (rho,)) < eps).sum(2))
            else:
                v = np.where(rng.random(cw.shape) < 2 * eps, rho * (rng.integers(0, 2, cw.shape) * 2 - 1), rho * sgn)
            Le = math.log((1 - eps) / eps)
            LLR0 = np.zeros(cw.shape)
            for f in range(cw.shape[0]):
                _, _, LLR0[f] = inputs_from_full(-v[f] * Le, rng)
            nbo, nbc, _ = nb.decode_many(LLR0)
            bto, btc, _ = NB.sp_decode_batch(code["H"], LLR0, E.LDPC_ITERS)
            c = collections.Counter()
            for f in range(cw.shape[0]):
                do, co = K._sp_decode(code["H"], LLR0[f], E.LDPC_ITERS)
                c["words"] += 1
                c["equal"] += np.array_equal(do, nbo[f])
                c["conv_equal"] += int(co == bool(nbc[f]))
                c["batch_equal"] += int(np.array_equal(do, bto[f]))
                c["batch_conv_equal"] += int(co == bool(btc[f]))
            per.append(dict(R=R, chan=chan, eps=eps, rho=rho, **c))
            tot.update(c)
            print(per[-1], flush=True)
    json.dump(dict(rows=per, totals=dict(tot)), open(out_f, "w"), indent=1)
    print("totals", dict(tot))


if __name__ == "__main__":
    if sys.argv[1] not in ("emu", "emuclip"):
        {"c1": c1, "c2": c2, "port": port}[sys.argv[1]](sys.argv[2])


# =============================================================================================== emu: PKG's arithmetic
# Disclosed addition (after c1 showed that flooring the channel input does not restore agreement): an emulation of the
# package's product-sum update as we understand ldpc v2 (received-vector input decoded as an error pattern against
# the received word's syndrome; per check, the product of tanh(m / 2) over the other edges by forward / backward
# products; message = sign(syndrome) log((1 + p) / (1 - p)) with no clipping; bit-to-check = prior + the other edges'
# messages by forward / backward sums; decision 'error' when the posterior <= 0). IEEE semantics: p = +-1 gives +-inf,
# inf - inf and inf x 0 give NaN, and NaN <= 0 is false. If this emulation reproduces PKG's outputs, the discrepancy
# is the unclipped arithmetic; OWN clips (|tanh| floor 1e-15, |p| <= 1 - 1e-12, inputs +-40).
from numba import njit as _njit


@_njit(cache=True, error_model="numpy")
def _emu(rp, rc, cp, ce, M, N, prior, syn, iters, dec, clip):
    E_ = rc.shape[0]
    b2c = np.empty(E_)
    c2b = np.empty(E_)
    for j in range(N):
        for q in range(cp[j], cp[j + 1]):
            b2c[ce[q]] = prior[j]
    lpr = np.empty(N)
    for it in range(iters):
        for i in range(M):
            tmp = 1.0
            for e in range(rp[i], rp[i + 1]):
                c2b[e] = tmp
                x = b2c[e]
                if clip:
                    x = min(max(x, -40.0), 40.0)
                tmp *= math.tanh(x / 2.0)
            tmp = 1.0
            sgn = -1.0 if syn[i] else 1.0
            for e in range(rp[i + 1] - 1, rp[i] - 1, -1):
                c2b[e] *= tmp
                if clip:
                    c2b[e] = min(max(c2b[e], -(1.0 - 1e-12)), 1.0 - 1e-12)
                c2b[e] = sgn * math.log((1.0 + c2b[e]) / (1.0 - c2b[e]))
                x = b2c[e]
                if clip:
                    x = min(max(x, -40.0), 40.0)
                tmp *= math.tanh(x / 2.0)
        for j in range(N):
            tmp = prior[j]
            for q in range(cp[j], cp[j + 1]):
                b2c[ce[q]] = tmp
                tmp += c2b[ce[q]]
            lpr[j] = tmp
            dec[j] = 1 if tmp <= 0 else 0
        for j in range(N):
            tmp = 0.0
            for q in range(cp[j + 1] - 1, cp[j] - 1, -1):
                b2c[ce[q]] += tmp
                tmp += c2b[ce[q]]
        ok = True
        for i in range(M):
            s = 0
            for e in range(rp[i], rp[i + 1]):
                s += dec[rc[e]]
            if (s % 2) != syn[i]:
                ok = False
                break
        if ok:
            return True, lpr
    return False, lpr


class Emu:
    def __init__(self, H):
        Hr = H.tocsr()
        self.rp, self.rc = Hr.indptr.astype(np.int64), Hr.indices.astype(np.int64)
        E_ = len(self.rc)
        eid = np.arange(E_)
        Hcsc_order = np.lexsort((np.repeat(np.arange(H.shape[0]), np.diff(self.rp)), self.rc))
        self.ce = eid[Hcsc_order].astype(np.int64)               # edges grouped by column, rows ascending
        self.cp = np.r_[0, np.cumsum(np.bincount(self.rc, minlength=H.shape[1]))].astype(np.int64)
        self.H = H
        self.M, self.N = H.shape

    def decode(self, pr, hard, clip=False):
        syn = (self.H @ hard.astype(np.int64) % 2).astype(np.int64)
        prior = np.log((1 - pr) / pr)
        dec = np.zeros(self.N, np.int64)
        if not hard.any():
            return np.zeros(self.N, np.uint8), True, prior
        ok, lpr = _emu(self.rp, self.rc, self.cp, self.ce, self.M, self.N, prior, syn, E.LDPC_ITERS, dec, clip)
        return (hard ^ dec.astype(np.uint8)), bool(ok), lpr


def emu(out_f):
    """PKG vs its emulation vs OWN on c1's 1,007 codewords and on c2's channel-B frames (K = 16)."""
    stores = dict(E.load_stores("b34_out/test_stores.json.gz"))
    tot = collections.Counter()
    for sid, B, cond, R in c1_cells():
        S38 = E.Store38(sid, stores[sid])
        A = E.build_arm(S38, "E-LDPC", B, R)
        lay = A["lay"]
        tag = "%s|%s" % ("E-LDPC", json.dumps(R))
        ch = E.Channel(sid, tag, B, len(A["cells"]), int(B * S38.n))
        sums = E.rep_sum(ch.apply(A["cells"], cond), lay["counts"])
        code = E.ira_code(lay["K"], lay["R"])
        e_ = min(max(E.parse_cond(cond)["eps"], 1e-6), 0.4999)
        full = np.zeros(lay["ncw"] * lay["N"])
        full[A["keep"]] = sums * math.log((1 - e_) / e_)
        full[~A["keep"]] = -60.0
        pr, hard, llr0 = inputs_from_full(full, np.random.default_rng(1))
        bpd = pkg_decoder(code["H"])
        em = Emu(code["H"])
        for w in range(lay["ncw"]):
            sl = slice(w * lay["N"], (w + 1) * lay["N"])
            dp, cp_, _, lpr = pkg_decode(bpd, pr[sl], hard[sl])
            de, ce_, lpe = em.decode(pr[sl], hard[sl])
            tot["c1_words"] += 1
            tot["c1_emu_eq_pkg"] += np.array_equal(dp, de)
            tot["c1_emu_conv_eq_pkg"] += (cp_ == ce_)
            tot["c1_emu_nonfinite"] += bool(~np.isfinite(lpe).all())
            tot["c1_pkg_nonfinite"] += bool(~np.isfinite(lpr).all())
        print(sid, B, cond, dict(tot), flush=True)
    for R in (0.25, 0.5):
        code = E.ira_code(16, R)
        bpd = pkg_decoder(code["H"])
        em = Emu(code["H"])
        N = code["N"]
        rng = np.random.default_rng(E.seed_of("b58emu", R))
        for eps in (0.05, 0.12263475):
            for rho in (4, 16, 30):
                Le = math.log((1 - eps) / eps)
                U = rng.integers(0, 2, (300, 16)).astype(np.uint8)
                cw = E.ira_encode(code, U)
                sgn = 1 - 2 * cw.astype(np.int64)
                v = np.where(rng.random(cw.shape) < 2 * eps, rho * (rng.integers(0, 2, cw.shape) * 2 - 1), rho * sgn)
                for f in range(cw.shape[0]):
                    pr, hard, _ = inputs_from_full(-v[f] * Le, rng)
                    dp, _, _, _ = pkg_decode(bpd, pr, hard)
                    de, _, _ = em.decode(pr, hard)
                    tot["c2B_frames"] += 1
                    tot["c2B_emu_eq_pkg"] += np.array_equal(dp, de)
        print("c2B", R, dict(tot), flush=True)
    json.dump(dict(tot), open(out_f, "w"), indent=1)


def emuclip(out_f):
    """Mechanism test: the emulation with OWN's guards (inputs to tanh within +-40, |product| <= 1 - 1e-12) against
    OWN and PKG on c1's 1,007 codewords (agreement and decoding success)."""
    stores = dict(E.load_stores("b34_out/test_stores.json.gz"))
    tot = collections.Counter()
    for sid, B, cond, R in c1_cells():
        S38 = E.Store38(sid, stores[sid])
        A = E.build_arm(S38, "E-LDPC", B, R)
        lay = A["lay"]
        tag = "%s|%s" % ("E-LDPC", json.dumps(R))
        ch = E.Channel(sid, tag, B, len(A["cells"]), int(B * S38.n))
        sums = E.rep_sum(ch.apply(A["cells"], cond), lay["counts"])
        code = E.ira_code(lay["K"], lay["R"])
        e_ = min(max(E.parse_cond(cond)["eps"], 1e-6), 0.4999)
        full = np.zeros(lay["ncw"] * lay["N"])
        full[A["keep"]] = sums * math.log((1 - e_) / e_)
        full[~A["keep"]] = -60.0
        coded = np.zeros(lay["ncw"] * lay["N"], np.uint8)
        coded[A["keep"]] = A["coded"]
        pr, hard, llr0 = inputs_from_full(full, np.random.default_rng(1))
        bpd = pkg_decoder(code["H"])
        em = Emu(code["H"])
        ob, _, _ = NB.sp_decode_batch(code["H"], llr0.reshape(lay["ncw"], lay["N"]), E.LDPC_ITERS)
        kk = lay["K"]
        for w in range(lay["ncw"]):
            sl = slice(w * lay["N"], (w + 1) * lay["N"])
            dp, _, _, _ = pkg_decode(bpd, pr[sl], hard[sl])
            dc, _, lpc = em.decode(pr[sl], hard[sl], clip=True)
            tru = coded[sl][:kk]
            tot["words"] += 1
            tot["clip_eq_own"] += np.array_equal(dc[:kk], ob[w, :kk])
            tot["clip_eq_pkg"] += np.array_equal(dc[:kk], dp[:kk])
            tot["clip_ok"] += np.array_equal(dc[:kk], tru)
            tot["own_ok"] += np.array_equal(ob[w, :kk], tru)
            tot["pkg_ok"] += np.array_equal(dp[:kk], tru)
            tot["clip_nonfinite"] += bool(~np.isfinite(lpc).all())
        print(sid, B, cond, dict(tot), flush=True)
    json.dump(dict(tot), open(out_f, "w"), indent=1)


if __name__ == "__main__" and sys.argv[1] == "emu":
    emu(sys.argv[2])
if __name__ == "__main__" and sys.argv[1] == "emuclip":
    emuclip(sys.argv[2])
