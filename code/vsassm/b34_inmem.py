"""B34 (prereg/B34_inmemory_directory_prereg.md): the RULER VSA store against tables whose key directory lives on the
same noisy devices (fully in-memory), at an equal device count, on B25's phase-change-memory model (Joshi et al. 2020).

python3 b34_inmem.py --selftest                       # synthetic pairs only: code paths, identities, exactness at k = 0
python3 b34_inmem.py dev OUT.json [--workers N]       # synthetic dev stores only (split grid + full grid for predictions)
python3 b34_inmem.py test B21_DIR OUT.json CHOICE.json [--workers N]   # the 50 B22/B25 stores (registered run)

Every stored bit of every arm is one cell = one differential pair of PCM devices (+1 -> (G_ON, RESET), -1 -> (RESET,
G_ON)) read as a sign (B25/B26b's cell model). Query-side computation (hashing a key string, binding, majority gates,
comparisons, arithmetic) is stateless logic and is allowed in every arm; the value vocabulary (index -> value string)
is the shared output dictionary of every arm and is not counted (every arm returns a vocabulary index).

Arms (one row per stored (key, j) pair for the tables; budget C = the VSA store's cell count S x 8192 unless noted):
  V-S     B26b's VSA: bipolar store on the devices, digital clean-up, threshold 0.5 at every j (held as registered).
  V-Snt   the same reads, but the first value is accepted by argmax without the threshold (the tables' assumption that
          a probed key is present); the stop at j >= 1 keeps the threshold.
  V-SC    store and clean-up codebook (N_values x 8192) on the devices, GDC on the clean-up crossbar (B25's B1-SC).
  V-SCnt  V-SC with V-Snt's first-value rule.
  T-iv    B26b's Trep reproduced (digital key -> row directory and value counts; values repetition-coded).
  T-i     CAM, exact match over the native key code (UTF-8 bytes of the key padded to the store's longest key, then j in
          binary); values repetition-coded with the remaining cells; stop when (key, j) finds no match.
  T-iih   CAM, native code and value bits all repeated r times, majority per bit, exact match of the decoded word.
  T-iis   CAM, native code repeated r_d times, nearest Hamming distance over all rows (soft); last flag and value bits
          repeated r_p times, majority. Share of cells for the directory: dev choice.
  T-iix   as T-iis with a short hash code of (key, j) (h bits, the smallest h >= 2 log2 n that is collision-free).
  T-v     minimal perfect hash (hash-and-displace): a displacement array on the devices maps (key, j) to its row; rows
          hold the last flag and the value bits; every logical bit repeated r times, majority.
  T-iiia  CAM with random high-dimensional key codes (HD(key) * HD(pos j), w_k bits, nearest Hamming); last flag and
          value bits repetition-coded. Hyperdimensional (boundary case).
  T-iiib  T-iiia's key codes plus random high-dimensional value codes (HD(value), times HD(last) on a key's last
          value), decoded by nearest code over the store's vocabulary (digital, like V-S). Hyperdimensional.
  T-vhd   T-v's stored hash mapping (displacements repetition-coded, share s of the cells) with T-iiib's random
          high-dimensional value codes in the rows (no stored key, no superposition). Hyperdimensional values.
  T-vhdd  T-vhd with the displacements also high-dimensional: each displacement split into two blocks of
          ceil(db / 2) bits, each block stored as a random code of w_b cells and decoded by the nearest of its
          2^bits candidate codes. Every field random high-dimensional, no superposition (the boundary case).
Equal-total budget (for V-SC): the non-hyperdimensional in-memory tables again at C' = (S + N_values) x 8192 cells,
with cell errors drawn i.i.d. at the device model's measured bit-error rate (the device cells are i.i.d.).
"""
import collections
import hashlib
import json
import math
import sys
import time

import numpy as np

import b22_robust as B
import b25_pcm as P
import b26b_pcm_table as T26
import ruler_vsa as V

D = P.D
LOAD = 16
KS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
TIMES = P.TIMES
TABLES = ("T-iv", "T-i", "T-iih", "T-iis", "T-iix", "T-v", "T-iiia", "T-iiib", "T-vhd", "T-vhdd")
NONHD = ("T-i", "T-iih", "T-iis", "T-iix", "T-v")
HD = ("T-iiia", "T-iiib", "T-vhd", "T-vhdd")
VSAS = ("V-S", "V-Snt", "V-SC", "V-SCnt")
SPLIT_ARMS = ("T-iis", "T-iix", "T-iiia", "T-iiib", "T-vhd", "T-vhdd")
SHARES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MPHF_LAMBDAS = (3, 4, 5, 6, 8)


# ---------------------------------------------------------------- the cell model (B25/B26b), several times per programming
def signs_over_times(bits, seed, k, times):
    """Identical, time by time, to b26b_pcm_table.cells_read(bits, seed, t, k) (same seeds, same draw order), but the
    programming step is drawn once and every time point read from it. bits: 2-D +-1 array. Returns {label: int8 signs}."""
    r = np.random.default_rng([seed, 3])
    gp = np.where(bits > 0, P.G_ON, 0.0).astype(np.float32)
    gm = np.where(bits < 0, P.G_ON, 0.0).astype(np.float32)
    nu_p, g1p = P.program(gp, gp.shape, r, k)
    nu_m, g1m = P.program(gm, gm.shape, r, k)
    out = {}
    for tl, t in times:
        rr = np.random.default_rng([seed, 4, int(t)])
        Gp, _, _ = P.at_time(gp, nu_p, g1p, t, rr, k)
        Gm, _, _ = P.at_time(gm, nu_m, g1m, t, rr, k)
        q = P.adc(Gp - Gm, P.G_ON, P.ADC_BITS)
        sgn = np.sign(q)
        ties = sgn == 0
        sgn[ties] = np.random.default_rng([seed, 5, int(t)]).choice([-1.0, 1.0], size=int(ties.sum()))
        out[tl] = sgn.astype(np.int8)
    return out


def signs_bernoulli(bits, seed, ber):
    """i.i.d. sign errors at the measured bit-error rates: one uniform per cell, shared by every (k, t) (monotone
    coupling, the analogue of common random numbers). ber: {(k, tl): p}. Returns {(k, tl): int8 signs}."""
    u = np.random.default_rng([seed, 11]).random(bits.shape, dtype=np.float32)
    b8 = bits.astype(np.int8)
    return {kt: np.where(u < p, -b8, b8) for kt, p in ber.items()}


def measured_ber(n_rows=1000, n_cols=20000, seed=34034):
    """The device model's cell bit-error rate per (k, time), on 2e7 random cells (no store)."""
    bits = np.where(np.random.default_rng(seed).random((n_rows, n_cols)) < 0.5, 1.0, -1.0).astype(np.float32)
    out = {}
    for k in KS:
        s = signs_over_times(bits, seed, k, TIMES)
        for tl, _ in TIMES:
            out[(k, tl)] = float((s[tl] != bits).mean())
    return out


# ---------------------------------------------------------------- codes
def hd_code(s, w):
    return np.where(np.random.default_rng(V._seed("b34hd:" + s)).random(w) < 0.5, 1, -1).astype(np.int8)


def bits_pm(x, nbits):
    """Unsigned integers -> (len, nbits) +-1 int8, MSB first (+1 = bit 1)."""
    x = np.asarray(x, np.int64)
    if nbits == 0:
        return np.zeros((len(x), 0), np.int8)
    b = (x[:, None] >> np.arange(nbits - 1, -1, -1)[None, :]) & 1
    return (2 * b - 1).astype(np.int8)


def pm_to_int(pm):
    if pm.shape[1] == 0:
        return np.zeros(len(pm), np.int64)
    return ((pm > 0).astype(np.int64) * (1 << np.arange(pm.shape[1] - 1, -1, -1))[None, :]).sum(1)


def native_code(keys, js, jb):
    """UTF-8 bytes of the key, zero-padded to the longest key, then j in jb bits; +-1 int8."""
    enc = [k.encode("utf-8") for k in keys]
    L = max(len(e) for e in enc)
    arr = np.zeros((len(enc), L), np.uint8)
    for i, e in enumerate(enc):
        arr[i, :len(e)] = np.frombuffer(e, np.uint8)
    kb = (2 * np.unpackbits(arr, axis=1).astype(np.int8) - 1)
    return np.concatenate([kb, bits_pm(js, jb)], 1), L


def hash_code(keys, js):
    """h-bit hash of (key, j); h = smallest h >= max(8, ceil(2 log2 n)) with no two pairs sharing a code."""
    dig = np.stack([np.frombuffer(hashlib.sha256(("b34h:%s\x00%d" % (k, j)).encode("utf-8")).digest(), np.uint8)
                    for k, j in zip(keys, js)])
    allb = np.unpackbits(dig, axis=1)
    h = max(8, math.ceil(2 * math.log2(max(len(keys), 2))))
    while len({allb[i, :h].tobytes() for i in range(len(keys))}) < len(keys):
        h += 1
    return (2 * allb[:, :h].astype(np.int8) - 1), h


def majority(signs, nbits, r, rng):
    """signs: (rows, nbits * r), copy c of bit i at column i * r + c. Majority per bit, ties at random. -> +-1 int8."""
    s = signs.reshape(len(signs), nbits, r).sum(2, dtype=np.int32)
    ties = s == 0
    if ties.any():
        s[ties] = rng.choice([-1, 1], size=int(ties.sum()))
    return np.sign(s).astype(np.int8)


# ---------------------------------------------------------------- MPHF (hash and displace; Belazzougui, Botelho & Dietzfelbinger 2009)
def _is_prime(n):
    if n < 2:
        return False
    for p in range(2, int(n ** 0.5) + 1):
        if n % p == 0:
            return False
    return True


def mphf_hash(k, j, R, Bk):
    """(bucket, first slot, step) of the identity (key, j): stateless, from SHA-256."""
    h = hashlib.sha256(("b34mphf:%s\x00%d" % (k, j)).encode("utf-8")).digest()
    return (int.from_bytes(h[0:8], "little") % Bk, int.from_bytes(h[8:16], "little") % R,
            1 + int.from_bytes(h[16:24], "little") % (R - 1) if R > 1 else 0)


def _mp_hash(s, mp, k, j):
    """(h1, h2, bucket) of (key, j) in a built MPHF (stored identity or not)."""
    i = s.row_of.get((k, j))
    if i is not None:
        return int(mp["h1"][i]), int(mp["h2"][i]), int(mp["h0"][i])
    g, h1, h2 = mphf_hash(k, j, mp["R"], mp["B"])
    return h1, h2, g


def mphf_slot(h1, h2, sigma, R):
    return (h1 + (sigma % R) * h2 + sigma // R) % R


def mphf_build(keys, js, lam, R=None):
    """Rows R = the smallest prime >= n (load ~1); if some bucket finds no free displacement among the R^2 pairs
    (tiny stores), R moves to the next prime (empty rows are cells too)."""
    n = len(keys)
    if R is None:
        R = n
    while not _is_prime(R):
        R += 1
    Bk = max(1, math.ceil(n / lam))
    hh = np.array([mphf_hash(k, j, R, Bk) for k, j in zip(keys, js)], np.int64).reshape(-1, 3)
    h0, h1, h2 = hh[:, 0], hh[:, 1], hh[:, 2]
    members = collections.defaultdict(list)
    for i, g in enumerate(h0):
        members[int(g)].append(i)
    order = sorted(members, key=lambda g: (-len(members[g]), g))
    occ = np.zeros(R, bool)
    sigma = np.zeros(Bk, np.int64)
    slot = np.zeros(n, np.int64)
    for g in order:
        idx = np.array(members[g])
        found = None
        for c0 in range(0, R * R, 4096):                     # sigma -> (d0, d1) = (sigma mod R, sigma div R), CHD's
            sg = np.arange(c0, c0 + 4096)                     # displacement pair: slot = (h1 + d0 h2 + d1) mod R
            sl = (h1[idx][None, :] + (sg % R)[:, None] * h2[idx][None, :] + (sg // R)[:, None]) % R
            ok = (~occ[sl]).all(1)
            if len(idx) > 1:
                ss = np.sort(sl, 1)
                ok &= (np.diff(ss, axis=1) != 0).all(1)
            hit = np.flatnonzero(ok)
            if len(hit):
                found = int(sg[hit[0]])
                break
        if found is None:
            return mphf_build(keys, js, lam, R + 1)
        sigma[g] = found
        s = mphf_slot(h1[idx], h2[idx], found, R)
        occ[s] = True
        slot[idx] = s
    db = max(1, int(math.ceil(math.log2(int(sigma.max()) + 1)))) if sigma.max() > 0 else 1
    return dict(R=R, B=Bk, h0=h0, h1=h1, h2=h2, sigma=sigma, slot=slot, db=db)


# ---------------------------------------------------------------- one store: layouts
class Store:
    def __init__(self, pairs, seed_name):
        self.pairs = pairs
        self.cd = P.codes(pairs)
        self.st = P.build(pairs, LOAD, self.cd)
        self.seed = V._seed(seed_name) % (2 ** 62)
        self.vi = {v: i for i, v in enumerate(self.cd["vocab"])}
        self.n = len(pairs)
        self.Nv = len(self.cd["vocab"])
        self.b = max(1, math.ceil(math.log2(self.Nv))) if self.Nv > 1 else 1
        self.maxj = max(j for _, _, j in pairs)               # largest stored j (the VSA's loop bound is maxj + 1)
        self.jb = math.ceil(math.log2(self.maxj + 1)) if self.maxj > 0 else 0
        self.lastbit = 1 if self.maxj > 0 else 0
        tab = V.table_of(pairs)
        self.count = {k: len(v) for k, v in tab.items()}
        self.rk = [k for k, _, _ in pairs]
        self.rj = np.array([j for _, _, j in pairs], np.int64)
        self.rv = np.array([self.vi[v] for _, v, _ in pairs], np.int64)
        self.rlast = np.array([1 if j == self.count[k] - 1 else 0 for k, _, j in pairs], np.int64)
        self.row_of = {(k, j): i for i, (k, _, j) in enumerate(pairs)}
        self.keys = self.st["keys"]                           # probed keys (B25's)
        self.want = self.st["want"]
        self.C = self.st["S"] * D                             # the VSA store's cell count

    def mphf(self, lam):
        if not hasattr(self, "_mphf"):
            self._mphf = {}
        if lam not in self._mphf:
            self._mphf[lam] = mphf_build(self.rk, self.rj, lam)
        return self._mphf[lam]

    def payload_bits(self):
        """last flag (if any key has > 1 value) + value index, per row: (n, lastbit + b) +-1."""
        parts = []
        if self.lastbit:
            parts.append(bits_pm(self.rlast, 1))
        parts.append(bits_pm(self.rv, self.b))
        return np.concatenate(parts, 1)

    def decode_payload(self, pm):
        """(rows, lastbit + b) +-1 -> (last, value index)."""
        last = (pm[:, 0] > 0).astype(np.int64) if self.lastbit else np.ones(len(pm), np.int64)
        return last, pm_to_int(pm[:, self.lastbit:])


def layout(store, arm, C, share=None, lam=None):
    """The arm's on-device content at a budget of C cells: a list of fields (name, logical +-1 bits, repetition r, mode)
    grouped into arrays that are simulated together. Returns dict with 'arrays': {name: (cells +-1 int8, [(field, nbits,
    r, offset)])}, cell count and parameters; None if infeasible."""
    s = store
    n = s.n
    W = C // n                                                # cells per row
    pay = s.payload_bits()
    pb = pay.shape[1]
    info = {"arm": arm, "W": int(W)}
    if arm == "T-i":
        kc, L = native_code(s.rk, s.rj, s.jb)
        wk = kc.shape[1]
        vb = bits_pm(s.rv, s.b)
        rv = (W - wk) // s.b
        if rv < 1:
            return None
        fields = [("key", kc, 1, "exact"), ("val", vb, rv, "hard")]
        info.update(wk=wk, rv=int(rv), Lmax=L)
    elif arm == "T-iih":
        kc, L = native_code(s.rk, s.rj, s.jb)
        vb = bits_pm(s.rv, s.b)
        r = W // (kc.shape[1] + s.b)
        if r < 1:
            return None
        fields = [("key", kc, r, "hardmatch"), ("val", vb, r, "hard")]
        info.update(wk=kc.shape[1], r=int(r), Lmax=L)
    elif arm in ("T-iis", "T-iix"):
        if arm == "T-iis":
            kc, L = native_code(s.rk, s.rj, s.jb)
            info.update(Lmax=L)
        else:
            kc, h = hash_code(s.rk, s.rj)
            info.update(h=h)
        wk = kc.shape[1]
        rd = max(1, int(round(share * W / wk)))
        while rd > 1 and W - rd * wk < pb:
            rd -= 1
        rp = (W - rd * wk) // pb
        if rp < 1:
            return None
        fields = [("key", kc, rd, "soft"), ("pay", pay, rp, "hard")]
        info.update(wk=wk, rd=int(rd), rp=int(rp))
    elif arm == "T-iiia":
        wk = max(1, min(W - pb, int(round(share * W))))
        kc = np.stack([hd_code("key:" + k, wk) * hd_code("pos:%d" % j, wk) for k, j in zip(s.rk, s.rj)])
        rp = (W - wk) // pb
        if rp < 1:
            return None
        fields = [("key", kc, 1, "soft"), ("pay", pay, rp, "hard")]
        info.update(wk=wk, rp=int(rp))
    elif arm == "T-iiib":
        wk = max(1, min(W - 1, int(round(share * W))))
        wv = W - wk
        kc = np.stack([hd_code("key:" + k, wk) * hd_code("pos:%d" % j, wk) for k, j in zip(s.rk, s.rj)])
        lastc = hd_code("last", wv)
        vc = np.stack([hd_code("value:" + s.cd["vocab"][v], wv) * (lastc if (s.lastbit and l) else 1)
                       for v, l in zip(s.rv, s.rlast)]).astype(np.int8)
        fields = [("key", kc, 1, "soft"), ("valhd", vc, 1, "soft")]
        info.update(wk=wk, wv=wv)
    elif arm == "T-v":
        mp = s.mphf(lam)
        disp = bits_pm(mp["sigma"], mp["db"])
        rowpay = np.full((mp["R"], pb), -1, np.int8)
        rowpay[mp["slot"]] = pay
        logical = disp.size + rowpay.size
        r = C // logical
        if r < 1:
            return None
        info.update(R=mp["R"], B=mp["B"], db=mp["db"], lam=lam, r=int(r), mphf=mp)
        arrays = {"disp": (np.repeat(disp, r, axis=1), [("disp", mp["db"], r, 0)]),
                  "rows": (np.repeat(rowpay, r, axis=1), [("pay", pb, r, 0)])}
        info["cells"] = int(sum(a[0].size for a in arrays.values()))
        info["arrays"] = arrays
        info["fields_logical"] = {"disp": disp, "pay": rowpay}
        return info
    elif arm == "T-vhd":
        mp = s.mphf(lam)
        disp = bits_pm(mp["sigma"], mp["db"])
        rd = max(1, int(share * C // disp.size))
        wv = (C - disp.size * rd) // mp["R"]
        if wv < 1:
            return None
        lastc = hd_code("last", wv)
        rows = np.stack([hd_code("empty:%d" % i, wv) for i in range(mp["R"])])
        for i in range(n):
            rows[mp["slot"][i]] = hd_code("value:" + s.cd["vocab"][s.rv[i]], wv) * (lastc if (s.lastbit and s.rlast[i])
                                                                                     else 1)
        info.update(R=mp["R"], B=mp["B"], db=mp["db"], lam=lam, rd=int(rd), wv=int(wv), mphf=mp)
        arrays = {"disp": (np.repeat(disp, rd, axis=1), [("disp", mp["db"], rd, 0)]),
                  "rows": (rows.astype(np.int8), [("valhd", wv, 1, 0)])}
        info["cells"] = int(sum(a[0].size for a in arrays.values()))
        info["arrays"] = arrays
        info["fields_logical"] = {"disp": disp, "valhd": rows}
        return info
    elif arm == "T-vhdd":
        mp = s.mphf(lam)
        hb = (mp["db"] + 1) // 2                              # bits per displacement block (two blocks)
        wb = max(1, int(share * C // (mp["B"] * 2)))
        wv = (C - mp["B"] * 2 * wb) // mp["R"]
        if wv < 1:
            return None
        sig_ = mp["sigma"]
        blocks = np.stack([sig_ >> hb, sig_ & ((1 << hb) - 1)], 1)       # (B, 2) block values
        dcodes = np.stack([np.concatenate([hd_code("dblk%d:%d" % (c, int(blocks[g, c])), wb) for c in range(2)])
                           for g in range(mp["B"])])
        lastc = hd_code("last", wv)
        rows = np.stack([hd_code("empty:%d" % i, wv) for i in range(mp["R"])])
        for i in range(n):
            rows[mp["slot"][i]] = hd_code("value:" + s.cd["vocab"][s.rv[i]], wv) * (lastc if (s.lastbit and s.rlast[i])
                                                                                     else 1)
        info.update(R=mp["R"], B=mp["B"], db=mp["db"], hb=hb, lam=lam, wb=int(wb), wv=int(wv), mphf=mp)
        arrays = {"disp": (dcodes.astype(np.int8), [("disphd", 2 * wb, 1, 0)]),
                  "rows": (rows.astype(np.int8), [("valhd", wv, 1, 0)])}
        info["cells"] = int(sum(a[0].size for a in arrays.values()))
        info["arrays"] = arrays
        info["fields_logical"] = {"disphd": dcodes, "valhd": rows}
        return info
    else:
        raise ValueError(arm)
    cols, meta, off = [], [], 0
    for name, bits, r, mode in fields:
        cols.append(np.repeat(bits, r, axis=1))
        meta.append((name, bits.shape[1], r, off, mode))
        off += bits.shape[1] * r
    cells = np.concatenate(cols, 1).astype(np.int8)
    info["arrays"] = {"rows": (cells, meta)}
    info["cells"] = int(cells.size)
    info["fields_logical"] = {name: bits for name, bits, _, _ in fields}
    return info


# ---------------------------------------------------------------- reading the tables
def _field(sig, meta, name):
    for m in meta:
        if m[0] == name:
            return sig[:, m[3]:m[3] + m[1] * m[2]], m[1], m[2]
    raise KeyError(name)


def read_table(store, info, sig, rng, diag=None):
    """sig: {array name: int8 signs as read}. Returns, per probed key, the list of value indices read."""
    s = store
    arm = info["arm"]
    got = [[] for _ in s.keys]
    if arm in ("T-v", "T-vhd", "T-vhdd"):
        mp = info["mphf"]
        dsig = sig["disp"]
        if arm == "T-vhdd":
            hb, wb = info["hb"], info["wb"]
            need = sorted({_mp_hash(s, mp, k, j)[2] for k in s.keys for j in range(s.maxj + 1)})
            disp = np.zeros(mp["B"], np.int64)
            dec = []
            for c in range(2):
                cand_d = np.stack([hd_code("dblk%d:%d" % (c, v), wb) for v in range(1 << hb)]).astype(np.float32)
                sims_d = dsig[need][:, c * wb:(c + 1) * wb].astype(np.float32) @ cand_d.T
                sims_d += rng.random(sims_d.shape, dtype=np.float32)
                dec.append(sims_d.argmax(1).astype(np.int64))
            disp[need] = (dec[0] << hb) | dec[1]
        else:
            disp = pm_to_int(majority(dsig, mp["db"], info["r"] if arm == "T-v" else info["rd"], rng))
        if arm == "T-v":
            pay = majority(sig["rows"], s.lastbit + s.b, info["r"], rng)
            last, val = s.decode_payload(pay)
        else:
            wv = info["wv"]
            cand = np.stack([hd_code("value:" + v, wv) for v in s.cd["vocab"]]).astype(np.float32)
            if s.lastbit:
                cand = np.concatenate([cand, cand * hd_code("last", wv)[None, :].astype(np.float32)], 0)
            nv = len(s.cd["vocab"])
            # decode only the rows a read lands on (all reads of the probed keys share one tie-break stream)
            slots = set()
            for k in s.keys:
                for j in range(s.maxj + 1):
                    h1, h2, g = _mp_hash(s, mp, k, j)
                    slots.add(mphf_slot(h1, h2, int(disp[g]), mp["R"]))
            slots = sorted(slots)
            sims = sig["rows"][slots].astype(np.float32) @ cand.T
            sims += rng.random(sims.shape, dtype=np.float32)
            cc = sims.argmax(1)
            val = np.zeros(mp["R"], np.int64)
            last = np.ones(mp["R"], np.int64)
            val[slots] = cc % nv
            if s.lastbit:
                last[slots] = (cc >= nv).astype(np.int64)
        for a, k in enumerate(s.keys):
            for j in range(s.maxj + 1):
                i = s.row_of.get((k, j))
                if i is not None:
                    g, h1, h2 = mp["h0"][i], mp["h1"][i], mp["h2"][i]
                else:                                         # asked only after a misread last flag: the hash still
                    g, h1, h2 = mphf_hash(k, j, mp["R"], mp["B"])   # maps it to some row
                slot = mphf_slot(h1, h2, int(disp[g]), mp["R"])
                got[a].append(int(val[slot]))
                if last[slot]:
                    break
        return got
    cells, meta = info["arrays"]["rows"][0], info["arrays"]["rows"][1]
    S_ = sig["rows"]
    kfield, wk, rk = _field(S_, meta, "key")
    logical_key = info["fields_logical"]["key"]
    mode = [m[4] for m in meta if m[0] == "key"][0]
    if arm == "T-i":
        vf, b, rv = _field(S_, meta, "val")
        val = pm_to_int(majority(vf, b, rv, rng))
        last = None
    elif arm == "T-iih":
        kdec = majority(kfield, wk, rk, rng)
        vf, b, rv = _field(S_, meta, "val")
        val = pm_to_int(majority(vf, b, rv, rng))
        last = None
    elif arm == "T-iiib":
        vfield, wv, _ = _field(S_, meta, "valhd")
        vocab = s.cd["vocab"]
        cand = np.stack([hd_code("value:" + v, wv) for v in vocab]).astype(np.float32)
        if s.lastbit:
            cand = np.concatenate([cand, cand * hd_code("last", wv)[None, :].astype(np.float32)], 0)
    else:
        pf, pbits, rp = _field(S_, meta, "pay")
        last, val = s.decode_payload(majority(pf, pbits, rp, rng))
    # query codes of every (probed key, j) are rows of the noiseless key field (a probed key's rows exist)
    active = list(range(len(s.keys)))
    for j in range(s.maxj + 1):
        if not active:
            break
        qrows = [s.row_of.get((s.keys[a], j)) for a in active]
        # a (key, j) that is not stored has no query code in the logical table: build it from the code family
        qs = [logical_key[i] if i is not None else query_code(s, info, s.keys[a], j) for a, i in zip(active, qrows)]
        keep = [qi for qi, q_ in enumerate(qs) if q_ is not None]   # exact arms: an unrepresentable j stops the key
        active = [active[qi] for qi in keep]
        qrows = [qrows[qi] for qi in keep]
        if not active:
            break
        q = np.stack([qs[qi] for qi in keep])
        if mode == "exact":
            dots = (kfield.astype(np.float32) @ q.T.astype(np.float32)).T      # (queries, rows)
            hit = dots >= wk - 0.5
        elif mode == "hardmatch":
            dots = (kdec.astype(np.float32) @ q.T.astype(np.float32)).T
            hit = dots >= wk - 0.5
        else:
            qr = np.repeat(q, rk, axis=1).astype(np.float32)
            dots = (kfield.astype(np.float32) @ qr.T).T
            dots += rng.random(dots.shape, dtype=np.float32)                  # random tie-break (dots step by 2)
        nxt = []
        for qi, a in enumerate(active):
            if mode in ("exact", "hardmatch"):
                rows = np.flatnonzero(hit[qi])
                if len(rows) == 0:
                    continue                                                    # no match: stop
                row = int(rows[0])
                got[a].append(int(val[row]))
                nxt.append(a)                                                   # absence stop: try j + 1
            else:
                row = int(np.argmax(dots[qi]))
                if diag is not None:
                    diag.append((a, j, row, qrows[qi]))
                if arm == "T-iiib":
                    sims = cand @ vfield[row].astype(np.float32)
                    sims += rng.random(sims.shape, dtype=np.float32)
                    c = int(np.argmax(sims))
                    v, l = (c % len(s.cd["vocab"]), int(c >= len(s.cd["vocab"]))) if s.lastbit else (c, 1)
                    got[a].append(v)
                    if not l:
                        nxt.append(a)
                else:
                    got[a].append(int(val[row]))
                    if not last[row]:
                        nxt.append(a)
        active = nxt
    return got


def query_code(s, info, k, j):
    """The arm's code of a (key, j) that is not stored (asked after a misread last flag, or by the exact-match arms
    when they look for the next j): the key's own code with j's bits swapped in, or the code family's hash."""
    arm = info["arm"]
    if arm in ("T-i", "T-iih", "T-iis"):
        c = info["fields_logical"]["key"][s.row_of[(k, 0)]].copy()
        if s.jb:
            if j >= 2 ** s.jb:
                return None                                   # j not representable: no row can carry it
            c[-s.jb:] = bits_pm([j], s.jb)[0]
        return c
    if arm == "T-iix":
        dig = np.frombuffer(hashlib.sha256(("b34h:%s\x00%d" % (k, j)).encode("utf-8")).digest(), np.uint8)
        return (2 * np.unpackbits(dig)[:info["h"]].astype(np.int8) - 1)
    wk = info["wk"]
    return hd_code("key:" + k, wk) * hd_code("pos:%d" % j, wk)


def read_tiv(store, t, k, sigs=None):
    """B26b's Trep (T-iv), reproduced with b26b's own layout, seeds and decoder."""
    s = store
    rows, owner, b, r, _, _ = T26.table_layout(s.pairs, s.st, s.cd)
    seed = s.seed + 2
    cells = np.repeat(rows, r, axis=1)
    out = {}
    sg_t = signs_over_times(cells, seed, k, TIMES)
    for tl, tt in TIMES:
        dec = T26.decode(sg_t[tl].astype(np.float32), b, r, seed, tt)
        ok = collections.defaultdict(list)
        for a, d in zip(owner, dec):
            ok[a].append(int(d))
        out[tl] = sum(ok[a] == w for a, w in enumerate(s.want))
    return out, r


# ---------------------------------------------------------------- reading the VSA (B25's functions)
def read_keys_nt(rows, scale, sh, kidx, Kc, Pc, cleanup, maxj):
    """B25's read_keys, except that the first value is accepted by argmax (no threshold at j = 0)."""
    X = rows[sh] * scale[sh][:, None] * Kc[kidx]
    got = [[] for _ in sh]
    active = np.arange(len(sh))
    j = 0
    while len(active) and j < maxj:
        sims = cleanup(X[active] * Pc[j])
        b = sims.argmax(1)
        ok = sims[np.arange(len(active)), b] >= P.THR if j > 0 else np.ones(len(active), bool)
        for a, bb, o in zip(active, b, ok):
            if o:
                got[a].append(int(bb))
        active = active[ok]
        j += 1
    return got


def exact(got, want):
    return int(sum(g == w for g, w in zip(got, want)))


# ---------------------------------------------------------------- one job: (store, k)
def run_job(job):
    """job: (store id, pairs, k, choice, ber, flags). Returns {cell name: [exact, n]} and per-arm parameters."""
    sid, pairs, k, choice, ber, flags = job
    t0 = time.time()
    s = Store(pairs, "b26b:%s" % sid.replace("/", ":"))   # the VSA and T-iv use B26b's seeds (reproduction check)
    tseed = V._seed("b34:%s" % sid) % (2 ** 62)
    res, params, diag = {}, {}, {}
    n = len(s.keys)
    st = s.st
    dig = lambda Y, Vc=st["Vc"]: Y @ Vc.T / D
    # ---- VSA arms
    if "vsa" in flags:
        for tl, t in TIMES:
            out, dg = P.store_array("B1", st["mem"], st["b1"], s.seed, t, k)
            rows = out["gdc"]
            res["V-S|%s" % tl] = [exact(P.read_keys(rows, st["rms0"], st["sh"], st["kidx"], st["Kc"], st["Pc"], dig,
                                                    st["maxj"]), s.want), n]
            res["V-Snt|%s" % tl] = [exact(read_keys_nt(rows, st["rms0"], st["sh"], st["kidx"], st["Kc"], st["Pc"], dig,
                                                       st["maxj"]), s.want), n]
            diag["bit_err|%s" % tl] = dg["bit_err"]
            if "sc" in flags:
                W, S2, a = P.codebook_array(st["Vc"], s.seed, t, k)
                clean = P.pcm_cleanup(W, a, 1, P.ADC_BITS)
                res["V-SC|%s" % tl] = [exact(P.read_keys(rows, st["rms0"], st["sh"], st["kidx"], st["Kc"], st["Pc"],
                                                         clean, st["maxj"]), s.want), n]
                res["V-SCnt|%s" % tl] = [exact(read_keys_nt(rows, st["rms0"], st["sh"], st["kidx"], st["Kc"],
                                                            st["Pc"], clean, st["maxj"]), s.want), n]
                diag["alpha_cb|%s" % tl] = a
                del W
    # ---- T-iv (reproduction of B26b's Trep)
    if "tiv" in flags:
        o, r = read_tiv(s, None, k)
        for tl, _ in TIMES:
            res["T-iv|%s" % tl] = [o[tl], n]
        params["T-iv"] = {"r": int(r)}
    # ---- in-memory tables at budget C (device model) and, if asked, at C' (i.i.d. at the measured BER)
    budgets = [("C", s.C, "device")]
    if "eqtotal" in flags:
        budgets.append(("Ctot", (st["S"] + s.Nv) * D, "bernoulli"))
    for bname, C, noise in budgets:
        arms = flags["arms"] if bname == "C" else NONHD
        for arm in arms:
            if arm == "T-iv":
                continue
            share = choice.get(arm, {}).get("share") if choice else None
            lam = choice.get("T-v", {}).get("lam", 4) if choice else 4
            if isinstance(share, dict):                       # dev grid: several shares
                share_list = share["grid"]
            else:
                share_list = [share]
            for sh_ in share_list:
                info = layout(s, arm, C, share=sh_, lam=lam)
                tag = arm if len(share_list) == 1 else "%s@%.1f" % (arm, sh_)
                if bname == "Ctot":
                    tag += "@tot"
                if info is None:
                    for tl, _ in TIMES:
                        res["%s|%s" % (tag, tl)] = [0, n]
                    params[tag] = {"infeasible": True}
                    continue
                aseed = V._seed("b34:%s:%s:%s" % (sid, tag, bname)) % (2 ** 62)
                sigs = {}
                for an, (cells, _) in info["arrays"].items():
                    asd = V._seed("b34:%s:%s:%s:%s" % (sid, tag, bname, an)) % (2 ** 62)
                    if noise == "device":
                        sigs[an] = signs_over_times(cells, asd, k, TIMES)
                    else:
                        sb = signs_bernoulli(cells, asd, {(k, tl): ber[(k, tl)] for tl, _ in TIMES})
                        sigs[an] = {tl: sb[(k, tl)] for tl, _ in TIMES}
                for tl, tt in TIMES:
                    rng = np.random.default_rng([aseed, 9, int(tt), int(k * 10)])
                    got = read_table(s, info, {an: sigs[an][tl] for an in sigs}, rng)
                    res["%s|%s" % (tag, tl)] = [exact(got, s.want), n]
                p = {kk: v for kk, v in info.items() if kk not in ("arrays", "fields_logical", "mphf")}
                p["cells_frac"] = info["cells"] / C
                params[tag] = p
                del sigs, info
    return {"store": sid, "k": k, "res": res, "params": params, "diag": diag, "n_pairs": s.n, "n_values": s.Nv,
            "S": int(st["S"]), "C": int(s.C), "sec": time.time() - t0}


# ---------------------------------------------------------------- synthetic dev stores (no test data)
def dev_stores():
    import random
    import uuid as uuidlib
    import wonderwords
    nouns = sorted(set(wonderwords.random_word._get_words_from_text_file("nounlist.txt")))
    adjs = sorted(set(wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")))
    out = []
    spec = [("word", 6000, 0, "mk2big"), ("word", 6000, 2, "mk2big2"), ("word", 1500, 0, "mk2mid"),
            ("uuid", 1700, 0, "mk3"), ("multi", 18, 0, "mv"), ("vt", 9, 0, "vt")]
    for rep in range(2):
        for kind, nk, nmulti, name in spec:
            rng = random.Random("b34dev:%s:%d" % (name, rep))
            keys, seen = [], set()
            while len(keys) < nk:
                if kind in ("word", "multi"):
                    k = "%s-%s" % (rng.choice(adjs), rng.choice(nouns))
                elif kind == "uuid":
                    k = str(uuidlib.UUID(int=rng.getrandbits(128), version=4))
                else:
                    k = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 5)))
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
            pairs = []
            counts = [1] * nk
            if kind == "multi":
                counts[0], counts[1], counts[2] = 4, 2, 2
            for i in range(nmulti):
                counts[(i + 1) * 7] = 2
            for k, c in zip(keys, counts):
                for j in range(c):
                    v = str(uuidlib.UUID(int=rng.getrandbits(128), version=4)) if kind == "uuid" else \
                        str(rng.randint(10 ** 6, 10 ** 7 - 1)) if kind != "vt" else str(rng.randint(10 ** 4, 10 ** 5 - 1))
                    pairs.append((k.lower(), v, j))
            out.append(("dev/%s/%d" % (name, rep), pairs))
    return out


def pool_run(jobs, workers):
    from multiprocessing import Pool
    out = []
    with Pool(workers) as pool:
        for r in pool.imap_unordered(run_job, jobs):
            out.append(r)
            print("done %-34s x%-4g %6.0fs  %d/%d" % (r["store"], r["k"], r["sec"], len(out), len(jobs)), flush=True)
    return out


def totals(per):
    tot = collections.defaultdict(lambda: [0, 0])
    for r in per:
        for name, (e, n) in r["res"].items():
            key = "%s|x%g" % (name, r["k"])
            tot[key][0] += e
            tot[key][1] += n
    return {k: [v[0], v[1], v[0] / v[1]] for k, v in tot.items()}


# ---------------------------------------------------------------- self-test (synthetic pairs only)
def selftest():
    rng = np.random.default_rng(1)
    pairs, seen = [], collections.defaultdict(int)
    for i in range(1500):
        k = "key%d" % rng.integers(0, 1200)
        pairs.append((k, "val%d" % (i % 900), seen[k]))
        seen[k] += 1
    # 1. signs_over_times == b26b cells_read, time by time
    bits = np.where(rng.random((37, 211)) < 0.5, 1.0, -1.0).astype(np.float32)
    for k in (1.0, 8.0):
        so = signs_over_times(bits, 77, k, TIMES)
        for tl, t in TIMES:
            assert np.array_equal(so[tl].astype(np.float32), T26.cells_read(bits, 77, t, k)), (k, tl)
    print("signs_over_times == b26b cells_read: ok")
    # 2. every table arm reads every probed key exactly with no noise
    s = Store(pairs, "selftest")
    for arm in TABLES[1:]:
        for share in ((0.3, 0.7) if arm in SPLIT_ARMS else (None,)):
            info = layout(s, arm, s.C, share=share, lam=4)
            sig = {an: c for an, (c, _) in info["arrays"].items()}
            got = read_table(s, info, sig, np.random.default_rng(0))
            e = exact(got, s.want)
            print("%-7s share %-4s cells/C %.3f  noiseless exact %d/%d  %s" % (
                arm, share, info["cells"] / s.C, e, len(s.want),
                {k: v for k, v in info.items() if k in ("W", "wk", "rv", "r", "rd", "rp", "h", "R", "B", "db", "wv")}))
            assert e == len(s.want), arm
    o, r = read_tiv(s, None, 1.0)
    assert all(v == len(s.want) for v in o.values()), o
    print("T-iv noiseless ok, r =", r)
    # 3. one noisy job on the synthetic store (all arms, two k)
    global KS
    choice = {a: {"share": 0.5} for a in SPLIT_ARMS}
    choice["T-v"] = {"lam": 4}
    flags = {"vsa": 1, "sc": 1, "tiv": 1, "eqtotal": 1, "arms": TABLES}
    ber = {(k, tl): 0.0 for k in KS for tl, _ in TIMES}
    ber.update({(8.0, tl): 0.1 for tl, _ in TIMES})
    t0 = time.time()
    out = run_job(("selftest", pairs, 8.0, choice, ber, flags))
    for kk in sorted(out["res"]):
        if kk.endswith("30 d"):
            print("%-22s %s" % (kk, out["res"][kk]))
    print("params", json.dumps(out["params"])[:600])
    print("selftest ok (%.0fs)" % (time.time() - t0))


def load_store_file(path):
    import gzip
    return [(sid, [tuple(p) for p in pairs]) for sid, pairs in json.load(gzip.open(path, "rt"))]


def read_ber(path):
    return {(float(k.split("|")[0]), k.split("|")[1]): v for k, v in json.load(open(path)).items()}


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit()
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    mode = sys.argv[1]
    if mode == "ber":                                         # python3 b34_inmem.py ber OUT.json
        b = measured_ber()
        json.dump({"%g|%s" % kt: v for kt, v in b.items()}, open(sys.argv[2], "w"), indent=1)
        print(b)
    elif mode == "dump_dev":                                  # python3 b34_inmem.py dump_dev OUT.json.gz
        import gzip
        json.dump(dev_stores(), gzip.open(sys.argv[2], "wt"))
    elif mode == "dump_test":                                 # python3 b34_inmem.py dump_test B21_DIR OUT.json.gz
        import gzip
        json.dump([("%d/%s/%d" % (L, t, i), p) for L, t, i, p in B.load_stores(sys.argv[2])], gzip.open(sys.argv[3], "wt"))
    elif mode == "mphf_lambda":                               # python3 b34_inmem.py mphf_lambda DEV.json.gz OUT.json
        res = {}
        stores = load_store_file(sys.argv[2])
        for lam in MPHF_LAMBDAS:
            tot = 0
            for sid, pairs in stores:
                s = Store(pairs, sid)
                mp = s.mphf(lam)
                tot += (mp["B"] * mp["db"] + (mp["R"] - s.n) * (s.lastbit + s.b)) / s.n
                print(lam, sid, "R", mp["R"], "B", mp["B"], "db", mp["db"], flush=True)
            res[lam] = tot / len(stores)
        print("mean overhead bits per pair (displacements + empty rows):", res)
        json.dump(res, open(sys.argv[3], "w"), indent=1)
    elif mode in ("dev", "test"):
        # python3 b34_inmem.py dev|test STORES.json.gz BER.json OUT.json STAGE [CHOICE.json]
        #   dev stage "grid": the split arms over the share grid at k = 1, 4, 8, 16, 32 (choice of shares)
        #   stage "full": every arm, every k (dev: to inform the predictions; test: the registered run)
        stores = load_store_file(sys.argv[2])
        ber = read_ber(sys.argv[3])
        out, stage = sys.argv[4], sys.argv[5]
        jobs = []
        if stage == "grid":                                   # [ARMS comma-separated] [MPHF lambda]
            assert mode == "dev"
            arms = tuple(sys.argv[6].split(",")) if len(sys.argv) > 6 and not sys.argv[6].startswith("--") \
                else SPLIT_ARMS[:4]
            choice = {a: {"share": {"grid": list(SHARES)}} for a in arms}
            if len(sys.argv) > 7 and not sys.argv[7].startswith("--"):
                choice["T-v"] = {"lam": int(sys.argv[7])}
            flags = {"arms": arms}
            ks = (1.0, 4.0, 8.0, 16.0, 32.0)
        else:
            choice = json.load(open(sys.argv[6]))
            flags = {"vsa": 1, "sc": 1, "tiv": 1, "eqtotal": 1, "arms": TABLES}
            ks = KS
        for sid, pairs in stores:
            for k in ks:
                jobs.append((sid, pairs, k, choice, ber, flags))
        jobs.sort(key=lambda j: -len(j[1]))
        per = pool_run(jobs, workers)
        json.dump({"totals": totals(per), "per_job": per, "choice": choice}, open(out, "w"), indent=1)
