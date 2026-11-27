# B38 dev-stage check: our BCH codec (b38_eq.BCH) vs the galois library on 17 random designs (generator polynomial,
# systematic encoding of random messages, decoding of 30 words with 0 .. 2t+2 random errors each). Synthetic only.
import sys, time, numpy as np, galois
sys.path.insert(0, "/path/to/vsassm")
import b38_eq as E
rng = np.random.default_rng(0)
des = E.bch_designs()
print("designs", len(des))
pick = [d for d in des if d[0] in (63, 127, 255, 511, 1023)]
sel = [pick[i] for i in rng.choice(len(pick), 14, replace=False)] + [(1023, 193, 118), (255, 131, 18), (63, 36, 5)]
for n, k, t in sel:
    own = E.BCH(n, k, t)
    gb = galois.BCH(n, d=2 * t + 1)
    assert gb.k == k, (n, k, t, gb.k)
    assert gb.t == t, (n, k, t, gb.t)
    g_g = np.array(gb.generator_poly.coeffs, dtype=np.uint8)            # highest degree first
    same_g = np.array_equal(g_g, own.g_low[::-1])
    ks = max(1, k - int(rng.integers(0, k // 2 + 1)))
    M = rng.integers(0, 2, (30, ks)).astype(np.uint8)
    C_own = own.encode(M)
    C_g = np.asarray(gb.encode(galois.GF2(M)))
    same_c = np.array_equal(C_own, C_g)
    agree = tot = 0
    t0 = time.time()
    for i in range(30):
        ne = int(rng.integers(0, 2 * t + 3))
        w = C_own[i].copy()
        pos = rng.choice(len(w), min(ne, len(w)), replace=False)
        w[pos] ^= 1
        ok, mo, no = own.decode(w)
        dg, ng = gb.decode(galois.GF2(w), errors=True)
        okg = int(ng) >= 0
        agree += (ok == okg) and (not ok or np.array_equal(mo, np.asarray(dg)))
        tot += 1
    print(n, k, t, "ks", ks, "g same", same_g, "enc same", same_c, "decode agree %d/%d" % (agree, tot), "%.2fs" % (time.time() - t0))
