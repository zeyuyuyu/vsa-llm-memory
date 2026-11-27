"""Clean-room table: minimal perfect hash by displacement (hash-and-displace),
rows of (last flag, value index).  Written from the paper's description only.

Logical bit stream of a store = [displacement directory | rows].
Rows are indexed by the MPH of (key, position-in-key's-value-list).
"""
import hashlib, json, gzip, math, collections
import numpy as np

M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def splitmix(x):
    x = (x + np.uint64(0x9E3779B97F4A7C15))
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def h3(item, salt):
    d = hashlib.blake2b(f"{salt}\x1f{item}".encode(), digest_size=16).digest()
    a = int.from_bytes(d[:8], 'little'); b = int.from_bytes(d[8:], 'little')
    return a, b


def slot_of(h1, disp, n):
    """h1: uint64 array, disp: int array -> slot in [0, n)"""
    with np.errstate(over='ignore'):
        return (splitmix(h1 ^ splitmix(disp.astype(np.uint64) + np.uint64(1))) % np.uint64(n)).astype(np.int64)


def load_stores(path):
    d = json.load(gzip.open(path))
    out = []
    for name, triples in d:
        out.append((name, [(t[0], t[1], int(t[2])) for t in triples]))
    return out


def probed_keys(triples):
    order = []
    seen = set()
    for k, v, j in triples:
        if k not in seen:
            seen.add(k); order.append(k)
    nk = len(order)
    if nk <= 150:
        return order
    step = nk // 150
    return order[::step][:150]


class Table:
    def __init__(self, name, triples, lam=5.0, salt=0):
        self.name = name
        self.n = n = len(triples)
        vals = sorted(set(v for _, v, _ in triples))
        self.N = len(vals)
        self.b = max(1, math.ceil(math.log2(self.N))) if self.N > 1 else 1
        vidx = {v: i for i, v in enumerate(vals)}
        bykey = collections.defaultdict(list)
        for k, v, j in triples:
            bykey[k].append((j, v))
        for k in bykey:
            bykey[k].sort()
        self.bykey = bykey
        items = [(k, j) for k in bykey for (j, _) in bykey[k]]
        assert len(items) == n
        self.salt = salt
        self.r = r = max(1, math.ceil(n / lam))
        H = np.array([h3(f"{k}\x1e{j}", salt) for k, j in items], dtype=np.uint64)
        bucket = (H[:, 0] % np.uint64(r)).astype(np.int64)
        h1 = H[:, 1]
        # hash-and-displace, largest buckets first
        members = collections.defaultdict(list)
        for i, bk in enumerate(bucket):
            members[bk].append(i)
        order = sorted(members.keys(), key=lambda bk: -len(members[bk]))
        free = np.ones(n, dtype=bool)
        disp = np.zeros(r, dtype=np.int64)
        slot = np.full(n, -1, dtype=np.int64)
        CH = 512
        for bk in order:
            mem = np.array(members[bk])
            s = len(mem)
            d0 = 0
            while True:
                ds = np.arange(d0, d0 + CH, dtype=np.int64)
                sl = slot_of(h1[mem][None, :], ds[:, None], n)  # CH x s
                ok = free[sl].all(axis=1)
                if s > 1:
                    srt = np.sort(sl, axis=1)
                    ok &= (np.diff(srt, axis=1) != 0).all(axis=1)
                w = np.nonzero(ok)[0]
                if len(w):
                    d = int(ds[w[0]]); break
                d0 += CH
            disp[bk] = d
            sl = slot_of(h1[mem], np.full(s, d), n)
            free[sl] = False
            slot[mem] = sl
        assert (~free).all()
        self.w = max(1, int(disp.max()).bit_length())
        self.disp = disp
        # rows
        self.rowbits = 1 + self.b
        row_last = np.zeros(n, dtype=np.int64)
        row_val = np.zeros(n, dtype=np.int64)
        self.item_index = {}
        for i, (k, j) in enumerate(items):
            vals_k = bykey[k]
            row_val[slot[i]] = vidx[vals_k[j][1]]
            row_last[slot[i]] = 1 if j == len(vals_k) - 1 else 0
            self.item_index[(k, j)] = i
        self.vidx = vidx
        # logical bits
        dbits = ((disp[:, None] >> np.arange(self.w)[None, :]) & 1).astype(np.uint8).ravel()
        rb = np.concatenate([row_last[:, None], (row_val[:, None] >> np.arange(self.b)[None, :]) & 1], axis=1).astype(np.uint8).ravel()
        self.bits = np.concatenate([dbits, rb])
        self.K = len(self.bits)
        self.dir_bits = len(dbits)
        # probe structure: for each probed key, its hashes for j = 0..maxlen
        self.probe = probed_keys(triples)
        self.MAXREAD = 8
        pb = []
        ph1 = []
        for k in self.probe:
            hh = [h3(f"{k}\x1e{j}", salt) for j in range(self.MAXREAD)]
            pb.append([x[0] for x in hh]); ph1.append([x[1] for x in hh])
        pb = np.array(pb, dtype=np.uint64); ph1 = np.array(ph1, dtype=np.uint64)
        self.p_bucket = (pb % np.uint64(r)).astype(np.int64)
        self.p_h1 = ph1
        self.p_truth = [[vidx[v] for _, v in bykey[k]] for k in self.probe]
        self.p_len = np.array([len(t) for t in self.p_truth])
        tr = np.full((len(self.probe), self.MAXREAD), -1, dtype=np.int64)
        for i, t in enumerate(self.p_truth):
            tr[i, :len(t)] = t
        self.p_truth_arr = tr

    def read_exact(self, bits):
        """bits: decoded logical bit stream (uint8, len K). Returns bool array over probed keys."""
        dbits = bits[:self.dir_bits].reshape(self.r, self.w).astype(np.int64)
        disp = (dbits << np.arange(self.w)[None, :]).sum(1)
        rows = bits[self.dir_bits:self.dir_bits + self.n * self.rowbits].reshape(self.n, self.rowbits).astype(np.int64)
        last = rows[:, 0]
        val = (rows[:, 1:] << np.arange(self.b)[None, :]).sum(1)
        P = len(self.probe)
        sl = slot_of(self.p_h1, disp[self.p_bucket], self.n)  # P x MAXREAD
        v = val[sl]; l = last[sl]
        # read until first last flag
        ok = np.ones(P, dtype=bool)
        stopped = np.zeros(P, dtype=bool)
        for j in range(self.MAXREAD):
            active = ~stopped
            exp = self.p_truth_arr[:, j]
            # active reads must match the truth at position j
            ok &= ~active | (v[:, j] == exp)
            # exact stopping: last flag must be 1 exactly at position len-1
            stop_now = active & (l[:, j] == 1)
            ok &= ~(stop_now & (j != self.p_len - 1))
            ok &= ~(active & (l[:, j] == 0) & (j == self.p_len - 1))
            stopped |= stop_now
        ok &= stopped
        return ok
