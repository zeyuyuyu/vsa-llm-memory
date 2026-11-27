"""R1 step 0 (development, oracle facts): can a VSA rule memory learn relation composition from short chains (k = 2, 3)
and apply it to long ones (k = 2-10)? Uses CLUTRR's gold relation sequences (f_comb), so the reader is not involved.

python3 r1_oracle.py r1_data   (CPU, numpy)
"""
import collections
import csv
import sys

import numpy as np

D = 2048
rng = np.random.default_rng(0)


class FHRR:
    """Unit-modulus complex codes; bind = elementwise product, permute = roll; cleanup = nearest code."""

    def __init__(self):
        self.codes = {}

    def code(self, name):
        if name not in self.codes:
            self.codes[name] = np.exp(1j * rng.uniform(-np.pi, np.pi, D))
        return self.codes[name]

    def cleanup(self, v):
        names = list(self.codes)
        sims = np.array([np.real(np.vdot(self.codes[n], v)) / D for n in names])
        i = int(sims.argmax())
        return names[i], float(sims[i])


class RuleMemory:
    """Composition rules r1 o r2 -> r3 as a superposition of key -> value bindings (one complex vector of size D).
    The first slot, the second slot and the value use three independent codebooks: with one shared codebook a rule
    (x, b) -> a reads back as code(x) when (a, b) is queried (exact crosstalk, found in development)."""

    def __init__(self):
        self.first, self.second, self.value = FHRR(), FHRR(), FHRR()
        self.M, self.n = np.zeros(D, complex), 0

    def key(self, r1, r2):
        return self.first.code(r1) * self.second.code(r2)

    def write(self, r1, r2, r3):
        self.M += self.key(r1, r2) * self.value.code(r3)
        self.n += 1

    def read(self, r1, r2):
        return self.value.cleanup(np.conj(self.key(r1, r2)) * self.M)


def compose(rels, known, read):
    """Fold a chain with any bracketing (kinship composition is associative): CYK over sub-chains; known(a, b) says a
    rule exists, read(a, b) returns its value. Returns the relation for the whole chain, or None."""
    k = len(rels)
    table = {(i, i): {rels[i]} for i in range(k)}
    for span in range(2, k + 1):
        for i in range(k - span + 1):
            j = i + span - 1
            out = set()
            for m in range(i, j):
                for x in table[(i, m)]:
                    for y in table[(m + 1, j)]:
                        if known(x, y):
                            out.add(read(x, y))
            table[(i, j)] = out
    res = table[(0, k - 1)]
    return sorted(res)[0] if len(res) == 1 else (None if not res else ("AMBIG", tuple(sorted(res))))


def load(path):
    return [{"rels": r["f_comb"].split("-"), "target": r["target_text"]} for r in csv.DictReader(open(path))]


def main():
    root = sys.argv[1]
    train, test = load(f"{root}/train.csv"), load(f"{root}/test.csv")
    # 1. rules from k = 2 stories (majority vote), 2. new rules from k = 3 stories via known k = 2 rules (closure)
    votes = collections.defaultdict(collections.Counter)
    for r in train:
        if len(r["rels"]) == 2:
            votes[tuple(r["rels"])][r["target"]] += 1
    rules = {k: c.most_common(1)[0][0] for k, c in votes.items()}
    conflicts = sum(len(c) > 1 for c in votes.values())
    for _ in range(3):
        new = collections.defaultdict(collections.Counter)
        for r in train:
            if len(r["rels"]) == 3:
                a, b, c = r["rels"]
                if (a, b) in rules and (rules[(a, b)], c) not in rules:
                    new[(rules[(a, b)], c)][r["target"]] += 1
                if (b, c) in rules and (a, rules[(b, c)]) not in rules:
                    new[(a, rules[(b, c)])][r["target"]] += 1
        rules.update({k: c.most_common(1)[0][0] for k, c in new.items()})
    print(f"rules learned: {len(rules)} ({len(votes)} from k=2, {conflicts} with conflicting votes)")

    mem = RuleMemory()
    for (a, b), c in rules.items():
        mem.write(a, b, c)
    exact = sum(mem.read(a, b)[0] == c for (a, b), c in rules.items())
    print(f"VSA rule memory: {exact}/{len(rules)} rules read back exactly (D={D})")

    known = lambda a, b: (a, b) in rules
    by_k = collections.defaultdict(lambda: [0, 0, 0, 0, 0])  # left-fold, any-bracketing table, VSA, unknown, total
    for r in test:
        k = len(r["rels"])
        cur, left_ok = r["rels"][0], True
        for nxt in r["rels"][1:]:
            if (cur, nxt) not in rules:
                left_ok = False
                break
            cur = rules[(cur, nxt)]
        any_t = compose(r["rels"], known, lambda a, b: rules[(a, b)])
        any_v = compose(r["rels"], known, lambda a, b: mem.read(a, b)[0])
        b = by_k[k]
        b[0] += left_ok and cur == r["target"]
        b[1] += any_t == r["target"]
        b[2] += any_v == r["target"]
        b[3] += any_t is None
        b[4] += 1
    print(" k    n   left-fold  any-bracket(table)  any-bracket(VSA)  no-rule-path")
    tot = [0] * 5
    for k in sorted(by_k):
        b = by_k[k]
        tot = [x + y for x, y in zip(tot, b)]
        print(f"{k:2d} {b[4]:4d}   {b[0] / b[4]:.3f}      {b[1] / b[4]:.3f}               {b[2] / b[4]:.3f}             {b[3] / b[4]:.3f}")
    print(f"all {tot[4]}  {tot[0] / tot[4]:.3f}      {tot[1] / tot[4]:.3f}               {tot[2] / tot[4]:.3f}             {tot[3] / tot[4]:.3f}")


if __name__ == "__main__":
    main()
