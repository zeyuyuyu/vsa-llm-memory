"""B50 hand-read sampler (prereg 'Checks from other angles', hand-read). Prints the items to be read by hand.
usage: python b50_handread.py <b50_root> <b40_pred_root> <best_gate> [<second_gate>]  > results/b50_handread_raw.txt
(a) seed 0 of each named gate: every held-out item on which it opens (up to 30) and every trained-type item on which it
    closes (up to 30), with DISC's decoded read, its output and BASE's output;
(b) COT2048: 20 items B40 truncated that now end, 10 still at 2,048 tokens, 10 random others (fixed seed 50)."""
import json, sys
import numpy as np
import b40_data as D


def load(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)}


def main():
    b50, b40 = sys.argv[1], sys.argv[2]
    gates = sys.argv[3:]
    rows = {r["id"]: r for r in D.gen("N", "test")}
    base = load(f"{b40}/N/base.jsonl")
    for g in gates:
        f = load(f"{b50}/preds/N/disc{g}_s0.jsonl")
        ho_open = [i for i in f if rows[i]["op"] in D.OPS_HELD and f[i]["mod"]["gate"]]
        tr_closed = [i for i in f if rows[i]["op"] in D.OPS_TRAIN and not f[i]["mod"]["gate"]]
        print(f"===== gate {g}, seed 0: held-out items OPEN {len(ho_open)}, trained-type items CLOSED {len(tr_closed)}")
        for title, ids in (("held-out, gate open", ho_open[:30]), ("trained-type, gate closed", tr_closed[:30])):
            print(f"--- {title}")
            for i in ids:
                r, m = rows[i], f[i]["mod"]
                print(f"  {i}: {r['op']} x={r['x']} y={r['y']} gold={r['ans']} | DISC read {m['op']} x={m['x']} y={m['y']} "
                      f"ans={m['ans']} score={m[g]:.4g} | out={f[i]['out']!r} correct={f[i]['correct']} | BASE out="
                      f"{base[i]['out']!r} correct={base[i]['correct']}")
    rng = np.random.default_rng(50)
    for task in ("N", "R"):
        c2 = load(f"{b50}/preds/{task}/cot2048_cot.jsonl")
        c5 = load(f"{b40}/{task}/cot.jsonl")
        tr = [i for i in c5 if "final answer" not in c5[i]["out"].lower()]
        now_end = [i for i in tr if not c2[i]["hit_limit"]]
        still = [i for i in c2 if c2[i]["hit_limit"]]
        other = [i for i in c2 if i not in set(tr) and not c2[i]["hit_limit"]]
        pick = lambda L, k: [L[j] for j in sorted(rng.choice(len(L), min(k, len(L)), replace=False))] if L else []
        for title, ids, show in (("B40-truncated, now ending", pick(now_end, 20), "tail"),
                                 ("still at 2,048 tokens", pick(still, 10), "both"),
                                 ("random other", pick(other, 10), "tail")):
            print(f"===== CoT2048 task {task}: {title} ({len(ids)} shown)")
            for i in ids:
                x = c2[i]
                o = x["out"].replace("\n", " / ")
                seg = o[-400:] if show == "tail" else (o[600:900] + "  ...  " + o[-300:])
                print(f"  {i}: gold={x['gold']} pred={x['pred']!r} correct={x['correct']} n_gen={x['n_gen']} "
                      f"B40 pred={c5[i]['pred']!r} | {seg}")


if __name__ == "__main__":
    main()
