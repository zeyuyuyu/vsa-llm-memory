"""B47: replacement data for Fig. 2b,c (prereg/B47_phi2_finetunes_hierboot_consistency_prereg.md), every number with
its source file and the verbatim source line (or JSON path). Also writes b47_out/observed_arms_C.json (the arena-C
learned arms' accuracy, geometry and clean-up rates, parsed from results/b42_verdict_C.txt, which carries B31's and
B42's arena-C finals at each seed's dev-selected epoch).

Contents of b47_out/b47_fig2bc.json
  b.coherence   the imposed coherence sweep (rho = 1), perm and comm: B44's registered output (results/b44_fig2bc.json
                -> results/b44_xtalk_sweep.json) and B47's recomputation in its own code path (gate G1b)
  b.consistency the new consistency sweep at c = 0 (mention and form models; perm and comm), x = measured same-symbol
                cosine of the codes as bound, y = test accuracy (per seed), dev beside it, hop-wise clean-up rates
  b.grid        coherence x consistency (perm, mention)
  b.points      the learned arms (B31 / B42 finals, dev-selected epochs) with both geometry coordinates, accuracy and
                its hierarchical interval (b47_hierboot.json) where computed; the binding-loss map at initialisation
                (B44) as in results/b44_fig2bc.json
  b.floor       B31's Phi-2 equalised probe (0.472) with its verdict line
  c             B44's Fig. 2c data (k-sweep: exact, naming head at dev-selected epochs, probe), copied with sources,
                plus H-nest intervals of each point recomputed from B44's saved per-instance outputs (b44_out/k)

  python b47_fig2bc.py
"""
import json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "b47_out")
ARM_RE = re.compile(r"^    (Phi-2|Qwen3-8B)\s{2,}(.+?)\s{2,}mean (-?\d\.\d+)(?: \[.*?\])?\s+seeds (.*)$")
SEEDV = re.compile(r"(-?\d\.\d+) \[")


def parse_b42_C():
    p = os.path.join(HERE, "results", "b42_verdict_C.txt")
    lines = open(p).read().split("\n")
    i0 = next(i for i, l in enumerate(lines) if l.strip() == "arena C (n = 500)")
    arms, cur = [], None
    for i in range(i0 + 1, len(lines)):
        l = lines[i]
        if l.startswith("## "):
            break
        m = ARM_RE.match(l)
        if m:
            cur = {"encoder": m.group(1), "label": "%s %s" % (m.group(1), m.group(2).strip()), "acc_mean": float(m.group(3)),
                   "acc_per_seed": [float(x) for x in SEEDV.findall(m.group(4))], "source": "results/b42_verdict_C.txt",
                   "line": l}
            arms.append(cur); continue
        if cur is None:
            continue
        s = l.strip()
        if s.startswith("cross-talk different/same"):
            pr = [tuple(float(v) for v in x.split("/")) for x in s.replace("cross-talk different/same", "").split(",")]
            cur["different_per_seed"] = [x[0] for x in pr]; cur["same_per_seed"] = [x[1] for x in pr]
            cur["different_mean"] = float(np.mean(cur["different_per_seed"])); cur["same_mean"] = float(np.mean(cur["same_per_seed"]))
            cur["geom_line"] = l
        elif s.startswith("clean-up correct at hop 1 / hop 2 / both:"):
            pr = [tuple(float(v) for v in x.split("/")) for x in s.split(":", 1)[1].split(",")]
            cur["hop1_per_seed"] = [x[0] for x in pr]; cur["hop2_per_seed"] = [x[1] for x in pr]
            cur["hop1_mean"] = float(np.mean(cur["hop1_per_seed"])); cur["hop2_mean"] = float(np.mean(cur["hop2_per_seed"]))
            cur["cleanup_line"] = l
    return arms


LABELFREE = ("binding-loss", "InfoNCE", "VQ", "key-value", "Gumbel", "straight-through", "centred", "whitening", "centring")


def main():
    arms = parse_b42_C()
    lf = [a for a in arms if any(k in a["label"] for k in LABELFREE) and "oracle" not in a["label"].lower()
          and "different_mean" in a]
    json.dump(lf, open(os.path.join(OUT, "observed_arms_C.json"), "w"), indent=1)
    b44 = json.load(open(os.path.join(HERE, "results", "b44_fig2bc.json")))
    sw = {w: (json.load(open(os.path.join(OUT, "sweep_%s.json" % w))) if os.path.exists(os.path.join(OUT, "sweep_%s.json" % w))
              else None) for w in ("consistency", "grid", "coherence")}
    hb = json.load(open(os.path.join(OUT, "b47_hierboot.json"))) if os.path.exists(os.path.join(OUT, "b47_hierboot.json")) else {}
    fig = {"prereg": "prereg/B47_phi2_finetunes_hierboot_consistency_prereg.md", "b": {}, "c": {}}
    b = fig["b"]
    b["floor_phi2"] = b44["b"]["floor_phi2"]
    b["coherence_b44"] = {k: v for k, v in b44["b"].items() if k.startswith("sweep_raw")}
    if sw["coherence"]:
        for enc in ("perm", "comm"):
            rows = sorted([x for x in sw["coherence"]["cells"] if x["enc"] == enc], key=lambda x: x["c"])
            b["coherence_b47/%s" % enc] = {
                "c": [x["c"] for x in rows], "acc_per_seed": [[r["acc"] for r in x["per_seed"]] for x in rows],
                "acc_mean": [x["acc_mean"] for x in rows], "different_mean": [x["geom_different_mean"] for x in rows],
                "same_mean": [x["geom_same_mean"] for x in rows],
                "source": "b47_out/sweep_coherence.json cells[enc=%s]" % enc}
    if sw["consistency"]:
        for st in ("mention", "form"):
            for enc in ("perm", "comm"):
                rows = sorted([x for x in sw["consistency"]["cells"] if x["structure"] == st and x["enc"] == enc],
                              key=lambda x: -x["rho"])
                b["consistency/%s/%s" % (st, enc)] = {
                    "rho": [x["rho"] for x in rows], "same_mean": [x["geom_same_mean"] for x in rows],
                    "different_mean": [x["geom_different_mean"] for x in rows],
                    "acc_per_seed": [[r["acc"] for r in x["per_seed"]] for x in rows], "acc_mean": [x["acc_mean"] for x in rows],
                    "dev_acc_mean": [x["dev_acc_mean"] for x in rows], "auroc_mean": [x["auroc_mean"] for x in rows],
                    "hop1_mean": [x["hop1_mean"] for x in rows], "hop2_mean": [x["hop2_mean"] for x in rows],
                    "both_mean": [x["both_mean"] for x in rows],
                    "source": "b47_out/sweep_consistency.json cells[structure=%s, enc=%s]" % (st, enc)}
    if sw["grid"]:
        b["grid/mention/perm"] = [{"c": x["c"], "rho": x["rho"], "acc_mean": x["acc_mean"],
                                   "acc_per_seed": [r["acc"] for r in x["per_seed"]], "different_mean": x["geom_different_mean"],
                                   "same_mean": x["geom_same_mean"], "hop1_mean": x["hop1_mean"]} for x in sw["grid"]["cells"]]
        b["grid_source"] = "b47_out/sweep_grid.json"
    pts = []
    for a in arms:
        if "different_mean" not in a:
            continue
        p = {k: a[k] for k in ("encoder", "label", "acc_mean", "acc_per_seed", "different_per_seed", "same_per_seed",
                               "different_mean", "same_mean", "source", "line", "geom_line") if k in a}
        for k in ("hop1_mean", "hop2_mean", "cleanup_line"):
            if k in a:
                p[k] = a[k]
        pts.append(p)
    b["points_C"] = pts
    b["points_b44"] = {k: b44["b"][k] for k in ("points_phi2", "points_qwen8")}
    for enc in ("phi2", "qwen8"):
        fig["c"][enc] = b44["c"][enc]
    fig["c"]["source"] = "results/b44_fig2bc.json c (B44 k-sweep: b44_out/k/<enc>_k<k>_<arm>.json)"
    # H-nest intervals of the 2c points from B44's per-instance outputs
    try:
        import b44_data as BD
        import b47_boot as HB
        for enc in ("phi2", "qwen8"):
            for row in fig["c"][enc]:
                k = row["k"]
                y = np.array([s["label"] for s in BD.ksplit(k, "test", struct=False)])
                for arm in ("name", "exact", "probe"):
                    pth = os.path.join(HERE, "b44_out", "k", "%s_k%d_%s.json" % (enc, k, arm))
                    if not os.path.exists(pth):
                        continue
                    if arm == "probe":
                        d = json.load(open(pth))
                        C = [((np.array(r["test_scores"]) > 0).astype(int) == y).astype(int) for r in d["runs"]]
                        c = HB.cell(C, [r["test_acc"] for r in d["runs"]], [r["seed"] for r in d["runs"]], pth)
                    else:
                        c = HB.mem(pth, "perm/named", y)
                    row["%s_hnest" % arm] = HB.pct(HB.draws([c], "hnest", 7000 + k)[0])
                    row["%s_recomputed_equals_logged" % arm] = c["recomputed_equals_logged"]
    except Exception as e:                                                    # reported, never silent
        fig["c"]["hnest_error"] = repr(e)
    fig["hierboot_accuracies"] = hb.get("accuracies", {})
    json.dump(fig, open(os.path.join(OUT, "b47_fig2bc.json"), "w"), indent=1)
    print("observed arms:", len(lf), "points:", len(pts), "2c hnest error:", fig["c"].get("hnest_error"))


if __name__ == "__main__":
    main()
