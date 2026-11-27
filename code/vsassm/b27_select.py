"""B27 selection on the dev split (rules fixed in prereg/B27_arena_referee_baselines_prereg.md, section "Selection").
Reads only tuning outputs (train + dev; the test split is never built in tuning mode) and writes the selected
configurations. No test number exists when this runs.

  python b27_select.py lf   <tune_dir> <out.json>     label-free arms: per arena, per family, then the primary family
  python b27_select.py lora <tune_dir> <out.json>     LoRA: stage rules (edge extension, rank stage, fit rule)
"""
import glob, json, os, sys

FAM_ORDER = ["orth", "vic", "nce"]


def best_epoch(epochs, key="dev"):
    """(best dev accuracy, epoch); ties go to the earliest epoch."""
    b, be = -1.0, None
    for e in epochs:
        v = e[key]["acc"] if isinstance(e[key], dict) else e[key]
        if v > b + 1e-12:
            b, be = v, e["epoch"]
    return b, be


def select_lf(tune_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(tune_dir, "*.json"))):
        d = json.load(open(p)); a = d["args"]; r = d["runs"][0]
        dev, ep = best_epoch(r["epochs"])
        rows.append({"file": os.path.basename(p), "arena": a["arena"], "reg": a["reg"], "lam": a["lam"],
                     "lr": a["lr"], "qc": a["qc"], "tau": a["tau"], "dev": dev, "epoch": ep,
                     "train_at_epoch": r["epochs"][ep - 1]["train_acc"]})
    out = {"configs": rows, "selected": {}}
    for ar in "ABC":
        sel = {}
        for fam in FAM_ORDER:
            c = [x for x in rows if x["arena"] == ar and x["reg"] == fam]
            if not c:
                continue
            top = max(x["dev"] for x in c)
            tied = [x for x in c if x["dev"] >= top - 0.002]
            # ties: smaller lambda, then lr 3e-4 (the published rate), then qc index (the published interface)
            tied.sort(key=lambda x: (x["lam"], 0 if abs(x["lr"] - 3e-4) < 1e-12 else 1, 0 if x["qc"] == "index" else 1))
            sel[fam] = tied[0]
        if sel:
            top = max(v["dev"] for v in sel.values())
            prim = [f for f in FAM_ORDER if f in sel and sel[f]["dev"] >= top - 0.002][0]
            out["selected"][ar] = {"families": sel, "primary": prim}
    return out


def select_lora_a2(tune_dir):
    """Amendment 2: one stage (rank 16, lr in {1e-3, 1e-4}, 30 epochs); the registered fit rule and tie rules."""
    rows = []
    for p in sorted(glob.glob(os.path.join(tune_dir, "*.json"))):
        d = json.load(open(p)); a = d["args"]; r = d["runs"][0]
        dev, ep = best_epoch(r["epochs"])
        rows.append({"file": os.path.basename(p), "arena": a["arena"], "rank": a["rank"], "lr": a["lr"],
                     "epochs": a["epochs"], "dev": dev, "epoch": ep,
                     "train_max": max(e["train"]["acc"] for e in r["epochs"] if "train" in e),
                     "train_final": r["epochs"][-1]["train"]["acc"],
                     "diverged": bool(r.get("diverged"))})
    out = {"configs": rows, "selected": {}}
    for ar in "ABC":
        c = [x for x in rows if x["arena"] == ar]
        if c:
            out["selected"][ar] = _pick(c)
    return out


def select_lora(tune_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(tune_dir, "*.json"))):
        d = json.load(open(p)); a = d["args"]; r = d["runs"][0]
        dev, ep = best_epoch(r["epochs"])
        rows.append({"file": os.path.basename(p), "arena": a["arena"], "rank": a["rank"], "lr": a["lr"],
                     "sched": a.get("sched", "cosine"), "epochs": a["epochs"], "dev": dev, "epoch": ep,
                     "train_max": max(e["train"]["acc"] for e in r["epochs"] if "train" in e),
                     "train_at_epoch": r["epochs"][ep - 1]["train"]["acc"] if "train" in r["epochs"][ep - 1] else None,
                     "diverged": bool(r.get("diverged"))})
    out = {"configs": rows, "selected": {}, "next": {}}
    for ar in "ABC":
        c = [x for x in rows if x["arena"] == ar and x["epochs"] == 12]
        s1 = [x for x in c if x["rank"] == 16]
        if not s1:
            continue
        pick = _pick(s1)
        # amendment 4: the edge rule and the rank stage apply within the selected schedule
        lrs = sorted(set(x["lr"] for x in s1 if x["sched"] == pick["sched"]))
        nxt = []
        if abs(pick["lr"] - min(lrs)) < 1e-15 and min(lrs) > 1.5e-5:
            nxt.append({"rank": 16, "lr": min(lrs) / 3.0, "epochs": 12, "sched": pick["sched"]})
        elif abs(pick["lr"] - max(lrs)) < 1e-15 and max(lrs) < 2e-3:
            nxt.append({"rank": 16, "lr": max(lrs) * 3.0, "epochs": 12, "sched": pick["sched"]})
        if not nxt:
            for rk in (4, 64):
                if not any(x["rank"] == rk and abs(x["lr"] - pick["lr"]) < 1e-15 and x["sched"] == pick["sched"]
                           for x in c):
                    nxt.append({"rank": rk, "lr": pick["lr"], "epochs": 12, "sched": pick["sched"]})
        allp = _pick(c)
        out["selected"][ar] = allp
        if not nxt and allp["train_max"] < 0.95:
            # extension stage: nothing fits in 12 epochs -> the configuration closest to fitting (highest training
            # accuracy at any epoch; ties -> higher dev, then rank 16, then smaller rate) again with 30 epochs
            ext = [x for x in rows if x["arena"] == ar and x["epochs"] == 30]
            if not ext:
                ok = [x for x in c if not x["diverged"]] or c
                close = sorted(ok, key=lambda x: (-x["train_max"], -x["dev"], 0 if x["rank"] == 16 else 1, x["lr"]))[0]
                nxt.append({"rank": close["rank"], "lr": close["lr"], "epochs": 30, "sched": close["sched"]})
            else:
                e = dict(ext[0], fit=ext[0]["train_max"] >= 0.95)
                if e["fit"]:
                    out["selected"][ar] = e
                out["selected"][ar]["extension"] = e
        out["next"][ar] = nxt
    return out


def _pick(c):
    fit = [x for x in c if x["train_max"] >= 0.95 and not x["diverged"]]
    pool = fit if fit else [x for x in c if not x["diverged"]] or c
    top = max(x["dev"] for x in pool)
    tied = [x for x in pool if x["dev"] >= top - 0.004]
    # ties: rank 16, then the smaller rate, then the registered schedule (cosine)
    tied.sort(key=lambda x: (0 if x["rank"] == 16 else 1, x["lr"], 0 if x.get("sched", "cosine") == "cosine" else 1))
    return dict(tied[0], fit=tied[0]["train_max"] >= 0.95)


if __name__ == "__main__":
    kind, d, outp = sys.argv[1:4]
    res = select_lf(d) if kind == "lf" else (select_lora_a2(d) if kind == "lora_a2" else select_lora(d))
    json.dump(res, open(outp, "w"), indent=1)
    print(json.dumps(res["selected"], indent=1))
    if kind == "lora":
        print("next:", json.dumps(res["next"]))
