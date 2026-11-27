"""V-131 Part B: give the fine-tune the tuning the manuscript admits it never had.

The bAbI task-2 comparator is LoRA at rank 8, two epochs, lr 1e-4 -- one point in hyperparameter space --
and the manuscript concedes "a larger or longer one may close part of the gap". This sweeps twelve
configurations at three seeds each and applies the P1 bar to the BEST of them.

`v50.py` is invoked as a subprocess rather than imported, so the instance set (775 expressible training and
420 expressible evaluation rows), the externally supplied ordering rule and the answer-candidate restriction
are byte-identical to the arm the manuscript reports; only --rank, --epochs and --lr move.

Selection is on the test set, deliberately and disclosed: this harness has no validation split, inventing one
would change the training set and break comparability, and selecting the OPPONENT's configuration on the set
it is scored on is generous to the opponent. A gap that survives that survives a fortiori.
"""
import argparse, itertools, json, os, subprocess, sys, time

PY = sys.executable
V50 = "/path/to/work/ngcls/v50.py"
HANDOVER = 1.0          # banked, bAbI task 2, all three seeds (v36/v37b)
BANKED_LORA = [0.8500, 0.6952380952380952, 0.8571428571428571]   # v50_task2_BANKED.json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--epochs", type=int, nargs="+", default=[2, 6])
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 3e-4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default="v131_logs/v131b.json")
    a = ap.parse_args()

    if os.environ.get("PYTHONHASHSEED") != "0":
        print("ABORT: PYTHONHASHSEED must be 0 (standing rule, defect 024d9e0). Got %r."
              % os.environ.get("PYTHONHASHSEED"), flush=True)
        sys.exit(2)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    os.makedirs("v131_logs/b_runs", exist_ok=True)

    grid = list(itertools.product(a.ranks, a.epochs, a.lrs))
    print("===== V-131 Part B: tuned LoRA opponent, bAbI task 2 =====", flush=True)
    print("  %d configurations x %d seeds; manuscript's point is rank 8 / 2 epochs / 1e-4 -> %s"
          % (len(grid), len(a.seeds), [round(x, 4) for x in BANKED_LORA]), flush=True)

    out = {"prereg": "6a2a52d", "handover": HANDOVER, "banked_lora_per_seed": BANKED_LORA,
           "grid": [{"rank": r, "epochs": e, "lr": l} for r, e, l in grid],
           "seeds": list(a.seeds), "runs": [], "selection": "best mean on the TEST set (disclosed handicap)"}

    for i, (rank, ep, lr) in enumerate(grid):
        tag = "r%d_e%d_lr%g" % (rank, ep, lr)
        dst = "v131_logs/b_runs/%s.json" % tag
        cmd = [PY, V50, "--task", "2", "--arms", "lora_direct",
               "--seeds"] + [str(s) for s in a.seeds] + [
              "--rank", str(rank), "--epochs", str(ep), "--lr", str(lr),
              "--device", a.device, "--out", dst]
        t0 = time.time()
        print("  [%2d/%d] %s ..." % (i + 1, len(grid), tag), flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True)
        mins = (time.time() - t0) / 60
        rec = {"rank": rank, "epochs": ep, "lr": lr, "tag": tag, "minutes": round(mins, 1),
               "returncode": p.returncode, "accs": None, "n_train": None, "n_eval": None}
        if p.returncode != 0 or not os.path.exists(dst):
            rec["stderr_tail"] = (p.stderr or "")[-400:]
            print("      FAILED rc=%d (%.1f min)" % (p.returncode, mins), flush=True)
        else:
            d = json.load(open(dst))
            rec["accs"] = d["arms"].get("lora_direct")
            rec["n_train"], rec["n_eval"] = d.get("n_train"), d.get("n_eval")
            m = sum(rec["accs"]) / len(rec["accs"]) if rec["accs"] else float("nan")
            print("      %s mean %.4f  seeds %s  (%.1f min, n_train %s / n_eval %s)"
                  % (tag, m, [round(x, 4) for x in rec["accs"]], mins, rec["n_train"], rec["n_eval"]),
                  flush=True)
        out["runs"].append(rec)
        json.dump(out, open(a.out, "w"), indent=1)

    json.dump(out, open(a.out, "w"), indent=1)
    print("V131B-DONE", flush=True)


if __name__ == "__main__":
    main()
