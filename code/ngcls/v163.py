"""V-163 (prereg in ledger.md, committed before this file): the V-95 imposed cross-talk sweep re-scored under the
decision rule the Table arms use, and a diagnostic of what limits the exact-row arm in arena C.

No language model is loaded: the parser arm reads no hidden states (v13.BindArm.vec, mode "parser"), so the hidden
states handed to it are a dummy tensor. Arena, binding chain and cone codebook are v13's and v95's own code, imported.

Part A (sweep). For each imposed cross-talk c and codebook seed s, the chain's final cosine is computed on the V-95 test
arena (v13.Arena(tok, 600, 5000 + s)) and scored three ways:
  fixed    V-95's rule: the untrained logit 20*cos - 4 > 0, i.e. positive iff cos > 0.2            (regression gate G0)
  fitted   v13.run()'s rule for every Table arm: the threshold that maximises accuracy on the training arena
           (v13.Arena(tok, 800, 42)) under the same codebook, applied to the test cosines
  auroc    area under the ROC curve of the test cosine against the label (no decision rule)
Part B (ceiling). At c = 0, on the Table configuration (codebook seeds 42/123/456, test arena v13.Arena(tok, 500, 4242)):
  G1       fitted-rule accuracy reproduces the banked parser arm of v13_logs/v13_fixed.json
  alias    first-hop clean-up errors split by the commutativity-alias condition: some other pair of the level has the
           query symbol as its FILLER and its role symbol is itself a candidate filler of that level.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/path/to/work/ngcls")
import v13                                                            # noqa: E402
from v95 import ConeArm                                               # noqa: E402

DEV = "cpu"
BAR = 0.615                                                           # V-95's bar: floor 0.515 + 0.10
FLOOR = 0.515


def cosines(arm, rows):
    """Final-hop cosine per instance, by the arm's own sims(); H is a dummy because the parser arm never reads it."""
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            b = rows[i:i + 64]
            out.append(arm.sims(torch.zeros(len(b), 1, 1), b, DEV).cpu())
    return torch.cat(out).numpy().astype(np.float64), np.array([r["label"] for r in rows])


def fixed_rule(arm, se, ye):
    """Exactly V-95: logits [-lp, lp] with the arm's untrained scale and bias, arg-max (ties go to class 0)."""
    lp = torch.tensor(se, dtype=torch.float32) * arm.scale.detach() + arm.bias.detach()
    pred = torch.stack([-lp, lp], dim=1).argmax(-1).numpy()
    return float((pred == ye).astype(np.float32).mean())


def fitted_rule(st, yt, se, ye):
    """Exactly v13.run().acc(): threshold = the training cosine that maximises training accuracy."""
    c = np.unique(st)
    thr = float(c[int(np.argmax([(((st >= t) == (yt == 1)).mean()) for t in c]))])
    return float((((se >= thr) == (ye == 1)).mean())), thr


def auroc(s, y):
    order = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); i = 0
    ss = s[order]
    while i < len(ss):                                                # average ranks over ties
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def hop1(arm, rows):
    """First-hop clean-up with the true query key, split by the alias condition."""
    n = {"alias": 0, "clean": 0}; err = {"alias": 0, "clean": 0}; onto = 0
    with torch.no_grad():
        for r in rows:
            pairs = [(aa, bb) for aa, bb, _, _ in r["ctx"][0]]
            q = r["q"]
            true = [b for a, b in pairs if a == q]
            if len(true) != 1:
                continue
            s1 = true[0]; fillers = {b for _, b in pairs}
            alias_roles = [a for a, b in pairs if b == q and a != q]
            cond = "alias" if any(a in fillers and a != s1 for a in alias_roles) else "clean"
            mem = torch.stack([v13.bind(arm.sym[b], arm.sym[a]) for a, b in pairs], 0).sum(0)
            hat = F.normalize(v13.unbind(mem, arm.sym[q]), dim=-1)
            cands = torch.stack([arm.sym[b] for _, b in pairs], 0)
            pick = pairs[int(F.cosine_similarity(cands, hat.unsqueeze(0)).argmax().item())][1]
            n[cond] += 1; err[cond] += int(pick != s1)
            onto += int(cond == "alias" and pick != s1 and pick in alias_roles)
    return {"n_alias": n["alias"], "err_alias": err["alias"], "n_clean": n["clean"], "err_clean": err["clean"],
            "err_onto_aliased_role": onto}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cs", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="v163_logs/v163.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("===== V-163: V-95 sweep under the fitted rule; arena-C ceiling diagnostic =====", flush=True)
    tr = v13.Arena(tok, 800, 42)
    out = {"curve": [], "table_config": []}
    for c in a.cs:
        rec = {"target_c": c, "crosstalk": [], "fixed": [], "fitted": [], "thr": [], "auroc": []}
        for s in a.seeds:
            torch.manual_seed(s); np.random.seed(s)
            te = v13.Arena(tok, 600, 5000 + s)
            m = ConeArm(s, c)
            st, yt = cosines(m, tr.rows); se, ye = cosines(m, te.rows)
            fa, thr = fitted_rule(st, yt, se, ye)
            rec["crosstalk"].append(m.realised_crosstalk()); rec["fixed"].append(fixed_rule(m, se, ye))
            rec["fitted"].append(fa); rec["thr"].append(thr); rec["auroc"].append(auroc(se, ye))
        for k in ("crosstalk", "fixed", "fitted", "auroc"):
            rec[k + "_mean"] = float(np.mean(rec[k]))
        out["curve"].append(rec)
        print("  c=%.2f  crosstalk %.4f | fixed %.4f %s | fitted %.4f %s | auroc %.4f"
              % (c, rec["crosstalk_mean"], rec["fixed_mean"], [round(x, 4) for x in rec["fixed"]],
                 rec["fitted_mean"], [round(x, 4) for x in rec["fitted"]], rec["auroc_mean"]), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)

    te = v13.Arena(tok, 500, 4242)
    for s in (42, 123, 456):
        m = v13.BindArm("parser", s)
        st, yt = cosines(m, tr.rows); se, ye = cosines(m, te.rows)
        fa, thr = fitted_rule(st, yt, se, ye)
        rec = {"seed": s, "fitted": fa, "thr": thr, "auroc": auroc(se, ye), "positive_rate": float(ye.mean())}
        rec.update(hop1(m, te.rows))
        out["table_config"].append(rec)
        print("  table config seed %d: fitted %.4f | hop-1 alias %d/%d err, clean %d/%d err, onto aliased role %d"
              % (s, fa, rec["err_alias"], rec["n_alias"], rec["err_clean"], rec["n_clean"],
                 rec["err_onto_aliased_role"]), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("V163-RUN-DONE", flush=True)


if __name__ == "__main__":
    main()
