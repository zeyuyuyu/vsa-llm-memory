"""V-21: does an arg-max onto the codebook REPAIR an inaccurate code? One projection, read two ways.

The observation this tests, and why it needs its own run. Across everything now on record the continuous
handover's recovery tracks the fitted cosine fidelity of its projection (.875 -> .94, .906 -> .976, .91 ->
~1.00, .66 -> .715) while the discrete handover sits at 1.00 throughout. That would say exactness buys
nothing when the code is already accurate and a great deal when it is not -- but the discrete and continuous
arms differ in TWO ways at once, the read (arg-max versus continuous) and the objective that produced the
code (cross-entropy on identity versus cosine fit to the codebook), so no existing measurement isolates the
arg-max. And the pattern was noticed after the fact, which is the failure mode this paper is about.

The design removes both problems. A single projection is fitted to the codebook outside the binding loss, at
a range of fitting budgets that produce a range of fidelities. At each budget the SAME projection is read two
ways: continuously (`frozen_cont`) and snapped to its nearest codebook row (`proj_argmax`). Same weights,
same fidelity, same chain, same seeds. The only difference is the arg-max.

Prediction, written before the run: the arg-max buys a lot at low fidelity and little at high fidelity. If it
buys nothing anywhere, the exactness claim is dead for good and the manuscript keeps its training-signal
framing unchanged.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/path/to/work/ngcls")
import v13                                                            # noqa: E402

_OrigArm = v13.BindArm
_OrigArena = v13.Arena
FIT_EPOCHS = [0, 1, 3, 10, 30]      # 0 = untouched random projection


class Arm(_OrigArm):
    """v13's arm plus one mode: the frozen_cont projection, snapped to its nearest codebook row."""

    def __init__(self, mode, cfg_seed, *a, **k):
        self._pa = (mode == "proj_argmax")
        super().__init__("frozen_cont" if self._pa else mode, cfg_seed, *a, **k)
        self.mode_public = mode

    def vec(self, H, sid, sp, device):
        if not self._pa:
            return super().vec(H, sid, sp, device)
        v = F.normalize(self.proj(self.sf(H, sp)), dim=-1)          # the identical continuous vector
        sym = self.sym.to(device)
        idx = int(F.cosine_similarity(sym, v.unsqueeze(0).expand_as(sym)).argmax().item())
        return sym[idx]                                              # ... snapped to an exact row


class LongArena(_OrigArena):
    def __init__(self, tok, n, seed, max_len=2048):
        super().__init__(tok, n, seed, max_len=max_len)
        bad = sum(1 for r in self.rows
                  if len(r["ctx"]) != v13.HOPS or any(len(l) != v13.N_CTX for l in r["ctx"]))
        if bad:
            raise SystemExit("TRUNCATION: %d/%d rows lost bindings. Aborting." % (bad, len(self.rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--n_ctx", type=int, default=32)
    ap.add_argument("--n_sym", type=int, default=256)
    ap.add_argument("--fits", type=int, nargs="+", default=FIT_EPOCHS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="v21_logs/v21.json")
    a = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                    torch_dtype=torch.float16).to(a.device).eval()
    for p in llm.parameters():
        p.requires_grad_(False)

    v13.Arena, v13.BindArm = LongArena, Arm
    v13.HOPS, v13.N_CTX, v13.N_SYM = a.hops, a.n_ctx, a.n_sym
    print("===== V-21  HOPS=%d N_CTX=%d N_SYM=%d VSA_DIM=%d =====" %
          (v13.HOPS, v13.N_CTX, v13.N_SYM, v13.VSA_DIM), flush=True)

    out = {"hops": a.hops, "n_ctx": a.n_ctx, "n_sym": a.n_sym, "cells": []}
    for arm in ["floor", "parser"]:
        accs = [v13.run(arm, s, tok, llm, a.device, a.n_train, a.n_test,
                        a.epochs, a.batch_size, a.lr)["acc"] for s in a.seeds]
        out[arm] = accs
        print("  %-13s %s mean=%.4f" % (arm, accs, float(np.mean(accs))), flush=True)
    fl, pa = float(np.mean(out["floor"])), float(np.mean(out["parser"]))
    H = pa - fl
    out["floor_mean"], out["parser_mean"], out["headroom"] = fl, pa, H
    print("  floor=%.4f parser=%.4f headroom=%.4f" % (fl, pa, H), flush=True)

    for fe in a.fits:
        v13.FIT_EPOCHS_OVERRIDE = fe
        cell = {"fit_epochs": fe}
        for arm in ["frozen_cont", "proj_argmax"]:
            accs = []
            for s in a.seeds:
                accs.append(v13.run(arm, s, tok, llm, a.device, a.n_train, a.n_test,
                                    a.epochs, a.batch_size, a.lr)["acc"])
            cell[arm] = accs
            print("  fit=%-3d %-13s %s mean=%.4f" % (fe, arm, [round(x, 4) for x in accs],
                                                     float(np.mean(accs))), flush=True)
        cell["R_cont"] = (float(np.mean(cell["frozen_cont"])) - fl) / H
        cell["R_pa"] = (float(np.mean(cell["proj_argmax"])) - fl) / H
        cell["gain_H"] = cell["R_pa"] - cell["R_cont"]
        print("  fit=%-3d R_cont=%.4f  R_proj_argmax=%.4f  arg-max gain=%.4f of headroom"
              % (fe, cell["R_cont"], cell["R_pa"], cell["gain_H"]), flush=True)
        out["cells"].append(cell)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)

    # ---- scorer, per the prereg, written before the run ----
    print("\n===== V-21 SCORING (bars registered before this run) =====", flush=True)
    lo, hi = out["cells"][0], out["cells"][-1]
    print("  lowest budget  fit=%d: R_cont=%.4f R_pa=%.4f gain=%.4f"
          % (lo["fit_epochs"], lo["R_cont"], lo["R_pa"], lo["gain_H"]), flush=True)
    print("  highest budget fit=%d: R_cont=%.4f R_pa=%.4f gain=%.4f"
          % (hi["fit_epochs"], hi["R_cont"], hi["R_pa"], hi["gain_H"]), flush=True)
    per_seed_lo = all(p > c for p, c in zip(sorted(lo["proj_argmax"]), sorted(lo["frozen_cont"])))
    print("  low-budget gain holds seed-for-seed: %s" % per_seed_lo, flush=True)
    gains = [c["gain_H"] for c in out["cells"]]
    print("  gain by budget: %s" % [round(g, 3) for g in gains], flush=True)
    if lo["gain_H"] >= 0.20 and abs(hi["gain_H"]) <= 0.05:
        v = "CONFIRMED (arg-max repairs an inaccurate code and is redundant on an accurate one)"
    elif all(abs(g) <= 0.05 for g in gains):
        v = "REFUTED (the arg-max buys nothing at any fidelity; the exactness claim stays withdrawn)"
    else:
        v = "INTERMEDIATE (report the measured gains; license neither reading)"
    print("  >>> V-21 VERDICT: %s" % v, flush=True)
    print("V21-DONE", flush=True)


if __name__ == "__main__":
    main()
