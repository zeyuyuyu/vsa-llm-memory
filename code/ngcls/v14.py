"""V-14 (prereg e293493): LOCATE the remedy's boundary by sweeping ONLY the number of symbols an example
requires the model to identify. Three banked points (few/32/64 -> 99%/26%/3%) show a boundary exists; this
finds where it is."""
import argparse, json, os, sys
import numpy as np
import torch
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/path/to/work/ngcls")
import v13 as V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8, 16, 24, 32])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="v14_logs/v14.json")
    a = ap.parse_args()
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                    torch_dtype=torch.float16).to(a.device).eval()
    for p in llm.parameters():
        p.requires_grad_(False)
    rows = []
    for k in a.ks:
        V.N_CTX = k                      # the ONLY thing that varies
        cell = {"k": k, "n_identify": k * V.HOPS}
        for arm in ["floor", "parser", "probe_argmax"]:
            accs = []
            for s in a.seeds:
                r = V.run(arm, s, tok, llm, a.device, a.n_train, a.n_test, a.epochs, 4, 3e-4)
                accs.append(r["acc"])
            cell[arm] = accs
            print("  k=%-3d %-13s %s mean=%.4f" % (k, arm, [round(x, 4) for x in accs],
                                                   float(np.mean(accs))), flush=True)
        fl, pa, pq = (float(np.mean(cell[x])) for x in ["floor", "parser", "probe_argmax"])
        cell["R"] = (pq - fl) / max(pa - fl, 1e-9)
        print("  k=%-3d  n_identify=%-3d  floor=%.3f parser=%.3f probe=%.3f  R=%.3f"
              % (k, cell["n_identify"], fl, pa, pq, cell["R"]), flush=True)
        rows.append(cell)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(rows, open(a.out, "w"), indent=1)

    print("\n=== V-14 SCORING (prereg e293493) ===", flush=True)
    Rs = [c["R"] for c in rows]
    print("  R(k): %s" % [(c["k"], round(c["R"], 3)) for c in rows], flush=True)
    inv = [(rows[i]["k"], round(Rs[i + 1] - Rs[i], 3)) for i in range(len(Rs) - 1)
           if Rs[i + 1] > Rs[i] + 0.05]
    print("  MONOTONE (one inversion <= 0.05 allowed) -> %s   inversions=%s"
          % (len(inv) <= 1, inv), flush=True)
    bracketed = Rs[0] >= 0.50 and Rs[-1] <= 0.20
    print("  BOUNDARY BRACKETED (R>=.50 at smallest k, R<=.20 at largest) -> %s" % bracketed, flush=True)
    kstar = None
    for i in range(len(Rs) - 1):
        if Rs[i] >= 0.50 >= Rs[i + 1] and Rs[i] > Rs[i + 1]:
            f = (Rs[i] - 0.50) / (Rs[i] - Rs[i + 1])
            kstar = rows[i]["k"] + f * (rows[i + 1]["k"] - rows[i]["k"])
            break
    print("  k* (R crosses 0.50) = %s  -> symbols identified at crossover = %s"
          % (("%.1f" % kstar) if kstar else "not bracketed",
             ("%.0f" % (kstar * V.HOPS)) if kstar else "n/a"), flush=True)
    print("  >>> V-14: %s" % ("BOUNDARY LOCATED" if (bracketed and len(inv) <= 1)
                              else "COMPOUNDING ACCOUNT NOT SUPPORTED"), flush=True)
    print("V14-DONE", flush=True)


if __name__ == "__main__":
    main()
