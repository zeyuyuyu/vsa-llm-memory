"""B31 queue files (prereg/B31_encoder_equalised_baselines_prereg.md). Each phase writes plain command lines for
b31_worker.sh into q/<phase>_<class>; heavy = one Qwen3-8B LoRA per GPU, medium = Phi-2 LoRA (2 per GPU), light = frozen-
state arms and probes (4 per GPU). Phases: p1 (test features, G0, untuned finals, tuning), p2 (after b31_select: tuned
finals), p2b (per encoder and arena once its LoRA grid is complete: LoRA finals, stage 2, stage 3), p3 (LoRA + identity
finals and deferred stage-3 finals). p2b and p3 are incremental and append to live queues under the workers' flock.

  python b31_queues.py p1 | p2a | p2b | p3   (run in /path/to/work/b31)
"""
import json, os, sys

SEEDS = "42 123 456"
ENCS = ["phi2", "qwen8"]
LR1 = {"phi2": [3e-5, 1e-4, 3e-4], "qwen8": [1e-4, 3e-4]}
LR2 = [1e-4, 3e-4]


def w(name, lines):
    os.makedirs("q", exist_ok=True)
    open(os.path.join("q", name), "w").write("".join(l + "\n" for l in lines))
    print(name, len(lines))


def lora_cmd(enc, ar, rank, lr, idl, mode, out, epochs=8, seeds="42"):
    return ("b31_lora.py --enc %s --arena %s --mode %s --rank %d --lr %g --id_lambda %d --epochs %d --seeds %s "
            "--out %s" % (enc, ar, mode, rank, lr, idl, epochs, seeds, out))


def p1():
    light, med, heavy = [], [], []
    for enc in ENCS:
        light.append("b31_prep.py feats --enc %s --items A:test B:test C:test" % enc)
    w("p1_feats", light)
    w("p1_g0", ["b31_arms.py --enc_model phi2 --arena %s --arm %s --mode g0 --seeds %s --out out/g0/%s_%s.json"
                % (ar, arm, SEEDS, ar, arm) for ar in "ABC" for arm in ("exact", "name", "bind")])
    for ar in "ABC":
        light.append("b31_arms.py --enc_model phi2 --arena %s --arm exact --mode final --enc perm --readings comm/named "
                     "--seeds %s --out out/arms_final/X_%s_exact.json" % (ar, SEEDS, ar))
        for enc in ENCS:
            light += [
                "b31_arms.py --enc_model %s --arena %s --arm name --mode final --epochs 4 --readings perm/index "
                "comm/named comm/index --seeds %s --out out/arms_final/%s_%s_name.json" % (enc, ar, SEEDS, enc, ar),
                "b31_arms.py --enc_model %s --arena %s --arm soft --mode final --fit_epochs 6 --readings comm/named "
                "perm/index --seeds %s --out out/arms_final/%s_%s_soft.json" % (enc, ar, SEEDS, enc, ar),
                "b31_arms.py --enc_model %s --arena %s --arm bind --mode final --enc comm --conv index --lr 3e-4 "
                "--epochs 10 --seeds %s --out out/arms_final/%s_%s_bindpub.json" % (enc, ar, SEEDS, enc, ar),
                "b31_probe.py --enc %s --arena %s --mode tune --out out/probe_tune/%s_%s.json" % (enc, ar, enc, ar)]
            for lr in (3e-4, 1e-3):
                light.append("b31_arms.py --enc_model %s --arena %s --arm bind --mode tune --lr %g --epochs 10 "
                             "--out out/arms_tune/%s_%s_bind_lr%g.json" % (enc, ar, lr, enc, ar, lr))
                for lam in (1, 10, 100):
                    light.append("b31_arms.py --enc_model %s --arena %s --arm nce --mode tune --lam %g --lr %g "
                                 "--epochs 10 --out out/arms_tune/%s_%s_nce_lam%g_lr%g.json" % (
                                     enc, ar, lam, lr, enc, ar, lam, lr))
            for rank in (8, 64):
                for lr in LR1[enc]:
                    c = lora_cmd(enc, ar, rank, lr, 0, "tune", "out/lora_tune/%s_%s_r%d_lr%g_id0.json" % (enc, ar, rank, lr))
                    (heavy if enc == "qwen8" else med).append(c)
    # longest first: arena A before B and C
    key = lambda c: (0 if "--arena A" in c else 1 if "--arena B" in c else 2)
    w("p1_light", [l for l in light if "feats" not in l])
    w("p1_medium", sorted(med, key=key))
    w("p1_heavy", sorted(heavy, key=key))


NGRID = {"phi2": 6, "qwen8": 4}


def append(name, lines):
    """Append to a live queue under the worker's flock (b31_worker.sh pops with flock on <queue>.lock)."""
    import fcntl
    os.makedirs("q", exist_ok=True)
    with open(os.path.join("q", name + ".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with open(os.path.join("q", name), "a") as f:
            f.write("".join(l + "\n" for l in lines))
    print(name, "+%d" % len(lines))


def done_keys(tag):
    p = os.path.join("q", "done_%s" % tag)
    return set(open(p).read().split()) if os.path.exists(p) else set()


def mark(tag, k):
    open(os.path.join("q", "done_%s" % tag), "a").write(k + "\n")


def p2a():
    """Tuned memory-arm and probe finals (after b31_select has the bind / nce / probe selections)."""
    sel = json.load(open("out/select.json"))
    light = []
    for enc in ENCS:
        for ar in "ABC":
            k = "%s_%s" % (enc, ar)
            b = sel["bind"][k]
            light.append("b31_arms.py --enc_model %s --arena %s --arm bind --mode final --lr %g --epochs 10 --seeds %s "
                         "--out out/arms_final/%s_%s_bind.json" % (enc, ar, b["lr"], SEEDS, enc, ar))
            n = sel["nce"][k]
            light.append("b31_arms.py --enc_model %s --arena %s --arm nce --mode final --lam %g --lr %g --epochs 10 "
                         "--seeds %s --out out/arms_final/%s_%s_nce.json" % (enc, ar, n["lam"], n["lr"], SEEDS, enc, ar))
            json.dump(sel["probe"][k], open("out/probe_sel_%s.json" % k, "w"))
            light.append("b31_probe.py --enc %s --arena %s --mode final --select out/probe_sel_%s.json "
                         "--out out/probe_final/%s_%s.json" % (enc, ar, k, enc, ar))
    w("p2_light", light)


def p2b():
    """Per (encoder, arena) whose stage-1 LoRA grid is complete: equalised-LoRA finals (unless the arena goes to
    stage 3), LoRA + identity tuning (stage 2) at the selected rank, and stage 3 where no configuration fitted."""
    sel = json.load(open("out/select.json"))
    dk = done_keys("p2b")
    for enc in ENCS:
        for ar in "ABC":
            k = "%s_%s" % (enc, ar)
            if k in dk or len(sel["grids"]["lora"].get(k, [])) < NGRID[enc] or k not in sel["lora"]:
                continue
            L = sel["lora"][k]
            q, q3 = [], []
            stage3 = ar in "AC" and not L["any_fit"]
            if not stage3:
                for s in SEEDS.split():
                    q.append(lora_cmd(enc, ar, L["rank"], L["lr"], 0, "final", "out/lora_final/%s_%s_id0_s%s.json" % (
                        enc, ar, s), epochs=L.get("epochs", 8), seeds=s))
            for lr in LR2:
                q.append(lora_cmd(enc, ar, L["rank"], lr, 1, "tune", "out/lora_tune/%s_%s_r%d_lr%g_id1.json" % (
                    enc, ar, L["rank"], lr)))
            if stage3:
                grid = sel["grids"]["lora"][k]
                c3 = max(grid, key=lambda c: (c["train_max"], c["dev"], -c["rank"], -c["lr"]))
                q3.append(lora_cmd(enc, ar, c3["rank"], c3["lr"], 0, "tune", "out/lora_tune3/%s_%s_r%d_lr%g_id0.json" % (
                    enc, ar, c3["rank"], c3["lr"]), epochs=24))
            append("p2_heavy" if enc == "qwen8" else "p2_medium", q3 + q)
            mark("p2b", k)


def p3():
    """Per (encoder, arena) whose LoRA + identity grid is complete (and, for stage-3 arenas, whose stage-3 run is
    done): LoRA + identity finals and the deferred equalised-LoRA finals."""
    sel = json.load(open("out/select.json"))
    dk = done_keys("p3")
    for enc in ENCS:
        for ar in "ABC":
            k = "%s_%s" % (enc, ar)
            if k in dk or len(sel["grids"]["loraid"].get(k, [])) < len(LR2) or k not in sel["lora"]:
                continue
            L0 = sel["lora"][k]
            stage3 = ar in "AC" and not sel["grids"]["lora"].get(k) is None and \
                not any(c["fit"] for c in sel["grids"]["lora"][k])
            if stage3 and k not in sel["stage3"]:
                continue
            q = []
            L = sel["loraid"][k]
            for s in SEEDS.split():
                q.append(lora_cmd(enc, ar, L["rank"], L["lr"], 1, "final", "out/lora_final/%s_%s_id1_s%s.json" % (
                    enc, ar, s), epochs=L.get("epochs", 8), seeds=s))
            if stage3:
                for s in SEEDS.split():
                    q.append(lora_cmd(enc, ar, L0["rank"], L0["lr"], 0, "final", "out/lora_final/%s_%s_id0_s%s.json" % (
                        enc, ar, s), epochs=24 if L0.get("stage") == 3 else 8, seeds=s))
            append("p3_heavy" if enc == "qwen8" else "p3_medium", q)
            mark("p3", k)


if __name__ == "__main__":
    {"p1": p1, "p2a": p2a, "p2b": p2b, "p3": p3}[sys.argv[1]]()
