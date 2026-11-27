"""B40 queue generator (one command per line; run by b40_worker.sh). Phases follow the prereg's dev selection and test
protocol. usage: python b40_queues.py <phase> [key=value ...]   (writes q/<phase>_* files)"""
import sys, os

A = dict(kv.split("=") for kv in sys.argv[2:])
ph = sys.argv[1]
os.makedirs("q", exist_ok=True)
R = "b40_run.py"


def w(name, lines):
    open(f"q/{name}", "w").write("\n".join(lines) + "\n")
    print(name, len(lines))


if ph == "C":  # reader length grid at the chosen layers + full R train cache
    LN, LR = A["LN"], A["LR"]
    w("C_cache", [f"{R} cache --task R --split train --layers {LR} --bs 16"])
    lines = []
    for e in (100, 300, 1000):
        lines.append(f"{R} reader --task N --arm vsa --layer {LN} --epochs {e} --eval_every {e} --seed 0 --out out/rd/N_vsa_e{e}_s0")
    for e in (30, 100, 300):
        lines.append(f"{R} reader --task N --arm disc --layer {LN} --epochs {e} --eval_every {e} --seed 0 --out out/rd/N_disc_e{e}_s0")
    w("C_readN", lines)
    lines = []
    for e in (100, 300, 1000):
        lines.append(f"{R} reader --task R --arm vsa --layer {LR} --epochs {e} --eval_every {e} --seed 0 --out out/rd/R_vsa_e{e}_s0")
    for e in (30, 100, 300):
        lines.append(f"{R} reader --task R --arm disc --layer {LR} --epochs {e} --eval_every {e} --seed 0 --out out/rd/R_disc_e{e}_s0")
    w("C_readR", lines)

if ph == "D":  # write-back / LoRA / E2E dev grids, seed 0
    LN, LR = A["LN"], A["LR"]
    rd = {k: A[k] for k in ("NV", "ND", "RV", "RD")}  # chosen reader dirs
    L = []
    for lr in ("1e-3", "3e-4"):
        for init in ("recon", "zero"):
            L.append(f"{R} train --task N --arm vsa --layer {LN} --reader {rd['NV']} --dec_init {init} --lr {lr} --steps 1000 --eval_every 250 --seed 0 --out out/dev/N_vsa_{init}_lr{lr}_s0")
        L.append(f"{R} train --task N --arm disc --layer {LN} --reader {rd['ND']} --lr {lr} --steps 1000 --eval_every 250 --seed 0 --out out/dev/N_disc_lr{lr}_s0")
        L.append(f"{R} train --task N --arm lora1l --layer {LN} --lr {lr} --steps 1000 --eval_every 250 --seed 0 --out out/dev/N_lora1l_lr{lr}_s0")
        L.append(f"{R} train --task R --arm vsa --layer {LR} --reader {rd['RV']} --lr {lr} --steps 1000 --eval_every 250 --seed 0 --out out/dev/R_vsa_lr{lr}_s0")
        L.append(f"{R} train --task R --arm disc --layer {LR} --reader {rd['RD']} --lr {lr} --steps 1000 --eval_every 250 --seed 0 --out out/dev/R_disc_lr{lr}_s0")
    w("D_mod", L)
    L = []
    for lr in ("1e-4", "3e-4", "1e-3"):
        L.append(f"{R} train --task N --arm lora --lr {lr} --steps 3000 --eval_every 500 --seed 0 --out out/dev/N_lora_lr{lr}_s0")
        L.append(f"{R} train --task R --arm lora --lr {lr} --steps 3000 --eval_every 500 --seed 0 --out out/dev/R_lora_lr{lr}_s0")
    w("D_lora", L)
    L = []
    for lr in ("1e-3", "3e-3"):
        for arm in ("e2e_vsa", "e2e_tab", "e2e_const"):
            L.append(f"{R} train --task R --arm {arm} --layer {LR} --lr {lr} --steps 3000 --eval_every 500 --seed 0 --out out/dev/R_{arm}_lr{lr}_s0")
    w("D_e2e", L)

# ------------------------------------------------------------------ test phase (after the dev choices)
if ph in ("Dsoft", "T", "E"):
    import json
    ch = json.load(open("out/dev_choices.json"))
    LN, LR = ch["N_layer"], ch["R_layer"]
    rdir = lambda t, arm, s: f"out/rd/{t}_{arm}_e{ch[f'{t}_{arm}_epochs']}_s{s}"

    def cfg(pre):  # chosen dev run suffix, e.g. 'recon_lr1e-3' or 'lr3e-4'
        return ch[pre]

    def lr_of(pre):
        return cfg(pre).split("lr")[1]

    def ck(t, arm, s):  # seed 0 = the chosen dev run; seeds 1, 2 = out/fin
        return f"out/dev/{t}_{arm}_{cfg(f'{t}_{arm}_')}_s0" if s == 0 else f"out/fin/{t}_{arm}_s{s}"

if ph == "Dsoft":
    w("Dsoft", [f"{R} eval --task R --arm vsa --layer {LR} --reader {rdir('R', 'vsa', 0)} --ckpt {ck('R', 'vsa', 0)} --split dev "
                f"--soft_beta {b} --bs 32 --pred out/devpred/R_vsasoft_b{b}.jsonl" for b in (10, 30, 100)])

if ph == "T":
    L = []
    for s in (1, 2):
        init = cfg("N_vsa_").split("_")[0]
        L.append(f"{R} train --task N --arm vsa --layer {LN} --reader {rdir('N', 'vsa', s)} --dec_init {init} --lr {lr_of('N_vsa_')} --steps 1000 --eval_every 250 --seed {s} --out {ck('N', 'vsa', s)}")
        L.append(f"{R} train --task N --arm disc --layer {LN} --reader {rdir('N', 'disc', s)} --lr {lr_of('N_disc_')} --steps 1000 --eval_every 250 --seed {s} --out {ck('N', 'disc', s)}")
        L.append(f"{R} train --task N --arm lora1l --layer {LN} --lr {lr_of('N_lora1l_')} --steps 1000 --eval_every 250 --seed {s} --out {ck('N', 'lora1l', s)}")
        L.append(f"{R} train --task R --arm vsa --layer {LR} --reader {rdir('R', 'vsa', s)} --lr {lr_of('R_vsa_')} --steps 1000 --eval_every 250 --seed {s} --out {ck('R', 'vsa', s)}")
        L.append(f"{R} train --task R --arm disc --layer {LR} --reader {rdir('R', 'disc', s)} --lr {lr_of('R_disc_')} --steps 1000 --eval_every 250 --seed {s} --out {ck('R', 'disc', s)}")
    w("T_mod", L)
    L = []
    for s in (1, 2):
        for arm in ("e2e_vsa", "e2e_tab", "e2e_const"):
            L.append(f"{R} train --task R --arm {arm} --layer {LR} --lr {lr_of(f'R_{arm}_')} --steps 3000 --eval_every 500 --seed {s} --out {ck('R', arm, s)}")
    w("T_e2e", L)
    w("T_loraN", [f"{R} train --task N --arm lora --lr {lr_of('N_lora_')} --steps 3000 --eval_every 500 --seed {s} --out {ck('N', 'lora', s)}" for s in (1, 2)]
      if ch.get("N_lora_") else [])
    w("T_loraR", [f"{R} train --task R --arm lora --lr {lr_of('R_lora_')} --steps 3000 --eval_every 500 --seed {s} --out {ck('R', 'lora', s)}" for s in (1, 2)]
      if ch.get("R_lora_") else [])

if ph == "E":
    beta = A.get("BETA")
    E = []
    for t, L_ in (("N", LN), ("R", LR)):
        E.append(f"{R} eval --task {t} --arm none --split test --bs 32 --pred preds/{t}/base.jsonl")
        for s in (0, 1, 2):
            for arm in (("vsa", "disc", "lora1l") if t == "N" else ("vsa", "disc")):
                rd = f"--reader {rdir(t, arm, s)}" if arm in ("vsa", "disc") else ""
                E.append(f"{R} eval --task {t} --arm {arm} --layer {L_} {rd} --ckpt {ck(t, arm, s)} --split test --bs 32 --pred preds/{t}/{arm}_s{s}.jsonl")
            if t == "R":
                E.append(f"{R} eval --task R --arm vsa --layer {LR} --reader {rdir('R', 'vsa', s)} --ckpt {ck('R', 'vsa', s)} --split test --bs 32 --soft_beta {beta} --pred preds/R/vsasoft_s{s}.jsonl")
    w("E_mod", E)
    E = []
    for s in (0, 1, 2):
        for arm in ("e2e_vsa", "e2e_tab", "e2e_const"):
            E.append(f"{R} eval --task R --arm {arm} --layer {LR} --ckpt {ck('R', arm, s)} --split test --bs 32 --pred preds/R/{arm}_s{s}.jsonl")
            if arm != "e2e_const":
                E.append(f"{R} eval --task R --arm {arm} --layer {LR} --ckpt {ck('R', arm, s)} --split test --bs 32 --shuffle_facts 1 --pred preds/R/{arm}_s{s}_shuf.jsonl")
    w("E_e2e", E)
    E = []
    for t, L_ in (("N", LN), ("R", LR)):
        for s in (0, 1, 2):
            for rho in ("0.25", "0.5", "1.0", "2.0", "4.0", "8.0"):
                for arm in ("vsa", "disc"):
                    E.append(f"{R} eval --task {t} --arm {arm} --layer {L_} --reader {rdir(t, arm, s)} --ckpt {ck(t, arm, s)} --split test --bs 32 --noise {rho} --pred preds/{t}/{arm}_s{s}_rho{rho}.jsonl")
                if t == "R":
                    E.append(f"{R} eval --task R --arm vsa --layer {LR} --reader {rdir('R', 'vsa', s)} --ckpt {ck('R', 'vsa', s)} --split test --bs 32 --soft_beta {beta} --noise {rho} --pred preds/R/vsasoft_s{s}_rho{rho}.jsonl")
    w("E_noise", E)
    w("E_cot", [f"{R} eval --task N --arm cot --split test --cot_bs 48 --pred preds/N/cot.jsonl",
                f"{R} eval --task R --arm cot --split test --cot_bs 32 --pred preds/R/cot.jsonl"])
    w("E_lora", [f"{R} eval --task {t} --arm lora --ckpt {ck(t, 'lora', s)} --split test --bs 32 --pred preds/{t}/lora_s{s}.jsonl"
                 for t in ("N", "R") for s in (0, 1, 2)])

# ------------------------------------------------------------------ amendment 2: depth-3 label-free arms
if ph in ("T3", "E3"):
    import json
    ch = json.load(open("out/dev_choices.json"))
    ck3 = lambda arm, s: f"out/dev/R3_{arm}_{ch[f'R3_{arm}_']}_s0" if s == 0 else f"out/fin/R3_{arm}_s{s}"
if ph == "T3":
    w("T3", [f"{R} train --task R --arm {arm} --layer {ch['R_layer']} --hops 3 --lr {ch[f'R3_{arm}_'].split('lr')[1]} --steps 3000 --eval_every 500 --seed {s} --out {ck3(arm, s)}"
             for s in (1, 2) for arm in ("e2e_vsa", "e2e_tab", "e2e_const")])
if ph == "E3":
    E = []
    for s in (0, 1, 2):
        for arm in ("e2e_vsa", "e2e_tab", "e2e_const"):
            E.append(f"{R} eval --task R --arm {arm} --hops 3 --layer {ch['R_layer']} --ckpt {ck3(arm, s)} --split test --bs 32 --pred preds/R/e2e3_{arm[4:]}_s{s}.jsonl")
            if arm != "e2e_const":
                E.append(f"{R} eval --task R --arm {arm} --hops 3 --layer {ch['R_layer']} --ckpt {ck3(arm, s)} --split test --bs 32 --shuffle_facts 1 --pred preds/R/e2e3_{arm[4:]}_s{s}_shuf.jsonl")
    w("E3", E)
