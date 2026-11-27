"""B55 runner (prereg/B55_argmax_read_prereg.md). One process loads the frozen Llama-3.1-8B-Instruct once and runs a
list of test evaluations in order. Each evaluation is B40's cmd_eval (b40_run.py) unchanged in everything but the
module: the same test rows and prompts (b40_data / b40_core.build), the same order (longest prompt first), batch 32,
torch.manual_seed(12345) before the first batch (so the read noise is drawn exactly as in B40's runs on the same GPU
type), greedy generation of 6 new tokens, the same parser and the same per-item record (plus n_gen). b40_core.py,
b40_data.py, b40_run.py are imported unchanged.

Modules (all load B40's trained seeds: reader.pt, then best.pt, strict=False, as b40_run does):
  N vsa      B40's VSAN; digit clean-up threshold from the job: -inf = ARGMAX (the B55 arm), 0.5 = DE25 / B40
  N disc     B40's DiscN unchanged (softmax gate > 0.8)
  N discmsp  B40's DiscN with B50's MSP gate: open iff log p_max - log(1 - p_max) >= tau_s (B50 out/thresholds.json,
             computed exactly as b50_run.DiscNOOD.scores)
  R vsaarg   B40's VSAR with every read (both mentions of every fact and the question mention) cleaned up by argmax
             over the 50 entity codes BEFORE the memory is built (the B55 arm); the hop clean-up is unchanged
  R vsa      B40's VSAR unchanged (hop clean-up by argmax; mentions bound uncleaned)
  R disc     B40's DiscR unchanged
usage: b55_run.py plan --where a800|pace --out jobs.json
       b55_run.py run --jobs jobs.json [--lane i --lanes k]      (jobs i, i+k, ...; a job whose file exists is skipped)
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import b40_data as D
import b40_core as C
import b40_run as R40

ROOT = os.environ.get("B55_ROOT", "/path/to/work/b55")
B40 = os.environ.get("B40_ROOT", "/path/to/work/b40")
B50 = os.environ.get("B50_ROOT", "/path/to/work/b50")
RHOS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
LAYER = {"N": 20, "R": 17}  # B40 dev choices (out/dev_choices.json)
RD = {("N", "vsa"): "N_vsa_e1000", ("N", "disc"): "N_disc_e300", ("R", "vsa"): "R_vsa_e100", ("R", "disc"): "R_disc_e30"}
CK0 = {("N", "vsa"): "N_vsa_recon_lr1e-3", ("N", "disc"): "N_disc_lr1e-3", ("R", "vsa"): "R_vsa_lr1e-3",
       ("R", "disc"): "R_disc_lr1e-3"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def paths(task, base, s):  # B40's reader / write-back directories of seed s (b40_queues.py rdir / ck)
    rd = f"{B40}/out/rd/{RD[(task, base)]}_s{s}"
    ck = f"{B40}/out/dev/{CK0[(task, base)]}_s0" if s == 0 else f"{B40}/out/fin/{task}_{base}_s{s}"
    return rd, ck


# ------------------------------------------------------------------------------------------------ modules
class ArgLinear(nn.Linear):
    """B40's linear encoder followed by an argmax clean-up against the entity codebook: x -> K[argmax (W x) . K]."""

    def forward(self, x):
        v = F.linear(x, self.weight)
        K = self.codes.K
        return K[(v @ K.T).argmax(-1)]


class VSARArg(C.VSAR):
    def __init__(self, codes):
        super().__init__(codes)
        self.enc = ArgLinear(C.HID, C.VSA_DIM, bias=False)  # same parameter name 'enc.weight'
        self.enc.codes = codes


class DiscMSP(C.DiscN):
    def __init__(self, tau):
        super().__init__(thr=0.8)
        self.tau = tau

    def msp(self, h_last):  # = b50_run.DiscNOOD.scores(...)["msp"]
        lo, _, _ = self.logits(h_last)
        z = lo.double()
        top = z.max(-1)
        rest = z.clone()
        rest[torch.arange(len(z), device=z.device), top.indices] = -float("inf")
        return top.values - torch.logsumexp(rest, -1)

    def forward(self, h, ctx):
        w, _, info = super().forward(h, ctx)
        with torch.no_grad():
            s = self.msp(h[:, -1, :].float())
        gate = s >= self.tau
        for d, v, g in zip(info, s.tolist(), gate.tolist()):
            d["msp"], d["gate"] = float(v), bool(g)
        return w, gate, info


def make(job):
    t, arm, s = job["task"], job["arm"], job["seed"]
    base = "disc" if arm.startswith("disc") else "vsa"
    if t == "N":
        if arm == "vsa":
            mod = C.VSAN(C.NumEngine().to(C.DEV), digit_thr=job["digit_thr"])
        elif arm == "disc":
            mod = C.DiscN()
        elif arm == "discmsp":
            mod = DiscMSP(job["tau"])
    else:
        codes = C.EntCodes().to(C.DEV)
        mod = {"vsa": lambda: C.VSAR(codes), "vsaarg": lambda: VSARArg(codes), "disc": lambda: C.DiscR()}[arm]()
    mod = mod.to(C.DEV)
    rd, ck = paths(t, base, s)
    mod.load_state_dict(torch.load(f"{rd}/reader.pt"), strict=False)
    mod.load_state_dict(torch.load(f"{ck}/best.pt"), strict=False)
    return mod.eval()


# ------------------------------------------------------------------------------------------------ eval (B40's)
@torch.no_grad()
def eval_job(job, tok, model, cache):
    task = job["task"]
    if task not in cache:
        rows = R40.rows_for(task, "test")
        cache[task] = (rows, [C.build(tok, r, cot=False) for r in rows])
    rows, items = cache[task]
    mod = make(job)
    inj = C.Injector(model, LAYER[task])
    inj.read_noise = job["rho"]
    torch.manual_seed(12345)  # B40's --noise_seed default
    order = sorted(range(len(items)), key=lambda i: -len(items[i]["ids"]))
    res, bs, t0 = [None] * len(items), 32, time.time()
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        ids, am, pos, ctx, gold = C.collate(tok, [items[i] for i in idx])
        inj.arm(mod, ctx)
        out = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=6, do_sample=False,
                             temperature=None, top_p=None, pad_token_id=tok.pad_id,
                             return_dict_in_generate=True, output_logits=True)
        info = inj.info
        inj.armed = False
        first = out.logits[0].float()
        lp = torch.log_softmax(first, -1)
        gen = out.sequences[:, ids.shape[1]:]
        for j, i in enumerate(idx):
            txt = tok.decode(gen[j], skip_special_tokens=True)
            r = rows[i]
            pred = C.parse(task, txt, cot=False)
            res[i] = {"id": r["id"], "gold": D.gold(r), "pred": pred, "correct": pred.strip().lower() == D.gold(r).lower(),
                      "out": txt[:40], "argmax_ok": int(first[j].argmax()) == int(gold[j]), "ce": float(-lp[j, gold[j]]),
                      **({"op": r["op"], "x": r["x"], "y": r["y"]} if task == "N" else
                         {"depth": r["depth"], "size": r["size"]}),
                      "mod": info[j]}
    inj.handle.remove()
    tmp = job["pred"] + ".part"
    os.makedirs(os.path.dirname(job["pred"]), exist_ok=True)
    with open(tmp, "w") as f:
        for x in res:
            f.write(json.dumps(x) + "\n")
    os.replace(tmp, job["pred"])
    wall = time.time() - t0
    with open(f"{ROOT}/logs/gpu_hours.txt", "a") as f:
        f.write(json.dumps({"job": job["name"], "gpu_h": round(wall / 3600, 4), "host": os.uname().nodename,
                            "gpu": torch.cuda.get_device_name(0), "cvd": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "end": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
    log(job["name"], "acc", round(float(np.mean([x["correct"] for x in res])), 4), f"{wall:.0f}s")


# ------------------------------------------------------------------------------------------------ plan
def plan(where):
    tau = json.load(open(f"{B50}/out/thresholds.json"))
    J = []

    def add(task, arm, s, rho, name, **kw):
        J.append({"task": task, "arm": arm, "seed": s, "rho": rho, "name": f"{task}/{name}_s{s}_rho{rho}",
                  "pred": f"{ROOT}/preds/{task}/{name}_s{s}_rho{rho}.jsonl", **kw})

    for rho in RHOS:  # registered B55 arms first (clean level first), then the comparison re-runs (PACE only)
        for s in (0, 1, 2):
            add("N", "vsa", s, rho, "vsaarg", digit_thr=float("-inf"))
            add("R", "vsaarg", s, rho, "vsaarg")
            if rho > 0:
                add("N", "discmsp", s, rho, "discmsp", tau=float(tau["seeds"][str(s)]["tau"]["msp"]))
    if where == "pace":  # same-GPU comparison arms (B40's files are from A800s; noise draws depend on the GPU)
        for rho in RHOS:
            for s in (0, 1, 2):
                add("N", "vsa", s, rho, "vsathr", digit_thr=0.5)
                add("N", "disc", s, rho, "disc")
                add("R", "vsa", s, rho, "vsa")
                add("R", "disc", s, rho, "disc")
                if rho == 0:
                    add("N", "discmsp", s, rho, "discmsp", tau=float(tau["seeds"][str(s)]["tau"]["msp"]))
    return J


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd"); p.add_argument("--where", default="a800"); p.add_argument("--out", default="")
    p.add_argument("--jobs", default=""); p.add_argument("--lane", type=int, default=0); p.add_argument("--lanes", type=int, default=1)
    p.add_argument("--limit_split", default="")  # smoke only: 'train:16' runs the jobs on 16 TRAINING rows
    a = p.parse_args()
    if a.cmd == "plan":
        J = plan(a.where)
        json.dump(J, open(a.out, "w"), indent=0)
        print(len(J), "jobs")
        return
    J = json.load(open(a.jobs))[a.lane::a.lanes]
    os.makedirs(f"{ROOT}/logs", exist_ok=True)
    tok, model = C.load()
    cache = {}
    if a.limit_split:  # smoke on training rows: never touches the test split
        sp, n = a.limit_split.split(":")
        assert sp == "train"
        for t in ("N", "R"):
            rows = D.gen(t, "train")[:int(n)]
            cache[t] = (rows, [C.build(tok, r, cot=False) for r in rows])
    if a.limit_split:  # smoke: seed 0 of every job kind at rho 0 and 1, written under smoke/ (never preds/)
        J = [dict(j, pred=j["pred"].replace("/preds/", "/smoke/")) for j in J if j["seed"] == 0 and j["rho"] in (0.0, 1.0)]
    for j in J:
        if os.path.exists(j["pred"]):
            log("skip (exists)", j["name"])
            continue
        eval_job(j, tok, model, cache)


if __name__ == "__main__":
    main()
