"""B40 dev-only noise calibration: read accuracy of the chosen seed-0 readers (VSA, DISC) on the cached DEV states under
the read noise of Injector (relative norm rho per position), to check that the registered rho grid spans the range
where reads degrade. No LLM forward, no write-back, dev split only.
usage: python b40_noisecal.py <dev_choices.json>"""
import json, sys
import torch
import b40_core as C
import b40_data as D
import b40_run as R

ch = json.load(open(sys.argv[1]))
torch.manual_seed(12345)
RH = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def noisy(H, rho):
    if rho == 0:
        return H
    e = torch.randn_like(H)
    return H + e / e.norm(dim=-1, keepdim=True) * H.norm(dim=-1, keepdim=True) * rho


out = {}
LN, LR = ch["N_layer"], ch["R_layer"]
Hn, mn = R.load_cache("N", "dev", LN)
eng = C.NumEngine().to(C.DEV)
for arm, e in (("vsa", ch["N_vsa_epochs"]), ("disc", ch["N_disc_epochs"])):
    mod = (C.VSAN(eng) if arm == "vsa" else C.DiscN()).to(C.DEV)
    mod.load_state_dict(torch.load(f"out/rd/N_{arm}_e{e}_s0/reader.pt"), strict=False)
    for rho in RH:
        out[f"N_{arm}_{rho}"] = R.read_acc_N(mod, arm, noisy(Hn, rho), mn, eng)["exact"]
Hr, mr = R.load_cache("R", "dev", LR)
codes = C.EntCodes().to(C.DEV)
ent = torch.tensor(mr["ent"])
for arm, e in (("vsa", ch["R_vsa_epochs"]), ("disc", ch["R_disc_epochs"])):
    mod = (C.VSAR(codes) if arm == "vsa" else C.DiscR()).to(C.DEV)
    mod.load_state_dict(torch.load(f"out/rd/R_{arm}_e{e}_s0/reader.pt"), strict=False)
    for rho in RH:
        with torch.no_grad():
            h = noisy(Hr, rho).to(C.DEV)
            p = mod.read_codes(h) if arm == "vsa" else mod.read_ids(h)
        out[f"R_{arm}_{rho}"] = (p.cpu() == ent).float().mean().item()
for k in ("N_vsa", "N_disc", "R_vsa", "R_disc"):
    print(k, " ".join(f"rho{r}:{out[f'{k}_{r}']:.3f}" for r in RH))
json.dump(out, open("out/noisecal_dev.json", "w"), indent=1)
