"""B65 checks (prereg 'Checks from other angles'), run on the box in $B65_ROOT after the test:
  1. reader reproduction: LAST lr-1e-3 reader snapshots vs B40's separately trained readers (weights and dev reads)
  2. write-back reproduction: LAST VSA-PUB / DISC dev curves at lr 1e-3 and 3e-4 vs B40's dev runs (same code path)
  3. dev-optimum-interior: every chosen write-back lr is not an endpoint of its final grid
  4. determinism: the three seed-0 LAST test evaluations re-run (preds_rerun/) vs the registered files
  5. second scorer: b40_check.py on every registered test file
usage: python b65_check.py   -> prints a report (saved by the caller to results/b65_check.txt)
"""
import glob, json, os, subprocess, sys
import torch

B40 = os.environ.get("B40_ROOT", "/path/to/work/b40")
ch = json.load(open("out/choices.json"))
print("1. reader reproduction (LAST, lr 1e-3, seed 0; snapshot of one run vs B40's separate runs)")
for arm, eps, key in (("vsa", (100, 300, 1000), "enc.weight"), ("disc", (30, 100, 300), "head.weight")):
    for e in eps:
        a = torch.load(f"out/rd/last_{arm}_lr1e-3_s0/e{e}/reader.pt")[key]
        b = torch.load(f"{B40}/out/rd/N_{arm}_e{e}_s0/reader.pt")[key]
        r40 = json.load(open(f"{B40}/out/rd/N_{arm}_e{e}_s0/reader.json"))["final"]["exact"]
        h = [x for x in json.load(open(f"out/rd/last_{arm}_lr1e-3_s0/reader.json"))["hist"] if x["epoch"] == e][0]
        mine = h["pub"] if arm == "vsa" else h["disc"]
        print(f"   {arm} e{e}: max |W - W_B40| = {float((a - b).abs().max()):.3e}; dev read {mine:.5f} vs B40 {r40:.5f}")
print("2. write-back reproduction (LAST, seed 0; B40's readers = these readers if 1. is exact)")
for mine, theirs in (("last_vsapub_lr1e-3_s0", "N_vsa_recon_lr1e-3_s0"), ("last_vsapub_lr3e-4_s0", "N_vsa_recon_lr3e-4_s0"),
                     ("last_disc_lr1e-3_s0", "N_disc_lr1e-3_s0"), ("last_disc_lr3e-4_s0", "N_disc_lr3e-4_s0")):
    a = json.load(open(f"out/dev/{mine}/train.json"))
    b = json.load(open(f"{B40}/out/dev/{theirs}/train.json"))
    ca = [round(h["dev"], 4) for h in a["hist"]]; cb = [round(h["dev"], 4) for h in b["hist"]]
    print(f"   {mine}: {ca} | B40 {theirs}: {cb} | identical: {ca == cb}")
print("3. dev-optimum-interior")
for k, v in sorted(ch["wb"].items()):
    g = sorted(ch["wb_grid"][k], key=float)
    print(f"   {k}: chosen {v['lr']} in grid {g}: {'INTERIOR' if v['lr'] not in (g[0], g[-1]) else 'EDGE'} ({v['status']})")
print("4. determinism (seed-0 LAST test evaluations re-run)")
for arm in ("vsapub", "vsabm", "disc"):
    f1, f2 = f"preds/last/{arm}_s0.jsonl", f"preds_rerun/{arm}_s0.jsonl"
    if not os.path.exists(f2):
        print(f"   {arm}: re-run missing"); continue
    A = [json.loads(l) for l in open(f1)]; B = [json.loads(l) for l in open(f2)]
    same_pred = sum(a["pred"] == b["pred"] for a, b in zip(A, B)); same_mod = sum(a["mod"] == b["mod"] for a, b in zip(A, B))
    print(f"   {arm}: predictions identical {same_pred}/{len(A)}, module records identical {same_mod}/{len(A)}")
print("5. second scorer (b40_check.py)")
files = sorted(glob.glob("preds/*/*_s[012].jsonl"))
out = subprocess.run([sys.executable, "b40_check.py"] + files, capture_output=True, text=True).stdout
print("\n".join("   " + l for l in out.strip().splitlines()))
