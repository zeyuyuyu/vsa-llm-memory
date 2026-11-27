"""B50 queue generator (one command per line; lanes of b50_worker.sh pop q/<worker>_L<i>).
usage: python b50_queues.py dev|cot|test <worker> <lane> [<lane2>]   (run in /path/to/work/b50)
  dev  : live dev runs of the three VSA seeds and the three DISC seeds with the gate forced open; the test-state cache
  cot  : COT2048 on the task-N and task-R test splits (B40's batch sizes)
  test : per DISC seed, gate open / p08 / msp / energy / maha / mahalw / maxlogit on the test split at the dev thresholds
"""
import json, sys

RD = "/path/to/work/b40/out"
R = "b50_run.py"
NV = {0: (f"{RD}/rd/N_vsa_e1000_s0", f"{RD}/dev/N_vsa_recon_lr1e-3_s0"),
      1: (f"{RD}/rd/N_vsa_e1000_s1", f"{RD}/fin/N_vsa_s1"), 2: (f"{RD}/rd/N_vsa_e1000_s2", f"{RD}/fin/N_vsa_s2")}
ND = {0: (f"{RD}/rd/N_disc_e300_s0", f"{RD}/dev/N_disc_lr1e-3_s0"),
      1: (f"{RD}/rd/N_disc_e300_s1", f"{RD}/fin/N_disc_s1"), 2: (f"{RD}/rd/N_disc_e300_s2", f"{RD}/fin/N_disc_s2")}


def emit(path, lines):
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")
    print(path, len(lines))


ph, wk = sys.argv[1], sys.argv[2]
lanes = sys.argv[3:]
if ph == "dev":
    a = [f"{R} eval --task N --arm vsa --split dev --layer 20 --bs 32 --reader {NV[s][0]} --ckpt {NV[s][1]} "
         f"--pred preds/dev/vsa_s{s}.jsonl" for s in (0, 1, 2)]
    a += [f"{R} eval --task N --arm discood --gate open --split dev --layer 20 --bs 32 --reader {ND[s][0]} --ckpt {ND[s][1]} "
          f"--pred preds/dev/discopen_s{s}.jsonl" for s in (0, 1, 2)]
    emit(f"q/{wk}_L{lanes[0]}", a)
    emit(f"q/{wk}_L{lanes[1]}", [f"{R} cache --task N --split test --layers 20 --bs 16"])
if ph == "cot":
    emit(f"q/{wk}_L{lanes[0]}", [f"{R} eval --task N --arm cot --split test --cot_bs 48 --cot_max_new 2048 --pred preds/N/cot2048_cot.jsonl",
                                 f"{R} eval --task R --arm cot --split test --cot_bs 32 --cot_max_new 2048 --pred preds/R/cot2048_cot.jsonl"])
if ph == "test":
    th = json.load(open("out/thresholds.json"))["seeds"]
    L = [[], []]
    for s in (0, 1, 2):
        for j, g in enumerate(("open", "p08", "msp", "energy", "maha", "mahalw", "maxlogit")):
            t = f" --thr={th[str(s)]['tau'][g]!r}" if g not in ("open", "p08") else ""  # '=' form: thresholds can be negative
            L[j % 2].append(f"{R} eval --task N --arm discood --gate {g}{t} --split test --layer 20 --bs 32 --reader {ND[s][0]} "
                            f"--ckpt {ND[s][1]} --pred preds/N/disc{g}_s{s}.jsonl")
    emit(f"q/{wk}_L{lanes[0]}", L[0])
    emit(f"q/{wk}_L{lanes[1]}", L[1])
