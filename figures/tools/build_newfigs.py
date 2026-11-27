"""Builds figdata/reasoning.json and figdata/conditions.json for main Figs 5 and 7 (added in r3).
Reads the neuro-symbolic session's banked outputs READ-ONLY (python3 -I -B, nothing written there).
Every value carries its source; values computed here are re-checked against the verified numbers in the verdicts.
    python3 -I -B tools/build_newfigs.py"""
import json, os, re, sys
NSR = "/path/to/vsassm/nsr"
sys.path.insert(0, NSR)
from nsr1_score import gold                     # noqa: E402
from nsr2_score import auroc, selective         # noqa: E402
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1, V2, V3 = (open(f"/path/to/vsassm/results/{n}_verdict.txt").read() for n in ("nsr1", "nsr2", "nsr3"))

def cite(text, needle, value, label):
    assert needle in text, (label, needle)
    line = [l for l in text.splitlines() if needle in l][0].strip()
    return {"value": value, "source": f"{label}: '{line[:160]}'"}

# ---------------- selective accuracy (NSR2 held-out split, min reliability over the answer's facts) ------------
logs = {}
for sc in ("warehouse", "biobank", "home", "ops"):
    for l in open(f"{NSR}/data/{sc}_nsr2.jsonl"):
        d = json.loads(l); logs[d["id"]] = d
Q = [json.loads(l) for l in open(f"{NSR}/out/nsr2_nsr2_records.jsonl")]
ok = [sorted(r["pred_det"]) == gold(logs[r["log"]], r["qtype"], r["target"]) for r in Q]
conf = [r["conf"]["min_rel"] for r in Q]
grid = [round(0.05 * k, 2) for k in range(2, 21)]
curve = [selective(conf, ok, c) for c in grid]
S = json.load(open(f"{NSR}/out/nsr2_score_nsr2.json"))
assert abs(sum(ok) / len(ok) - S["acc_det"]) < 1e-12
assert abs(selective(conf, ok, 0.7) - S["arms"]["min_rel"]["sel@70"]) < 1e-12
assert abs(auroc(conf, ok) - S["arms"]["min_rel"]["auroc"]) < 1e-12
assert "%.3f" % S["arms"]["min_rel"]["sel@70"] == "0.957" and "%.3f" % S["acc_det"] == "0.917"
selective_block = {"coverage": grid, "accuracy": curve, "n": len(Q), "wrong": len(Q) - sum(ok),
                   "auroc": S["arms"]["min_rel"]["auroc"], "all_answers": S["acc_det"],
                   "source": "computed from vsassm/nsr/out/nsr2_nsr2_records.jsonl (conf.min_rel) with nsr2_score.selective and "
                             "nsr1_score.gold; endpoints asserted equal to nsr/out/nsr2_score_nsr2.json (acc_det, min_rel sel@70, auroc)"}

# ---------------- one real traced answer (NSR1 held-out, arm 1) --------------------------------------------------
LID, QT, TG = "biobank-402-2", "freezer_down", "FRZ-02"
L = [json.loads(l) for l in open(f"{NSR}/data/biobank_nsr1.jsonl") if json.loads(l)["id"] == LID][0]
R = [json.loads(l) for l in open(f"{NSR}/out/held/nsr1_nsr1_arm1.jsonl")
     if json.loads(l)["log"] == LID and json.loads(l)["target"] == TG][0]
G = gold(L, QT, TG)
assert sorted(R["pred"]) == G
P = [json.loads(l) for l in open(f"{NSR}/out/held/nsr1_llm_problems_nsr1_arm3.jsonl")]
P = [p for p in P if p["log"] == LID and p["target"] == TG][0]["prompt"]
assert "Question: %s has defrosted. Which specimens are affected?" % TG in P
Ls = dict(L); Ls["truth"] = L["sor"]
sor_answer = gold(Ls, QT, TG)
hops = {}
for item, tr in R["trace"].items():
    hops[item] = [{"x": h[0], "rel": h[1], "y": h[2], "line": h[3],
                   "src": L["lines"][h[3]]["src"] if isinstance(h[3], int) else "catalog",
                   "kind": L["lines"][h[3]]["kind"] if isinstance(h[3], int) else "catalog",
                   "who": L["lines"][h[3]].get("who") if isinstance(h[3], int) else None,
                   "time": L["lines"][h[3]].get("time") if isinstance(h[3], int) else None,
                   "text": L["lines"][h[3]]["text"] if isinstance(h[3], int) else None} for h in tr]
trace_block = {"log": LID, "site": L["title"], "question": "%s has defrosted. Which specimens are affected?" % TG,
               "qtype": QT, "target": TG, "pred": R["pred"], "gold": G, "system_of_record_answer": sor_answer,
               "sor_place": {k: L["sor"].get(k) for k in R["pred"]}, "hops": hops,
               "source": f"vsassm/nsr/out/held/nsr1_nsr1_arm1.jsonl ({LID}, {TG}); log lines from vsassm/nsr/data/biobank_nsr1.jsonl; "
                         "gold and the system-of-record answer from nsr1_score.gold on the truth and sor states"}

# ---------------- read state against true state; the language model given the read state ------------------------
read_true = [
    {"id": "clutrr", "label": "CLUTRR", "read": {"value": 0.717, "source": "results.tex: 'This system reached an accuracy of $71.7\\%$'"},
     "true": {"value": 0.975, "source": "results.tex / appendix.tex: memory given the gold relations answers 97.5% of the 1,146 test stories"},
     "read_label": "relations written by Qwen3-8B", "true_label": "gold relations"},
    {"id": "impact", "label": "impact questions", "read": cite(V1, "| 1  RM state + VSA rule memory", 0.930, "results/nsr1_verdict.txt"),
     "true": cite(V1, "sanity: truth + graph query", 1.000, "results/nsr1_verdict.txt"),
     "read_label": "state read from the log", "true_label": "true state"},
    {"id": "window", "label": "past-window questions", "read": {"value": 0.874, "source": "results/nsr3_verdict.txt: history engine 0.874"},
     "true": {"value": 1.000, "source": "results/nsr3_verdict.txt: truth history 1.000"},
     "read_label": "history read from the log", "true_label": "true history"},
]
assert "0.874" in V3 and "1.000" in V3
llm_state = {"log": cite(V1, "Qwen3-32B", 0.278, "results/nsr1_verdict.txt (Qwen3-32B over the log, forced)"),
             "state": cite(V1, "| 3b Qwen3-32B thinking over the RM state as sentences", 0.826, "results/nsr1_verdict.txt"),
             "ours": cite(V1, "| 1  RM state + VSA rule memory", 0.930, "results/nsr1_verdict.txt"), "n": 230}
assert "0.278" in V1
json.dump({"_about": "Main Fig. 7 (reasoning over the stored state); built by tools/build_newfigs.py",
           "trace": trace_block, "read_true": read_true, "llm_state": llm_state, "selective": selective_block},
          open(os.path.join(HERE, "figdata/reasoning.json"), "w"), indent=1)

# ---------------- when the memory helps: memory against a LoRA fine-tune -------------------------------------------
A = " ".join(open(os.path.join(HERE, "appendix.tex")).read().split())
def tab(needle):
    assert needle in A, needle
    return f"appendix.tex: '{needle}'"
conditions = {"_about": "Main Fig. 5; values transcribed from the Supplementary tables tab:babi and tab:threearena and the bAbI "
                        "screening paragraph, each with the string it was read from (re-checked by make_figures.verify_sources)",
  "groups": [
   {"label": "single pass succeeds", "rows": [
     {"label": "bAbI task 1 (one hop)", "memory": [1.0, 1.0, 1.0], "lora": [0.9980, 1.0000, 1.0000], "prompt": 0.7920,
      "source": tab("Handover & 1.0000 (1.0000, 1.0000, 1.0000)") + "; " + tab("0.9993 (0.9980, 1.0000, 1.0000)")}]},
   {"label": "a tuned fine-tune learns the task", "rows": [
     {"label": "bAbI task 2 (two hops)", "memory": [1.0, 1.0, 1.0], "lora": [0.9714, 0.9738, 0.9714], "lora_first": [0.8500, 0.6952, 0.8571],
      "prompt": 0.3860, "source": tab("0.9722 (0.9714, 0.9738, 0.9714)") + "; " + tab("0.8008 (0.8500, 0.6952, 0.8571)")}]},
   {"label": "pairings given in each input", "rows": [
     {"label": "arena B (queried pairs held out)", "memory": [0.990, 1.000, 0.998], "lora": [0.980, 0.895, 0.510],
      "source": "appendix.tex: 'on B ($0.990$, $1.000$ and $0.998$)' ; 'reached $0.980$, $0.895$ and $0.510$'"},
     {"label": "arena C (pairing drawn per instance)", "memory": [0.932, 0.932, 0.938], "lora": [0.515, 0.518, 0.515],
      "source": "appendix.tex: 'on C ($0.932$, $0.932$ and $0.938$)' ; '($0.515$, $0.518$ and $0.515$; majority $0.515$)'"}]},
   {"label": "algebra not exact on the task", "rows": [
     {"label": "bAbI task 15", "memory": [0.836], "lora": [0.991], "source": tab("reached only $0.836$") + "; " + tab("reached $0.991$")},
     {"label": "bAbI task 4", "memory": [0.691], "lora": [1.000], "source": tab("Exact codes reached $0.691$ and the fine-tune reached $1.000$")}]},
  ]}
R_ = " ".join(open(os.path.join(HERE, "results.tex")).read().split())
for s in ("on B ($0.990$, $1.000$ and $0.998$) and on C ($0.932$, $0.932$ and $0.938$)", "reached $0.980$, $0.895$ and $0.510$"):
    assert s in A, s
assert "($0.515$, $0.518$ and $0.515$; majority $0.515$)" in A
json.dump(conditions, open(os.path.join(HERE, "figdata/conditions.json"), "w"), indent=1)
print("selective:", [round(x, 3) for x in curve])
print("trace:", R["pred"], "sor answer:", sor_answer, {k: L["sor"].get(k) for k in R["pred"]})
for it, hs in hops.items():
    for h in hs: print("  ", it, h["x"], "->", h["y"], h["line"], h["kind"], h["src"], h["who"], h["time"], (h["text"] or "")[:70])

# ---------------- multi-day records (NSR5): set-exact accuracy against the number of days --------------------------
V5 = open("/path/to/vsassm/results/nsr5_verdict.txt").read()
def row(prefix):
    l = [x for x in V5.splitlines() if x.startswith(prefix)]
    assert len(l) == 1, prefix
    cells = [c.strip() for c in l[0].split("|")[2:5]]
    vals = [float(c.split()[0]) for c in cells]
    f1 = [float(c.split("(")[1].rstrip(")")) for c in cells]
    return {"set_exact": vals, "micro_f1": f1, "source": "results/nsr5_verdict.txt: '%s'" % l[0].strip()[:150]}
multiday = {"_about": "Main Fig. 6d; set-exact accuracy on impact questions over records of 1, 4 and 8 days (NSR5), built by "
                      "tools/build_newfigs.py from results/nsr5_verdict.txt (scored once with nsr/nsr5_score.py)",
            "days": [1, 4, 8], "n": [230, 161, 69], "tokens_k": [9.5, 33.5, 69.8],
            "pipeline": row("| pipeline (reader + replay)"),
            "gpt_registered": row("| gpt-6.1-sol xhigh, registered"),
            "gpt_repaired": row("|   repaired         "),
            "opus_registered": row("| Opus 5.5 medium, registered"),
            "opus_normalised": row("|   normalised (= repaired+normalised)")}
assert multiday["pipeline"]["set_exact"] == [0.930, 0.801, 0.710]
assert multiday["gpt_registered"]["set_exact"] == [0.922, 0.677, 0.739]
assert multiday["gpt_repaired"]["set_exact"] == [0.943, 0.702, 0.783]
assert multiday["opus_normalised"]["set_exact"] == [0.935, 0.845, 0.754]
assert "(230)" in V5 and "(161)" in V5 and "(69)" in V5
json.dump(multiday, open(os.path.join(HERE, "figdata/multiday.json"), "w"), indent=1)
print("multiday:", {k: v["set_exact"] for k, v in multiday.items() if isinstance(v, dict) and "set_exact" in v})
