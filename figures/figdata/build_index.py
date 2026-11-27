"""Build figdata_v2/: copy the archived run artifacts and emit INDEX.json (panel -> [{label,file,key,value,manuscript,match}]).

Every value is read from JSON via the recorded `key` expression (evaluated with d = json.load(file), plus helpers
mean/np). Nothing is hand-typed except the manuscript literals, which are what the values are checked against.
"""
import json, os, re, shutil
import numpy as np

SRC = "/path/to/scratch"
OUT = "/path/to/paper_v54/figdata_v2"
ROOT_D = "/path/to/work/ngcls"
ROOT_B = "/path/to/work/sync_h20/rewardbench_experiment/zeyu/hybrid_neurosymbolic"

# relative path -> source root ("D" = ngcls, "B" = hybrid_neurosymbolic)
FILES = {
    # arena A (original variable-tracking pipeline, lives in hybrid_neurosymbolic)
    "v1_logs/frozen_llm.json": "B", "v1_logs/vsa_hidden.json": "B", "v1_logs/vsa_parser.json": "B",
    "v2_logs/v2.json": "B",
    "v4_logs/v4_perocc.json": "B", "v4_logs/v4_consistent.json": "B", "v4_logs/v4_extract.json": "B",
    "v8_logs/v8.json": "B",
    "v9_logs/v9_gpu1_rerun_partial.json": "B",
    "v11_logs/v18_spanfix.json": "B",
    # ngcls
    "runs/v9/v9.json": "D", "runs/v9/v9.log": "D",
    "v6_logs/v6.json": "D", "v7_logs/v7_last.json": "D", "v12_logs/v12.json": "D",
    "v13_logs/floor_survey.json": "D", "v13_logs/v13_fixed.json": "D",
    "v14_logs/v14_fixed.json": "D", "v15_logs/v15.json": "D", "v15b_logs/v15b.json": "D",
    "v21_logs/v21.json": "D", "v23_logs/v23.json": "D",
    "v35_logs/v35.json": "D", "v36_logs/v36.json": "D", "v37_logs/v37b.json": "D",
    "v39_logs/v39.json": "D", "v39_logs/v39b.json": "D", "v40_logs/v40.json": "D",
    "v41_logs/v41.json": "D", "v42_logs/v42.json": "D", "v48_logs/v48.json": "D",
    "v49_logs/v49.json": "D", "v50_logs/v50_task2_BANKED.json": "D", "v50_logs/v50_task1_VOID_locoffset.json": "D",
    "v57_logs/v57.json": "D",
    "v62_logs/v62_BANKED.json": "D", "v63_logs/v63_PROVISIONAL.json": "D", "v63b_logs/v63b_BANKED.json": "D",
    "v64_logs/v64_BANKED.json": "D",
    "v65_logs/v65_BANKED.json": "D", "v67_logs/v67_BANKED.json": "D", "v69_logs/v69_BANKED.json": "D",
    "v72b_logs/v72b_BANKED.json": "D", "v73_logs/v73_BANKED.json": "D",
    "v79_logs/v79_BANKED.json": "D", "v79_logs/v79a_BANKED.json": "D",
    "v95_logs/v95_scored.json": "D", "v100_logs/v100_scored.json": "D",
    "v101_logs/v101_scored.json": "D", "v102_logs/v102_scored.json": "D",
    "crosstalk_audit2.json": "D", "geometry_banked.json": "D",
}

os.makedirs(OUT, exist_ok=True)
copied = []
for rel, root in FILES.items():
    src = os.path.join(SRC, root, rel)
    dst = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(rel)

_cache = {}
def load(rel):
    if rel not in _cache:
        _cache[rel] = json.load(open(os.path.join(OUT, rel)))
    return _cache[rel]

mean = lambda xs: float(np.mean(xs))
def rows(d, **kw):
    return [r for r in d if all(r.get(k) == v for k, v in kw.items())]
def accs(d, **kw):
    return [r["acc"] for r in rows(d, **kw)]
NS = {"mean": mean, "np": np, "rows": rows, "accs": accs}


def _dec(lit):
    m = re.search(r"\.(\d+)", lit)
    return len(m.group(1)) if m else 0

def _match_scalar(v, lit):
    lit = lit.strip().rstrip("%")
    p = _dec(lit)
    return abs(round(float(v), p) - float(lit)) < 1e-9 or abs(float(v) - float(lit)) <= 0.5 * 10 ** (-p) + 1e-9

def match(value, manuscript):
    if manuscript is None:
        return None
    if isinstance(manuscript, list):
        if not isinstance(value, (list, tuple)) or len(value) != len(manuscript):
            return False
        return all(_match_scalar(v, m) for v, m in zip(value, manuscript))
    if isinstance(value, (list, tuple)):
        return False
    return _match_scalar(value, manuscript)


INDEX = {}
def add(panel, label, file, key, manuscript, note=None):
    d = load(file)
    # eval of author-written key expressions over the loaded JSON (not user input); the key string is the audit trail
    value = eval(key, dict(NS, d=d))
    if isinstance(value, np.generic):
        value = float(value)
    if isinstance(value, (list, tuple)):
        value = [float(x) if isinstance(x, (int, float, np.generic)) else x for x in value]
    e = {"label": label, "file": file, "key": key, "value": value, "manuscript": manuscript,
         "match": match(value, manuscript)}
    if note:
        e["note"] = note
    INDEX.setdefault(panel, []).append(e)
    return e

def unresolved(panel, label, manuscript, searched):
    INDEX.setdefault("UNRESOLVED", []).append(
        {"panel": panel, "label": label, "manuscript": manuscript, "searched": searched})


# ------------------------------------------------------------------ F1b natural text (bAbI 1 / 2)
P = "F1b_natural_text"
add(P, "bAbI1 prompting zero-shot, per seed", "v49_logs/v49.json", 'd["cells"]["task1/zeroshot/all"]["acc"]', ["0.792"] * 3)
add(P, "bAbI1 prompting zero-shot (mean)", "v49_logs/v49.json", 'mean(d["cells"]["task1/zeroshot/all"]["acc"])', "0.792")
add(P, "bAbI1 prompting four-shot, per seed", "v49_logs/v49.json", 'd["cells"]["task1/fewshot/all"]["acc"]', None)
add(P, "bAbI1 prompting four-shot (mean)", "v49_logs/v49.json", 'mean(d["cells"]["task1/fewshot/all"]["acc"])', "0.763")
add(P, "bAbI1 LoRA fine-tune, per seed", "v50_logs/v50_task1_VOID_locoffset.json", 'd["arms"]["lora_direct"]', None,
    note="File is named VOID because its lora_through_vsa arm (0.08 x3) was voided for a location-offset bug; the lora_direct arm is the one the manuscript quotes (0.999). No other artifact holds task-1 LoRA.")
add(P, "bAbI1 LoRA fine-tune (mean)", "v50_logs/v50_task1_VOID_locoffset.json", 'mean(d["arms"]["lora_direct"])', "0.999")
add(P, "bAbI1 discrete handover (banked)", "v50_logs/v50_task1_VOID_locoffset.json", 'd["banked"]["handover"]', "1.000")
add(P, "bAbI1 discrete handover, per seed (v36 permuted cell)", "v36_logs/v36.json", 'd["cells"][0]["chain"]', ["1.000"] * 3)
add(P, "bAbI1 last-token probe floor", "v50_logs/v50_task1_VOID_locoffset.json", 'd["banked"]["probe"]', "0.771")
add(P, "bAbI1 last-token probe floor (v35 floor field)", "v35_logs/v35.json", 'd["floor"]', "0.771")
add(P, "bAbI1 prompting zero-shot (banked in v50)", "v50_logs/v50_task1_VOID_locoffset.json", 'd["banked"]["prompt"]', "0.792")
add(P, "bAbI2 prompting zero-shot, per seed", "v49_logs/v49.json", 'd["cells"]["task2/zeroshot/all"]["acc"]', ["0.386"] * 3)
add(P, "bAbI2 prompting four-shot, per seed", "v49_logs/v49.json", 'd["cells"]["task2/fewshot/all"]["acc"]', None)
add(P, "bAbI2 LoRA fine-tune, per seed", "v50_logs/v50_task2_BANKED.json", 'd["arms"]["lora_direct"]', ["0.850", "0.695", "0.857"])
add(P, "bAbI2 LoRA fine-tune (mean)", "v50_logs/v50_task2_BANKED.json", 'mean(d["arms"]["lora_direct"])', "0.801")
add(P, "bAbI2 discrete handover (banked)", "v50_logs/v50_task2_BANKED.json", 'd["banked"]["handover"]', "1.000")
add(P, "bAbI2 discrete handover, per seed", "v37_logs/v37b.json", 'd["arms"]["probe_argmax"]', ["1.000"] * 3)
add(P, "bAbI2 through-the-loss projection (frozen), per seed", "v37_logs/v37b.json", 'd["arms"]["spans"]', None)
add(P, "bAbI2 through-the-loss projection (frozen, mean)", "v37_logs/v37b.json", 'mean(d["arms"]["spans"])', "0.502")
add(P, "bAbI2 through-the-loss, encoder unfrozen (LoRA through VSA), per seed", "v50_logs/v50_task2_BANKED.json", 'd["arms"]["lora_through_vsa"]', None)
add(P, "bAbI2 through-the-loss, encoder unfrozen (mean)", "v50_logs/v50_task2_BANKED.json", 'mean(d["arms"]["lora_through_vsa"])', "0.600")
add(P, "bAbI2 mean-pool probe floor, per seed", "v37_logs/v37b.json", 'd["arms"]["floor"]', None)
add(P, "bAbI2 mean-pool probe floor (mean)", "v37_logs/v37b.json", 'd["floor"]', "0.203")
add(P, "bAbI2 last-token probe floor, per seed", "v48_logs/v48.json", 'd["floor_last_token"]', None)
add(P, "bAbI2 last-token probe floor (mean)", "v48_logs/v48.json", 'd["floor_mean"]', "0.245")
add(P, "bAbI2 last-token probe floor (banked in v50)", "v50_logs/v50_task2_BANKED.json", 'd["banked"]["probe"]', "0.245")
add(P, "bAbI2 label prior (majority, eval)", "v48_logs/v48.json", 'd["majority_eval"]', "0.205")

# ------------------------------------------------------------------ F2a label priors (tab:invalid)
P = "F2a_label_priors"
TASKS = [("variable_tracking", "Variable tracking", "0.464"), ("rule_priority", "Rule priority", "0.706"),
         ("state_update", "State updates", "0.659"), ("binding_test", "Entity binding", "0.573")]
for t, name, lit in TASKS:
    add(P, f"{name}: frozen-LLM probe, per seed", "v2_logs/v2.json",
        f'[r["test_per_task"]["{t}"] for r in rows(d, arm="llm_only")]', None)
    add(P, f"{name}: frozen-LLM probe (mean)", "v2_logs/v2.json",
        f'mean([r["test_per_task"]["{t}"] for r in rows(d, arm="llm_only")])', lit,
        note=("artifact .5718 rounds to .572; manuscript prints .573 (ledger 57406f1 quotes .5726). Within verify_paper3's .0015 tolerance, off by one in the last digit at manuscript precision." if t == "binding_test" else None))
# majority / P(label=1) columns: ledger only
for name, lit_p, lit_m in [("Variable tracking", "0.498", "0.502"), ("Rule priority", "0.327", "0.673"),
                           ("State updates", "0.342", "0.658"), ("Entity binding", "0.513", "0.513")]:
    unresolved(P, f"{name}: P(label=1) on eval split", lit_p,
               "No JSON artifact. Only source is ngcls/ledger.md lines 7280-7287 (2026-08-19 BENCHMARK DIAGNOSIS, commit 57406f1), computed inline from run_compositional.py _gen_* over 8,000 samples; verify_paper2/3 hard-code the ledger values. Searched: grep 0.5022/0.6729/0.5871 across ngcls and hybrid_neurosymbolic *.json/*.txt/*.log (hits only in unrelated cifar/fmnist/dissect runs).")
    unresolved(P, f"{name}: majority baseline on eval split", lit_m, "Same as P(label=1): ledger-only (ngcls/ledger.md 7283-7286, 57406f1).")
unresolved(P, "per-task majority predictor overall accuracy", "0.5871", "Ledger-only (ngcls/ledger.md line 7287).")
# probe on rebuilt tasks (tab:valid single-hop arenas): v7_last probe test per seed
for arena, name, lit in [("rule_priority", "Rule priority (rebuilt)", "0.998"), ("state_update", "State updates (rebuilt)", "1.000"),
                         ("binding_test", "Entity binding (rebuilt)", "0.996")]:
    add(P, f"{name}: probe per seed", "v7_logs/v7_last.json",
        f'[a["test"] for a in rows(d, arena="{arena}")[0]["arms"]["probe"]]', None)
    add(P, f"{name}: probe (mean)", "v7_logs/v7_last.json",
        f'mean([a["test"] for a in rows(d, arena="{arena}")[0]["arms"]["probe"]])', lit)
    add(P, f"{name}: majority", "v7_logs/v7_last.json", f'rows(d, arena="{arena}")[0]["majority"]', None)

# ------------------------------------------------------------------ F2b per-arena arms (tab:valid)
P = "F2b_per_arena_arms"
VALID = {  # arena -> (name, majority, probe, mlp, vsa, hd)
    "variable_tracking": ("Variable tracking", "0.512", ["0.790", "0.766", "0.785"], ["0.785", "0.786", "0.795"], ["0.745", "0.736", "0.720"], "0.629"),
    "rule_priority": ("Rule priority", "0.517", ["1.000", "0.995", "0.999"], None, None, None),
    "state_update": ("State updates", "0.502", ["1.000", "1.000", "1.000"], None, None, None),
    "binding_test": ("Entity binding", "0.503", ["0.995", "0.997", "0.995"], None, None, None),
    "babi1": ("bAbI 1", "0.187", ["0.752", "0.759", "0.742"], ["0.751", "0.725", "0.721"], ["0.744", "0.780", "0.731"], "0.750"),
    "babi2": ("bAbI 2", "0.187", ["0.303", "0.324", "0.262"], ["0.287", "0.275", "0.308"], ["0.261", "0.167", "0.203"], "0.249"),
    "babi3": ("bAbI 3", "0.215", ["0.191", "0.158", "0.186"], None, None, None),
    "babi16": ("bAbI 16", "0.257", ["0.546", "0.533", "0.555"], ["0.526", "0.567", "0.542"], ["0.513", "0.532", "0.526"], "0.532"),
}
v7 = load("v7_logs/v7_last.json")
v7_arenas = [r["arena"] for r in v7]
for arena, (name, maj, probe, mlp, vsa, hd) in VALID.items():
    if arena not in v7_arenas:
        unresolved(P, f"{name}: tab:valid row", {"majority": maj, "probe": probe}, f"arena key '{arena}' not in v7_last.json (has {v7_arenas})")
        continue
    add(P, f"{name}: majority", "v7_logs/v7_last.json", f'rows(d, arena="{arena}")[0]["majority"]', maj)
    for arm, lit in (("probe", probe), ("mlp", mlp), ("vsa", vsa)):
        r = rows(v7, arena=arena)[0]
        if arm in r["arms"]:
            add(P, f"{name}: {arm} test per seed", "v7_logs/v7_last.json",
                f'[a["test"] for a in rows(d, arena="{arena}")[0]["arms"]["{arm}"]]', lit)
        elif lit is not None:
            unresolved(P, f"{name}: {arm}", lit, f"arm '{arm}' absent from v7_last.json arena {arena}")
V12 = {"variable_tracking": "variable_tracking", "babi1": "babi1", "babi2": "babi2", "babi16": "babi16"}
for arena, (name, maj, probe, mlp, vsa, hd) in VALID.items():
    if hd is None:
        continue
    k = V12[arena]
    add(P, f"{name}: HD reported column (raw/online) per seed", "v12_logs/v12.json", f'rows(d, arena="{k}")[0]["hd"]["raw/online"]', None)
    add(P, f"{name}: HD reported column (raw/online, mean)", "v12_logs/v12.json", f'mean(rows(d, arena="{k}")[0]["hd"]["raw/online"])', hd)
    add(P, f"{name}: HD single-pass (raw/single)", "v12_logs/v12.json", f'mean(rows(d, arena="{k}")[0]["hd"]["raw/single"])',
        {"variable_tracking": "0.515", "babi1": "0.656", "babi2": "0.166", "babi16": "0.381"}[arena])
    add(P, f"{name}: v12 probe per seed", "v12_logs/v12.json", f'rows(d, arena="{k}")[0]["probe"]', None)
HD_CFG = {"raw/single": "0.430", "raw/retrain": "0.362", "raw/online": "0.540", "rp_sign/single": "0.431", "rp_sign/retrain": "0.330", "rp_sign/online": "0.520"}
for cfg, lit in HD_CFG.items():
    add(P, f"HD six configurations: {cfg} mean over 4 arenas", "v12_logs/v12.json",
        f'mean([mean(r["hd"]["{cfg}"]) for r in d])', lit)
    add(P, f"HD six configurations: {cfg} per-arena means (VT, bAbI1, bAbI2, bAbI16)", "v12_logs/v12.json",
        f'[mean(r["hd"]["{cfg}"]) for r in d]', None)
add(P, "HD six configurations: range of means", "v12_logs/v12.json",
    '[min(mean([mean(r["hd"][c]) for r in d]) for c in d[0]["hd"]), max(mean([mean(r["hd"][c]) for r in d]) for c in d[0]["hd"])]', ["0.330", "0.540"])
add(P, "linear probe mean over the same 4 arenas", "v12_logs/v12.json", 'mean([mean(r["probe"]) for r in d])', "0.593")

# ------------------------------------------------------------------ F2c pooling
P = "F2c_pooling"
add(P, "Variable tracking: mean-pool probe per seed", "v6_logs/v6.json", 'rows(d, arena="variable_tracking")[0]["probe"]["mean"]', None)
add(P, "Variable tracking: mean-pool probe (mean)", "v6_logs/v6.json", 'mean(rows(d, arena="variable_tracking")[0]["probe"]["mean"])', "0.656")
add(P, "Variable tracking: last-token probe per seed", "v6_logs/v6.json", 'rows(d, arena="variable_tracking")[0]["probe"]["last"]', None)
add(P, "Variable tracking: last-token probe (mean)", "v6_logs/v6.json", 'mean(rows(d, arena="variable_tracking")[0]["probe"]["last"])', "0.787")
add(P, "Variable tracking: max-pool probe per seed", "v6_logs/v6.json", 'rows(d, arena="variable_tracking")[0]["probe"]["max"]', None)
add(P, "bAbI 1: mean-pool probe per seed", "v6_logs/v6.json", 'rows(d, arena="babi_task1")[0]["probe"]["mean"]', None)
add(P, "bAbI 1: mean-pool probe (mean)", "v6_logs/v6.json", 'mean(rows(d, arena="babi_task1")[0]["probe"]["mean"])', "0.358")
add(P, "bAbI 1: last-token probe per seed", "v6_logs/v6.json", 'rows(d, arena="babi_task1")[0]["probe"]["last"]', None)
add(P, "bAbI 1: last-token probe (mean)", "v6_logs/v6.json", 'mean(rows(d, arena="babi_task1")[0]["probe"]["last"])', "0.752")
add(P, "bAbI 1: max-pool probe per seed", "v6_logs/v6.json", 'rows(d, arena="babi_task1")[0]["probe"]["max"]', None)

# ------------------------------------------------------------------ F2d placement (tab:placement)
P = "F2d_placement"
PLACE = ["0.936", "0.654", "0.909", "0.615", "0.659", "0.499", "0.601", "0.474", "0.659", "0.536", "0.594", "0.528"]
for i, lit in enumerate(PLACE):
    add(P, f"row {i}: floor (mean of 2 seeds)", "v13_logs/floor_survey.json", f'd[{i}]["floor"]', lit,
        note=("artifact .52749997 (float32 of .5275) rounds to .527; manuscript .528 = round-half-up of .5275. Not a data discrepancy." if i == 11 else None))
    add(P, f"row {i}: floor per seed", "v13_logs/floor_survey.json", f'd[{i}]["floor_per_seed"]', None)
    add(P, f"row {i}: config (hops, n_ctx, query_at_end)", "v13_logs/floor_survey.json", f'[d[{i}]["hops"], d[{i}]["n_ctx"], d[{i}]["query_at_end"]]', None)

# ------------------------------------------------------------------ F3a three arenas (tab:threearena)
P = "F3a_three_arenas"
# arena A
add(P, "A floor per seed", "v1_logs/frozen_llm.json", 'accs(d)', None)
add(P, "A floor (mean)", "v1_logs/frozen_llm.json", 'mean(accs(d))', "0.509")
add(P, "A parser per seed", "v4_logs/v4_perocc.json", 'accs(d, epsilon=0.0)', None)
add(P, "A parser (mean)", "v4_logs/v4_perocc.json", 'mean(accs(d, epsilon=0.0))', "0.918")
add(P, "A relabelled per seed", "v4_logs/v4_consistent.json", 'accs(d, epsilon=1.0)', None)
add(P, "A relabelled (mean)", "v4_logs/v4_consistent.json", 'mean(accs(d, epsilon=1.0))', "0.869")
add(P, "A through-the-loss per seed", "v8_logs/v8.json", 'accs(d)', None)
add(P, "A through-the-loss (mean)", "v8_logs/v8.json", 'mean(accs(d))', "0.501")
add(P, "A discrete handover per seed", "v9_logs/v9_gpu1_rerun_partial.json", 'accs(d, arm="extract_then_bind")', None)
add(P, "A discrete handover (mean)", "v9_logs/v9_gpu1_rerun_partial.json", 'mean(accs(d, arm="extract_then_bind"))', "0.914")
add(P, "A frozen continuous projection per seed (V-9 Arm A best)", "runs/v9/v9.json", '[r["armA"]["best"] for r in d]', None,
    note="Arm A = projection frozen, 2 trainable readout scalars (V-9, prereg 7c90962). Init (untrained scalars) .629/.629/.633. Ledger 10101: '.673 is the frozen projection WITH its two readout scalars trained'.")
e = add(P, "A frozen continuous projection (mean of best)", "runs/v9/v9.json", 'mean([r["armA"]["best"] for r in d])', "0.670",
        note="MISMATCH: artifact mean .6733 rounds to .673, manuscript tab:threearena prints 0.670 (its 39% recovery is consistent with .670; the artifact gives 40%).")
add(P, "A frozen continuous projection, init per seed", "runs/v9/v9.json", '[r["armA"]["init"] for r in d]', None)
add(P, "A recovery: discrete handover (%)", "v9_logs/v9_gpu1_rerun_partial.json",
    '100*(mean(accs(d, arm="extract_then_bind")) - 0.509)/(0.918 - 0.509)', "99", note="uses floor .509 and parser .918 from v1/v4 (literals as rounded in the table)")
add(P, "A recovery: continuous (%)", "runs/v9/v9.json", '100*(mean([r["armA"]["best"] for r in d]) - 0.509)/(0.918 - 0.509)', "39",
    note="artifact gives 40.2%; manuscript 39% follows from the printed 0.670")
# arena B
for arm, name, lit in [("floor", "floor", "0.815"), ("parser", "parser", "0.996"), ("consistent", "relabelled", "0.994"),
                       ("spans", "through-the-loss", "0.507"), ("probe_argmax", "discrete handover", "0.996"), ("frozen_cont", "frozen continuous", "0.985")]:
    add(P, f"B {name} per seed", "v11_logs/v18_spanfix.json", f'accs(d, arm="{arm}")',
        ["0.990", "1.000", "0.998"] if arm == "probe_argmax" else None)
    add(P, f"B {name} (mean)", "v11_logs/v18_spanfix.json", f'mean(accs(d, arm="{arm}"))', lit)
add(P, "B recovery: discrete handover (%)", "v11_logs/v18_spanfix.json",
    '100*(mean(accs(d, arm="probe_argmax")) - mean(accs(d, arm="floor")))/(mean(accs(d, arm="parser")) - mean(accs(d, arm="floor")))', "100")
add(P, "B recovery: continuous (%)", "v11_logs/v18_spanfix.json",
    '100*(mean(accs(d, arm="frozen_cont")) - mean(accs(d, arm="floor")))/(mean(accs(d, arm="parser")) - mean(accs(d, arm="floor")))', "94")
# arena C
for arm, name, lit in [("floor", "floor", "0.515"), ("parser", "parser", "0.934"), ("consistent", "relabelled", "0.929"),
                       ("spans", "through-the-loss", "0.483"), ("probe_argmax", "discrete handover", "0.934"), ("frozen_cont", "frozen continuous", "0.924")]:
    add(P, f"C {name} per seed", "v13_logs/v13_fixed.json", f'accs(d, arm="{arm}")',
        ["0.932", "0.932", "0.938"] if arm == "probe_argmax" else None)
    add(P, f"C {name} (mean)", "v13_logs/v13_fixed.json", f'mean(accs(d, arm="{arm}"))', lit)
add(P, "C recovery: discrete handover (%)", "v13_logs/v13_fixed.json",
    '100*(mean(accs(d, arm="probe_argmax")) - mean(accs(d, arm="floor")))/(mean(accs(d, arm="parser")) - mean(accs(d, arm="floor")))', "100")
add(P, "C recovery: continuous (%)", "v13_logs/v13_fixed.json",
    '100*(mean(accs(d, arm="frozen_cont")) - mean(accs(d, arm="floor")))/(mean(accs(d, arm="parser")) - mean(accs(d, arm="floor")))', "98")
# LoRA on B and C, ten-epoch extension
add(P, "LoRA on B per seed", "v15_logs/v15.json", 'accs(d, arena="B")', ["0.980", "0.895", "0.510"])
add(P, "LoRA on B (mean)", "v15_logs/v15.json", 'mean(accs(d, arena="B"))', "0.795")
add(P, "LoRA on C per seed", "v15_logs/v15.json", 'accs(d, arena="C")', ["0.515", "0.518", "0.515"])
add(P, "LoRA on C (mean)", "v15_logs/v15.json", 'mean(accs(d, arena="C"))', "0.516")
add(P, "LoRA on C, ten-epoch extension per seed", "v15b_logs/v15b.json", 'accs(d, arena="C")', ["0.538", "0.520", "0.515"])
add(P, "LoRA on C, ten-epoch extension (mean)", "v15b_logs/v15b.json", 'mean(accs(d, arena="C"))', "0.524")

# ------------------------------------------------------------------ F3b arena A decomposition (tab:where)
P = "F3b_arena_A_decomposition"
add(P, "hand scan (parser) per seed", "v4_logs/v4_perocc.json", 'accs(d, epsilon=0.0)', None)
add(P, "hand scan (parser)", "v4_logs/v4_perocc.json", 'mean(accs(d, epsilon=0.0))', "0.918")
add(P, "consistent relabelling per seed", "v4_logs/v4_consistent.json", 'accs(d, epsilon=1.0)', None)
add(P, "consistent relabelling", "v4_logs/v4_consistent.json", 'mean(accs(d, epsilon=1.0))', "0.869")
add(P, "10% occurrences corrupted per seed", "v4_logs/v4_perocc.json", 'accs(d, epsilon=0.1)', None)
add(P, "10% occurrences corrupted", "v4_logs/v4_perocc.json", 'mean(accs(d, epsilon=0.1))', "0.720")
add(P, "supervised linear identity map per seed", "v4_logs/v4_extract.json", '[r["A_extract"] for r in d]', None)
add(P, "supervised linear identity map", "v4_logs/v4_extract.json", 'mean([r["A_extract"] for r in d])', "0.9986")
add(P, "trained through the binding loss per seed", "v8_logs/v8.json", 'accs(d)', None)
add(P, "trained through the binding loss", "v8_logs/v8.json", 'mean(accs(d))', "0.501")
add(P, "raw hidden states per seed", "v1_logs/vsa_hidden.json", 'accs(d)', None)
add(P, "raw hidden states", "v1_logs/vsa_hidden.json", 'mean(accs(d))', "0.505")
add(P, "per-occurrence corruption curve: epsilons", "v4_logs/v4_perocc.json", 'sorted(set(r["epsilon"] for r in d))', None)
add(P, "per-occurrence corruption curve: mean acc per epsilon", "v4_logs/v4_perocc.json",
    '[mean(accs(d, epsilon=e)) for e in sorted(set(r["epsilon"] for r in d))]', None)
add(P, "per-occurrence corruption curve: per-seed acc per epsilon", "v4_logs/v4_perocc.json",
    '[accs(d, epsilon=e) for e in sorted(set(r["epsilon"] for r in d))]', None)
add(P, "consistent relabelling curve: mean acc per epsilon", "v4_logs/v4_consistent.json",
    '[mean(accs(d, epsilon=e)) for e in sorted(set(r["epsilon"] for r in d))]', None)
add(P, "consistent relabelling curve: per-seed acc per epsilon", "v4_logs/v4_consistent.json",
    '[accs(d, epsilon=e) for e in sorted(set(r["epsilon"] for r in d))]', None)
add(P, "corruption rate at which per-occurrence arm reaches the floor 0.509 (linear interpolation)", "v4_logs/v4_perocc.json",
    'float(np.interp(-0.509, [-mean(accs(d, epsilon=e)) for e in sorted(set(r["epsilon"] for r in d))], sorted(set(r["epsilon"] for r in d))))', "0.46",
    note="floor 0.509 from v1_logs/frozen_llm.json; interpolation on the mean curve")

# ------------------------------------------------------------------ F3c symbol-count sweep
P = "F3c_symbol_count_sweep"
add(P, "k values", "v14_logs/v14_fixed.json", '[r["k"] for r in d]', None)
add(P, "parser mean per k", "v14_logs/v14_fixed.json", '[mean(r["parser"]) for r in d]', ["1.000", "0.998", "0.997", "0.972", "0.951", "0.934"])
add(P, "parser per seed per k", "v14_logs/v14_fixed.json", '[r["parser"] for r in d]', None)
add(P, "discrete handover mean per k", "v14_logs/v14_fixed.json", '[mean(r["probe_argmax"]) for r in d]', None,
    note="manuscript: handover ties the parser from k=8 (its literals 1.000/0.998/0.997/0.972/0.951/0.934 are the parser column)")
add(P, "discrete handover per seed per k", "v14_logs/v14_fixed.json", '[r["probe_argmax"] for r in d]', None)
add(P, "floor mean per k", "v14_logs/v14_fixed.json", '[mean(r["floor"]) for r in d]', None)
add(P, "floor per seed per k", "v14_logs/v14_fixed.json", '[r["floor"] for r in d]', None)
add(P, "recovery fraction R per k", "v14_logs/v14_fixed.json", '[r["R"] for r in d]', None)

# ------------------------------------------------------------------ F3d read swept by budget
P = "F3d_read_by_budget"
add(P, "fit epochs", "v21_logs/v21.json", '[c["fit_epochs"] for c in d["cells"]]', None)
add(P, "continuous read: headroom recovered (%)", "v21_logs/v21.json", '[100*c["R_cont"] for c in d["cells"]]', ["-8", "30", "50", "67", "75"])
add(P, "snapped read: headroom recovered (%)", "v21_logs/v21.json", '[100*c["R_pa"] for c in d["cells"]]', None)
add(P, "snap gain (points of headroom) at the four non-degenerate budgets", "v21_logs/v21.json", '[100*c["gain_H"] for c in d["cells"][1:]]', ["67", "48", "31", "24"])
add(P, "snap gain at fit_epochs 0", "v21_logs/v21.json", '100*d["cells"][0]["gain_H"]', None)
add(P, "continuous read per seed", "v21_logs/v21.json", '[c["frozen_cont"] for c in d["cells"]]', None)
add(P, "snapped read per seed", "v21_logs/v21.json", '[c["proj_argmax"] for c in d["cells"]]', None)
add(P, "floor per seed", "v21_logs/v21.json", 'd["floor"]', None)
add(P, "parser per seed", "v21_logs/v21.json", 'd["parser"]', None)
unresolved(P, "fidelity saturates at 0.76", "0.76", "v21_logs/v21.json carries no fidelity field; not searched further (not in the requested panel list).")

# ------------------------------------------------------------------ F3e depth
P = "F3e_depth"
add(P, "depths", "v57_logs/v57.json", 'd["depths"]', None)
for arm, lit in [("probe_argmax", ["0.958", "0.934", "0.880"]), ("lora_direct", ["0.507", "0.501", "0.515"]), ("parser", None), ("floor", None)]:
    add(P, f"{arm} mean per depth", "v57_logs/v57.json", f'[mean(d["cells"][str(k)]["{arm}"]) for k in d["depths"]]', lit)
    add(P, f"{arm} per seed per depth", "v57_logs/v57.json", f'[d["cells"][str(k)]["{arm}"] for k in d["depths"]]', None)

# ------------------------------------------------------------------ F4a seesaw
P = "F4a_seesaw"
for file, arms, fam in [("v62_logs/v62_BANKED.json", ["fsq_spans", "fsq_inj"], "task-only"),
                        ("v63_logs/v63_PROVISIONAL.json", ["fsq_id", "fsq_id_only"], "classification"),
                        ("v63b_logs/v63b_BANKED.json", ["fsq_id", "fsq_id_only"], "classification"),
                        ("v64_logs/v64_BANKED.json", ["fsq_con", "fsq_con_only"], "equivalence")]:
    for arm in arms:
        add(P, f"{fam} / {arm}: separation (1 - distinct-symbol code cosine) per seed", file,
            f'[1.0 - g["distinct_symbol_code_cosine"] for g in d["diag"]["{arm}"]]', None)
        add(P, f"{fam} / {arm}: same-symbol consistency per seed", file, f'[g["same_symbol_consistency"] for g in d["diag"]["{arm}"]]', None)
        add(P, f"{fam} / {arm}: chain accuracy per seed", file, f'd["arms"]["{arm}"]', None)

# ------------------------------------------------------------------ F4b hashing cosines
P = "F4b_hashing_cosines"
add(P, "raw sign hash: within-symbol cosine (within instance)", "v39_logs/v39b.json", 'd["arms"]["lsh_raw"]["same_within"]', "0.568")
add(P, "raw sign hash: between-symbol cosine (within instance)", "v39_logs/v39b.json", 'd["arms"]["lsh_raw"]["diff_within"]', "0.498")
add(P, "subspace hash (36 comp.): within-symbol cosine", "v40_logs/v40.json", 'd["diagnostic"]["pca_hash_36"]["same_within"]', "0.243")
add(P, "subspace hash (36 comp.): between-symbol cosine", "v40_logs/v40.json", 'd["diagnostic"]["pca_hash_36"]["diff_within"]', "0.091",
    note="artifact .0905 rounds to .090 (or .091 half-up on .0905); manuscript prints .091. Last-digit rounding, not a data discrepancy.")
add(P, "subspace hash (128 comp.): within/between cosine", "v40_logs/v40.json", '[d["diagnostic"]["pca_hash_128"]["same_within"], d["diagnostic"]["pca_hash_128"]["diff_within"]]', None)
add(P, "raw sign hash: chain per seed", "v39_logs/v39.json", 'd["arms"]["lsh_raw"]', None)
add(P, "raw sign hash: chain (mean)", "v39_logs/v39.json", 'mean(d["arms"]["lsh_raw"])', "0.523")
add(P, "subspace hash 36: chain per seed", "v40_logs/v40.json", 'd["arms"]["pca_hash_36"]', None)
add(P, "subspace hash 36: chain (mean)", "v40_logs/v40.json", 'mean(d["arms"]["pca_hash_36"])', "0.503")
add(P, "subspace hash 128: chain per seed", "v40_logs/v40.json", 'd["arms"]["pca_hash_128"]', None)
add(P, "subspace hash 128: chain (mean)", "v40_logs/v40.json", 'mean(d["arms"]["pca_hash_128"])', "0.507")
add(P, "k-means clustering: chain per seed", "v23_logs/v23.json", 'd["clustered"]', None)
add(P, "k-means clustering: chain (mean)", "v23_logs/v23.json", 'd["clustered_mean"]', "0.527")
add(P, "k-means clustering: floor (mean)", "v23_logs/v23.json", 'mean(d["arms"]["floor"])', "0.515")
add(P, "k-means clustering: eval purity per seed", "v23_logs/v23.json", '[c["eval_purity"] for c in d["cluster"]]', None)
add(P, "k-means clustering: eval purity range", "v23_logs/v23.json", '[min(c["eval_purity"] for c in d["cluster"]), max(c["eval_purity"] for c in d["cluster"])]', ["0.29", "0.36"])
add(P, "reference: floor/parser/handover per seed (same arena)", "v39_logs/v39.json", '[d["arms"]["floor"], d["arms"]["parser"], d["arms"]["probe_argmax"]]', None)

# ------------------------------------------------------------------ F4c pairwise head
P = "F4c_pairwise_head"
add(P, "pairwise head: groups recovered per seed", "v41_logs/v41.json", '[d["grouping"][s]["groups"] for s in ("42","123","456")]', None)
add(P, "pairwise head: groups recovered range", "v41_logs/v41.json", '[min(d["grouping"][s]["groups"] for s in d["grouping"]), max(d["grouping"][s]["groups"] for s in d["grouping"])]', ["35", "39"])
add(P, "pairwise head: true symbols present", "v41_logs/v41.json", 'd["grouping"]["42"]["true_symbols"]', "58")
add(P, "pairwise head: chain per seed", "v41_logs/v41.json", 'd["arms"]["group_argmax"]', None)
add(P, "pairwise head: chain (mean)", "v41_logs/v41.json", 'mean(d["arms"]["group_argmax"])', "0.581")
add(P, "pairwise head: eval pair F1 per seed", "v41_logs/v41.json", '[d["grouping"][s]["eval_pair_f1"] for s in ("42","123","456")]', None)
add(P, "pairwise head: eval pair F1 range", "v41_logs/v41.json", '[min(d["grouping"][s]["eval_pair_f1"] for s in d["grouping"]), max(d["grouping"][s]["eval_pair_f1"] for s in d["grouping"])]', ["0.400", "0.450"])
add(P, "pairwise head: train pair F1 range", "v41_logs/v41.json", '[min(d["matcher"][s]["train_pair_f1"] for s in d["matcher"]), max(d["matcher"][s]["train_pair_f1"] for s in d["matcher"])]', ["0.63", "0.65"])
add(P, "chain-selected head: groups recovered per seed", "v42_logs/v42.json", '[d["grouping"][s]["groups"] for s in ("42","123","456")]', None)
add(P, "chain-selected head: groups recovered range", "v42_logs/v42.json", '[min(d["grouping"][s]["groups"] for s in d["grouping"]), max(d["grouping"][s]["groups"] for s in d["grouping"])]', ["50", "61"])
add(P, "chain-selected head: chain per seed", "v42_logs/v42.json", 'd["arms"]["group_argmax"]', None)
add(P, "chain-selected head: chain (mean)", "v42_logs/v42.json", 'mean(d["arms"]["group_argmax"])', "0.583")
add(P, "chain-selected head: eval pair F1 per seed", "v42_logs/v42.json", '[d["grouping"][s]["eval_pair_f1"] for s in ("42","123","456")]', None)
add(P, "chain-selected head: eval pair F1 range", "v42_logs/v42.json", '[min(d["grouping"][s]["eval_pair_f1"] for s in d["grouping"]), max(d["grouping"][s]["eval_pair_f1"] for s in d["grouping"])]', ["0.46", "0.48"])
add(P, "chain-selected head: merge / split rates per seed", "v42_logs/v42.json", '[[d["grouping"][s]["merge_rate"], d["grouping"][s]["split_rate"]] for s in ("42","123","456")]', None)
add(P, "reference floor (mean)", "v41_logs/v41.json", 'd["floor"]', "0.515")

# ------------------------------------------------------------------ F4d aliasing / corruption on bAbI 1
P = "F4d_aliasing_babi"
add(P, "permuted codebook: chain per seed", "v36_logs/v36.json", 'rows(d["cells"], cell="permuted")[0]["chain"]', None)
add(P, "permuted codebook: chain (mean)", "v36_logs/v36.json", 'rows(d["cells"], cell="permuted")[0]["mean"]', "1.000")
add(P, "alias names (merge-persons, rate .25): chain per seed", "v36_logs/v36.json", 'rows(d["cells"], cell="merge-persons", rate=0.25)[0]["chain"]', None)
add(P, "alias names (merge-persons, rate .25): chain (mean)", "v36_logs/v36.json", 'rows(d["cells"], cell="merge-persons", rate=0.25)[0]["mean"]', "0.880")
add(P, "alias locations (merge-locations, rate .25): chain per seed", "v36_logs/v36.json", 'rows(d["cells"], cell="merge-locations", rate=0.25)[0]["chain"]', None)
add(P, "alias locations (merge-locations, rate .25): chain (mean)", "v36_logs/v36.json", 'rows(d["cells"], cell="merge-locations", rate=0.25)[0]["mean"]', "0.658")
add(P, "all v36 cells (cell, rate, mean)", "v36_logs/v36.json", '[[c["cell"], c["rate"], c["mean"]] for c in d["cells"]]', None)
add(P, "1-in-10 occurrences corrupted (split .1): chain per seed", "v35_logs/v35.json", 'rows(d["cells"], family="split", rate=0.1)[0]["chain"]', None)
add(P, "1-in-10 occurrences corrupted (split .1): chain (mean)", "v35_logs/v35.json", 'rows(d["cells"], family="split", rate=0.1)[0]["chain_mean"]', "0.946")
add(P, "1-in-5 occurrences corrupted (split .2): chain per seed", "v35_logs/v35.json", 'rows(d["cells"], family="split", rate=0.2)[0]["chain"]', None)
add(P, "1-in-5 occurrences corrupted (split .2): chain (mean)", "v35_logs/v35.json", 'rows(d["cells"], family="split", rate=0.2)[0]["chain_mean"]', "0.825")
add(P, "all v35 cells (family, rate, chain_mean)", "v35_logs/v35.json", '[[c["family"], c["rate"], c["chain_mean"]] for c in d["cells"]]', None)
add(P, "all v35 cells: chain per seed", "v35_logs/v35.json", '[c["chain"] for c in d["cells"]]', None)
add(P, "bAbI 1 floor / parser", "v35_logs/v35.json", '[d["floor"], d["parser"]]', ["0.771", "1.000"])

# ------------------------------------------------------------------ F5a/b multi-hop QA
P = "F5a_2wiki_three_arms"
add(P, "2Wiki compositional (phi-2): prompt per seed", "v65_logs/v65_BANKED.json", 'd["arms"]["prompt"]["covered"]', None)
add(P, "2Wiki compositional (phi-2): prompt (mean)", "v65_logs/v65_BANKED.json", 'mean(d["arms"]["prompt"]["covered"])', "0.253")
add(P, "2Wiki compositional (phi-2): gold-facts prompt (mean)", "v65_logs/v65_BANKED.json", 'mean(d["arms"]["prompt_ev"]["covered"])', "0.950")
add(P, "2Wiki compositional (phi-2): HRR store per seed", "v65_logs/v65_BANKED.json", 'd["arms"]["hrr_gold"]["covered"]', ["0.970", "0.977", "0.977"])
add(P, "2Wiki compositional (phi-2): HRR store (mean)", "v65_logs/v65_BANKED.json", 'mean(d["arms"]["hrr_gold"]["covered"])', "0.974")
add(P, "2Wiki compositional (Qwen2.5-7B): prompt", "v67_logs/v67_BANKED.json", 'mean(d["arms"]["prompt"]["covered"])', "0.227")
add(P, "2Wiki compositional (Qwen2.5-7B): gold-facts prompt", "v67_logs/v67_BANKED.json", 'mean(d["arms"]["prompt_ev"]["covered"])', "0.953")
add(P, "2Wiki inference (phi-2): prompt (mean)", "v69_logs/v69_BANKED.json", 'mean(d["arms"]["prompt"]["covered"])', "0.093")
add(P, "2Wiki inference (phi-2): gold-facts prompt (mean)", "v69_logs/v69_BANKED.json", 'mean(d["arms"]["prompt_ev"]["covered"])', "0.683")
add(P, "2Wiki inference (phi-2): HRR store per seed", "v69_logs/v69_BANKED.json", 'd["arms"]["hrr_gold"]["covered"]', None)
add(P, "2Wiki inference (phi-2): HRR store (mean)", "v69_logs/v69_BANKED.json", 'mean(d["arms"]["hrr_gold"]["covered"])', "0.762")
add(P, "Qwen3-8B prompt at 128-fact pool (V-100)", "v100_logs/v100_scored.json", 'rows(d["curve"], n_facts=128)[0]["prompt_acc"]', "0.875")
add(P, "store at d=4096, 128 facts (V-101)", "v101_logs/v101_scored.json", 'd["grid"]["4096"]["128"]', "0.919")
P = "F5b_bulky_evidence"
for typ, lit in [("compositional", {"prompt_ev_big": "0.353", "hrr_gold_big": "0.953", "hrr_then_prompt_big": "0.900", "hall_prompt": "22", "hall_hybrid": "4"}),
                 ("inference", {"prompt_ev_big": "0.007", "hrr_gold_big": "0.723", "hrr_then_prompt_big": "0.517", "hall_prompt": "77", "hall_hybrid": "2"})]:
    add(P, f"{typ}: all facts in prompt EM", "v72b_logs/v72b_BANKED.json", f'd["arms"]["{typ}"]["prompt_ev_big"]["em"]', lit["prompt_ev_big"])
    add(P, f"{typ}: retrieved facts in prompt (hybrid) EM", "v72b_logs/v72b_BANKED.json", f'd["arms"]["{typ}"]["hrr_then_prompt_big"]["em"]', lit["hrr_then_prompt_big"])
    add(P, f"{typ}: HRR store readout EM", "v72b_logs/v72b_BANKED.json", f'd["arms"]["{typ}"]["hrr_gold_big"]["em"]', lit["hrr_gold_big"])
    add(P, f"{typ}: hallucination, all facts in prompt (%)", "v72b_logs/v72b_BANKED.json", f'100*d["arms"]["{typ}"]["prompt_ev_big"]["hallucination"]', lit["hall_prompt"])
    add(P, f"{typ}: hallucination, hybrid (%)", "v72b_logs/v72b_BANKED.json", f'100*d["arms"]["{typ}"]["hrr_then_prompt_big"]["hallucination"]', lit["hall_hybrid"])
    add(P, f"{typ}: hallucination, store readout", "v72b_logs/v72b_BANKED.json", f'd["arms"]["{typ}"]["hrr_gold_big"]["hallucination"]', "0")
    add(P, f"{typ}: minimal-evidence refs", "v72b_logs/v72b_BANKED.json", f'd["refs"]["{typ}"]', None)
P = "F5b_musique"
add(P, "MuSiQue: n instances", "v79_logs/v79_BANKED.json", 'd["n"]', "212")
add(P, "MuSiQue: prompt per seed", "v79_logs/v79_BANKED.json", 'd["arms"]["prompt"]["covered"]', None)
add(P, "MuSiQue: prompt (mean)", "v79_logs/v79_BANKED.json", 'mean(d["arms"]["prompt"]["covered"])', "0.165")
add(P, "MuSiQue: supporting-paragraph prompt (mean)", "v79_logs/v79_BANKED.json", 'mean(d["arms"]["prompt_sup"]["covered"])', "0.401")
add(P, "MuSiQue: gold-triples prompt +0 per seed", "v79_logs/v79_BANKED.json", 'd["arms"]["prompt_ev_p0"]["covered"]', None)
add(P, "MuSiQue: gold-triples prompt +0 (mean)", "v79_logs/v79_BANKED.json", 'mean(d["arms"]["prompt_ev_p0"]["covered"])', None)
add(P, "MuSiQue: HRR store +0 per seed", "v79_logs/v79_BANKED.json", 'd["arms"]["hrr_gold_p0"]["covered"]', None)
add(P, "MuSiQue: HRR store +0 (mean)", "v79_logs/v79_BANKED.json", 'mean(d["arms"]["hrr_gold_p0"]["covered"])', "0.948")
add(P, "MuSiQue: gold-triples prompt +50 per seed", "v79_logs/v79_BANKED.json", 'd["arms"]["prompt_ev_p50"]["covered"]', None)
add(P, "MuSiQue: HRR store +50 per seed", "v79_logs/v79_BANKED.json", 'd["arms"]["hrr_gold_p50"]["covered"]', None)
add(P, "MuSiQue: gold-triples prompt +150 per seed", "v79_logs/v79_BANKED.json", 'd["arms"]["prompt_ev_p150"]["covered"]', None)
add(P, "MuSiQue: gold-triples prompt +150 (mean)", "v79_logs/v79_BANKED.json", 'mean(d["arms"]["prompt_ev_p150"]["covered"])', "0.187")
add(P, "MuSiQue: HRR store +150 per seed", "v79_logs/v79_BANKED.json", 'd["arms"]["hrr_gold_p150"]["covered"]', None)
add(P, "MuSiQue: HRR store +150 (mean)", "v79_logs/v79_BANKED.json", 'mean(d["arms"]["hrr_gold_p150"]["covered"])', "0.656")
add(P, "MuSiQue: excluded rate (%), 1 - kept", "v79_logs/v79_BANKED.json", '100*(1 - d["excl_rate"])', "8.8",
    note="manuscript keeps 8.8% of instances; excl_rate .9123 is the excluded fraction")
add(P, "MuSiQue: hop distribution", "v79_logs/v79_BANKED.json", 'd["hop_dist"]', None)
add(P, "MuSiQue by hop: store EM at +0/+50/+150 per hop (2,3,4)", "v79_logs/v79a_BANKED.json",
    '{p: [d["by_hop"][p]["cells"][h]["em"] for h in ("2","3","4")] for p in ("p0","p50","p150")}', None)
add(P, "MuSiQue by hop: overall per pool", "v79_logs/v79a_BANKED.json", '[d["by_hop"][p]["overall"] for p in ("p0","p50","p150")]', None)
add(P, "MuSiQue by hop: distractors sharing a gold relation per pool", "v79_logs/v79a_BANKED.json",
    '[d["collisions"][p]["mean_distractors_sharing_a_gold_relation"] for p in ("p0","p50","p150")]', None)

# ------------------------------------------------------------------ F5c capacity
P = "F5c_capacity"
add(P, "v73 K values", "v73_logs/v73_BANKED.json", '[int(k) for k in d["Ks"]]', None)
add(P, "v73 compositional: mean bindings per K", "v73_logs/v73_BANKED.json", '[d["Ks"][k]["compositional"]["mean_bindings"] for k in d["Ks"]]', None)
add(P, "v73 compositional: EM per K", "v73_logs/v73_BANKED.json", '[d["Ks"][k]["compositional"]["em"] for k in d["Ks"]]', None)
add(P, "v73 inference: EM per K", "v73_logs/v73_BANKED.json", '[d["Ks"][k]["inference"]["em"] for k in d["Ks"]]', None)
add(P, "v73 compositional EM at K=100", "v73_logs/v73_BANKED.json", 'd["Ks"]["100"]["compositional"]["em"]', "0.780")
add(P, "v73 compositional EM at K=400", "v73_logs/v73_BANKED.json", 'd["Ks"]["400"]["compositional"]["em"]', "0.063")
add(P, "v73 d", "v73_logs/v73_BANKED.json", 'd["d"]', "4096")
add(P, "v101 dims", "v101_logs/v101_scored.json", '[int(k) for k in d["grid"]]', None)
add(P, "v101 ks", "v101_logs/v101_scored.json", '[int(k) for k in d["grid"]["512"]]', None)
add(P, "v101 grid acc[d][k]", "v101_logs/v101_scored.json", '{dd: [d["grid"][dd][k] for k in d["grid"][dd]] for dd in d["grid"]}', None)
add(P, "v101 k* at .70 for d=512..4096", "v101_logs/v101_scored.json", '[d["kstar"][k] for k in ("512","1024","2048","4096")]', ["32", "64", "128", "256"])
add(P, "v101 k* at d=8192", "v101_logs/v101_scored.json", 'd["kstar"]["8192"]', None)
add(P, "constant-ratio readout d=512,k=32", "v101_logs/v101_scored.json", 'd["grid"]["512"]["32"]', "0.844")
add(P, "constant-ratio readout d=8192,k=512", "v101_logs/v101_scored.json", 'd["grid"]["8192"]["512"]', "0.603")
add(P, "3-way at d=4096/128 facts: store", "v101_logs/v101_scored.json", 'd["grid"]["4096"]["128"]', "0.919")
add(P, "3-way at 128 facts: Qwen3-8B prompt", "v100_logs/v100_scored.json", 'rows(d["curve"], n_facts=128)[0]["prompt_acc"]', "0.875")
add(P, "3-way: phi-2 prompt on 102-152-fact pool (inference)", "v72b_logs/v72b_BANKED.json", 'd["arms"]["inference"]["prompt_ev_big"]["em"]', "0.007")

# ------------------------------------------------------------------ F5d sharding + cost
P = "F5d_sharding_cost"
add(P, "v102 grid keys", "v102_logs/v102_scored.json", 'list(d["grid"].keys())', None)
add(P, "K=512 d=2048 S=1 per seed", "v102_logs/v102_scored.json", 'd["grid"]["d2048_K512_S1"]', None)
add(P, "K=512 d=2048 S=1 (mean)", "v102_logs/v102_scored.json", 'mean(d["grid"]["d2048_K512_S1"])', "0.039")
add(P, "K=512 d=2048 S=8 per seed", "v102_logs/v102_scored.json", 'd["grid"]["d2048_K512_S8"]', None)
add(P, "K=512 d=2048 S=8 (mean)", "v102_logs/v102_scored.json", 'mean(d["grid"]["d2048_K512_S8"])', "0.919")
add(P, "K=512 d=2048: mean per S (1,2,4,8,16)", "v102_logs/v102_scored.json", '[mean(d["grid"]["d2048_K512_S%d" % s]) for s in (1,2,4,8,16)]', None)
add(P, "all cells mean", "v102_logs/v102_scored.json", '{k: mean(v) for k, v in d["grid"].items()}', None)
add(P, "cross-shard cost: |K1024 S16 - K512 S8| (both 64 facts/shard)", "v102_logs/v102_scored.json",
    'abs(mean(d["grid"]["d2048_K1024_S16"]) - mean(d["grid"]["d2048_K512_S8"]))', "0.01",
    note="manuscript literal is the phrase 'cross-shard selection costs about a point' (line 407), encoded here as 0.01; verify_paper_capacity bounds it at <= .02. The v102 verdict string quotes .0139 for its P3 (different pair).")
add(P, "V-100 curve: n_facts", "v100_logs/v100_scored.json", '[r["n_facts"] for r in d["curve"]]', None)
add(P, "V-100 curve: prompt acc (Qwen3-8B)", "v100_logs/v100_scored.json", '[r["prompt_acc"] for r in d["curve"]]', None)
add(P, "V-100 curve: store acc (d=2048)", "v100_logs/v100_scored.json", '[r["store_acc"] for r in d["curve"]]', None)
add(P, "V-100 curve: prompt tokens", "v100_logs/v100_scored.json", '[r["prompt_tokens"] for r in d["curve"]]', None)
add(P, "V-100 curve: prompt latency (s)", "v100_logs/v100_scored.json", '[r["prompt_latency_s"] for r in d["curve"]]', None)
add(P, "V-100 curve: store latency (s)", "v100_logs/v100_scored.json", '[r["store_latency_s"] for r in d["curve"]]', None)
add(P, "V-100 latency ratio prompt/store per scale", "v100_logs/v100_scored.json", '[r["prompt_latency_s"]/r["store_latency_s"] for r in d["curve"]]', None,
    note="The manuscript contains NO latency literal (line 600: 'we did not profile wall-clock overhead'); the '1/70th' in the request is not in main_npj.tex. Ratios here range 14x-186x; verdict string quotes 186x at the 16-fact crossover.")

# ------------------------------------------------------------------ F4e cross-talk geometry (optional)
P = "F4e_crosstalk_geometry"
add(P, "cross-talk threshold (largest still clearing .615)", "v95_logs/v95_scored.json", 'd["threshold_crosstalk"]', "0.099")
add(P, "v95 curve: target c", "v95_logs/v95_scored.json", '[c["target_c"] for c in d["curve"]]', None)
add(P, "v95 curve: measured cross-talk", "v95_logs/v95_scored.json", '[c["crosstalk"] for c in d["curve"]]', None)
add(P, "v95 curve: acc (mean)", "v95_logs/v95_scored.json", '[c["acc"] for c in d["curve"]]', None)
add(P, "v95 curve: acc per seed", "v95_logs/v95_scored.json", '[c["acc_per_seed"] for c in d["curve"]]', None)
add(P, "v95 acc at cross-talk .099", "v95_logs/v95_scored.json", 'rows(d["curve"], target_c=0.1)[0]["acc"]', "0.903")
add(P, "v95 acc at cross-talk .199", "v95_logs/v95_scored.json", 'rows(d["curve"], target_c=0.2)[0]["acc"]', "0.604")
add(P, "classifier logits (continuous) cross-talk", "geometry_banked.json", 'd["crosstalk_probe_logits"]', "0.3904")
add(P, "discrete arg-max lookup cross-talk", "geometry_banked.json", 'd["crosstalk_discrete"]', "0.0001")
add(P, "raw states cross-talk / codebook cross-talk / probe accuracy", "geometry_banked.json", '[d["crosstalk_raw_states"], d["crosstalk_codebook"], d["probe_accuracy"]]', None)
add(P, "fitted frozen-continuous projection (3 epochs): cross-talk", "crosstalk_audit2.json", 'd["fit3"]["crosstalk"]', "0.032")
add(P, "fitted frozen-continuous projection (3 epochs): cross-talk per seed", "crosstalk_audit2.json", 'd["fit3"]["xt_seeds"]', None)
add(P, "fitted frozen-continuous projection (3 epochs): acc / per seed", "crosstalk_audit2.json", '[d["fit3"]["acc"], d["fit3"]["acc_seeds"]]', None)
add(P, "binding-loss projection (0 fit epochs): cross-talk", "crosstalk_audit2.json", 'd["fit0"]["crosstalk"]', "0.699")
add(P, "binding-loss projection (0 fit epochs): acc / per seed", "crosstalk_audit2.json", '[d["fit0"]["acc"], d["fit0"]["acc_seeds"]]', None)

# ------------------------------------------------------------------ meta + write
INDEX["_meta"] = {
    "built": "2026-09-02",
    "roots": {"D": ROOT_D, "B": ROOT_B},
    "file_roots": FILES,
    "key_semantics": "python expression evaluated with d = json.load(file); helpers: mean(xs), np, rows(list, **eq_filters), accs(list, **eq_filters) -> [r['acc']]",
    "match_rule": "value rounded to the manuscript literal's decimal places equals the literal (lists element-wise); null manuscript = per-seed/support data with no literal",
    "manuscript": "/path/to/paper_v54/main_npj.tex",
}
json.dump(INDEX, open(os.path.join(OUT, "INDEX.json"), "w"), indent=1)

# ------------------------------------------------------------------ summary
tot = ok = bad = nolit = 0
for p, es in INDEX.items():
    if p in ("_meta", "UNRESOLVED"):
        continue
    for e in es:
        tot += 1
        if e["match"] is None: nolit += 1
        elif e["match"]: ok += 1
        else:
            bad += 1
            print("MISMATCH  [%s] %s: value=%s manuscript=%s" % (p, e["label"], json.dumps(e["value"])[:120], e["manuscript"]))
print("\nfiles copied: %d" % len(copied))
print("panels: %d" % len([p for p in INDEX if p not in ("_meta", "UNRESOLVED")]))
print("entries: %d  (matched %d, mismatched %d, no-literal/support %d)" % (tot, ok, bad, nolit))
print("unresolved: %d" % len(INDEX.get("UNRESOLVED", [])))
for u in INDEX.get("UNRESOLVED", []):
    print("  UNRESOLVED [%s] %s -> %s" % (u["panel"], u["label"], u["manuscript"]))
