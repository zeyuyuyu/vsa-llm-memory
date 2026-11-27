"""v52 addenda to figdata_v2/INDEX.json (the figure revision that followed deck/review_v51/FIGURE_audit_v51.md).

    python3 figdata_v2/add_index_v52.py

build_index.py can no longer rebuild INDEX.json (its source directory is gone and the tuned-LoRA entry was added by
hand on 2026-09-15), so the entries below are ADDED to the existing file, in the existing schema
(label, file, key, value, manuscript, match). The value is never typed: each key is evaluated on the artifact, exactly
as build_index.py did (d = json.load(file); helpers mean, np, rows, accs). The script is idempotent: an entry whose
(panel, label) already exists is left alone. It also writes INDEX_ADDENDA_v52.md, the list of everything it added.

Two existing entries are annotated (a `note` field is added, nothing else changes): their labels call crosstalk_audit2's
`fit0` a "binding-loss projection", which it is not (it is the untrained projection of the frozen_cont arm).
"""
import json, os
import numpy as np

D2 = os.path.dirname(os.path.abspath(__file__))
IDX_PATH = os.path.join(D2, "INDEX.json")
mean = lambda xs: float(np.mean(xs))
def rows(d, **kw):
    return [r for r in d if all(r.get(k) == v for k, v in kw.items())]
def accs(d, **kw):
    return [r["acc"] for r in rows(d, **kw)]
NS = {"mean": mean, "np": np, "rows": rows, "accs": accs}

ADD = [  # (panel, label, file, key, note)
    # ---------------- main Fig. 1b (fig1_overview) ----------------
    ("F1b_natural_text", "bAbI1 frozen-state (last-token) probe, per seed (v30 arms.floor)", "v30_logs/v30.json",
     'd["arms"]["floor"]',
     "Seed dots of the task-1 probe bar. Same probe as the bar: v35.json's `floor` (0.7713) is the constant "
     "FLOOR = 0.7713 hard-coded in v35.py as 'banked reference points, V-30/V-33', i.e. the mean of this list."),
    ("F1b_natural_text", "bAbI1 frozen-state (last-token) probe (mean, v30 floor)", "v30_logs/v30.json", 'd["floor"]', None),
    ("F1b_natural_text", "bAbI1 handover, per seed (v35 cells[0]: family none, rate 0)", "v35_logs/v35.json",
     'rows(d["cells"], family="none")[0]["chain"]',
     "Replaces the v36 permuted-codebook cell as the source of the task-1 handover bar (same values, 1.0 x3)."),
    ("F1b_natural_text", "bAbI1 v30 n_train / n_eval", "v30_logs/v30.json", '[d["n_train"], d["n_eval"]]', None),
    ("F1b_natural_text", "bAbI prompting zero-shot: n per cell (task 1, task 2)", "v49_logs/v49.json",
     '[d["cells"]["task1/zeroshot/all"]["n"], d["cells"]["task2/zeroshot/all"]["n"]]', None),
    ("F1b_natural_text", "bAbI2 expressible instances: n_train / n_eval", "v37_logs/v37b.json",
     '[d["n_train_expressible"], d["n_eval_expressible"]]', None),
    # ---------------- main Fig. 2a (fig3_mechanism) ----------------
    ("F3a_three_arenas", "LoRA on A per seed", "v1_logs/lora.json", "accs(d)",
     "LoRA rank 16 on arena A, V-1 run (seeds 42/123/456); was drawn as 'n/a' in v51."),
    ("F3a_three_arenas", "LoRA on A (mean)", "v1_logs/lora.json", "mean(accs(d))", None),
    ("F3a_three_arenas", "LoRA on A: seeds / trainable parameters", "v1_logs/lora.json",
     '[[r["seed"] for r in d], d[0]["trainable_params"]]', None),
    # ---------------- main Fig. 2b (fig3_mechanism) and Supplementary identity panel b ----------------
    ("F4e_crosstalk_geometry", "v163 curve: target c", "v163_logs/v163.json", '[c["target_c"] for c in d["curve"]]', None),
    ("F4e_crosstalk_geometry", "v163 curve: measured cross-talk (mean of 3 codebook seeds)", "v163_logs/v163.json",
     '[c["crosstalk_mean"] for c in d["curve"]]', None),
    ("F4e_crosstalk_geometry", "v163 curve: measured cross-talk per seed", "v163_logs/v163.json",
     '[c["crosstalk"] for c in d["curve"]]', None),
    ("F4e_crosstalk_geometry", "v163 curve: acc under the fitted rule (mean)", "v163_logs/v163.json",
     '[c["fitted_mean"] for c in d["curve"]]', None),
    ("F4e_crosstalk_geometry", "v163 curve: acc under the fitted rule per seed", "v163_logs/v163.json",
     '[c["fitted"] for c in d["curve"]]', None),
    ("F4e_crosstalk_geometry", "v163 tolerance under the fitted rule (largest tested cross-talk clearing .615)",
     "v163_logs/v163_scored.json", 'd["tolerance"]', None),
    ("F4e_crosstalk_geometry", "v165 projection before training: different-symbol cosine per seed", "v165_logs/v165.json",
     '[s["init"]["different"] for s in d["seeds"]]', None),
    ("F4e_crosstalk_geometry", "v165 projection before training: acc per seed (fitted threshold)", "v165_logs/v165.json",
     '[s["init"]["acc"] for s in d["seeds"]]', None),
    ("F4e_crosstalk_geometry", "v165 projection trained through the binding loss: different-symbol cosine per seed (final epoch)",
     "v165_logs/v165.json", '[s["epochs"][-1]["different"] for s in d["seeds"]]', None),
    ("F4e_crosstalk_geometry", "v165 projection trained through the binding loss: acc_best per seed", "v165_logs/v165.json",
     '[s["acc_best"] for s in d["seeds"]]', None),
    ("F4e_crosstalk_geometry", "v165 means: different init / different trained / acc_best", "v165_logs/v165.json",
     '[d["different_init_mean"], d["different_trained_mean"], d["acc_best_mean"]]', None),
    ("F4e_crosstalk_geometry", "v165 seeds / gates G0, B1, B2", "v165_logs/v165.json",
     '[[s["seed"] for s in d["seeds"]], d["G0"], d["B1"], d["B2"]]', None),
    ("F4e_crosstalk_geometry", "fitted frozen-continuous projection (3 epochs): acc per seed", "crosstalk_audit2.json",
     'd["fit3"]["acc_seeds"]', None),
    ("F4e_crosstalk_geometry", "untrained random projection (frozen_cont arm, 0 fit epochs; crosstalk_audit2 fit0): cross-talk",
     "crosstalk_audit2.json", 'd["fit0"]["crosstalk"]',
     "Correctly labelled duplicate of 'binding-loss projection (0 fit epochs): cross-talk'. Not plotted in v52: the "
     "V-165 before-training point replaces it."),
    # ---------------- main Fig. 3a, 3b (fig5_deployment) ----------------
    ("F5a_2wiki_three_arms", "2Wiki compositional (phi-2): gold-facts prompt per seed", "v65_logs/v65_BANKED.json",
     'd["arms"]["prompt_ev"]["covered"]', "Deterministic arm: three identical values, drawn as a bar without dots."),
    ("F5a_2wiki_three_arms", "2Wiki inference (phi-2): prompt per seed", "v69_logs/v69_BANKED.json",
     'd["arms"]["prompt"]["covered"]', "Deterministic arm: three identical values, drawn as a bar without dots."),
    ("F5a_2wiki_three_arms", "2Wiki inference (phi-2): gold-facts prompt per seed", "v69_logs/v69_BANKED.json",
     'd["arms"]["prompt_ev"]["covered"]', "Deterministic arm: three identical values, drawn as a bar without dots."),
    ("F5a_2wiki_three_arms", "2Wiki compositional / inference: n instances", "v65_logs/v65_BANKED.json", 'd["n"]',
     "v69_BANKED.json has the same n (300)."),
    ("F5b_bulky_evidence", "compositional: all facts in prompt, share of prompts truncated", "v72b_logs/v72b_BANKED.json",
     'd["arms"]["compositional"]["prompt_ev_big"]["trunc_rate"]', None),
    ("F5b_bulky_evidence", "inference: all facts in prompt, share of prompts truncated", "v72b_logs/v72b_BANKED.json",
     'd["arms"]["inference"]["prompt_ev_big"]["trunc_rate"]', None),
    ("F5b_musique", "MuSiQue (Qwen3-8B, 31k-token window, v131c): single pass per seed", "v131_logs/v131c.json",
     'd["arms"]["single_pass"]["covered"]', "Deterministic arm: three identical values, drawn as a bar without dots."),
    ("F5b_musique", "MuSiQue (Qwen3-8B, 31k-token window, v131c): gold-triples prompt +150 per seed", "v131_logs/v131c.json",
     'd["arms"]["prompt_ev_p150"]["covered"]', None),
    ("F5b_musique", "MuSiQue (Qwen3-8B, v131c): share of prompts truncated, single pass / +150", "v131_logs/v131c.json",
     '[mean(d["diag"]["single_pass_trunc"]), mean(d["diag"]["prompt_ev_p150_trunc"])]', None),
    ("F5b_musique", "MuSiQue (Qwen3-8B, v131c): n / max_len / model", "v131_logs/v131c.json",
     '[d["n"], d["config"]["max_len"], d["config"]["model"]]', None),
    ("F5b_musique", "MuSiQue (phi-2, v79): share of prompts truncated, passages / +0 / +50 / +150", "v79_logs/v79_BANKED.json",
     '[mean(d["diag"]["prompt_trunc"]), mean(d["diag"]["prompt_ev_p0_trunc"]), mean(d["diag"]["prompt_ev_p50_trunc"]), mean(d["diag"]["prompt_ev_p150_trunc"])]',
     None),
    # ---------------- main Fig. 3c (fig5_deployment) ----------------
    ("F5d_sharding_cost", "all cells per seed", "v102_logs/v102_scored.json", 'd["grid"]',
     "The cells keyed K1024 hold at most 800 facts: pooled() runs out of gold triples at 400 instances x 2."),
    # ---------------- Supplementary sweeps figure, panels c and d (figS_sweeps) ----------------
    ("FS_capacity_v151", "v151 dims", "v151_logs/v151.json", '[int(x) for x in d["config"]["dims"]]', None),
    ("FS_capacity_v151", "v151 ks", "v151_logs/v151.json", '[int(x) for x in d["config"]["ks"]]',
     "The largest grid value, 1024, is a request: the pool holds at most 800 facts (400 instances x 2 gold triples)."),
    ("FS_capacity_v151", "v151 models", "v151_logs/v151.json", 'd["config"]["models"]', None),
    ("FS_capacity_v151", "v151 n instances per cell / capacity seeds", "v151_logs/v151.json",
     '[d["config"]["n"], d["config"]["cap_seeds"]]', None),
    ("FS_capacity_v151", "v151 readout[model][d] per k", "v151_logs/v151.json",
     '{m: {dd: [d["capacity"][m][dd][str(k)] for k in d["config"]["ks"]] for dd in d["capacity"][m]} for m in d["config"]["models"]}',
     "Mean over 200 instances x 3 capacity seeds; per-seed values are not banked."),
    # ---------------- main Fig. 4b (fig6_benchmarks) ----------------
    ("F6b_clutrr", "ours: n per k (k = 2..10)", "r1_logs/ours_by_k.json", '[d["by_k"][str(k)]["n"] for k in range(2, 11)]', None),
    ("F6b_clutrr", "ours: correct per k (k = 2..10)", "r1_logs/ours_by_k.json",
     '[d["by_k"][str(k)]["correct"] for k in range(2, 11)]', None),
    ("F6b_clutrr", "ours: accuracy per k (k = 2..10)", "r1_logs/ours_by_k.json",
     '[d["by_k"][str(k)]["correct"] / d["by_k"][str(k)]["n"] for k in range(2, 11)]', None),
    ("F6b_clutrr", "ours: overall accuracy", "r1_logs/ours_by_k.json", 'd["correct"] / d["n_rows"]', None),
]

ANNOTATE = [  # (panel, label, note)
    ("F4e_crosstalk_geometry", "binding-loss projection (0 fit epochs): cross-talk",
     "v52: MISLABELLED. crosstalk_audit2 fit0 is the frozen_cont arm with zero fitting epochs, an UNTRAINED random "
     "projection, not an arm trained through the binding loss (FIGURE_audit_v51.md, Fig. 2b finding 1). The trained "
     "arm is in v165_logs/v165.json. Label kept so that older scripts still resolve; not plotted in v52."),
    ("F4e_crosstalk_geometry", "binding-loss projection (0 fit epochs): acc / per seed",
     "v52: MISLABELLED, see the cross-talk entry. Accuracy of the untrained projection under the untrained (fixed) readout."),
]


def main():
    I = json.load(open(IDX_PATH))
    cache, added, skipped = {}, [], 0
    for panel, label, file, key, note in ADD:
        ents = I.setdefault(panel, [])
        if any(e["label"] == label for e in ents):
            skipped += 1
            continue
        if not os.path.exists(os.path.join(D2, file)):
            print("  SKIPPED (artifact absent): %s / %s  [%s]" % (panel, label, file))
            continue
        if file not in cache:
            cache[file] = json.load(open(os.path.join(D2, file)))
        d = cache[file]
        value = eval(key, dict(NS, d=d))      # author-written key over a banked artifact, as in build_index.py
        if isinstance(value, np.generic):
            value = float(value)
        e = {"label": label, "file": file, "key": key, "value": value, "manuscript": None, "match": None}
        if note:
            e["note"] = note
        ents.append(e)
        added.append((panel, e))
    annotated = []
    for panel, label, note in ANNOTATE:
        for e in I[panel]:
            if e["label"] == label and e.get("note") != note:
                e["note"] = note
                annotated.append((panel, label))
    if "_meta" in I:                              # keep _meta last, as build_index.py wrote it
        I["_meta"] = I.pop("_meta")
        I["_meta"]["v52_addenda"] = "entries added by add_index_v52.py (2026-11-11); listed in INDEX_ADDENDA_v52.md"
    with open(IDX_PATH, "w") as f:
        f.write(json.dumps(I, indent=1))
    print("INDEX.json: %d added, %d already present, %d annotated" % (len(added), skipped, len(annotated)))
    write_md(I)


def write_md(I):
    """List every v52 entry (whatever run added it), read back from INDEX.json itself."""
    def short(v):
        s = json.dumps(v)
        return s if len(s) <= 150 else s[:147] + "..."
    L = ["# INDEX.json addenda, v52 (2026-11-11)", "",
         "`build_index.py` cannot rebuild `INDEX.json` any more, so these entries were added to the existing file by",
         "`figdata_v2/add_index_v52.py` (idempotent; schema unchanged: `label`, `file`, `key`, `value`, `manuscript`, `match`,",
         "optional `note`). No value is typed: each `key` is a Python expression evaluated on the artifact with",
         "`d = json.load(file)` and the helpers `mean`, `np`, `rows`, `accs`, as in `build_index.py`. `manuscript` and `match`",
         "are `null` for all of them (no manuscript literal was matched at the time of adding).", "",
         "## Artifacts added to `figdata_v2/` for v52", "",
         "| File | Copied from | Used for |", "|---|---|---|",
         "| `v30_logs/v30.json` | `ngcls/v30_logs/v30.json` | Fig. 1b, seed dots of the task-1 frozen-state probe (`arms.floor`). `arms.spans` (0.976) is NOT plotted: it is a supervised cosine regression onto the codebook, not a projection trained through the binding loss |",
         "| `v1_logs/lora.json` | `hybrid_neurosymbolic/v1_logs/lora.json` | Fig. 2a, LoRA on arena A |",
         "| `v131_logs/v131c.json` | `ngcls/v131_logs/v131c.json` | Fig. 3a, MuSiQue references read without truncation (Qwen3-8B, 31,000-token window) |",
         "| `v151_logs/v151.json` | `ngcls/v151_logs/v151.json` | Supplementary sweeps figure, panels c and d (fine-grid capacity) |",
         "| `v163_logs/v163.json`, `v163_scored.json`, `v163_score.out` | banked by the V-163 session | Fig. 2b curve and tolerance; Supplementary identity panel b tolerance line |",
         "| `v165_logs/v165.json` | banked by the V-165 session | Fig. 2b and Supplementary identity panel b, the projection before and after training |",
         "| `r1_logs/ours_by_k.json` | written by `r1_logs/bank_ours_by_k.py` from `vsassm/r1_out/test_8B_think.jsonl` | Fig. 4b, ours at k = 2..10 |",
         "| `benchmarks/babilong.json`, `benchmarks/clutrr.json` | transcribed from the verdict files, one `source` string per value | Fig. 4 (read directly by `make_fig6_benchmarks.py`; not in INDEX) |",
         "", "## Entries added", ""]
    labels = {(p, l) for p, l, *_ in ADD}
    for panel in dict.fromkeys(p for p, *_ in ADD):
        L += ["### `%s`" % panel, "", "| label | file | key | value |", "|---|---|---|---|"]
        for e in I[panel]:
            if (panel, e["label"]) in labels:
                L.append("| %s | `%s` | `%s` | `%s` |" % (e["label"], e["file"], e["key"].replace("|", "\\|"), short(e["value"])))
        L.append("")
    L += ["## Existing entries annotated (a `note` field added; label, key and value unchanged)", ""]
    for panel, label, note in ANNOTATE:
        L.append("- `%s` / \"%s\": %s" % (panel, label, note))
    L += ["", "## `_meta`", "", "- `_meta.v52_addenda` added (one sentence pointing to this file)."]
    open(os.path.join(D2, "INDEX_ADDENDA_v52.md"), "w").write("\n".join(L) + "\n")
    print("wrote INDEX_ADDENDA_v52.md")


if __name__ == "__main__":
    main()
