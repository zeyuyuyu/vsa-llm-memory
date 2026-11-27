# INDEX.json addenda, v52 (2026-11-11)

`build_index.py` cannot rebuild `INDEX.json` any more, so these entries were added to the existing file by
`figdata_v2/add_index_v52.py` (idempotent; schema unchanged: `label`, `file`, `key`, `value`, `manuscript`, `match`,
optional `note`). No value is typed: each `key` is a Python expression evaluated on the artifact with
`d = json.load(file)` and the helpers `mean`, `np`, `rows`, `accs`, as in `build_index.py`. `manuscript` and `match`
are `null` for all of them (no manuscript literal was matched at the time of adding).

## Artifacts added to `figdata_v2/` for v52

| File | Copied from | Used for |
|---|---|---|
| `v30_logs/v30.json` | `ngcls/v30_logs/v30.json` | Fig. 1b, seed dots of the task-1 frozen-state probe (`arms.floor`). `arms.spans` (0.976) is NOT plotted: it is a supervised cosine regression onto the codebook, not a projection trained through the binding loss |
| `v1_logs/lora.json` | `hybrid_neurosymbolic/v1_logs/lora.json` | Fig. 2a, LoRA on arena A |
| `v131_logs/v131c.json` | `ngcls/v131_logs/v131c.json` | Fig. 3a, MuSiQue references read without truncation (Qwen3-8B, 31,000-token window) |
| `v151_logs/v151.json` | `ngcls/v151_logs/v151.json` | Supplementary sweeps figure, panels c and d (fine-grid capacity) |
| `v163_logs/v163.json`, `v163_scored.json`, `v163_score.out` | banked by the V-163 session | Fig. 2b curve and tolerance; Supplementary identity panel b tolerance line |
| `v165_logs/v165.json` | banked by the V-165 session | Fig. 2b and Supplementary identity panel b, the projection before and after training |
| `r1_logs/ours_by_k.json` | written by `r1_logs/bank_ours_by_k.py` from `vsassm/r1_out/test_8B_think.jsonl` | Fig. 4b, ours at k = 2..10 |
| `benchmarks/babilong.json`, `benchmarks/clutrr.json` | transcribed from the verdict files, one `source` string per value | Fig. 4 (read directly by `make_fig6_benchmarks.py`; not in INDEX) |

## Entries added

### `F1b_natural_text`

| label | file | key | value |
|---|---|---|---|
| bAbI1 frozen-state (last-token) probe, per seed (v30 arms.floor) | `v30_logs/v30.json` | `d["arms"]["floor"]` | `[0.78, 0.782, 0.752]` |
| bAbI1 frozen-state (last-token) probe (mean, v30 floor) | `v30_logs/v30.json` | `d["floor"]` | `0.7713333333333333` |
| bAbI1 handover, per seed (v35 cells[0]: family none, rate 0) | `v35_logs/v35.json` | `rows(d["cells"], family="none")[0]["chain"]` | `[1.0, 1.0, 1.0]` |
| bAbI1 v30 n_train / n_eval | `v30_logs/v30.json` | `[d["n_train"], d["n_eval"]]` | `[900, 500]` |
| bAbI prompting zero-shot: n per cell (task 1, task 2) | `v49_logs/v49.json` | `[d["cells"]["task1/zeroshot/all"]["n"], d["cells"]["task2/zeroshot/all"]["n"]]` | `[500, 500]` |
| bAbI2 expressible instances: n_train / n_eval | `v37_logs/v37b.json` | `[d["n_train_expressible"], d["n_eval_expressible"]]` | `[775, 420]` |

### `F3a_three_arenas`

| label | file | key | value |
|---|---|---|---|
| LoRA on A per seed | `v1_logs/lora.json` | `accs(d)` | `[0.493, 0.508, 0.515]` |
| LoRA on A (mean) | `v1_logs/lora.json` | `mean(accs(d))` | `0.5053333333333333` |
| LoRA on A: seeds / trainable parameters | `v1_logs/lora.json` | `[[r["seed"] for r in d], d[0]["trainable_params"]]` | `[[42, 123, 456], 10490882]` |

### `F4e_crosstalk_geometry`

| label | file | key | value |
|---|---|---|---|
| v163 curve: target c | `v163_logs/v163.json` | `[c["target_c"] for c in d["curve"]]` | `[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]` |
| v163 curve: measured cross-talk (mean of 3 codebook seeds) | `v163_logs/v163.json` | `[c["crosstalk_mean"] for c in d["curve"]]` | `[0.00014959025793359615, 0.04921026900410652, 0.0989002212882042, 0.14872662723064423, 0.19863509138425192, 0.2486018588145574, 0.29861300190289813...` |
| v163 curve: measured cross-talk per seed | `v163_logs/v163.json` | `[c["crosstalk"] for c in d["curve"]]` | `[[0.00013061509525869042, 0.0003610173298511654, -4.2861651309067383e-05], [0.04977203533053398, 0.050192009657621384, 0.0476667620241642], [0.0996...` |
| v163 curve: acc under the fitted rule (mean) | `v163_logs/v163.json` | `[c["fitted_mean"] for c in d["curve"]]` | `[0.9222222222222222, 0.9166666666666666, 0.9022222222222221, 0.8316666666666667, 0.7016666666666667, 0.62, 0.5705555555555555, 0.5461111111111111, ...` |
| v163 curve: acc under the fitted rule per seed | `v163_logs/v163.json` | `[c["fitted"] for c in d["curve"]]` | `[[0.9183333333333333, 0.9216666666666666, 0.9266666666666666], [0.9033333333333333, 0.9266666666666666, 0.92], [0.8883333333333333, 0.9083333333333...` |
| v163 tolerance under the fitted rule (largest tested cross-talk clearing .615) | `v163_logs/v163_scored.json` | `d["tolerance"]` | `0.2486018588145574` |
| v165 projection before training: different-symbol cosine per seed | `v165_logs/v165.json` | `[s["init"]["different"] for s in d["seeds"]]` | `[0.7204259690565941, 0.7138714912992258, 0.7188225894020154]` |
| v165 projection before training: acc per seed (fitted threshold) | `v165_logs/v165.json` | `[s["init"]["acc"] for s in d["seeds"]]` | `[0.514, 0.5, 0.534]` |
| v165 projection trained through the binding loss: different-symbol cosine per seed (final epoch) | `v165_logs/v165.json` | `[s["epochs"][-1]["different"] for s in d["seeds"]]` | `[0.9879103341927895, 0.9894992642677747, 0.9878066407564359]` |
| v165 projection trained through the binding loss: acc_best per seed | `v165_logs/v165.json` | `[s["acc_best"] for s in d["seeds"]]` | `[0.472, 0.502, 0.476]` |
| v165 means: different init / different trained / acc_best | `v165_logs/v165.json` | `[d["different_init_mean"], d["different_trained_mean"], d["acc_best_mean"]]` | `[0.7177066832526117, 0.9884054130723333, 0.48333333333333334]` |
| v165 seeds / gates G0, B1, B2 | `v165_logs/v165.json` | `[[s["seed"] for s in d["seeds"]], d["G0"], d["B1"], d["B2"]]` | `[[42, 123, 456], true, true, true]` |
| fitted frozen-continuous projection (3 epochs): acc per seed | `crosstalk_audit2.json` | `d["fit3"]["acc_seeds"]` | `[0.9233333468437195, 0.9383333325386047, 0.9333333373069763]` |
| untrained random projection (frozen_cont arm, 0 fit epochs; crosstalk_audit2 fit0): cross-talk | `crosstalk_audit2.json` | `d["fit0"]["crosstalk"]` | `0.6987163659599093` |

### `F5a_2wiki_three_arms`

| label | file | key | value |
|---|---|---|---|
| 2Wiki compositional (phi-2): gold-facts prompt per seed | `v65_logs/v65_BANKED.json` | `d["arms"]["prompt_ev"]["covered"]` | `[0.95, 0.95, 0.95]` |
| 2Wiki inference (phi-2): prompt per seed | `v69_logs/v69_BANKED.json` | `d["arms"]["prompt"]["covered"]` | `[0.09333333333333334, 0.09333333333333334, 0.09333333333333334]` |
| 2Wiki inference (phi-2): gold-facts prompt per seed | `v69_logs/v69_BANKED.json` | `d["arms"]["prompt_ev"]["covered"]` | `[0.6833333333333333, 0.6833333333333333, 0.6833333333333333]` |
| 2Wiki compositional / inference: n instances | `v65_logs/v65_BANKED.json` | `d["n"]` | `300` |

### `F5b_bulky_evidence`

| label | file | key | value |
|---|---|---|---|
| compositional: all facts in prompt, share of prompts truncated | `v72b_logs/v72b_BANKED.json` | `d["arms"]["compositional"]["prompt_ev_big"]["trunc_rate"]` | `0.0` |
| inference: all facts in prompt, share of prompts truncated | `v72b_logs/v72b_BANKED.json` | `d["arms"]["inference"]["prompt_ev_big"]["trunc_rate"]` | `0.8266666666666667` |

### `F5b_musique`

| label | file | key | value |
|---|---|---|---|
| MuSiQue (Qwen3-8B, 31k-token window, v131c): single pass per seed | `v131_logs/v131c.json` | `d["arms"]["single_pass"]["covered"]` | `[0.28773584905660377, 0.28773584905660377, 0.28773584905660377]` |
| MuSiQue (Qwen3-8B, 31k-token window, v131c): gold-triples prompt +150 per seed | `v131_logs/v131c.json` | `d["arms"]["prompt_ev_p150"]["covered"]` | `[0.5943396226415094, 0.5660377358490566, 0.5754716981132075]` |
| MuSiQue (Qwen3-8B, v131c): share of prompts truncated, single pass / +150 | `v131_logs/v131c.json` | `[mean(d["diag"]["single_pass_trunc"]), mean(d["diag"]["prompt_ev_p150_trunc"])]` | `[0.0, 0.0]` |
| MuSiQue (Qwen3-8B, v131c): n / max_len / model | `v131_logs/v131c.json` | `[d["n"], d["config"]["max_len"], d["config"]["model"]]` | `[212, 31000, "Qwen/Qwen3-8B"]` |
| MuSiQue (phi-2, v79): share of prompts truncated, passages / +0 / +50 / +150 | `v79_logs/v79_BANKED.json` | `[mean(d["diag"]["prompt_trunc"]), mean(d["diag"]["prompt_ev_p0_trunc"]), mean(d["diag"]["prompt_ev_p50_trunc"]), mean(d["diag"]["prompt_ev_p150_trunc"])]` | `[0.8867924528301887, 0.0, 0.0, 1.0]` |

### `F5d_sharding_cost`

| label | file | key | value |
|---|---|---|---|
| all cells per seed | `v102_logs/v102_scored.json` | `d["grid"]` | `{"d2048_K128_S1": [0.7, 0.775, 0.7083333333333334], "d2048_K128_S2": [0.925, 0.9083333333333333, 0.925], "d2048_K128_S4": [0.925, 0.925, 0.93333333...` |

### `FS_capacity_v151`

| label | file | key | value |
|---|---|---|---|
| v151 dims | `v151_logs/v151.json` | `[int(x) for x in d["config"]["dims"]]` | `[512, 1024, 2048, 4096]` |
| v151 ks | `v151_logs/v151.json` | `[int(x) for x in d["config"]["ks"]]` | `[8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512, 724, 1024]` |
| v151 models | `v151_logs/v151.json` | `d["config"]["models"]` | `["HRR", "MAP", "BSC", "FHRR"]` |
| v151 n instances per cell / capacity seeds | `v151_logs/v151.json` | `[d["config"]["n"], d["config"]["cap_seeds"]]` | `[200, [0, 1, 2]]` |
| v151 readout[model][d] per k | `v151_logs/v151.json` | `{m: {dd: [d["capacity"][m][dd][str(k)] for k in d["config"]["ks"]] for dd in d["capacity"][m]} for m in d["config"]["models"]}` | `{"HRR": {"512": [0.945, 0.9466666666666667, 0.9366666666666666, 0.9100000000000001, 0.8449999999999999, 0.6583333333333333, 0.4033333333333333, 0.1...` |

### `F6b_clutrr`

| label | file | key | value |
|---|---|---|---|
| ours: n per k (k = 2..10) | `r1_logs/ours_by_k.json` | `[d["by_k"][str(k)]["n"] for k in range(2, 11)]` | `[38, 105, 190, 174, 107, 144, 150, 119, 119]` |
| ours: correct per k (k = 2..10) | `r1_logs/ours_by_k.json` | `[d["by_k"][str(k)]["correct"] for k in range(2, 11)]` | `[36, 70, 153, 136, 84, 98, 94, 77, 74]` |
| ours: accuracy per k (k = 2..10) | `r1_logs/ours_by_k.json` | `[d["by_k"][str(k)]["correct"] / d["by_k"][str(k)]["n"] for k in range(2, 11)]` | `[0.9473684210526315, 0.6666666666666666, 0.8052631578947368, 0.7816091954022989, 0.7850467289719626, 0.6805555555555556, 0.6266666666666667, 0.6470...` |
| ours: overall accuracy | `r1_logs/ours_by_k.json` | `d["correct"] / d["n_rows"]` | `0.7172774869109948` |

## Existing entries annotated (a `note` field added; label, key and value unchanged)

- `F4e_crosstalk_geometry` / "binding-loss projection (0 fit epochs): cross-talk": v52: MISLABELLED. crosstalk_audit2 fit0 is the frozen_cont arm with zero fitting epochs, an UNTRAINED random projection, not an arm trained through the binding loss (FIGURE_audit_v51.md, Fig. 2b finding 1). The trained arm is in v165_logs/v165.json. Label kept so that older scripts still resolve; not plotted in v52.
- `F4e_crosstalk_geometry` / "binding-loss projection (0 fit epochs): acc / per seed": v52: MISLABELLED, see the cross-talk entry. Accuracy of the untrained projection under the untrained (fixed) readout.

## `_meta`

- `_meta.v52_addenda` added (one sentence pointing to this file).
