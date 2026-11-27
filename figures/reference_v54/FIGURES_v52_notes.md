# Figures of v52: what each panel shows (for the captions)

Written 2026-11-11 with the figure revision that followed `FIGURE_audit_v51.md`. Every number in the tables below is
copied by program from `deck/review_v51/FIGURES_v52_values.json`, the file the figure scripts write as they plot; it holds
the same values at full precision together with the source string of each series.

## Regenerating every figure

```
cd /path/to/paper_v54
python3 figdata_v2/r1_logs/bank_ours_by_k.py   # CLUTRR counts per k from vsassm/r1_out/test_8B_think.jsonl (read-only)
python3 figdata_v2/add_index_v52.py            # INDEX.json addenda; idempotent (0 added on a second run)
python3 make_figs_v2.py                        # fig1_overview fig3_mechanism fig5_deployment fig2_audit arch fig4_identity figS_sweeps
python3 make_fig6_benchmarks.py                # fig6_benchmarks
```

Subsets: `python3 make_figs_v2.py 1 3 5 2 arch 4 S`. Each run prints a layout check per figure and rewrites
`figs/<name>.pdf`, `figs/preview/<name>.png` and `FIGURES_v52_values.json`.

Checks done on the final files:

- The scripts and `figdata_v2/` were copied to an empty scratch directory and run from scratch; all eight PDFs are
  pixel-identical (200 dpi, maximum difference 0) to the ones in `figs/`.
- All eight PDFs are 180.0 mm wide (510.24 pt), vector only (`pdfimages -list` finds nothing), creator
  "Matplotlib v3.10.8", fonts embedded TrueType (Liberation Sans; Computer Modern for the mathematical symbols of
  Fig. 1a and of the module schematic).
- Smallest text at the design size: 6.5 pt in every data panel; 6.0 pt for the sub- and superscripts of the two
  schematics (base 8.6 pt). At `\linewidth` = 6.5 in the figures are reduced to 0.917, so 6.5 pt prints at 5.96 pt and the
  subscripts at 5.5 pt.
- `figcheck.py` (run on every save) finds no text outside the canvas, no overlapping text boxes and no schematic label
  that leaves its box. Each PDF was rendered with `pdftoppm` and looked at; label-to-data collisions were fixed by hand.
- The PDFs are no longer cropped to their content (`bbox_inches="tight"` is off): the page is the design size.

## Conventions that hold in every figure

- **One name per arm:** frozen-state probe, exact codebook rows, projection trained through the binding loss, handover,
  continuous handover, LoRA fine-tune, prompting.
- **Colours** (plain name, hex): prompting blue `#0072B2`; LoRA fine-tune orange `#E69F00`; handover and HRR store
  vermilion `#D55E00`; projection trained through the binding loss pink `#CC79A7`; continuous handover sky blue
  `#56B4E9`; exact codebook rows black `#000000`; frozen-state probe grey `#7F7F7F` (green `#009E73` in the audit
  figure, see there). New in v52: the audited binding module violet `#6A3D9A` (`figstyle.C["module"]`). Vermilion is
  now used only for the handover and the store.
- **Seed dots:** small dark dots with a white edge on bars; faint dots in the series colour on line plots. A dot is
  drawn only where the arm has per-seed values that depend on the seed. Arms that repeat one deterministic number
  (greedy decoding of a fixed prompt, the HD readout) and single runs are drawn without dots. This is the rule you asked
  for in Fig. 1b, applied everywhere; it means that the three identical values of the 2Wiki prompt arms
  (`v65`/`v69` `prompt`, `prompt_ev`) get no dots. `DOTS_FOR_DETERMINISTIC = True` at the top of `make_figs_v2.py`
  draws them.
- **No titles** inside the figures. What the old titles said is now in the axis labels.
- Accuracies are fractions, except Fig. 3b and Fig. 4, which are per cent as before.

---

## Main Fig. 1, `figs/fig1_overview.pdf` (180 x 133 mm)

### a, schematic (`schematics.py` `draw_overview`)

No data. Three lanes from the same input ("Mary took the football. Mary went to the hallway. Where is the football?",
chain football -> Mary -> hallway): prompting (blue `#0072B2`, box "generate"), fine-tune (LoRA) (orange `#E69F00`, box
"low-rank update of the weights"), vector symbolic memory (this paper) (vermilion `#D55E00`, dashed frame). The memory
lane has four boxes: identity head (filled pale red `#FBE9E3`, "supervised on symbol labels; the one trained
component"), codebook rows (white, "fixed random unit vectors"), bind & bundle ("**one superposed vector per hop**"),
unbind & clean up ("two hops = two unbind-and-clean-up steps"), then "answer, always a codebook row". The frozen
language model is the blue-edged box filled `#E8EEF5`; "hidden states H" labels the arrow into the identity head.

Changed from v51: every label now fits its box (boxes 12 mm high, lane titles clear of the boxes and of the frame, the
memory lane's title beside the down arrow and not on the frame, the answer of the memory lane to the right of the last
box), "generate" has the size of its neighbours, and the bundle is labelled per hop.

### b, bAbI tasks 1 and 2 (Phi-2)

- x: two groups, "bAbI task 1 (one hop)", "bAbI task 2 (two hops)". y: accuracy, 0 to 1.05.
- Bars are means; dots are seeds. Two arms have no task-1 bar and carry a grey "n/a" (7 pt).

| Legend name | Colour | Task 1: mean (per seed) | Task 2: mean (per seed) | n and seeds | Source |
|---|---|---|---|---|---|
| prompting | blue `#0072B2` | 0.7920 (0.7920, 0.7920, 0.7920), no dots | 0.3860 (0.3860, 0.3860, 0.3860), no dots | 500 questions per task; one deterministic zero-shot decoding (the file stores the number three times), no dots. Task 2 is scored on all 500, the other task-2 arms on the 420 expressible questions | `v49_logs/v49.json` `cells["task1/zeroshot/all"].acc`, `cells["task2/zeroshot/all"].acc` |
| LoRA fine-tune | orange `#E69F00` | 0.9993 (0.9980, 1.0000, 1.0000) | 0.8008 (0.8500, 0.6952, 0.8571) | task 1: 900 train / 500 eval; task 2: 775 / 420; seeds 42, 123, 456 | `v50_logs/v50_task1_VOID_locoffset.json` `arms.lora_direct`; `v50_logs/v50_task2_BANKED.json` `arms.lora_direct` |
| LoRA fine-tune, best of 12 configurations | brown `#9C6500` | n/a | 0.9722 (0.9714, 0.9738, 0.9714) | task 2 only: 775 / 420, three seeds; configuration with the highest test mean of 12 (rank 16, 6 epochs, lr 1e-4). Task 1 was never run: n/a | `v131_logs/v131b.json`, run `r16_e6_lr0.0001`, `accs` |
| frozen-state probe | grey `#7F7F7F` | 0.7713 (0.7800, 0.7820, 0.7520) | 0.2452 (0.2524, 0.2357, 0.2476) | last-token linear probe. Task 1: 900 / 500, three seeds; task 2: 775 / 420, three seeds | `v30_logs/v30.json` `arms.floor` (new); `v48_logs/v48.json` `floor_last_token` |
| projection trained through the binding loss | pink `#CC79A7` | n/a | 0.5024 (0.4952, 0.5048, 0.5071) | task 2: 775 / 420, three seeds. Task 1: n/a, withheld (see below) | `v37_logs/v37b.json` `arms.spans` |
| handover | vermilion `#D55E00` | 1.0000 (1.0000, 1.0000, 1.0000) | 1.0000 (1.0000, 1.0000, 1.0000) | task 1: 2,000 train / 500 eval (defaults of v35.py), three seeds; task 2: 775 / 420, three seeds | `v35_logs/v35.json` `cells[0].chain` (family none; replaces the permuted-codebook cell of `v36.json`, same values); `v37_logs/v37b.json` `arms.probe_argmax` |

Notes for the caption:

- **Task-1 bar of the projection trained through the binding loss: withheld, as you instructed.** I checked the reason
  in the code: in `v30.py` the arm called `spans` is trained with `loss = 1 - dot(normalize(proj(state)), sym[sid])`
  (line 194), a supervised cosine regression of the projection onto the codebook row of the labelled symbol, not a
  loss passed through the binding chain. Its 0.980, 0.970, 0.978 are in `figdata_v2/v30_logs/v30.json` and are not read
  by any script.
- **Task-1 probe dots: added, and they are the bar's own probe.** `v30.py` lines 121-142 take the state at the last
  non-pad token and fit a linear probe on it; `v30.json` `arms.floor` = 0.780, 0.782, 0.752, mean 0.77133. The v51 bar read
  `v35.json` `floor` = 0.7713, which is the constant `FLOOR = 0.7713` typed in `v35.py` ("banked reference points,
  V-30/V-33"), i.e. this mean rounded. The bar is now the mean of the list (0.77133); the script asserts that the two
  agree to 5e-4.
- The task-1 arms do not share one training set: prompting 500 questions, LoRA and probe 900 train / 500 eval
  (V-50, V-30), handover 2,000 train / 500 eval (V-35). In V-30 itself the handover scored 0.778, 0.774, 0.770 because
  its identity head had not seen query positions (V-33); the 1.000 plotted is V-35's.
- The legend typo "Lora" is not in the figure (it was in the caption).

---

## Main Fig. 2, `figs/fig3_mechanism.pdf` (180 x 88 mm), three panels

### a, three arenas

- x: three groups with the labels "A: 3 hops, 16 pairs per level", "B: 1 hop, 64 bindings", "C: 2 hops, 32 bindings
  per level". y: chain accuracy, 0 to 1.05. Bars are means of three seeds, dots are the seeds (white-edged, so they
  show on the black bars).
- The label of A follows the audit and the generator (`systematic_chain_dataset.py`: three lists of `num_pairs = 16`
  pairs, R->X, X->Y, Y->filler; `v1_run.py:55`); you wrote "16 pairs", the counts are the same.

| Legend name | Colour | A: mean (per seed) | B: mean (per seed) | C: mean (per seed) | Source |
|---|---|---|---|---|---|
| frozen-state probe | grey `#7F7F7F` | 0.5090 (0.5080, 0.5110, 0.5080) | 0.8147 (0.7940, 0.8180, 0.8320) | 0.5153 (0.5120, 0.5300, 0.5040) | A `v1_logs/frozen_llm.json` `acc`; B `v11_logs/v18_spanfix.json` arm `floor`; C `v13_logs/v13_fixed.json` arm `floor` |
| projection trained through the binding loss | pink `#CC79A7` | 0.5010 (0.5110, 0.4990, 0.4930) | 0.5073 (0.5020, 0.4980, 0.5220) | 0.4833 (0.4720, 0.5020, 0.4760) | A `v8_logs/v8.json`; B `v18_spanfix` arm `spans`; C `v13_fixed` arm `spans` |
| LoRA fine-tune | orange `#E69F00` | 0.5053 (0.4930, 0.5080, 0.5150) | 0.7950 (0.9800, 0.8950, 0.5100) | 0.5158 (0.5150, 0.5175, 0.5150) | A `v1_logs/lora.json` `acc` (new; rank 16, 10,490,882 trainable parameters); B, C `v15_logs/v15.json` (rank 16) |
| continuous handover | sky blue `#56B4E9` | 0.6733 (0.6710, 0.6720, 0.6770) | 0.9853 (0.9780, 0.9900, 0.9880) | 0.9240 (0.9220, 0.9440, 0.9060) | A `runs/v9/v9.json` `armA.best`; B `v18_spanfix` arm `frozen_cont`; C `v13_fixed` arm `frozen_cont` |
| handover | vermilion `#D55E00` | 0.9137 (0.9150, 0.9130, 0.9130) | 0.9960 (0.9900, 1.0000, 0.9980) | 0.9340 (0.9320, 0.9320, 0.9380) | A `v9_logs/v9_gpu1_rerun_partial.json` arm `extract_then_bind`; B `v18_spanfix` arm `probe_argmax`; C `v13_fixed` arm `probe_argmax` |
| exact codebook rows | black `#000000` | 0.9183 (0.9180, 0.9190, 0.9180) | 0.9960 (0.9900, 1.0000, 0.9980) | 0.9340 (0.9320, 0.9320, 0.9380) | A `v4_logs/v4_perocc.json` epsilon 0; B `v18_spanfix` arm `parser`; C `v13_fixed` arm `parser` |

n and seeds: three seeds per bar (42, 123, 456). Arena A, probe and LoRA: V-1 protocol, 2,000 train / 1,000 eval
(audit). Arena C: train arena 800, test arena 500 (`v13.py`, as V-165 restates it). The LoRA bars on B and C are from
`v15.py`: rank 16, 600 train / 400 test, best test epoch; on C the three seeds share one data draw (train arena seed
42, test arena seed 4242). The artifacts do not record n for the other arms of arenas A and B.
The LoRA bar on arena A is new (0.493, 0.508, 0.515; it was "n/a" in v51).

### b, imposed cross-talk

- x: cross-talk, the mean cosine between different symbols' codes. **The axis is linear from 0 to 0.1 and logarithmic
  from 0.1 to 1** (matplotlib `symlog`, `linthresh = 0.1`); labelled ticks 0, 0.05, 0.1, 0.2, 0.5, 1, minor ticks at 0.3,
  0.4, 0.6, 0.7, 0.8, 0.9. A fully linear axis puts the handover (0.0001) and the continuous handover (0.03) on top of
  each other; a fully logarithmic one squeezes the fall of the curve. Please say so in the caption.
- y: chain accuracy on arena C under the fitted decision rule (threshold that maximizes accuracy on the training
  arena, as every arm of Table 1), 0.4 to 1.0.
- Black curve with dots: exact codebook rows with cross-talk imposed, V-163, mean of three codebook seeds at each of 14
  levels; faint grey dots are the three seeds, each at its own measured cross-talk. Test arena 600 instances per seed,
  threshold fitted on a training arena of 800. The fixed-rule (V-95) curve is not drawn.
- Dashed grey line at 0.615, labelled "probe + 0.10"; dotted grey line at 0.515, labelled "frozen-state probe" (the
  arena-C floor of panel a, 0.5153, rounded as V-95 and V-163 fix it).
- Thin dashed vertical line from the x axis to the curve at 0.2486, labelled "tolerance 0.25": the largest imposed
  cross-talk whose mean accuracy still clears 0.615 (`v163_scored.json` `tolerance`; accuracy there 0.620, next level
  0.2986 at 0.571).

| Imposed cross-talk (mean) | Accuracy, fitted rule (mean) | Per seed |
|---|---|---|
| 0.0001 | 0.9222 | 0.9183, 0.9217, 0.9267 |
| 0.0492 | 0.9167 | 0.9033, 0.9267, 0.9200 |
| 0.0989 | 0.9022 | 0.8883, 0.9083, 0.9100 |
| 0.1487 | 0.8317 | 0.8233, 0.8333, 0.8383 |
| 0.1986 | 0.7017 | 0.6917, 0.6883, 0.7250 |
| 0.2486 | 0.6200 | 0.6117, 0.5867, 0.6617 |
| 0.2986 | 0.5706 | 0.5700, 0.5367, 0.6050 |
| 0.3487 | 0.5461 | 0.5417, 0.5067, 0.5900 |
| 0.3987 | 0.5356 | 0.5283, 0.5017, 0.5767 |
| 0.4989 | 0.5322 | 0.5067, 0.5167, 0.5733 |
| 0.5992 | 0.5311 | 0.5333, 0.5000, 0.5600 |
| 0.6995 | 0.5206 | 0.5100, 0.4967, 0.5550 |
| 0.7997 | 0.5044 | 0.4867, 0.5083, 0.5183 |
| 0.8999 | 0.5256 | 0.5117, 0.5233, 0.5417 |

Source: `v163_logs/v163.json` `curve[].crosstalk_mean`, `fitted_mean`, `crosstalk`, `fitted`; `v163_scored.json`
`tolerance`.

Arms, each drawn at (its measured cross-talk, the accuracy it reached); the small dots of the same colour are seeds:

| Legend name | Marker, colour | Cross-talk: mean (per seed) | Accuracy: mean (per seed) | n and seeds | Source |
|---|---|---|---|---|---|
| handover | filled circle, vermilion `#D55E00` | 0.0001 (0.0001) | 0.9340 (0.9320, 0.9320, 0.9380) | cross-talk: one seed (seed 0), 200 test items; accuracy: three seeds, 500 test items (Table 1 configuration) | x `geometry_banked.json` `crosstalk_discrete`; y `v13_logs/v13_fixed.json` arm `probe_argmax` |
| continuous handover | filled square, sky blue `#56B4E9` | 0.0315 (0.0250, 0.0441, 0.0255) | 0.9317 (0.9233, 0.9383, 0.9333) | three seeds (0, 1, 2), 1,500 train / 600 test (audit-2 protocol), projection fitted for 3 epochs | `crosstalk_audit2.json` `fit3.xt_seeds`, `fit3.acc_seeds` |
| projection, before training | open diamond, pink `#CC79A7` | 0.7177 (0.7204, 0.7139, 0.7188) | 0.5160 (0.5140, 0.5000, 0.5340) | three seeds (42, 123, 456), train arena 800 / test arena 500; accuracy with the fitted threshold at initialization | `v165_logs/v165.json` `seeds[].init.different`, `seeds[].init.acc` |
| projection trained through the binding loss | filled diamond, pink `#CC79A7` | 0.9884 (0.9879, 0.9895, 0.9878) | 0.4833 (0.4720, 0.5020, 0.4760) | same three seeds and arenas; cross-talk after the final (second) epoch; accuracy at the best test epoch (`acc_best`, equal to the banked arm 0.472, 0.502, 0.476; gate G0 passes) | `v165_logs/v165.json` `seeds[].epochs[-1].different`, `seeds[].acc_best` |

Decisions in this panel:

- **The audit's 0.699 "untrained random projection" marker is dropped.** With V-165 present it would sit 0.02 from the
  "projection, before training" point (0.718) and hide it. The script still draws it, labelled "untrained random
  projection (0.70)", whenever `figdata_v2/v165_logs/v165.json` is absent, and prints a line saying so (tested in a
  scratch copy).
- The markers lie under the curve, except the open diamond, which is drawn over it so that the curve's own point at 0.70
  shows through. Nothing hides a curve point.
- The panel still mixes harnesses, as the audit noted: the curve (V-163, 600 test items), the handover (cross-talk from
  one seed of `geometry_banked.json`, accuracy from the Table 1 run), the continuous handover (audit 2, seeds 0, 1, 2) and
  the projection (V-165, seeds 42, 123, 456). The caption should say that each arm is plotted at its own measurement.
- For seed 123 the trained projection's best accuracy (0.502) is at epoch 1 and its cross-talk (0.9895) is taken after
  epoch 2 (0.9883 after epoch 1). The difference is invisible at this scale.

### c, one projection read two ways (V-21)

- x: epochs fitting the projection, five categories 0, 1, 3, 10, 30, equally spaced. y: chain accuracy, 0.4 to 1.06.
- Three seeds per point (faint dots). The run is the 256-symbol variant of the two-hop arena (`v21.json`: `hops` 2,
  `n_ctx` 32, `n_sym` 256); nothing in the panel says "arena C".

| Legend name / label | Colour, marker | Values at 0, 1, 3, 10, 30 epochs: mean | Per seed | Source |
|---|---|---|---|---|
| continuous handover | sky blue `#56B4E9`, circles | 0.4807, 0.6580, 0.7533, 0.8373, 0.8733 | 0.4960, 0.4580, 0.4880; 0.6660, 0.6500, 0.6580; 0.7480, 0.7660, 0.7460; 0.8280, 0.8520, 0.8320; 0.8720, 0.8680, 0.8800 | `v21_logs/v21.json` `cells[].frozen_cont` |
| the same projection, snapped to the nearest codebook row | vermilion `#D55E00`, squares | 0.4773, 0.9773, 0.9800, 0.9840, 0.9873 | 0.4880, 0.4560, 0.4880; 0.9760, 0.9740, 0.9820; 0.9800, 0.9800, 0.9800; 0.9860, 0.9800, 0.9860; 0.9880, 0.9840, 0.9900 | `v21_logs/v21.json` `cells[].proj_argmax` |
| exact codebook rows | black `#000000`, dashed horizontal line with a text label | 0.9920 | 0.9920, 0.9900, 0.9940 | `v21_logs/v21.json` `parser` |
| frozen-state probe | grey `#7F7F7F`, dashed horizontal line with a text label | 0.5173 | 0.4860, 0.5100, 0.5560 | `v21_logs/v21.json` `floor` |

The vermilion series is the same fitted projection snapped to its nearest codebook row (`proj_argmax`), not the
handover of panel a (which uses an identity classifier); the legend says "the same projection, snapped to the nearest
codebook row". n per point is not recorded in the artifact.

Moved out of this figure: bindings per level and chain depth are now panels a and b of `figs/figS_sweeps.pdf`.

---

## Main Fig. 3, `figs/fig5_deployment.pdf` (180 x 120 mm), three panels

### a, three routes on 2WikiMultihopQA and MuSiQue

- x: seven groups: "2Wiki compositional", "2Wiki inference", "2Wiki comp. +50 instances", "2Wiki inf. +50 instances",
  "MuSiQue", "MuSiQue +50 facts", "MuSiQue +150 facts"; a thin grey vertical line separates 2Wiki from MuSiQue.
  y: exact match, 0 to 1.05.
- Colour is the route: prompting (passages in the prompt) blue `#0072B2`; facts in the prompt orange `#E69F00`; facts
  in the HRR store, read by unbinding vermilion `#D55E00`.
- Bar style is the reader and whether its input was cut: **solid** = Phi-2, nothing truncated (and the store, which has
  no reader); **white with hatching in the route's colour** = Phi-2, input truncated (legend "Phi-2, input truncated");
  **solid with a black outline** = Qwen3-8B with a 31,000-token window, nothing truncated (legend "Qwen3-8B, 31k-token
  window, nothing truncated").
- **Choice between the two options you gave: the truncated Phi-2 bars are kept and hatched, not dropped.** One rule
  then covers all three truncated bars (the pooled 2Wiki inference prompt cannot be dropped, panel b depends on it),
  and no measured number disappears from the figure. On MuSiQue the solid outlined bars stand next to the store.
- A bar shorter than 0.02 carries its value: "0.007 (truncated)" for 2Wiki inference +50.
- The prompting route has no bar in the four pooled groups: the pool adds facts, not passages, and no such arm exists.

| Group | Route | Reader | Share of prompts truncated | Style | Mean | Per seed | Dots | Source |
|---|---|---|---|---|---|---|---|---|
| 2Wiki compositional | prompting (passages in the prompt) | Phi-2 | 0.000 | solid | 0.2533 | 0.2533, 0.2533, 0.2533 | no | `v65_logs/v65_BANKED.json arms.prompt.covered` |
| 2Wiki compositional | facts in the prompt | Phi-2 | 0.000 | solid | 0.9500 | 0.9500, 0.9500, 0.9500 | no | `v65 arms.prompt_ev.covered` |
| 2Wiki compositional | facts in the HRR store, read by unbinding | none (store readout) | 0.000 | solid | 0.9744 | 0.9700, 0.9767, 0.9767 | yes | `v65 arms.hrr_gold.covered` |
| 2Wiki inference | prompting (passages in the prompt) | Phi-2 | 0.000 | solid | 0.0933 | 0.0933, 0.0933, 0.0933 | no | `v69_logs/v69_BANKED.json arms.prompt.covered` |
| 2Wiki inference | facts in the prompt | Phi-2 | 0.000 | solid | 0.6833 | 0.6833, 0.6833, 0.6833 | no | `v69 arms.prompt_ev.covered` |
| 2Wiki inference | facts in the HRR store, read by unbinding | none (store readout) | 0.000 | solid | 0.7622 | 0.7567, 0.7433, 0.7867 | yes | `v69 arms.hrr_gold.covered` |
| 2Wiki comp. +50 instances | facts in the prompt | Phi-2 | 0.000 | solid | 0.3533 | single run | no | `v72b_logs/v72b_BANKED.json arms.compositional.prompt_ev_big.em (single run)` |
| 2Wiki comp. +50 instances | facts in the HRR store, read by unbinding | none (store readout) | 0.000 | solid | 0.9533 | single run | no | `v72b arms.compositional.hrr_gold_big.em (single run)` |
| 2Wiki inf. +50 instances | facts in the prompt | Phi-2 | 0.827 | hatched | 0.0067 | single run | no | `v72b arms.inference.prompt_ev_big.em (single run)` |
| 2Wiki inf. +50 instances | facts in the HRR store, read by unbinding | none (store readout) | 0.000 | solid | 0.7233 | single run | no | `v72b arms.inference.hrr_gold_big.em (single run)` |
| MuSiQue | prompting (passages in the prompt) | Phi-2 | 0.887 | hatched | 0.1651 | 0.1651, 0.1651, 0.1651 | no | `v79_logs/v79_BANKED.json arms.prompt.covered` |
| MuSiQue | prompting (passages in the prompt) | Qwen3-8B | 0.000 | outlined | 0.2877 | 0.2877, 0.2877, 0.2877 | no | `v131_logs/v131c.json arms.single_pass.covered` |
| MuSiQue | facts in the prompt | Phi-2 | 0.000 | solid | 0.8994 | 0.9009, 0.9009, 0.8962 | yes | `v79 arms.prompt_ev_p0.covered` |
| MuSiQue | facts in the HRR store, read by unbinding | none (store readout) | 0.000 | solid | 0.9481 | 0.9387, 0.9481, 0.9575 | yes | `v79 arms.hrr_gold_p0.covered` |
| MuSiQue +50 facts | facts in the prompt | Phi-2 | 0.000 | solid | 0.3475 | 0.3491, 0.3396, 0.3538 | yes | `v79 arms.prompt_ev_p50.covered` |
| MuSiQue +50 facts | facts in the HRR store, read by unbinding | none (store readout) | 0.000 | solid | 0.9324 | 0.9151, 0.9387, 0.9434 | yes | `v79 arms.hrr_gold_p50.covered` |
| MuSiQue +150 facts | facts in the prompt | Phi-2 | 1.000 | hatched | 0.1871 | 0.1557, 0.2264, 0.1792 | yes | `v79 arms.prompt_ev_p150.covered` |
| MuSiQue +150 facts | facts in the prompt | Qwen3-8B | 0.000 | outlined | 0.5786 | 0.5943, 0.5660, 0.5755 | yes | `v131c arms.prompt_ev_p150.covered` |
| MuSiQue +150 facts | facts in the HRR store, read by unbinding | none (store readout) | 0.000 | solid | 0.6557 | 0.6792, 0.6651, 0.6226 | yes | `v79 arms.hrr_gold_p150.covered` |

n and seeds: 2Wiki compositional and inference 300 instances each, three seeds (42, 123, 456; the prompt arms repeat
one deterministic value, only the store's values differ between seeds). The pooled 2Wiki groups are single runs
(`v72b_BANKED.json` holds one `em` per arm; the values are consistent with 300 instances: 0.3533 = 106/300, 0.0067 = 2/300). MuSiQue: 212 instances (181 two-hop, 5
three-hop, 26 four-hop), three seeds; the pooled arms vary with the seed because the distractor facts are redrawn.
`v131c.json`: Qwen/Qwen3-8B, `max_len` 31,000, 212 instances, three seeds, `diag.*_trunc` = 0.

For the caption: on MuSiQue the store (0.948) is to be compared with the untruncated single pass (0.288), and at +150
facts the store (0.656) with the untruncated facts prompt (0.579). The facts-in-prompt bars at +0 (0.899) and +50
(0.348) are Phi-2 and were not truncated; no Qwen3-8B run exists for them.

### b, answers naming an entity that is not in the evidence

- x: two groups, "2Wiki comp. +50 instances", "2Wiki inf. +50 instances". y: answers naming an absent entity (%), 0 to
  85. Single runs (the values are consistent with 300 instances each), no dots. The two zero bars carry a vermilion "0".

| Legend name | Colour | Compositional (%) | Inference (%) | Source |
|---|---|---|---|---|
| facts in the prompt | orange `#E69F00` | 22.00 | 77.33 (hatched: 0.827 of these prompts were truncated) | `v72b_logs/v72b_BANKED.json arms.<type>.prompt_ev_big.hallucination (single run)` |
| facts retrieved from the store, then placed in the prompt | pale orange `#F5C77E` | 4.00 | 2.33 | `v72b_logs/v72b_BANKED.json arms.<type>.hrr_then_prompt_big.hallucination (single run)` |
| facts in the HRR store | vermilion `#D55E00` | 0.00 | 0.00 | `v72b_logs/v72b_BANKED.json arms.<type>.hrr_gold_big.hallucination (single run)` |

The orange arm is the same arm as the orange bars of panel a at "+50 instances" and has the same name. The fourth legend
entry, "Phi-2, input truncated", explains the hatched inference bar.

### c, sharding at d = 2048 (V-102)

- x: stores the facts are spread over, S = 1, 2, 4, 8, 16 (log scale, base 2). y: readout exact match, 0 to 1.
- One curve per pool size, shades of orange from light to dark (matplotlib "Oranges"); faint dots are the three seeds
  of each cell. Dashed black horizontal line: "0.70 criterion".
- **The curve that v51 labelled "K = 1024" is labelled "K = 800 (all facts; 1,024 requested)".** `v102.py:96` loads
  400 instances and `pooled()` (`v100.py:38-53`) stops when the other instances' gold triples run out; with two triples
  per instance (V-73: 600 bindings from 300 instances) the pool holds 800 facts. `v102_scored.json` does not record the
  realized pool size, so 800 rests on the code and on V-73; its 16-store point is 50 facts per store, not 64.

| Legend name | Colour | Mean at S = 1, 2, 4, 8, 16 | Per seed | Source |
|---|---|---|---|---|
| K = 128 | shade of orange `#fda762` | 0.7278, 0.9194, 0.9278, 0.9306, 0.9278 | 0.7000, 0.7750, 0.7083; 0.9250, 0.9083, 0.9250; 0.9250, 0.9250, 0.9333; 0.9333, 0.9250, 0.9333; 0.9250, 0.9250, 0.9333 | `v102_logs/v102_scored.json grid['d2048_K128_S*']` |
| K = 256 | shade of orange `#f57520` | 0.2194, 0.6639, 0.9056, 0.9250, 0.9250 | 0.2250, 0.2583, 0.1750; 0.6750, 0.7250, 0.5917; 0.9083, 0.9000, 0.9083; 0.9250, 0.9167, 0.9333; 0.9250, 0.9250, 0.9250 | `v102_logs/v102_scored.json grid['d2048_K256_S*']` |
| K = 512 | shade of orange `#d14501` | 0.0389, 0.1889, 0.6472, 0.9194, 0.9222 | 0.0500, 0.0417, 0.0250; 0.1417, 0.2333, 0.1917; 0.6583, 0.5917, 0.6917; 0.9167, 0.9250, 0.9167; 0.9250, 0.9250, 0.9167 | `v102_logs/v102_scored.json grid['d2048_K512_S*']` |
| K = 800 (all facts; 1,024 requested) | shade of orange `#8e2d04` | 0.0083, 0.0444, 0.2556, 0.7222, 0.9083 | 0.0167, 0.0083, 0.0000; 0.0417, 0.0333, 0.0583; 0.2917, 0.2417, 0.2333; 0.7000, 0.7417, 0.7250; 0.9250, 0.9000, 0.9000 | `v102_logs/v102_scored.json grid['d2048_K1024_S*']` |

n and seeds: three seeds per cell (0, 1, 2) and 120 query instances per seed: the defaults of `v102.py` (`--n 120`),
consistent with every value being a multiple of 1/120. Neither is recorded in the artifact.

Moved out of this figure: capacity of one store and facts held at 0.70 against dimension are now panels c and d of
`figs/figS_sweeps.pdf`, redrawn from V-151.

---

## Main Fig. 4, `figs/fig6_benchmarks.pdf` (180 x 86 mm), `make_fig6_benchmarks.py`

No number is typed in the script any more. It reads `figdata_v2/benchmarks/babilong.json` and
`figdata_v2/benchmarks/clutrr.json` (one `source` string per value; I re-checked every transcribed value against
`vsassm/results/b18_verdict.txt`, `b18_api_verdict.txt` and `r1_verdict.txt` by program when writing the files) and
`figdata_v2/r1_logs/ours_by_k.json`.

### a, BABILong

- x: context length, eight equally spaced positions 0k, 4k, 16k, 64k, 128k, 512k, 1M, 10M; label "context length
  (tokens; positions not to scale)". y: BABILong accuracy, QA1-5 (%), 28 to 102.5.
- Under the accuracy axis there is a pale grey strip (`#F1F1F1`) that is not part of the accuracy scale. At 1M it holds
  two crosses, blue for GPT-6.1 Sol and pink for Claude Opus 5.5, beside the text "input rejected: exceeds context
  window". The v51 ticks at "33 %" are gone.

| Legend name | Colour, marker | 0k | 4k | 16k | 64k | 128k | 512k | 1M | 10M | n per point |
|---|---|---|---|---|---|---|---|---|---|---|
| ours (fine-tuned 1.7B reader + memory) | vermilion `#D55E00`, circles, thick line | 99.6 | 97.8 | 99.2 | 99.6 | 99.8 | 100.0 | 100.0 | 100.0 | 0k: 500; 4k: 500; 16k: 500; 64k: 500; 128k: 500; 512k: 125; 1M: 125; 10M: 25 |
| ARMT 145M, fine-tuned | black `#000000`, squares | 99.3 | 98.1 | 98.1 | 97.9 | 96.9 | 95.3 | 93.4 | 76.6 | not ours: leaderboard row |
| GPT-6.1 Sol, zero-shot* | blue `#0072B2`, triangles | 100.0 | 99.2 | 100.0 | 99.2 | 96.8 | 96.0 | rejected | - | 0k: 125; 4k: 125; 16k: 125; 64k: 125; 128k: 125; 512k: 25 |
| Claude Opus 5.5, zero-shot* | pink `#CC79A7`, diamonds | 100.0 | 97.6 | 95.2 | 96.0 | 97.6 | 88.0 | rejected | - | 0k: 125; 4k: 125; 16k: 125; 64k: 125; 128k: 125; 512k: 25 |
| GPT-4-0125 | grey `#7F7F7F`, inverted triangles | 87.0 | 74.0 | 64.0 | 43.0 | 36.0 | - | - | - | not ours: leaderboard row |
| Llama-3.1-70B | light grey `#BBBBBB`, crosses | 85.0 | 74.0 | 65.0 | 53.0 | 45.0 | - | - | - | not ours: leaderboard row |

Sources: ours `b18_verdict.txt`, table row "QA1-5" and "QA1-5 10M: 25/25 = 100.0"; ARMT the leaderboard line of the
same file and, at 10M, the ARMT paper's Table 1 as quoted there (mean of 89.4, 84.4, 37.0, 91.5, 80.9); GPT-6.1 Sol and
Claude Opus 5.5 `b18_api_verdict.txt` (first 25 questions per task, first 5 per task at 512k; the asterisk in the legend
stands for that subset); GPT-4-0125 and Llama-3.1-70B the leaderboard lines of `b18_verdict.txt`.

- Ours is labelled "ours (fine-tuned 1.7B reader + memory)".
- The ARMT line and its label "ARMT 145M, fine-tuned" are unchanged, and no other long-context system was added. The
  figure itself does not contain the words "best published".

### b, CLUTRR

- x: relations in the chain, k = 2..10. y: CLUTRR accuracy (%), 35 to 102.5.

| Legend name | Colour, style | k = 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| ours (frozen Qwen3-8B reader + rule memory) | vermilion `#D55E00`, circles, thick line | 0.9474 | 0.6667 | 0.8053 | 0.7816 | 0.7850 | 0.6806 | 0.6267 | 0.6471 | 0.6218 |
| n (correct) of ours | | 38 (36) | 105 (70) | 190 (153) | 174 (136) | 107 (84) | 144 (98) | 150 (94) | 119 (77) | 119 (74) |
| Qwen3-8B answering directly (reasoning on) | blue `#0072B2`, triangles, thin line | 0.921 | 0.400 | 0.768 | absent | 0.673 | absent | 0.580 | absent | 0.496 |

(The figure plots these as per cent.)

Reference lines, all chain lengths pooled, in the legend under the panel:

| Legend name | Colour, style | Value | Source |
|---|---|---|---|
| rule memory given gold relations, 97.5 | black `#000000`, dotted | 97.5 | vsassm/results/r1_verdict.txt: 'The composition engine is exact (oracle facts: 0.975, 1.000 at k=10)'; vsassm/results/nsr0_verdict.txt: 'T reproduces R1's 0.9747 exactly' |
| SatLM (code-davinci-002, rules given), 68.3 | dark grey `#555555`, dash-dotted | 68.3 | Ye et al., SatLM, NeurIPS 2023, Table 1 (CLUTRR, code-davinci-002, greedy; 1,042-story version of the test set, kinship rules given); transcribed from vsassm/docs/neurosymbolic_review.md lines 27, 98 and 147 |
| DSR-LM (rules learned from data), 61.0 | grey `#7F7F7F`, dashed | 61.0 | Zhang et al., DSR-LM, Findings of ACL 2023, Tables 5/6: 60.98 +- 2.64; transcribed from vsassm/docs/neurosymbolic_review.md lines 107 and 150 |

- **Ours at all nine k, recomputed** from `vsassm/r1_out/test_8B_think.jsonl` (1,146 rows, field `ok`, grouped by
  `len(gold_path)`); counts banked in `figdata_v2/r1_logs/ours_by_k.json` by `figdata_v2/r1_logs/bank_ours_by_k.py`.
  **No mismatch:** the five even-k values agree with the ones typed in the v51 script to three decimals (0.947, 0.805,
  0.785, 0.627, 0.622), and the odd ones are the expected 0.667 (n = 105), 0.782, 0.681, 0.647. Overall 822/1146 = 0.7173.
- The direct-answer arm has the five even k from the table of `r1_verdict.txt` and k = 3 = 0.400 from its line "(k=3:
  0.667 ours, 0.400 Qwen3-8B; ...)" (verified). k = 5, 7, 9 are absent. Neighbouring k are joined by a solid line; **a
  dashed line bridges each missing k**, so the reader sees where there is no point.
- `figdata_v2/r1_logs/direct_by_k.json` is picked up automatically when you add it: `{"5": {"n": 174, "correct": ..},
  ...}` (or the same under `"by_k"`; other keys are ignored). Every k in the file is drawn from the file, checked against
  the transcribed value where both exist, and the dashes disappear. Tested with a synthetic file in a scratch copy
  only; the script prints which case it is in.
- The curve is not monotone once the odd k are drawn: k = 3 (0.667) is 14 points under k = 4 (0.805), and the direct
  arm drops to 0.400 at k = 3.

---

## Supplementary, `figs/fig2_audit.pdf` (180 x 116 mm): audit of the original evaluation

### a, original generators

- x: four tasks (variable tracking, rule priority, state updates, entity binding). y: "accuracy, original generators",
  **0 to 1** (the bars now start at zero).
- Bars: majority class, light grey `#BBBBBB`. Green `#009E73` dots and a green tick: the **mean-pooled probe**, three
  seeds and their mean (legend "mean-pooled probe (seeds, mean)").

| Task | Majority class | Mean-pooled probe: mean (per seed) |
|---|---|---|
| Variable tracking | 0.502 | 0.4636 (0.4568, 0.4486, 0.4856) |
| Rule priority | 0.673 | 0.7057 (0.7344, 0.6562, 0.7266) |
| State updates | 0.658 | 0.6588 (0.6549, 0.6667, 0.6549) |
| Entity binding | 0.513 | 0.5718 (0.4837, 0.6260, 0.6057) |

Sources: majority `ledger_57406f1_label_priors.json` (transcribed from the ledger; 8,000 generated items); probe
`v2_logs/v2.json`, arm `llm_only`, `test_per_task` (1,000-item test split, i.e. 243, 256, 255, 246 items per task; three
seeds).

### b, rebuilt arenas

- x: four arenas (variable tracking, bAbI 1, bAbI 2, bAbI 16). y: "test accuracy, rebuilt arenas", 0 to 1. Bars are
  means of three seeds with seed dots; the HD readout is deterministic (one value repeated) and has no dots. Short
  dashed black line across each group: majority class.

| Legend name | Colour | variable tracking | bAbI 1 | bAbI 2 | bAbI 16 | Source |
|---|---|---|---|---|---|---|
| frozen-state probe | green `#009E73` | 0.7803 (0.7900, 0.7660, 0.7850) | 0.7510 (0.7520, 0.7590, 0.7420) | 0.2963 (0.3030, 0.3240, 0.2620) | 0.5447 (0.5460, 0.5330, 0.5550) | `v7_logs/v7_last.json` `arms.probe[].test` |
| parameter-matched MLP | yellow `#F0E442` | 0.7887 (0.7850, 0.7860, 0.7950) | 0.7323 (0.7510, 0.7250, 0.7210) | 0.2900 (0.2870, 0.2750, 0.3080) | 0.5450 (0.5260, 0.5670, 0.5420) | `v7_last.json` `arms.mlp[].test` |
| HD readout (adaptive) | reddish brown `#8C564B` | 0.6290 (deterministic) | 0.7500 (deterministic) | 0.2490 (deterministic) | 0.5320 (deterministic) | `v12_logs/v12.json` `hd["raw/online"]` |
| audited binding module (34.2M) | violet `#6A3D9A` | 0.7337 (0.7450, 0.7360, 0.7200) | 0.7517 (0.7440, 0.7800, 0.7310) | 0.2103 (0.2610, 0.1670, 0.2030) | 0.5237 (0.5130, 0.5320, 0.5260) | `v7_last.json` `arms.vsa[].test` |
| majority class | near-black `#222222`, dashed line | 0.512 | 0.187 | 0.187 | 0.257 | `v7_last.json` `majority` |

The audited module is violet `#6A3D9A` (it was vermilion). The probe of this panel is the last-token probe and is named
"frozen-state probe"; it keeps the green of this figure (grey in the main figures), because grey here is the majority
class and the pooling variants. n per arena is not recorded in the artifact.

### c, which state is read

- x: variable tracking, bAbI 1. y: probe accuracy, 0 to 1. Three seeds per bar.

| Legend name | Colour | variable tracking: mean (per seed) | bAbI 1: mean (per seed) |
|---|---|---|---|
| mean-pooled probe | light grey `#AAAAAA` | 0.6563 (0.6560, 0.6490, 0.6640) | 0.3577 (0.3750, 0.3490, 0.3490) |
| max-pooled probe | dark grey `#666666` | 0.7000 (0.6930, 0.6970, 0.7100) | 0.3583 (0.3510, 0.3300, 0.3940) |
| last-token probe (the frozen-state probe) | green `#009E73` | 0.7873 (0.7890, 0.7850, 0.7880) | 0.7523 (0.7570, 0.7350, 0.7650) |

Source: `v6_logs/v6.json` `probe.mean`, `probe.max`, `probe.last`.

### d, placing the question

- x: chain depth (hops), 1 to 3. y: frozen-state probe accuracy, 0.4 to 1.0. **Two seeds** per point (faint dots).
  Dotted black line: "chance, 0.5" (now in the legend).
- Colour is the position of the question, line style the context size. The mid-sequence lines are near-black now, not
  vermilion.

| Legend name | Colour, style | 1 hop: mean (per seed) | 2 hops | 3 hops |
|---|---|---|---|---|
| question at the end, 16 context items | grey `#7F7F7F`, solid, circles | 0.9363 (0.9425, 0.9300) | 0.6587 (0.6350, 0.6825) | 0.6587 (0.6375, 0.6800) |
| question at the end, 32 context items | grey `#7F7F7F`, dashed, squares | 0.9087 (0.9225, 0.8950) | 0.6013 (0.6000, 0.6025) | 0.5938 (0.6150, 0.5725) |
| question mid-sequence, 16 context items | near-black `#222222`, solid, circles | 0.6538 (0.6325, 0.6750) | 0.4988 (0.4975, 0.5000) | 0.5362 (0.5150, 0.5575) |
| question mid-sequence, 32 context items | near-black `#222222`, dashed, squares | 0.6150 (0.6175, 0.6125) | 0.4737 (0.4800, 0.4675) | 0.5275 (0.5250, 0.5300) |

Source: `v13_logs/floor_survey.json`, twelve configurations, `floor_per_seed`. The configuration chosen as arena C is
two hops, 32 context items, question mid-sequence (0.4737).

---

## Supplementary, `figs/arch.pdf` (180 x 70 mm): the binding module under audit (`schematics.py` `draw_binding_module`)

No data. Redrawn at 180 mm so that its text has the scale of the other figures (it was designed at 136 mm and enlarged).

- Frozen language model: box with blue `#0072B2` edge, filled `#E8EEF5`, "Phi-2, 2.7B"; legend swatch "frozen".
- Trained components: white boxes with a **violet `#6A3D9A`** edge; legend swatch "trained (34.2M parameters in
  total)". These are: projection `v_i = MLP_in(h_i)`; gate `g_i = sigma(MLP_g(h_i))`; role vectors `rho(1), ..., rho(R)`;
  bind top-3 and sum `M = norm sum_j v_j (x) rho(r_j)`, "j: the three largest gates g_j"; back-projection
  `m = MLP_out(M)`; gated fusion `h'_i = g_i MLP_f([h_i; m]) + (1 - g_i) h_i`; and now also **"mean pool, classifier"**
  (it was grey although it is trained).
- **Hard role choice** `r_i = arg max_r (W_role h_i)_r`: grey dashed edge `#888888` and the words "no gradient" inside
  the box (it was drawn as a trained box).
- Grey italic notes: "top-3" on the gate's arrow, "never unbound or cleaned up" under the bind box, "H (skip
  connection)" on the grey line into the fusion box.
- The role vectors are written rho(r) and not rho with a subscript, so that no nested subscript falls under 6 pt. The
  formulas are otherwise those of v51.
- Not changed: the last box still says "mean pool" as `model.py` does. The audit notes that the comparison of the audit
  figure reads the module at the final token (`v7.py`); that is for the caption.

---

## Supplementary, `figs/fig4_identity.pdf` (180 x 124 mm): identity (main-text Fig. 3 until v51)

### a, separation against consistency

- x: separation across symbols, 1 - cosine between different symbols' codes, 0 to 1.1. y: same-symbol consistency (the
  fraction of same-symbol pairs that land on the same grid point), 0 to 1.1. One open marker per trained head; 24 heads
  (six-dimensional FSQ heads on arena C, three seeds per variant). Pale green square at the top right with the note
  "what the algebra requires": an illustration, not a measured region.

| Legend name | Colour, marker | Runs | Separation | Consistency | Source |
|---|---|---|---|---|---|
| task loss only | blue `#0072B2`, open circles | 3 | 0.1105, 0.2450, 0.1760 | 0.2258, 0.6726, 0.3873 | v62_logs/v62_BANKED.json (fsq_spans) |
| task loss + label-free separation term | blue `#0072B2`, open diamonds | 3 | 0.6761, 0.7437, 0.5428 | 0.1437, 0.1686, 0.2717 | v62_logs/v62_BANKED.json (fsq_inj) |
| identity classification | orange `#E69F00`, open squares | 12 | 0.8704, 0.5921, 0.7104, 1.0016, 0.9923, 1.0130, 0.8462, 0.7550, 0.7207, 1.0026, 0.9936, 1.0131 | 0.0646, 0.0317, 0.0814, 0.0811, 0.0594, 0.0666, 0.0744, 0.0210, 0.0303, 0.0770, 0.0565, 0.0656 | v63_logs/v63_PROVISIONAL.json (fsq_id); v63_logs/v63_PROVISIONAL.json (fsq_id_only); v63b_logs/v63b_BANKED.json (fsq_id); v63b_logs/v63b_BANKED.json (fsq_id_only) |
| pairwise equivalence | pink `#CC79A7`, open triangles | 6 | 0.0266, 0.0509, 0.0354, 0.4964, 0.4891, 0.5150 | 0.5019, 0.5608, 0.5849, 0.4537, 0.4278, 0.3835 | v64_logs/v64_BANKED.json (fsq_con); v64_logs/v64_BANKED.json (fsq_con_only) |
| exact codebook rows | black `#000000`, filled star | 1 | 0.9999 | 1.0000 | x = 1 - geometry_banked.json crosstalk_codebook; y = 1 by construction (typed) |

- The v51 legend "task loss only (6 runs)" is split: three runs with the task loss only (`fsq_spans`) and three with the
  task loss plus a label-free separation term (`fsq_inj`), as the audit asked. The second group reaches separation 0.54 to
  0.74.
- The codebook itself is the black star named "exact codebook rows" (it was a vermilion star named "fixed codebook
  rows"). Its y = 1 is by construction and is typed in the script.

### b, cross-talk of every route from the frozen states to a code

- A dot plot, one row per route, sorted by value; the number is printed at the right of each row. x: cross-talk (mean
  cosine between different symbols' codes), **linear**, 0 to 1 (it was logarithmic with bars). Dashed vertical line
  labelled "tolerance 0.25" at 0.2486.

| Row label | Colour | Cross-talk | Source |
|---|---|---|---|
| projection trained through the binding loss | pink `#CC79A7` | 0.9884 | `v165_logs/v165.json different_trained_mean (three seeds)` |
| raw hidden states | mid grey `#888888` | 0.9149 | `geometry_banked.json crosstalk_raw_states` |
| projection, before training | pink `#CC79A7`, open circle | 0.7177 | `v165_logs/v165.json different_init_mean (three seeds)` |
| sign hash of the states | mid grey `#888888` | 0.4975 | `v39_logs/v39b.json arms.lsh_raw.diff_within` |
| classifier logits | mid grey `#888888` | 0.3904 | `geometry_banked.json crosstalk_probe_logits` |
| sign hash, 36-dim subspace | mid grey `#888888` | 0.0905 | `v40_logs/v40.json diagnostic.pca_hash_36.diff_within` |
| continuous handover | sky blue `#56B4E9` | 0.0315 | `crosstalk_audit2.json fit3.crosstalk` |
| handover | vermilion `#D55E00` | 0.0001 | `geometry_banked.json crosstalk_discrete` |
| exact codebook rows | black `#000000` | 0.0001 | `geometry_banked.json crosstalk_codebook` |
| tolerance (dashed line) | near-black `#222222` | 0.2486 | `v163_logs/v163_scored.json tolerance (fitted rule)` |

- **Failing routes are mid grey `#888888`**, not vermilion. Colour marks only the named arms: continuous handover sky
  blue, handover vermilion, exact codebook rows black, the projection pink (open before training, filled after).
- **The row that v51 called "binding-loss projection" (0.699) is replaced by the two V-165 rows:** "projection, before
  training" 0.718 and "projection trained through the binding loss" 0.988. I did not keep a separate "untrained random
  projection 0.699" row: it is the same quantity measured in the audit-2 run (seeds 0, 1, 2), and two rows 0.02 apart
  would only raise the question which is right. Without `v165.json` the script draws the 0.699 row under the name
  "untrained random projection". Say if you want the 0.699 row as well.
- **The tolerance line moved from 0.099 (V-95, fixed rule) to 0.2486 (V-163, fitted rule)** so that it agrees with
  Fig. 2b. No row changes side (0.090 below, 0.390 above).
- Single values (no seeds drawn): raw states, classifier logits, handover and codebook from `geometry_banked.json`
  (seed 0, 200 test items, per the audit); the sign hash of the states from 200 instances (`v39b.json` `n_instances`; not
  recorded for the 36-dim hash); the continuous handover and the two projection rows are means of three seeds.

### c, corrupting the code in arena A

- x: fraction of codes altered, epsilon, 0 to 1. y: chain accuracy, arena A, 0.4 to 1.0. Three seeds per point (faint
  dots). Grey dashed line with the label "frozen-state probe".
- Both curves start from exact codebook rows and are near-black; they differ in line style and marker, not in colour.
  The legend says "consistent relabelling (many-to-one)": each distinct symbol receives, with probability epsilon, an
  independently drawn wrong index, so two symbols can collide (`v4.py:54-70`); it is not a permutation.

| Legend name | Style | Mean at epsilon = 0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 | Source |
|---|---|---|---|
| consistent relabelling (many-to-one) | near-black `#222222`, solid line, filled squares | 0.9183, 0.9080, 0.9030, 0.8847, 0.8777, 0.8653, 0.8690 | `v4_logs/v4_consistent.json` |
| per-occurrence corruption | near-black `#222222`, dashed line, open circles | 0.9183, 0.7197, 0.6040, 0.5463, 0.4997, 0.4943, 0.4920 | `v4_logs/v4_perocc.json` |
| frozen-state probe | grey `#7F7F7F` dashed horizontal line | 0.5090 | `v1_logs/frozen_llm.json` |

Per seed, consistent relabelling: 0.9180, 0.9190, 0.9180; 0.9070, 0.9040, 0.9130; 0.9020, 0.9070, 0.9000; 0.8850, 0.8850, 0.8840; 0.8760, 0.8770, 0.8800; 0.8690, 0.8610, 0.8660; 0.8670, 0.8710, 0.8690. Per seed, per-occurrence corruption: 0.9180, 0.9190, 0.9180; 0.7160, 0.7150, 0.7280; 0.5900, 0.6110, 0.6110; 0.5440, 0.5480, 0.5470; 0.4950, 0.5050, 0.4990; 0.4940, 0.4960, 0.4930; 0.4930, 0.4910, 0.4920.

### d, tampering with the identity head's training labels, bAbI task 1

- x: five conditions; x label "tampering applied to the identity head's training labels". The first column, separated
  by a thin grey vertical line, is the control "none: codebook rows permuted, labels intact". y: chain accuracy, bAbI
  task 1, 0.4 to 1.03. **Dots and a mean tick instead of bars** (near-black), three seeds; no legend, the tick labels
  carry the conditions. Grey dashed line with the label "frozen-state probe" at 0.7713.

| Tick label | Mean | Per seed | Source |
|---|---|---|---|
| none: codebook rows permuted, labels intact | 1.0000 | 1.0000, 1.0000, 1.0000 | `v36_logs/v36.json cell permuted` |
| 1 in 10 mentions given a second label | 0.9460 | 0.9640, 0.9560, 0.9180 | `v35_logs/v35.json family split, rate 0.1` |
| 1 in 5 mentions given a second label | 0.8247 | 0.8420, 0.8180, 0.8140 | `v35_logs/v35.json family split, rate 0.2` |
| persons merged (each with p = 0.25) | 0.8800 | 0.8800, 0.8800, 0.8800 | `v36_logs/v36.json cell merge-persons, rate 0.25` |
| locations merged (each with p = 0.25) | 0.6580 | 0.8320, 0.4860, 0.6560 | `v36_logs/v36.json cell merge-locations, rate 0.25` |

What the conditions are, from `v35.py` and `v36.py`: "1 in 10 / 1 in 5 mentions given a second label" is V-35's `split`
family at rate 0.1 / 0.2 (a training mention of a symbol receives a shadow label; evaluation against the true label);
"persons / locations merged (each with p = 0.25)" is V-36's merge, applied to the labels the identity head is trained
on, each symbol of the type merged into another with probability 0.25 (realized merges per seed in `v36.json`
`merged`: persons 1, 1, 1; locations 1, 3, 2 of six). 2,000 train / 500 eval. The first column permutes the codebook
rows and leaves the labels alone; `v36.py` itself calls it an isomorphism check.

---

## Supplementary, `figs/figS_sweeps.pdf` (180 x 114 mm): sweeps (new)

### a, bindings per level in arena C (main Fig. 2c of v51)

- x: bindings per level, k = 2, 4, 8, 16, 24, 32 (log scale, base 2; the vocabulary stays at 64 symbols). y: chain
  accuracy, arena C, 0.4 to 1.02. Three seeds per point (faint dots). The black line is drawn thick under the vermilion
  one: from k = 8 on the two are identical in every seed.

| Legend name | Colour, marker | Mean at k = 2, 4, 8, 16, 24, 32 | Per seed | Source |
|---|---|---|---|---|
| exact codebook rows | black `#000000`, circles, thick | 1.0000, 0.9980, 0.9967, 0.9720, 0.9507, 0.9340 | 1.0000, 1.0000, 1.0000; 0.9980, 0.9980, 0.9980; 0.9940, 0.9960, 1.0000; 0.9700, 0.9720, 0.9740; 0.9500, 0.9540, 0.9480; 0.9320, 0.9320, 0.9380 | `v14_logs/v14_fixed.json [].parser` |
| handover | vermilion `#D55E00`, squares | 0.9887, 0.9947, 0.9967, 0.9720, 0.9507, 0.9340 | 0.9800, 0.9880, 0.9980; 0.9960, 0.9940, 0.9940; 0.9940, 0.9960, 1.0000; 0.9700, 0.9720, 0.9740; 0.9500, 0.9540, 0.9480; 0.9320, 0.9320, 0.9380 | `v14_logs/v14_fixed.json [].probe_argmax` |
| frozen-state probe | grey `#7F7F7F`, triangles | 0.7953, 0.5707, 0.5333, 0.5113, 0.5340, 0.5153 | 0.8060, 0.8080, 0.7720; 0.5760, 0.5760, 0.5600; 0.5240, 0.5420, 0.5340; 0.5100, 0.4940, 0.5300; 0.5340, 0.5460, 0.5220; 0.5120, 0.5300, 0.5040 | `v14_logs/v14_fixed.json [].floor` |

### b, chain depth (main Fig. 2e of v51)

- x: chain depth 1, 2, 3, label "chain depth (each arm trained at that depth)". y: chain accuracy, 0.4 to 1.02. Three
  seeds per point. Exact codebook rows and the handover are identical in every seed.

| Legend name | Colour | Mean at depth 1, 2, 3 | Per seed | Source |
|---|---|---|---|---|
| exact codebook rows | black `#000000` | 0.9580, 0.9340, 0.8800 | 0.9540, 0.9600, 0.9600; 0.9320, 0.9320, 0.9380; 0.8840, 0.8800, 0.8760 | `v57_logs/v57.json cells[depth].parser` |
| handover | vermilion `#D55E00` | 0.9580, 0.9340, 0.8800 | 0.9540, 0.9600, 0.9600; 0.9320, 0.9320, 0.9380; 0.8840, 0.8800, 0.8760 | `v57_logs/v57.json cells[depth].probe_argmax` |
| LoRA fine-tune | orange `#E69F00` | 0.5073, 0.5007, 0.5153 | 0.5200, 0.5220, 0.4800; 0.4880, 0.5400, 0.4740; 0.5200, 0.5200, 0.5060 | `v57_logs/v57.json cells[depth].lora_direct` |
| frozen-state probe | grey `#7F7F7F` | 0.5407, 0.5153, 0.5300 | 0.5400, 0.5280, 0.5540; 0.5120, 0.5300, 0.5040; 0.5280, 0.5180, 0.5440 | `v57_logs/v57.json cells[depth].floor` |

The LoRA here is rank 8, final epoch, 800 train / 500 test (audit, `v57.py:84`), not the rank-16 run of main Fig. 2a.

### c, capacity of one store on the fine grid of V-151

- x: facts in one store, k (log scale, base 2), ticks 8, 16, 32, 64, 128, 256, 512, 800. y: readout exact match, 0 to 1.
- HRR only, one curve per dimension d = 512, 1024, 2048, 4096 (shades of blue, light to dark; matplotlib "Blues").
  Dashed black horizontal line: "0.70 criterion". d = 8192 is not in V-151 and is not drawn.
- Each point is the mean over 200 query instances and three capacity seeds (0, 1, 2); per-seed values are not banked,
  so there are no dots.
- **The grid's last value, 1024, is drawn at 800.** V-151 runs `v149.py` with the pool of V-101 (`load_rows(400)`,
  `pooled()`), which holds at most 800 facts, so the cell requested as 1,024 holds 800. The other fourteen grid values
  (8 to 724) are real.

| d | Colour | Readout at k = 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512, 724, 800 |
|---|---|---|
| 512 | shade of blue `#94c4df` | 0.9450, 0.9467, 0.9367, 0.9100, 0.8450, 0.6583, 0.4033, 0.1683, 0.0850, 0.0283, 0.0067, 0.0100, 0.0050, 0.0000, 0.0017 |
| 1024 | shade of blue `#4f9bcb` | 0.9450, 0.9433, 0.9417, 0.9433, 0.9350, 0.9050, 0.7917, 0.5550, 0.3033, 0.1200, 0.0467, 0.0183, 0.0067, 0.0033, 0.0033 |
| 2048 | shade of blue `#1d6cb1` | 0.9450, 0.9400, 0.9450, 0.9400, 0.9383, 0.9400, 0.9383, 0.9017, 0.7433, 0.5150, 0.2450, 0.1217, 0.0400, 0.0083, 0.0067 |
| 4096 | shade of blue `#083c7d` | 0.9467, 0.9450, 0.9450, 0.9383, 0.9400, 0.9417, 0.9383, 0.9350, 0.9283, 0.8850, 0.7317, 0.4017, 0.1583, 0.0533, 0.0317 |

Source: `v151_logs/v151.json` `capacity.HRR[d][k]`.

### d, facts held at the 0.70 criterion against dimension

- x: dimension d = 512, 1024, 2048, 4096 (log scale, base 2). y: "facts held at 0.70 readout (interpolated)" (log
  scale, base 2, 24 to 540). k* is the 0.70 crossing, by linear interpolation in log k between the two grid points that
  bracket it.
- HRR is the main series (vermilion `#D55E00`, circles, thick line) with its values printed beside the points, rounded
  to integers (42, 73, 137, 265). MAP (dark grey `#444444`, squares), BSC (grey `#777777`, triangles) and FHRR (light
  grey `#A0A0A0`, diamonds) are thin secondary series. Dashed black line: least-squares power law fitted to the four
  HRR points, legend "power-law fit to HRR, exponent 0.89". **No linear reference line.**

| Model | k* at d = 512, 1024, 2048, 4096 | Exponent of k* ~ d^alpha | 95% bootstrap interval |
|---|---|---|---|
| HRR | 41.7, 73.3, 136.7, 264.7 | 0.890 | 0.815 to 0.953 |
| MAP | 43.5, 79.5, 139.1, 247.9 | 0.834 | 0.807 to 0.868 |
| BSC | 29.3, 54.3, 96.0, 177.1 | 0.861 | 0.821 to 0.891 |
| FHRR | 79.7, 139.7, 249.9, 456.9 | 0.840 | 0.810 to 0.870 |

- **These reproduce the ledger exactly:** HRR k* = 41.7, 73.3, 136.7, 264.7 and exponent 0.890 with interval 0.815 to
  0.953. The script prints all four models on every run. The interval is the frozen scorer's (`ngcls/v151_score.py`):
  20,000 bootstrap resamples of the four (d, k*) points, seed 20261008, models in the order HRR, MAP, BSC, FHRR on one
  random stream; the figure script repeats that computation.
- None of the four intervals contains 1.0, which is V-151's own verdict (its bar S1 fails): the growth is sub-linear.
- V-101's octave grid and its k* (32, 64, 128, 256, 256) are no longer drawn anywhere.

---

## Where I departed from the brief, or chose

1. **Fig. 3a, truncated Phi-2 bars: kept, hatched** (not dropped); reasons under the panel.
2. **Seed dots for deterministic arms: none, in every figure**, including the `v65`/`v69` `prompt_ev` lists you named
   (three identical values). One switch restores them.
3. **The 0.699 "untrained random projection" is drawn nowhere** now that V-165 exists: dropped from Fig. 2b as you
   allowed, and replaced by the V-165 before-training row in the identity panel b as well.
4. **Identity panel b: tolerance line at 0.2486** (V-163), was 0.099 (V-95); and the panel is a dot plot on a linear axis.
5. **Fig. 2b x axis: linear to 0.1, logarithmic above.**
6. **Arena A label:** "A: 3 hops, 16 pairs per level" (audit and generator), where you wrote "16 pairs".
7. **Fig. 3 legend: "Qwen3-8B, 31k-token window"**, not 32k: `v131c.json` `config.max_len` is 31,000.
8. **Fig. 3c and sweeps c: "800"**, with "(all facts; 1,024 requested)" in the sharding legend.
9. **Fig. 4b: dashed bridges** across the missing k of the direct-answer arm.
10. **Audit figure:** besides the two changes you listed, panel a's bars start at zero, panel d's vermilion became
    near-black and its 0.5 line is in the legend, the HD readout lost its (deterministic) dots, and the titles became
    axis labels. The probe stays green there.
11. **Identity panel a:** the legend split of the task-loss runs and the black star (audit, Fig. 3a).
12. **Blue arm of Fig. 3a** is named "prompting (passages in the prompt)" to keep the arm name.
13. **PDF pages are the design size** (180 mm), not cropped to content.

## Where the data disagreed with the brief, or could not be verified

- **Nothing disagreed numerically.** CLUTRR: all nine k recomputed, even k equal to the typed values to three decimals.
  V-151: k* and exponent equal to the ledger's. V-165: the values you quoted (0.7177, 0.9884; 0.9879, 0.9895, 0.9878;
  0.472, 0.502, 0.476) are the file's. LoRA on arena A: 0.493, 0.508, 0.515, key `acc` in a list of three records
  (`arm`, `seed`, `acc`, `trainable_params`, `lr`, `secs`). `v131c`: 0.2877 x3 and 0.5943, 0.5660, 0.5755 (mean 0.5786).
- **"32k tokens" for the Qwen3-8B reference:** the artifact says `max_len` 31,000 (budgets 28,000). Labelled 31k.
- **"K = 800":** not recorded in `v102_scored.json` or `v151.json`. It follows from `load_rows(400)`, `pooled()` and
  V-73's 600 bindings from the first 300 instances of the same stream; that the last 100 instances also carry exactly
  two gold triples each is not checked (the dataset is not local).
- **Fig. 2b mixes four harnesses** (listed under the panel). The handover's cross-talk is one seed.
- **Reference lines of Fig. 4b:** SatLM 68.3 and DSR-LM 61.0 are transcribed from `vsassm/docs/neurosymbolic_review.md`,
  not from the papers. The 97.5 line is one codebook draw; the audit's open point (NSR0: mean 0.962 over ten codebooks;
  V-164: 2.4 % of 1,000 codebooks misread one rule at D = 2,048) is not settled by anything I did.
- **ARMT:** the label is unchanged, as decided. The memory note `readable_memory_lit_check` records a later system
  (Q-RAG, about 95 % at 10M); the figure itself does not say "best published", so that wording lives only in the caption.
- **n per panel** is recorded in the artifacts only for some runs; where the notes say "not recorded" or "defaults of
  the script" I did not find it in the JSON.
- **Sub- and superscripts** of the two schematics are 6.0 pt (base 8.6 pt); everything else is at least 6.5 pt.

## Not done (yours, or out of scope)

- No `.tex`, `.bib` or caption was touched; nothing was committed. The `% SOURCE` comments in the `.tex` files and
  the strings in `splice_figs.py` still describe the v51 figures.
- `\includegraphics` for the new `figs/figS_sweeps.pdf` does not exist yet.
- `submission/figs/` and the flattened `submission/*.tex` were already stale in v51 and still are.
- `figdata_v2/build_index.py` was not repaired; `INDEX.json` was extended by `add_index_v52.py` instead (44 entries,
  two older entries annotated, all listed in `figdata_v2/INDEX_ADDENDA_v52.md`).
- Units: Fig. 3b and Fig. 4 are still per cent, the rest fractions.
- New measurements the audit lists under (iii) other than V-163, V-165 and V-151 (a Qwen3-8B facts prompt at +0 and
  +50 on MuSiQue, a pool that really reaches 1,024 facts, CLUTRR over ten codebooks) do not exist and are not drawn.

## Files

Changed: `make_figs_v2.py` (rewritten), `make_fig6_benchmarks.py` (rewritten), `schematics.py` (rewritten),
`figstyle.py` (one colour added: `C["module"] = "#6A3D9A"`), `figdata.py` (docstring only), `figdata_v2/INDEX.json`
(addenda), `figs/fig1_overview.pdf`, `figs/fig2_audit.pdf`, `figs/fig3_mechanism.pdf`, `figs/fig4_identity.pdf`,
`figs/fig5_deployment.pdf`, `figs/fig6_benchmarks.pdf`, `figs/arch.pdf`, and their previews in `figs/preview/`.

Added: `figcheck.py`, `figs/figS_sweeps.pdf`, `figs/preview/figS_sweeps.png`, `figdata_v2/add_index_v52.py`,
`figdata_v2/INDEX_ADDENDA_v52.md`, `figdata_v2/v30_logs/v30.json`, `figdata_v2/v1_logs/lora.json`,
`figdata_v2/v131_logs/v131c.json`, `figdata_v2/v151_logs/v151.json`, `figdata_v2/r1_logs/bank_ours_by_k.py`,
`figdata_v2/r1_logs/ours_by_k.json`, `figdata_v2/benchmarks/babilong.json`, `figdata_v2/benchmarks/clutrr.json`,
`deck/review_v51/FIGURES_v52_values.json`, `deck/review_v51/FIGURES_v52_notes.md` (this file).
(`figdata_v2/v163_logs/`, `v164_logs/` and `v165_logs/` were put there by other sessions.)

Rendered pages (280 dpi, from the final PDFs): `/path/to/scratch`
`fig1_overview.png`, `fig3_mechanism.png`, `fig5_deployment.png`, `fig6_benchmarks.png`, `fig2_audit.png`, `arch.png`,
`fig4_identity.png`, `figS_sweeps.png`.

