# Code, run artifacts and pre-registration ledger

This repository is the code release for the manuscript **"Vector symbolic memory improves compositional
reasoning in frozen language models"** (submitted to *npj Unconventional Computing*). It holds the task
generators, the comparison arms, the store, the scoring and verification scripts, one JSON artifact per
experiment, and the chronological pre-registration ledger (`ledger.md`). In the ledger, each experiment's
preregistration (question, bars, predicted direction) is committed before its code.

The repository is private during review and is shared with editors and reviewers on request.

## Layout

The layout is flat on purpose. The manuscript's `% SOURCE` comments, the figure index
(`figdata_v2/INDEX.json` in the paper sources) and the `verify_paper*.py` scripts cite files by these
paths, and so does every dated ledger entry. Files are therefore never moved or renamed.

| Path pattern | What it is |
|---|---|
| `ledger.md` | The pre-registration ledger: append-only and chronological, 2026-08-06 to now. The manuscript's experiments are the V-line, which starts with V-1 on 2026-08-19. |
| `vNN.py` | The runner for experiment V-NN, usually with its scorer inside. |
| `vNN_score.py` | A separate scorer. The protocol commits it before the run it scores (see below). |
| `vNN_prereg.md`, `vNN_amend.md`, `vNN_close.md` | Preregistrations, amendments and closing notes for the later experiments. Earlier ones are written in `ledger.md` itself. |
| `vNN_logs/` | That experiment's artifacts: JSON results, run logs, launch records. |
| `vNNgo.sh`, `vNN_run.sh`, `run_vNN*.sh`, `vNN*.sbatch` | Launchers: the exact command, device and output path. `*.sbatch` jobs ran on the Georgia Tech PACE cluster. |
| `runs/` | Copies of artifacts banked at verdict time (V-2, V-8, V-9, V-10, V-105, V-106 and V-122). Some files are banked twice by design (for example `runs/v105/` and `v105_logs/`). |
| `b_logs/`, `v162/` | Artifacts of V-155 to V-161, the line designed in `B_DESIGN.md`, and of V-162. |
| `verify_paper*.py`, `verify_coverage.py`, `verify_provenance.py`, `verify_npj.py` | Verification. The `verify_paper*` scripts assert manuscript numbers against artifacts. `verify_coverage.py` traces every numeric literal to an artifact or to the ledger. `verify_provenance.py` checks that every table declares a source that exists. `verify_npj.py` checks the journal's formatting limits. |
| `AUDIT_2026-10.md` | Pre-submission audit of the code that produced the manuscript's numbers. |
| `hybrid_neurosymbolic/` | Arenas A and B, the audited binding module (`model.py`, `vsa.py`) and the original task generators. These came from a second project directory; see `hybrid_neurosymbolic/README.md`. |
| Pre-V-line files (removed) | The work that came before the V-line (NgCLS continual learning, then the E-, N-, M-, A-, R-, T/TT-, U-, W-, D-, P-, Q- and O-series: the top-level `a*`, `e*`, `m*`, `n*`, `o*`, `p*`, `q*`, `r*`, `tt*`, ... scripts, `logs/`, `runs_local/`, `datasets/`, the NgCLS run records in `runs/` and the paper drafts), together with old manuscript drafts, the duplicate `runs/v3_arms/` copies and stray lock and download-log files, was removed from the tree on 2026-11-11 and remains at tag `archive-2026-11-11`, where the ledger's references to it resolve (`git checkout archive-2026-11-11 -- <path>` restores one). |

### File-name suffixes the ledger uses

- `_BANKED`: the copy a verdict was read from. In 22 of the 23 cases it is byte-identical to the run's own output beside it. `v61_logs/v61_ema_BANKED.json` has no raw twin.
- `_VOID_<reason>`: a run declared void. It is kept, but renamed so that no checker can read it as evidence.
- `_PROVISIONAL`: a run whose verdict waited on a declared rerun (V-63). Both runs are banked.
- `.bak` versions of a checker: the earlier state of a corrected checker.

## How the pre-registration works

A V-line experiment leaves this trail in `git log` (V-165 is the example):

1. **Preregistration, before any code.** A ledger entry, or a `vNN_prereg.md`, states the question, the
   bars, the direction of the prediction and the conditions under which the result reads as negative. It is
   committed on its own, with a message such as `V-165 prereg: ... (before any code)` (commit `a88f9dd`).
2. **Code and a frozen scorer, before the run.** The runner and scorer are committed next, with a message
   such as `V-165 code (after the prereg, before the run)` (`45daeb9`). Under the protocol, the scorer is not
   edited once any result exists. A correction is a new, dated entry (for example `ledger_v102_correction.md`).
3. **Launch record.** The command, host, device, seeds and output path go into the ledger and, for V-111
   to V-119, also into `vNN_logs/launch_record.txt`.
4. **Verdict.** A `CLOSED` or `VERDICT` entry names the prereg commit and reports each bar as pass, fail or
   hold, with the scorer's output quoted (`1ca2e96` closes V-165). Refuted predictions, void runs and
   withdrawn claims are recorded, not deleted.

To audit one experiment, take the prereg commit named in its verdict entry and run
`git show <hash>`. Then confirm with `git log --format='%h %ad %s' -- vNN.py vNN_score.py` that the code
came after it.

## Environment

- **Python 3.9.** This is the `dino_wm` conda environment on the lab GPU server; tracebacks in the run logs
  show `envs/dino_wm/lib/python3.9/site-packages`.
- **torch 2.4.1** and **transformers 4.57.6.** Both versions are recorded in the ledger (V-110, V-123 and
  V-126 entries) and in `v126_prereg.md`. Runs on PACE used torch 2.5.1. The V-159 runs on a DGX Spark used
  transformers 5.19 (`v159_prereg.md`).
- `requirements.txt` pins only those two versions. The other packages are inferred from the imports, and
  the file says so.
- **Hardware.** The manuscript's Methods give the hardware for each family of experiments. Ledger launch
  entries name the host and the GPU.
- **Models and data** are downloaded from Hugging Face at run time (for example `microsoft/phi-2`,
  `framolfese/2WikiMultihopQA`, `dgslibisey/MuSiQue`, `Muennighoff/babi`). The synthetic arenas are generated
  from the seeds recorded in each artifact. No model weights, adapters or datasets for the manuscript's
  experiments are tracked. The two exceptions, both from before the V-line, were removed from the tree on 2026-11-11 (see the size note below).
- The launchers set `HF_ENDPOINT=https://hf-mirror.com`, a Hugging Face mirror the lab's network needed.
  Drop it elsewhere.
- **API runs** (`or4.py` and the other OpenRouter runners) read the key from `~/.openrouter_key`. A pattern scan of the tree
  and of the full history on 2026-11-11 found no key.
- **Absolute paths.** Scripts and checkers use the server's paths: `D = /path/to/work/ngcls` (this
  repository) and `B = /path/to/work/sync_h20/rewardbench_experiment/zeyu/hybrid_neurosymbolic` (now
  `hybrid_neurosymbolic/`). To run them elsewhere without editing, make both paths symlinks:

      sudo mkdir -p /path/to/work/sync_h20/rewardbench_experiment/zeyu
      sudo ln -s "$PWD" /path/to/work/ngcls
      sudo ln -s "$PWD/hybrid_neurosymbolic" /path/to/work/sync_h20/rewardbench_experiment/zeyu/hybrid_neurosymbolic

- **Verification scripts** take the manuscript `.tex` as their first argument. The default is
  `main_npj.tex`, an older single-file build kept here. The paper sources write the current flattened
  manuscript with `flatten.py` (`build/main_flat.tex`).

## Manuscript-to-code map

The tables below are built from the manuscript's `% SOURCE` comments and from `figdata_v2/INDEX.json`
(including the `INDEX_ADDENDA_v52.md` entries). Every artifact path was checked against this repository's
tree when the map was made (2026-11-11).

- **Present:** the path exists in this repository. Artifacts from root `B` are listed under
  `hybrid_neurosymbolic/`.
- **NOT IN REPO: paper:** the file belongs to the manuscript's own sources: the figure scripts
  (`make_figs_v2.py`, `make_fig6_benchmarks.py`, `schematics.py`) and the transcribed benchmark values in
  `figdata_v2/`.
- **NOT IN REPO: vsassm:** the public-benchmark experiments (BABILong, CLUTRR, RULER, LongMemEval) ran in a
  separate repository. Per the Code availability statement, their code and per-question outputs are
  available from the authors on request.

"Producing script(s)" means the runner named by the experiment number, plus the launcher where one names
that output path. The exact command for every run is in its ledger launch entry. Labels are the
manuscript's `\label` keys. Figure numbers are left out because they change between builds.

### By manuscript element (`% SOURCE` comments)

| Manuscript element (`\label` key or description, and the .tex file holding its `% SOURCE` comment) | Artifact(s) | Producing / reading script(s) |
|---|---|---|
| `fig:overview` (intro.tex) | `v30_logs/v30.json`<br>`v35_logs/v35.json`<br>`v37_logs/v37b.json`<br>`v48_logs/v48.json`<br>`v49_logs/v49.json`<br>`v50_logs/v50_task1_VOID_locoffset.json`<br>`v50_logs/v50_task2_BANKED.json`<br>`v131_logs/v131b.json`<br>`figdata_v2/INDEX.json` (NOT IN REPO: paper) | `schematics.py` (NOT IN REPO: paper)<br>`v30.py`<br>`v30go.sh`<br>`v35.py`<br>`v35go.sh`<br>`v37b.py`<br>`v37bgo.sh`<br>`v48.py`<br>`v48_run.sh`<br>`v49.py`<br>`v49_run.sh`<br>`v50.py`<br>`v50_run.sh`<br>`v131b.py`<br>`run_v131b.sh`<br>`make_figs_v2.py` (NOT IN REPO: paper) |
| `fig:mechanism` (results.tex) | `runs/v9/v9.json`<br>`hybrid_neurosymbolic/v9_logs/v9_gpu1_rerun_partial.json`<br>`hybrid_neurosymbolic/v11_logs/v18_spanfix.json`<br>`v13_logs/v13_fixed.json`<br>`v15_logs/v15.json`<br>`v15b_logs/v15b.json`<br>`hybrid_neurosymbolic/v1_logs/lora.json`<br>`hybrid_neurosymbolic/v1_logs/frozen_llm.json`<br>`v163_logs/v163.json`<br>`geometry_banked.json`<br>`crosstalk_audit2.json`<br>`v165_logs/v165.json`<br>`v21_logs/v21.json` | `runs/v9/v9.py`<br>`runs/v9/launch_v9.sh`<br>`hybrid_neurosymbolic/v9.py`<br>`hybrid_neurosymbolic/v9go.sh`<br>`hybrid_neurosymbolic/v11.py`<br>`hybrid_neurosymbolic/v18go.sh`<br>`v13.py`<br>`v13fix.sh`<br>`v15.py`<br>`v15go.sh`<br>`launch_v15b.sh`<br>`v15b_score.py`<br>`hybrid_neurosymbolic/v1_run.py`<br>`hybrid_neurosymbolic/v1_driver.sh`<br>`v163.py`<br>`bank_geometry.py`<br>`crosstalk_audit2.py`<br>`run_audit2.sh`<br>`v165.py`<br>`v165_wait_gpu.sh`<br>`v21.py`<br>`v21go.sh` |
| `fig:deployment` (results.tex) | `v65_logs/v65_BANKED.json`<br>`v67_logs/v67_BANKED.json`<br>`v69_logs/v69_BANKED.json`<br>`v72b_logs/v72b_BANKED.json`<br>`v79_logs/v79_BANKED.json`<br>`v79_logs/v79a_BANKED.json`<br>`v131_logs/v131c.json`<br>`v102_logs/v102_scored.json` | `v65.py`<br>`v65_run.sh`<br>`v67.py`<br>`v67_run.sh`<br>`v69.py`<br>`v69_run.sh`<br>`v72b.py`<br>`v72b_run.sh`<br>`v79.py`<br>`v79_run.sh`<br>`v79a.py`<br>`v131c.py`<br>`run_v131c.sh`<br>`v131_score.py`<br>`v102.py`<br>`run_v102.sh` |
| `fig:benchmarks` (results.tex) | `figdata_v2/benchmarks/*.json` (NOT IN REPO: paper)<br>`figdata_v2/r1_logs/ours_by_k.json` (NOT IN REPO: paper)<br>`r1_out/test_8B_think.jsonl` (NOT IN REPO: vsassm)<br>`results/r1_verdict.txt` (NOT IN REPO: vsassm) | `make_fig6_benchmarks.py` (NOT IN REPO: paper) |
| `tab:threearena` (appendix.tex) | `runs/v9/v9.json`<br>`hybrid_neurosymbolic/v11_logs/v18_spanfix.json`<br>`v13_logs/v13_fixed.json` | `runs/v9/v9.py`<br>`runs/v9/launch_v9.sh`<br>`hybrid_neurosymbolic/v11.py`<br>`hybrid_neurosymbolic/v18go.sh`<br>`v13.py`<br>`v13fix.sh` |
| `bulky-evidence scale sweep (8..1024 pooled facts, prompt vs sharded store, tokens and latency)` (appendix.tex) | `v106_logs/v106.json`<br>`v100_logs/v100_scored.json`<br>`v101_logs/v101_scored.json`<br>ledger commit `82da296` | `v106.py`<br>`v106.sbatch`<br>`v106_score.py`<br>`verify_paper_scale.py`<br>`v100.py`<br>`v101.py`<br>`run_v101.sh` |
| `sec:benchmarks` (appendix.tex) | `~/vsassm/results/b18_verdict.txt` (NOT IN REPO: vsassm)<br>`b18_api_verdict.txt` (NOT IN REPO: vsassm)<br>`r1_verdict.txt` (NOT IN REPO: vsassm)<br>`~/vsassm/prereg/B18_prereg.md` (NOT IN REPO: vsassm)<br>`R1_prereg.md` (NOT IN REPO: vsassm) | `make_fig6_benchmarks.py` (NOT IN REPO: paper) |
| `tab:invalid` (appendix.tex) | `hybrid_neurosymbolic/v2_logs/v2.json`<br>ledger commit `57406f1` | `hybrid_neurosymbolic/v2_run.py`<br>`hybrid_neurosymbolic/v2go.sh` |
| `fig:audit` (appendix.tex) | `hybrid_neurosymbolic/v2_logs/v2.json`<br>`v7_logs/v7_last.json`<br>`figdata_v2/ledger_57406f1_label_priors.json` (NOT IN REPO: paper)<br>`v12_logs/v12.json`<br>`v6_logs/v6.json`<br>`v13_logs/floor_survey.json`<br>ledger commit `57406f1` | `hybrid_neurosymbolic/v2_run.py`<br>`hybrid_neurosymbolic/v2go.sh`<br>`v7.py`<br>`v7go.sh`<br>`v12.py`<br>`v12go.sh`<br>`v6.py`<br>`v6go.sh`<br>`floor_survey.py`<br>`survey_go.sh` |
| `fig:arch` (appendix.tex) | `hybrid_neurosymbolic/v2_logs/v2.json` | `schematics.py` (NOT IN REPO: paper)<br>`hybrid_neurosymbolic/model.py`<br>`hybrid_neurosymbolic/vsa.py`<br>`hybrid_neurosymbolic/v2_run.py`<br>`hybrid_neurosymbolic/v2go.sh` |
| `tab:valid` (appendix.tex) | `v7_logs/v7_last.json`<br>`v12_logs/v12.json` | `v7.py`<br>`v7go.sh`<br>`v12.py`<br>`v12go.sh` |
| `tab:placement` (appendix.tex) | `v13_logs/floor_survey.json` | `floor_survey.py`<br>`survey_go.sh` |
| `tab:where` (appendix.tex) | `hybrid_neurosymbolic/v4_logs/v4_perocc.json`<br>`hybrid_neurosymbolic/v4_logs/v4_consistent.json`<br>`hybrid_neurosymbolic/v4_logs/v4_extract.json`<br>`hybrid_neurosymbolic/v8_logs/v8.json` | `hybrid_neurosymbolic/v4.py`<br>`hybrid_neurosymbolic/v4go.sh`<br>`hybrid_neurosymbolic/v8.py`<br>`hybrid_neurosymbolic/v8go.sh` |
| `fig:sweeps` (appendix.tex) | `v14_logs/v14_fixed.json`<br>`v57_logs/v57.json`<br>`v151_logs/v151.json`<br>`figdata_v2/INDEX.json` (NOT IN REPO: paper) | `v14.py`<br>`v14fix.sh`<br>`v57.py`<br>`v57_run.sh`<br>`v149.py`<br>`run_v151.sh`<br>`v151_score.py`<br>`make_figs_v2.py` (NOT IN REPO: paper) |
| `fig:identity` (appendix.tex) | `v62_logs/v62_BANKED.json`<br>`v63_logs/v63_PROVISIONAL.json`<br>`v63b_logs/v63b_BANKED.json`<br>`v64_logs/v64_BANKED.json`<br>`geometry_banked.json`<br>`crosstalk_audit2.json`<br>`v23_logs/v23.json`<br>`v39_logs/v39.json`<br>`v39_logs/v39b.json`<br>`v40_logs/v40.json`<br>`hybrid_neurosymbolic/v4_logs/v4_consistent.json`<br>`hybrid_neurosymbolic/v4_logs/v4_perocc.json`<br>`runs/v9/v9.json`<br>`v35_logs/v35.json`<br>`v36_logs/v36.json` | `v62.py`<br>`v62_run.sh`<br>`v63.py`<br>`v63_run.sh`<br>`v63b.py`<br>`v63b_run.sh`<br>`v64.py`<br>`v64_run.sh`<br>`bank_geometry.py`<br>`crosstalk_audit2.py`<br>`run_audit2.sh`<br>`v23.py`<br>`v23go.sh`<br>`v39.py`<br>`v39go.sh`<br>`v39b.py`<br>`v39bgo.sh`<br>`v40.py`<br>`v40go.sh`<br>`hybrid_neurosymbolic/v4.py`<br>`hybrid_neurosymbolic/v4go.sh`<br>`runs/v9/v9.py`<br>`runs/v9/launch_v9.sh`<br>`v35.py`<br>`v35go.sh`<br>`v36.py`<br>`v36go.sh` |
| `app:benchmarks` (appendix.tex) | `~/vsassm/results/b18_verdict.txt` (NOT IN REPO: vsassm)<br>`b18_api_verdict.txt` (NOT IN REPO: vsassm)<br>`b18b_verdict.txt` (NOT IN REPO: vsassm)<br>`r1_verdict.txt` (NOT IN REPO: vsassm)<br>`b19_verdict.txt` (NOT IN REPO: vsassm)<br>`frontier_ruler_verdict.txt` (NOT IN REPO: vsassm)<br>`b20_verdict.txt` (NOT IN REPO: vsassm) | - |

### By figure-index panel (`figdata_v2/INDEX.json`)

The `INDEX.json` panel keys (`F1b_...`, `F3a_...`) follow the numbering of an earlier build (`main_npj.tex`),
and they are kept because the figure scripts read them. Each entry stores the Python expression that reads
its value from the artifact.

| `INDEX.json` panel | entries | Artifact file(s) | Producing script(s) |
|---|---|---|---|
| `F1b_natural_text` | 34 | `v49_logs/v49.json`<br>`v50_logs/v50_task1_VOID_locoffset.json`<br>`v36_logs/v36.json`<br>`v35_logs/v35.json`<br>`v50_logs/v50_task2_BANKED.json`<br>`v37_logs/v37b.json`<br>`v48_logs/v48.json`<br>`v131_logs/v131b.json`<br>`v30_logs/v30.json` | `v49.py`<br>`v49_run.sh`<br>`v50.py`<br>`v50_run.sh`<br>`v36.py`<br>`v36go.sh`<br>`v35.py`<br>`v35go.sh`<br>`v37b.py`<br>`v37bgo.sh`<br>`v48.py`<br>`v48_run.sh`<br>`v131b.py`<br>`run_v131b.sh`<br>`v30.py`<br>`v30go.sh` |
| `F2a_label_priors` | 17 | `hybrid_neurosymbolic/v2_logs/v2.json`<br>`v7_logs/v7_last.json` | `hybrid_neurosymbolic/v2_run.py`<br>`hybrid_neurosymbolic/v2go.sh`<br>`v7.py`<br>`v7go.sh` |
| `F2b_per_arena_arms` | 54 | `v7_logs/v7_last.json`<br>`v12_logs/v12.json` | `v7.py`<br>`v7go.sh`<br>`v12.py`<br>`v12go.sh` |
| `F2c_pooling` | 10 | `v6_logs/v6.json` | `v6.py`<br>`v6go.sh` |
| `F2d_placement` | 36 | `v13_logs/floor_survey.json` | `floor_survey.py`<br>`survey_go.sh` |
| `F3a_three_arenas` | 52 | `hybrid_neurosymbolic/v1_logs/frozen_llm.json`<br>`hybrid_neurosymbolic/v4_logs/v4_perocc.json`<br>`hybrid_neurosymbolic/v4_logs/v4_consistent.json`<br>`hybrid_neurosymbolic/v8_logs/v8.json`<br>`hybrid_neurosymbolic/v9_logs/v9_gpu1_rerun_partial.json`<br>`runs/v9/v9.json`<br>`hybrid_neurosymbolic/v11_logs/v18_spanfix.json`<br>`v13_logs/v13_fixed.json`<br>`v15_logs/v15.json`<br>`v15b_logs/v15b.json`<br>`hybrid_neurosymbolic/v1_logs/lora.json` | `hybrid_neurosymbolic/v1_run.py`<br>`hybrid_neurosymbolic/v1_driver.sh`<br>`hybrid_neurosymbolic/v4.py`<br>`hybrid_neurosymbolic/v4go.sh`<br>`hybrid_neurosymbolic/v8.py`<br>`hybrid_neurosymbolic/v8go.sh`<br>`hybrid_neurosymbolic/v9.py`<br>`hybrid_neurosymbolic/v9go.sh`<br>`runs/v9/v9.py`<br>`runs/v9/launch_v9.sh`<br>`hybrid_neurosymbolic/v11.py`<br>`hybrid_neurosymbolic/v18go.sh`<br>`v13.py`<br>`v13fix.sh`<br>`v15.py`<br>`v15go.sh`<br>`launch_v15b.sh`<br>`v15b_score.py` |
| `F3b_arena_A_decomposition` | 18 | `hybrid_neurosymbolic/v4_logs/v4_perocc.json`<br>`hybrid_neurosymbolic/v4_logs/v4_consistent.json`<br>`hybrid_neurosymbolic/v4_logs/v4_extract.json`<br>`hybrid_neurosymbolic/v8_logs/v8.json`<br>`hybrid_neurosymbolic/v1_logs/vsa_hidden.json` | `hybrid_neurosymbolic/v4.py`<br>`hybrid_neurosymbolic/v4go.sh`<br>`hybrid_neurosymbolic/v8.py`<br>`hybrid_neurosymbolic/v8go.sh`<br>`hybrid_neurosymbolic/v1_run.py`<br>`hybrid_neurosymbolic/v1_driver.sh` |
| `F3c_symbol_count_sweep` | 8 | `v14_logs/v14_fixed.json` | `v14.py`<br>`v14fix.sh` |
| `F3d_read_by_budget` | 9 | `v21_logs/v21.json` | `v21.py`<br>`v21go.sh` |
| `F3e_depth` | 9 | `v57_logs/v57.json` | `v57.py`<br>`v57_run.sh` |
| `F4a_seesaw` | 24 | `v62_logs/v62_BANKED.json`<br>`v63_logs/v63_PROVISIONAL.json`<br>`v63b_logs/v63b_BANKED.json`<br>`v64_logs/v64_BANKED.json` | `v62.py`<br>`v62_run.sh`<br>`v63.py`<br>`v63_run.sh`<br>`v63b.py`<br>`v63b_run.sh`<br>`v64.py`<br>`v64_run.sh` |
| `F4b_hashing_cosines` | 17 | `v39_logs/v39b.json`<br>`v40_logs/v40.json`<br>`v39_logs/v39.json`<br>`v23_logs/v23.json` | `v39b.py`<br>`v39bgo.sh`<br>`v40.py`<br>`v40go.sh`<br>`v39.py`<br>`v39go.sh`<br>`v23.py`<br>`v23go.sh` |
| `F4c_pairwise_head` | 16 | `v41_logs/v41.json`<br>`v42_logs/v42.json` | `v41.py`<br>`v41go.sh`<br>`v42.py`<br>`v42go.sh` |
| `F4d_aliasing_babi` | 14 | `v36_logs/v36.json`<br>`v35_logs/v35.json` | `v36.py`<br>`v36go.sh`<br>`v35.py`<br>`v35go.sh` |
| `F5a_2wiki_three_arms` | 17 | `v65_logs/v65_BANKED.json`<br>`v67_logs/v67_BANKED.json`<br>`v69_logs/v69_BANKED.json`<br>`v100_logs/v100_scored.json`<br>`v101_logs/v101_scored.json` | `v65.py`<br>`v65_run.sh`<br>`v67.py`<br>`v67_run.sh`<br>`v69.py`<br>`v69_run.sh`<br>`v100.py`<br>`v101.py`<br>`run_v101.sh` |
| `F5b_bulky_evidence` | 16 | `v72b_logs/v72b_BANKED.json` | `v72b.py`<br>`v72b_run.sh` |
| `F5b_musique` | 24 | `v79_logs/v79_BANKED.json`<br>`v79_logs/v79a_BANKED.json`<br>`v131_logs/v131c.json` | `v79.py`<br>`v79_run.sh`<br>`v79a.py`<br>`v131c.py`<br>`run_v131c.sh`<br>`v131_score.py` |
| `F5c_capacity` | 17 | `v73_logs/v73_BANKED.json`<br>`v101_logs/v101_scored.json`<br>`v100_logs/v100_scored.json`<br>`v72b_logs/v72b_BANKED.json` | `v73.py`<br>`v73_run.sh`<br>`v101.py`<br>`run_v101.sh`<br>`v100.py`<br>`v72b.py`<br>`v72b_run.sh` |
| `F5d_sharding_cost` | 16 | `v102_logs/v102_scored.json`<br>`v100_logs/v100_scored.json` | `v102.py`<br>`run_v102.sh`<br>`v100.py` |
| `F4e_crosstalk_geometry` | 29 | `v95_logs/v95_scored.json`<br>`geometry_banked.json`<br>`crosstalk_audit2.json`<br>`v163_logs/v163.json`<br>`v163_logs/v163_scored.json`<br>`v165_logs/v165.json` | `v95.py`<br>`bank_geometry.py`<br>`crosstalk_audit2.py`<br>`run_audit2.sh`<br>`v163.py`<br>`v163_score.py`<br>`v165.py`<br>`v165_wait_gpu.sh` |
| `FS_capacity_v151` | 5 | `v151_logs/v151.json` | `v149.py`<br>`run_v151.sh`<br>`v151_score.py` |
| `F6b_clutrr` | 4 | `r1_logs/ours_by_k.json` (NOT IN REPO: paper) | - |

`INDEX.json` also lists 10 `UNRESOLVED` entries, values with no JSON artifact behind them. Nine are the label priors, the per-task majority baselines and the overall majority-predictor accuracy for the four original tasks. The ledger computed them inline at commit `57406f1` from the `_gen_*` generators in `hybrid_neurosymbolic/run_compositional.py`, which is now in this repository, and `verify_paper2.py` and `verify_paper3.py` assert the ledger's values. The tenth is a fidelity value in `F3d_read_by_budget` that `v21_logs/v21.json` does not carry.

## Size note

The largest tracked files are four compressed score dumps of about 46 MB each, written by `v118.py` and
`v121.py`: `v118_logs/v118_scores_{moment,score}.npz` and `v121_logs/v121_scores_{moment,score}.npz`.
The next largest, `runs_local/w2/static.pt` (20 MB, the W-2 checkpoint the ledger names) and
`datasets/proto/o8_n100000_s0/*.npy` (about 21 MB, O-8), date from before the V-line and were removed from the tree on
2026-11-11. They remain at tag `archive-2026-11-11` and in the history, so a fresh clone is no smaller. `.gitignore` keeps weights, adapters, checkpoints and model caches out of
any future commit.

## Earlier project: NgCLS

The repository began on 2026-08-06 as NgCLS, a pre-registered audit of surprise-gated continual learning.
Its README is kept below unchanged, apart from smaller headings. `ngcls.py`, the NgCLS run records in `runs/`, `paper/` (the TMLR
draft) and `DESIGN.md` belonged to that project. They were removed from the tree on 2026-11-11 and remain at tag
`archive-2026-11-11`, so the layout below describes that tag.

<details>
<summary>Original README (NgCLS)</summary>

#### NgCLS — online surprise-neuromodulatory gating for continual learning

Code, experiment ledger, and paper for a **pre-registered audit** of whether a single online
surprise signal can usefully gate memory-write, replay, and per-synapse plasticity in one
train/deploy-free loop.

**Headline (honest):** the gating gives a real but modest, *boundary-triggered* gain in buffered
class-incremental learning (`ngcls_v3` vs DER++: +0.8–1.5pt average accuracy, 17/20 seeds, stable
across a 10x buffer range, consistently lower forgetting), and is **inert to harmful** on
boundary-free single-pass streams at every drift rate tested. A loss-z-score surprise signal is
structurally blind to smooth drift; the two-channel feature-drift detector we built to fix that
(`feat3`) passes every acceptance gate but does **not** rescue the mechanism. The reusable positive
artifact is the detector, not the learner.

##### Layout

| path | what |
|---|---|
| `ngcls.py` | single-file implementation: all methods, baselines, surprise signals, stream harness |
| `ledger.md` | append-only experiment ledger — every stage's pass/fail rule written *before* its runs, and the branch taken |
| `runs/` | 232 per-run JSON records (seed, full config, per-step metrics, realised mean LR, mean surprise) |
| `paper/` | TMLR submission source (`main.tex`, anonymous by default; `\usepackage[preprint]{tmlr}` for the de-anonymised build) |
| `DESIGN.md` | north-star spec: principle, positioning vs prior art, falsifiable experiment, guardrails |
| `aggregate.py`, `agg_buffer.py`, `dissect_surprise.py` | analysis / figure regeneration |

`data/` (CIFAR-10/100 npz) is not tracked; see the ledger for the HF-mirror fetch used.

##### Protocol

Baselines are the judge, at equal replay-memory and compute budget; >=3 (usually 5) seeds before any
claim; every pass/fail rule pre-registered in the ledger before launch; negatives recorded verbatim,
including two mis-specified bars we kept rather than quietly dropped.

</details>
