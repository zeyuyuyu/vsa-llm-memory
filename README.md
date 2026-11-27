# vsa-llm-memory

Code, pre-specifications and outputs for

> Zeyu Wang and Nabil Imam. **Vector symbolic architectures improve compositional reasoning and long-context recall in frozen large language models.** Submitted to *npj Unconventional Computing* (2026).

A frozen language model names the symbols, and a vector symbolic memory (VSA) composes them. This repository lets you
trace every number in the paper, its Supplementary Information and the Supplementary Record
(`supplementary_record.pdf`) to the code that produced it, the pre-specification that fixed the test before it ran,
and the output file the number was read from.

## Quick start

```sh
pip install -r requirements.txt        # Python 3.10
cd reproduce && sh run_all.sh          # recompute the headline numbers from archived outputs (numpy only, seconds)
cd .. && python3 tools/check_package.py   # checksums, paths and every figure pointer
```

`reproduce/run_all.sh` recomputes the equal-bit, equal-cell and forward-pass headline numbers; its output should
match `reproduce/expected_output.txt`. `cd figures && python3 make_figures.py` regenerates every figure from `figures/figdata/`.

## Layout

| Directory | Contents |
| --- | --- |
| `code/` | the scripts of each experiment, at their paths in the source repositories (`code/<repo>@<commit>/` where a result came from an earlier commit) |
| `data/` | inputs that are ours or that we drew (PopQA entities and Wikidata aliases as split, generated logs and solver problems) |
| `results/` | verdicts, scores and per-question outputs (large files gzipped) |
| `prereg/` | every pre-specification with its amendments and notes, and the OpenTimestamps proofs (`*.md.ots`) |
| `ledger/` | the timeline that shows each pre-specification was committed before its verdict |
| `figures/` | figure scripts, the figure data layer (`figdata/`, a pointer from each plotted value to its source) and the rendered figures |
| `reproduce/` | the one-command reproduction above |
| `tools/` | `check_package.py` (integrity) and `assemble_tree.py` (rebuild runnable source trees under `work/`) |
| `docs/` | `DETAILS.md` (full claim map and rerunning notes), `PROVENANCE.tsv` (where each file came from) |

## Where each result lives

Experiment identifiers (B…, R1) match the paper's Supplementary Table of pre-specified tests. Each has a
pre-specification in `prereg/vsassm/`, code in `code/vsassm/`, outputs in `results/vsassm/b<N>_out/` and a verdict in
`results/vsassm/results/`.

| In the paper | What it shows | Experiments | Start here |
| --- | --- | --- | --- |
| Fig. 1 | the coupling and a worked two-hop example | illustration | `figures/schematic.py`, `figures/figdata/worked_example.json` |
| Table 1 | the same frozen model without and with a VSA memory | B44 (arena C), B33 (fresh bAbI), B21 and B24 (RULER), R1 (CLUTRR), B40 and B65 (forward pass) | `results/vsassm/results/b33_verdict.txt`, `results/vsassm/results/b24_verdict.txt`, `results/vsassm/results/b40_verdict.txt` |
| Fig. 2 | arenas, consistency tolerance, load | controlled experiments V-1 to V-165; B31, B42, B44, B47, B48 | `figures/figdata/INDEX.json`, `code/ngcls/`, `results/vsassm/results/b44_verdict.txt` |
| Fig. 3 | what was compared, and how each memory was charged | schematic | `figures/figs/fig1_comparisons.pdf` |
| Table 2, Fig. 4, Table 3 | equal stored bits: the write rule | B49, B60, B62, B64, B72, B74; confirmatory run B73 | `results/vsassm/results/b60_verdict.json`, `results/vsassm/results/b72_verdict.json`, `results/vsassm/results/b74_verdict.json` |
| Fig. 5 | equal storage cells on simulated phase-change devices | B34, B38, B61, B63, B75; confirmatory run B73 | `results/vsassm/results/b61_verdict.json`, `results/vsassm/results/b63_verdict.json`, `results/vsassm/results/b75_verdict.json` |
| Supplementary Information and Record | all other analyses | see `docs/DETAILS.md` | `docs/DETAILS.md` |

For the controlled experiments, `figures/figdata/INDEX.json` maps each plotted value to its run artifact and the
expression that reads it; `python3 figures/check_sources.py` re-checks every pointer and transcribed value.

## Pre-specification and timing

Every test was written down before it was run: its pass criteria, predictions and the condition under which it would
count against the hypothesis. Amendments and notes written after a test are separate files next to the
pre-specification. Commit dates are internal records; the public evidence of timing is the OpenTimestamps proofs
(the `.md.ots` files next to the pre-specifications in `prereg/vsassm/`, from B47 on, and the 19 November 2026
digest in `prereg/vsassm/timestamps/`), checkable with
`ots verify`. `ledger/prereg_timeline.tsv` and `ledger/prereg_checks.txt` list, per test, the first commit of the
pre-specification and of the verdict.

## Rerunning

The GPU experiments download their models from the Hugging Face Hub (Qwen3, Phi-2, Granite-3.3, Llama-3.1,
Mistral-7B and three sentence encoders); no weights are included. Every verdict generator and scorer runs on CPU from
the saved outputs. Paths of the authors' machines are replaced by placeholders (`/path/to/...`). Frontier-model arms
read API keys from files named in the scripts; no key is included. `python3 tools/assemble_tree.py` writes runnable
source trees under `work/`. Per-experiment commands and the hardware used are in `docs/DETAILS.md`.

## Not included

Model weights, large regenerable intermediates (each with the command that regenerates it), third-party benchmark
text (replaced in per-question files by its length and SHA-256), run logs, and the parts of the source repositories
that the paper does not use. `WITHHELD.md` lists each with the reason.

## License

Code: MIT (`LICENSE`). Pre-specifications, outputs, data, figure data and figures: CC BY 4.0 (`LICENSE-data`), with
the third-party exceptions listed there. Base models keep their own licenses.

## Citation

Wang, Z. and Imam, N. Vector symbolic architectures improve compositional reasoning and long-context recall in frozen
large language models. Submitted to *npj Unconventional Computing* (2026). A Zenodo DOI will be added on acceptance.

## Use of AI tools

As stated in the paper (Methods), an AI assistant (Claude, through Claude Code) wrote and ran the experiment and
analysis code, proposed experiments, drafted their pre-specifications, produced the figures and assembled this
repository; the authors directed the work and checked results against the pre-specified criteria. Second
implementations were written by separate AI agents, one of another model family. No number was verified by a person.
