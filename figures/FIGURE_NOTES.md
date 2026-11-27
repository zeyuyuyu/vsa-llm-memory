# Figures of the rewrite: what each panel shows (for the captions)

Written 2026-11-11. Five main figures and one supplementary figure, all drawn by `make_figures.py` (matplotlib) from the
files under `figdata/`. Every number in the tables below was copied by program from `FIGURE_VALUES.json`, the file
`make_figures.py` writes as it plots; that file holds the same values at full precision with the source string of each
series.

| Figure | File | Size | Panels |
|---|---|---|---|
| Fig. 1 | `figs/fig1_overview.pdf` | 180 x 126 mm | a three couplings of a frozen language model (schematic); b a worked two-hop retrieval (illustration) |
| Fig. 2 | `figs/fig2_benchmarks.pdf` | 180 x 146 mm | a BABILong against text length; b BABILong per task at 1M and 10M tokens; c CLUTRR |
| Fig. 3 | `figs/fig5_oplogs.pdf` | 180 x 120 mm | a three question sets; b paired outcomes against GPT-6.1 Sol; c past-window questions by site |
| Fig. 4 | `figs/fig3_store.pdf` | 180 x 84 mm | a store against prompt as the evidence grows; b capacity of one store; c sharding |
| Fig. 5 | `figs/fig4_mechanism.pdf` | 180 x 132 mm | a three arenas; b imposed cross-talk; c bindings per level |
| Supplementary Fig. 1 | `figs/figS1_sweeps.pdf` | 180 x 88 mm | a chain depth; b one projection read two ways; c facts held against dimension |

## Regenerating

```
cd /path/to/paper
python3 make_figures.py            # all six PDFs into figs/, plotted values into FIGURE_VALUES.json
python3 make_figures.py 5 S1       # a subset: 1 2 3 4 5 S1
python3 make_figures.py --no-verify
```

`make_figures.py` imports `figstyle.py` (palette, arm names, sizes), `figdata.py` (reader of `figdata/`) and
`schematic.py` (Fig. 1a). No number is typed in the plotting code. Values with no banked JSON artifact are in small
files, each value with a `source` string: `figdata/oplogs.json`, `figdata/benchmarks/babilong_per_task.json` (new),
`figdata/benchmarks/babilong.json` and `clutrr.json` (as in v54), `figdata/constants.json` (new: the 0.615 bar, the 0.70
criterion and its bootstrap settings, the pool of 800 facts) and `figdata/worked_example.json` (new: seed, vocabulary and
story of Fig. 1b).

On every run, `verify_sources()` re-checks the transcribed values against their source text while the source files are
on the machine: the stored line must be in the file verbatim, the number must be in that line (in the stated table cell
for table rows), and the stored value must equal it. It checks 62 values today (all of `oplogs.json` and
`babilong_per_task.json`) and stops the run on any mismatch; I tested it by altering four values in a scratch copy
(a value, a table cell, the paired counts, a per-site n) and it caught each.

Checks done on the final files:

- From a clean state (the six PDFs and `FIGURE_VALUES.json` deleted) the script was run twice in place and once in an
  empty scratch directory holding only the four scripts and `figdata/`. The six PDFs are byte-identical across the three
  runs, and rendered at 200 dpi the pages are pixel-identical (maximum difference 0).
- Every PDF is 180.0 mm wide, vector only (`pdfimages -list` finds nothing), creator Matplotlib 3.10.8, fonts embedded
  TrueType (Liberation Sans regular, bold, italic; no other font).
- Smallest text at the design size: 7.0 pt (the notes under the boxes of Fig. 1a, "not run" in Fig. 3a, the two prompt
  lengths in Fig. 4a); tick labels and legends 7.5 pt; axis labels 8.5 pt; panel letters 10 pt bold. The script measures
  the glyph sizes written to each PDF and fails under 7 pt. If the manuscript sets the figures at a text width of 6.5 in (165 mm, as v54 did) they
  shrink to 0.92 and 7 pt prints at 6.4 pt.
- `check()` (run on every save) finds no text outside the canvas, no overlapping text boxes and no schematic label
  outside its box. Each page was rendered and looked at; label-to-data collisions were fixed by hand.

## Conventions that hold in every figure

- **Arm names in legends:** prompting, LoRA fine-tune, projection trained through the binding loss, handover, exact
  codebook rows, frozen-state probe (`figstyle.NAME`). The prompting arm carries its model in brackets where the model
  matters: "prompting (Qwen3-8B)" in Fig. 4a.
- **Colours** (`figstyle.C`): prompting blue `#0072B2`; LoRA fine-tune orange `#E69F00`; handover, the HRR store and
  "ours" vermilion `#D55E00`; projection trained through the binding loss pink `#CC79A7`; exact codebook rows black
  `#000000`; frozen-state probe grey `#7F7F7F`; system of record grey-brown `#A08C78`; GPT-6.1 Sol bluish green
  `#009E73`; Qwen3-32B and other open models read as external baselines sky blue `#56B4E9`. Text is dark grey `#222222`
  (secondary text `#666666`), the grid a hairline `#E6E6E6`.
- **Colours that had to be used twice** (the palette has eight; see "Decisions" at the end): sky blue is also the
  continuous handover (Fig. 5b, Supplementary Fig. 1b); in Fig. 2 black is also ARMT and the gold-relations line, pink
  also Claude Opus 5.5, grey also GPT-4-0125 and the DSR-LM line. No figure uses one colour for two things.
- **Seed dots:** small dark dots with a white edge on bars, faint dots in the series colour on lines. Drawn only where
  an arm has per-seed values. Single runs, deterministic arms and series whose per-seed values are not banked have none.
- **No titles** inside the figures; panel letters are bold lower case.
- Accuracies are fractions, except Fig. 2 (per cent, as the benchmarks report them) and the x axis of Fig. 3b (share of
  questions, per cent).

---

## Fig. 1, `figs/fig1_overview.pdf`

### a, schematic (`schematic.py`, `draw_overview`)

No data. One input ("Mary took the football. Mary went to the hallway." in italics, "Where is the football?" in bold)
enters a grey box "frozen language model". Three couplings leave it:

- **prompting**, a blue `#0072B2` arrow to "answer", with the note "evidence in the prompt";
- **LoRA fine-tune**, an orange `#E69F00` arrow to "answer", with the note "low-rank update of the weights";
- **vector symbolic memory**, the primary lane: a pale vermilion band (`#FBEDE5`) under the model, entered by a
  vermilion `#D55E00` arrow labelled "hidden states". Four white boxes with vermilion edges, joined by vermilion arrows,
  each with a grey note under it:

| Box | Note under the box |
|---|---|
| identity head | the only trained part |
| codebook rows | fixed random unit vectors |
| bind and bundle | one superposed vector per hop |
| unbind and clean up | one unbinding per hop |
| answer (bold, no box) | a codebook row |

Changed from the v54 schematic: no formulas, no dashed frame, no sub- or superscripts (so no glyph under 7 pt), the
frozen model is neutral grey (blue is now only the prompting arm), the two generated-text couplings are single arrows
without boxes, and the memory lane spans the full width. Please check the five notes against the text; they are my
wording of the v54 notes ("supervised on symbol labels; the one trained component", "fixed random unit vectors", "one
superposed vector per hop", "two hops = two unbind-and-clean-up steps", "answer, always a codebook row").

### b, a worked two-hop retrieval

**This panel is an illustration computed from the algebra, not an experiment.** No language model is involved and the
codebook rows are given exactly; `make_figures.py` (`worked_example()`) computes it with numpy on every run from
`figdata/worked_example.json`.

- Seed 0 (`numpy.random.RandomState(seed).standard_normal`), d = 2048. 18 codebook rows: random Gaussian vectors
  normalized to unit length, in the order objects, persons, locations.
- Vocabulary. Objects: football, apple, milk, book, keys, lamp. Persons: Mary, John, Daniel, Sandra, Bill, Fred. Locations: hallway, kitchen, garden, office, bedroom, park.
- Story (6 persons, two statements each, so **6 pairs in each bundle**): "Mary took the football. Mary went to the hallway. John took the apple. John went to the kitchen. Daniel took the milk. Daniel went to the garden. Sandra took the book. Sandra went to the office. Bill took the keys. Bill went to the bedroom. Fred took the lamp. Fred went to the park." Question: "Where is the football?"
- Operations, as in the arena code (`/path/to/HOME/v13.py`, `/path/to/HOME/v37b.py`): binding is circular convolution,
  `irfft(rfft(a) * rfft(b))`; a bundle is the plain sum of its bound pairs; unbinding is binding with the approximate
  inverse, `irfft(rfft(M) * conj(rfft(key)))`; the unbound vector is normalized and compared by cosine with the
  candidate rows; the nearest row is taken (clean-up) and is the key of the next hop.
  Holder bundle = sum over statements of bind(row(person), row(object)); location bundle = sum of bind(row(location),
  row(person)).
- Left chart, x label "hop 1: holder bundle unbound by “football”, compared with every person row"; right chart,
  "hop 2: location bundle unbound by “Mary”, compared with every location row". y: "cosine with codebook row", 0 to
  0.6. The selected row is vermilion `#D55E00`, the other rows light grey `#BDBDBD`. Each chart carries the label
  "clean-up selects the nearest row" at the tall bar and a bracket labelled "cross-talk" over the grey bars.
- Retrieved chain: football -> Mary -> hallway.

| Hop 1, person row | Mary | John | Daniel | Sandra | Bill | Fred |
|---|---|---|---|---|---|---|
| cosine | 0.4114 | -0.0096 | 0.0315 | 0.0359 | 0.0165 | 0.0039 |

| Hop 2, location row | hallway | kitchen | garden | office | bedroom | park |
|---|---|---|---|---|---|---|
| cosine | 0.3962 | 0.0507 | 0.0198 | -0.0066 | 0.0296 | 0.0196 |

For the caption: with 6 pairs in a bundle the right row scores about 0.4 (the algebra predicts roughly
1/sqrt(6 + 1) = 0.38) and the other rows scatter around zero with a standard deviation of about 1/sqrt(d) = 0.022; that
residue is what the panel labels "cross-talk". It is the noise the other pairs of the bundle leave on the read-out.
Seed 0 is an ordinary draw: over seeds 0 to 1,999 (checked once by hand-run script, not part of `make_figures.py`) the
chain is retrieved 2,000 times of 2,000, the winning cosine is 0.378 on average (s.d. 0.017) and the other rows' cosines
average 0.000 (s.d. 0.024). Fig. 5b uses "cross-talk" for a related but different quantity, the
mean cosine between different symbols' codes; in this example that is 0.0001 (largest absolute value 0.0580), because the
rows are exact. numpy 2.2.6; the cosines above are what that version produces (`RandomState` streams are frozen across
numpy versions).

Not in any figure any more: the bAbI accuracy bars (author's decision). Their numbers are at the end of this file for the
Supplementary Table.

---

## Fig. 2, `figs/fig2_benchmarks.pdf`

### a, BABILong against the length of the text

- x: eight equally spaced positions 0k, 4k, 16k, 64k, 128k, 512k, 1M, 10M; label "length of the text (tokens; positions not to scale)". y: "BABILong
  accuracy, QA1–5 (%)", 28 to 102.5 (mean over QA1-QA5, official rule).
- Under the accuracy axis a pale grey strip (`#F2F2F2`) that is not part of the accuracy scale holds two crosses at 1M,
  bluish green for GPT-6.1 Sol and pink for Claude Opus 5.5, beside the text "input rejected: exceeds context window".
- **GPT-4-0125 and Llama-3.1-70B are labelled where their curves end**, "Llama-3.1-70B, 45% at 128k" and "GPT-4-0125,
  36% at 128k" (the numbers are read from the file), and are drawn heavier (1.5 pt, larger markers) than the other
  reference systems (1.1 pt). They coincide at 4k and nearly at 0k and 16k.
- Single runs or published numbers: no dots.

| Legend name | Colour, marker | 0k | 4k | 16k | 64k | 128k | 512k | 1M | 10M | n per point |
|---|---|---|---|---|---|---|---|---|---|---|
| ours (fine-tuned 1.7B reader + memory) | vermilion `#D55E00`, circles, 1.9 pt line | 99.6 | 97.8 | 99.2 | 99.6 | 99.8 | 100.0 | 100.0 | 100.0 | 0k: 500; 4k: 500; 16k: 500; 64k: 500; 128k: 500; 512k: 125; 1M: 125; 10M: 25 |
| ARMT 145M, fine-tuned | black `#000000`, squares | 99.3 | 98.1 | 98.1 | 97.9 | 96.9 | 95.3 | 93.4 | 76.6 | not ours: published row |
| GPT-6.1 Sol, zero-shot* | bluish green `#009E73`, triangles | 100.0 | 99.2 | 100.0 | 99.2 | 96.8 | 96.0 | rejected | - | 0k: 125; 4k: 125; 16k: 125; 64k: 125; 128k: 125; 512k: 25 |
| Claude Opus 5.5, zero-shot* | pink `#CC79A7`, diamonds | 100.0 | 97.6 | 95.2 | 96.0 | 97.6 | 88.0 | rejected | - | 0k: 125; 4k: 125; 16k: 125; 64k: 125; 128k: 125; 512k: 25 |
| GPT-4-0125 | grey `#7F7F7F`, inverted triangles, 1.5 pt line | 87 | 74 | 64 | 43 | 36 | - | - | - | not ours: published row |
| Llama-3.1-70B | sky blue `#56B4E9`, plus signs, 1.5 pt line | 85 | 74 | 65 | 53 | 45 | - | - | - | not ours: published row |

Sources (one `source` string per point in `figdata/benchmarks/babilong.json`, unchanged from v54): ours
`vsassm/results/b18_verdict.txt`, table row "QA1-5" and "QA1-5 10M: 25/25 = 100.0"; ARMT the leaderboard line of the
same file and, at 10M, the ARMT paper's Table 1 as quoted there; GPT-6.1 Sol and Claude Opus 5.5
`vsassm/results/b18_api_verdict.txt` (first 25 questions per task, first 5 per task at 512k: the asterisk in the legend
stands for that subset); GPT-4-0125 and Llama-3.1-70B the leaderboard lines of `b18_verdict.txt`. The 36 and 45 at 128k
are also the cells of the public leaderboard file banked as `figdata/benchmarks/babilong_leaderboard_v0_results.csv`
(rows `gpt-4-0125-preview` and `Meta-Llama-3.1-70B-Instruct`, task `avg`).

Notes for the caption: ours is a 1.7B reader fine-tuned on the BABILong training split, as ARMT was; GPT-4-0125 and
Llama-3.1-70B are leaderboard rows that stop at 128k; the label "ARMT 145M, fine-tuned" follows the ARMT paper (the
leaderboard lists 137M). The figure does not contain the words "best published" (the memory note
`readable_memory_lit_check` records a later system, Q-RAG, at about 95% at 10M).

### b, BABILong per task at one and ten million tokens (new)

- x: "BABILong task", QA1 (single supporting fact), QA2 (two supporting facts), QA3 (three supporting facts), QA4 (two-argument relations), QA5 (three-argument relations). y: "BABILong accuracy (%)", 0 to 105.
- Four bars per task, in two pairs: at 1M tokens ours then ARMT (solid), at 10M tokens ours then ARMT (pale fill with
  an edge in the system's colour).
- Ours is the count of correct answers; ARMT is the mean of 3 runs published in its paper. ARMT's standard deviations
  are in the table and in the data file but are **not drawn**. No dots.

| Legend name | Colour | QA1 | QA2 | QA3 | QA4 | QA5 |
|---|---|---|---|---|---|---|
| ours, 1M tokens | vermilion `#D55E00`, solid | 100.0 (25/25) | 100.0 (25/25) | 100.0 (25/25) | 100.0 (25/25) | 100.0 (25/25) |
| ARMT, 1M tokens | black `#000000`, solid | 98.5 (s.d. 1.0) | 99.4 (s.d. 0.3) | 72.1 (s.d. 14.2) | 99.8 (s.d. 0.3) | 97.3 (s.d. 0.6) |
| ours, 10M tokens | pale vermilion `#f0c7a6`, edge `#D55E00` | 100.0 (5/5) | 100.0 (5/5) | 100.0 (5/5) | 100.0 (5/5) | 100.0 (5/5) |
| ARMT, 10M tokens | light grey `#C9C9C9`, edge `#000000` | 89.4 (s.d. 8.1) | 84.4 (s.d. 4.0) | 37.0 (s.d. 10.3) | 91.5 (s.d. 1.7) | 80.9 (s.d. 7.6) |

**n per bar for ours:** 25 questions per task at 1M (the first 25 of the 100 official test questions per task;
`b18_verdict.txt`: "25 per task at 512k / 1M"), 5 per task at 10M (the first 5 official test samples per task). All 150
answers are correct.

**Where each ARMT value was found** (you asked for exactly this):

- `vsassm/results/b18_verdict.txt` (line 62) quotes the paper's Table 1 **at 10M only**: 89.4, 84.4, 37.0, 91.5, 80.9, mean 76.6. It does **not**
  contain "QA3 = 72.1 at one million"; that number is in the old manuscript (`reference_v54/results.tex`, line 191)
  and in the paper.
- The public leaderboard file (`scratchpad/lb/lb.csv`, banked as
  `figdata/benchmarks/babilong_leaderboard_v0_results.csv`; Hugging Face space RMT-team/babilong, `data/leaderboard-v0_results.csv`, space sha 90b74f9e) has an ARMT row per task at both lengths, in whole
  per cent: 1M QA1 99, QA2 99, QA3 72, QA4 100, QA5 93.4, avg 93.4; 10M QA1 89, QA2 84, QA3 37, QA4 92, QA5 76.6, avg 76.6. **Its `qa5` row is a copy of its `avg` row from 4k onward** (4k 98.1 ... 1M 93.4, 10M 76.6),
  so the leaderboard holds no QA5 value for ARMT at 1M or 10M.
- **All ten ARMT bars are therefore the paper's own Table 1** (Rodkin et al., arXiv:2407.04841, page 10, row "ARMT
  (145M)", columns 1M and 10M, "3 runs averaged"). I downloaded the arXiv PDF (md5 `01f8bef5ef96841fdd3acd96bc9dce2e`, PDF created 2025-02-17) and banked the
  text layer of the table as `figdata/benchmarks/armt_table1_arxiv2407.04841.txt`; `babilong_per_task.json` was cut
  from that text by program. **This is the one place where I went to the network** (one PDF from arxiv.org; no ssh).
- Cross-checks the script asserts on every run: the paper's values agree with the leaderboard's whole-per-cent cells
  for QA1-QA4 at both lengths to within rounding (0.5); their mean over the five tasks is the leaderboard's `avg` (93.42
  against 93.4 at 1M, 76.64 against 76.6 at 10M); the 10M row equals the quote in `b18_verdict.txt`.

The paper's column "500k" (not drawn) reads 99.3, 99.4, 79.7, 99.9, 98.4. For the caption: ARMT is "145M" in its paper
and "137M" on the leaderboard; our n is small at 10M (5 per task; the Wilson 95% interval of 25/25 is 86.7 to 100, as
`b18_verdict.txt` states).

### c, CLUTRR

- x: "relations in the chain", k = 2..10. y: "CLUTRR accuracy (%)", 35 to 102.5. Single runs: no dots.

| Legend name | Colour, style | k = 2 | k = 3 | k = 4 | k = 5 | k = 6 | k = 7 | k = 8 | k = 9 | k = 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| ours (frozen Qwen3-8B reader + rule memory) | vermilion `#D55E00`, circles, 1.9 pt line | 0.9474 | 0.6667 | 0.8053 | 0.7816 | 0.7850 | 0.6806 | 0.6267 | 0.6471 | 0.6218 |
| n (correct) of ours |  | 38 (36) | 105 (70) | 190 (153) | 174 (136) | 107 (84) | 144 (98) | 150 (94) | 119 (77) | 119 (74) |
| Qwen3-8B answering directly (reasoning on) | blue `#0072B2`, triangles, 1.1 pt line | 0.921 | 0.400 | 0.768 | absent | 0.673 | absent | 0.580 | absent | 0.496 |

(The figure plots these as per cent.) Reference lines, all chain lengths pooled, in the legend under the panel:

| Legend name | Colour, style | Value | Source |
|---|---|---|---|
| rule memory given gold relations, 97.5 | black `#000000`, dotted | 97.5 | vsassm/results/r1_verdict.txt: 'The composition engine is exact (oracle facts: 0.975, 1.000 at k=10)'; vsassm/results/nsr0_verdict.txt: 'T reproduces R1's 0.9747 exactly' |
| SatLM (code-davinci-002, rules given), 68.3 | dark grey `#555555`, dash-dotted | 68.3 | Ye et al., SatLM, NeurIPS 2023, Table 1 (CLUTRR, code-davinci-002, greedy; 1,042-story version of the test set, kinship rules given); transcribed from vsassm/docs/neurosymbolic_review.md lines 27, 98 and 147 |
| DSR-LM (rules learned from data), 61.0 | grey `#7F7F7F`, dashed | 61.0 | Zhang et al., DSR-LM, Findings of ACL 2023, Tables 5/6: 60.98 +- 2.64; transcribed from vsassm/docs/neurosymbolic_review.md lines 107 and 150 |

- Ours: counts per k from `figdata/r1_logs/ours_by_k.json` (recomputed in v52 from
  `vsassm/r1_out/test_8B_think.jsonl`, 1,146 stories; overall 822/1146 = 0.7173). The script asserts that the five even-k
  values equal the ones of `r1_verdict.txt` to three decimals.
- The direct arm (the same frozen Qwen3-8B answering without the memory, reasoning on) has k = 2, 3, 4, 6, 8, 10 only; k = 5, 7, 9 are
  absent from `r1_verdict.txt`. Neighbouring k are joined by a solid line and **a dashed line bridges each missing
  k**. `figdata/r1_logs/direct_by_k.json` is picked up automatically if you add it.
- It is drawn in the prompting blue, not in the sky blue of the open models of Fig. 3: it is the frozen reader itself,
  prompted directly, as in Fig. 4a.
- SatLM and DSR-LM are transcribed from `vsassm/docs/neurosymbolic_review.md`, not from the papers (as in v54).

---

## Fig. 5, `figs/fig5_oplogs.pdf`

**r13 (2026-11-12): Fig. 6 GPT-6.1 Sol values replaced.** The first GPT-6.1 Sol runs went through the Sidrune reseller
gateway, which is not serving OpenAI's model or not counting its tokens (vsassm/results/nsr5_verdict.txt, route
replication). Every GPT-6.1 Sol value in Fig. 6 is now the re-run through OpenAI's API, where all parses agree, so the
open "repaired parse" markers are gone: a impact 0.952, window 0.904; b impact 205/9/14/2 (either 0.991), window
369/33/47/11 (either 0.976); c window by site 0.925/0.800/0.875/0.963; d 0.952/0.925/0.855 (end-of-line labels). The
gateway values stay in figdata under `gateway*` keys (not drawn) and in Supplementary Table 5. The tables below
this note describe the r12 figure and are superseded for GPT-6.1 Sol.

All values are single runs (one scoring of a held-out split): no dots anywhere in this figure. Every number below is in
`figdata/oplogs.json` with the verbatim source line, and is re-checked against the source file on every run.

### a, three question sets, four systems

- x: three groups, "where is it now (n = 414)", "impact (n = 230)", "past window (n = 460)". y: "questions answered
  exactly", 0 to 1 (set-exact for impact and past-window questions).
- Bars, left to right in each group: system of record, Qwen3-32B reading the whole log, GPT-6.1 Sol reading the whole
  log, ours. GPT-6.1 Sol has no bar in the first group and the words "not run" (7 pt, grey) in its place. On each
  GPT-6.1 Sol bar an **open circle** (white, bluish-green edge) marks the repaired-parse value.

| Legend name | Colour | where is it now (n = 414) | impact (n = 230) | past window (n = 460) |
|---|---|---|---|---|
| system of record | grey-brown `#A08C78` | 0.396 | 0.313 | 0.217 |
| Qwen3-32B reading the whole log | sky blue `#56B4E9` | 0.543 | 0.278 | 0.202 |
| GPT-6.1 Sol reading the whole log | bluish green `#009E73` | not run | 0.922 (open marker 0.943) | 0.830 (open marker 0.839) |
| ours: 1.7B reader + memory | vermilion `#D55E00` | 0.959 | 0.930 | 0.874 |

Source of each value (file, line, what the line says):

| Question set | System | Value | Source |
|---|---|---|---|
| where is it now | system of record | 0.396 | vsassm/apps/real/README.md line 52, table row 'All', column 2 (System of record alone) |
| where is it now | Qwen3-32B reading the whole log | 0.543 | vsassm/apps/real/README.md line 52, table row 'All', column 3 (Qwen3-32B, budget-forced score; strict 0.531 in brackets) |
| where is it now | ours: 1.7B reader + memory | 0.959 | vsassm/apps/real/README.md line 52, table row 'All', column 1 (Readable Memory) |
| where is it now | n | 414 | vsassm/apps/real/README.md line 52: '**0.959** (397/414)' (n of the held-out test2 split, 'All' row) |
| impact | system of record | 0.313 | vsassm/results/nsr1_verdict.txt line 16, table row '4  system of record + graph query', column 1 (all) |
| impact | Qwen3-32B reading the whole log | 0.278 | vsassm/results/nsr1_verdict.txt line 12, table row '3  Qwen3-32B thinking over the whole log (forced)', column 1 (all; budget-forced) |
| impact | GPT-6.1 Sol reading the whole log | 0.922 | vsassm/results/nsr1_verdict.txt line 119: 'Registered parse: frontier 0.917 -> 0.922' (registered parse after frontier amendment 1 (the one dropped stream re-run)) |
| impact | ours: 1.7B reader + memory | 0.930 | vsassm/results/nsr1_verdict.txt line 9, table row '1  RM state + VSA rule memory (rules from 1 worked trace)', column 1 (all) |
| impact | GPT-6.1 Sol reading the whole log, repaired parse | 0.943 | vsassm/results/nsr1_verdict.txt line 127: 'model scores **0.943**' (repaired parse, frontier amendment 2, post hoc) |
| impact | n | 230 | vsassm/results/nsr1_verdict.txt line 7: '## Results (set-exact accuracy, all 230 questions)' (n of the held-out nsr1 split) |
| past window | system of record | 0.217 | vsassm/results/nsr3_verdict.txt line 12, table row 'system-of-record history', column 1 (all) |
| past window | Qwen3-32B reading the whole log | 0.202 | vsassm/results/nsr3_verdict.txt line 9, table row 'Qwen3-32B thinking over the log (forced; registered)', column 1 (all; budget-forced, registered parse) |
| past window | GPT-6.1 Sol reading the whole log | 0.830 | vsassm/results/nsr3_verdict.txt line 52, table row 'gpt-6.1-sol over the log (registered parse)', column 1 (all; registered parse) |
| past window | ours: 1.7B reader + memory | 0.874 | vsassm/results/nsr3_verdict.txt line 8, table row 'history engine over the read state', column 1 (all; first results table) |
| past window | GPT-6.1 Sol reading the whole log, repaired parse | 0.839 | vsassm/results/nsr3_verdict.txt line 81: '(--repair) the frontier model scores 0.839' (repaired parse, frontier amendment 2, post hoc) |
| past window | n | 460 | vsassm/results/nsr3_verdict.txt line 2: 'seeds 500+), 460 temporal questions' (n of the held-out nsr2 logs) |

Notes for the caption:

- **All eleven numbers of the brief were found verbatim; none is missing.** 0.959 is 397/414.
- The three sets are three different held-out sets of 40 logs: `test2` (seeds 300+; where is it now), `nsr1` (seeds
  400+; impact), `nsr2` (seeds 500+; past window). "Ours" is the 1.7B reader with the memory in all three, with a
  different read-out on top: the tracked state (README, "Readable Memory"), the state plus the VSA rule memory
  (NSR1, arm 1) and the history engine over the read state (NSR3).
- The Qwen3-32B values are the **budget-forced** scores (a run that hits the 8,192-token reasoning limit is told to
  answer); the strict scores are 0.531, 0.265 and 0.194.
- GPT-6.1 Sol was not run on the first set. Its bars are the **registered parse**: 0.922 on impact is the value after
  frontier amendment 1 (the one dropped stream re-run; 0.917 before), 0.830 on past window. The open markers are the
  repaired parse of frontier amendment 2 (post hoc: glued "Answer:" lines restored), 0.943 and 0.839. Note that
  `nsr1_verdict.txt` also prints 0.922 for a different quantity (the underscore-normalising parse before amendment 1);
  the value drawn is taken from the line "Registered parse: frontier 0.917 -> 0.922".

### b, paired outcomes against GPT-6.1 Sol, registered parse (new)

- Two horizontal stacked bars, one per question set, each with its name and n above it ("impact (n = 230)", "past
  window (n = 460)"). x: "share of questions (%)", 0 to 100. Each segment carries its count; a count that does not fit
  its segment is written after the end of the bar (the 3 and the 17 of "both wrong").
- Under each bar a bracket spans the questions at least one of the two systems answers, labelled "either right: 0.987"
  and "either right: 0.963".

| Bar | both right, light grey `#D9D9D9` | only ours, vermilion `#D55E00` | only GPT-6.1 Sol, bluish green `#009E73` | both wrong, dark grey `#4D4D4D` | either right | Source |
|---|---|---|---|---|---|---|
| impact (n = 230) | 199 (86.5%) | 15 (6.5%) | 13 (5.7%) | 3 (1.3%) | 0.987 (= 227/230) | vsassm/results/nsr1_verdict.txt lines 120-121 (Correction 2026-11-11, frontier amendment 1; registered parse): 'Both right 199, only RM 15, only frontier 13, both wrong 3; either right 0.987' |
| past window (n = 460) | 341 (74.1%) | 61 (13.3%) | 41 (8.9%) | 17 (3.7%) | 0.963 (= 443/460) | vsassm/results/nsr3_verdict.txt line 58 (addendum, registered parse): 'both right 341, only history 61, only frontier 41, both wrong 17; either right 0.963' |

- "only ours" is "only RM" in `nsr1_verdict.txt` and "only history" in `nsr3_verdict.txt`; "only GPT-6.1 Sol" is
  "only frontier". The script checks that the four counts sum to n and that (n - both wrong)/n rounds to the printed
  "either right".
- **Recorded, not drawn:** the repaired-parse line of NSR1: both right 203, only ours 11, only GPT-6.1 Sol 14, both wrong
  2, either right 0.991 (nsr1_verdict.txt lines 128-129, Correction 2, post hoc). `nsr3_verdict.txt` gives no paired counts for the repaired parse.
- The paired differences, for the text: impact, ours minus GPT-6.1 Sol +0.009 (95% bootstrap interval -0.035 to
  +0.057); past window +0.044 (+0.002 to +0.087) under the registered parse (both from the same verdict files; not
  drawn).

### c, past-window questions by site (new)

- x: "past-window questions, by site", four groups with their n. y: "questions answered exactly", 0 to 1.
- Three bars per site, left to right: Qwen3-32B (a **pale tint** `#b3ddf5` of its sky blue, as you asked), GPT-6.1 Sol
  (registered parse), ours. The legend at the lower right serves panels a and c and shows Qwen3-32B in its full sky
  blue; say in the caption that it is drawn pale in c.

| System | warehouse (n = 80) | biobank (n = 60) | home robot (n = 160) | cloud incident (n = 160) | Source row |
|---|---|---|---|---|---|
| ours: 1.7B reader + memory | 0.725 (58/80) | 0.933 (56/60) | 0.838 (134/160) | 0.963 (154/160) | `nsr3_verdict.txt` line 8 |
| GPT-6.1 Sol reading the whole log | 0.850 (68/80) | 0.717 (43/60) | 0.750 (120/160) | 0.944 (151/160) | `nsr3_verdict.txt` line 52 |
| Qwen3-32B reading the whole log | 0.113 (9/80) | 0.083 (5/60) | 0.088 (14/160) | 0.406 (65/160) | `nsr3_verdict.txt` line 9 |

- Column names in the source are warehouse, biobank, home, ops; the figure writes "home robot" and "cloud incident"
  (the scenario names of `apps/real/README.md`).
- **n per site** is not in the verdict text except for warehouse ("frontier-only right 18 of 80 warehouse
  questions"). I counted it from the question ids of `vsassm/nsr/out/nsr3_nsr2_gpt61_strict.jsonl` (460 records): 80,
  60, 160, 160. The counts in brackets are round(accuracy x n); every printed accuracy equals count/n to 0.0005.
- For the text: GPT-6.1 Sol leads on warehouse (0.850 against 0.725), ours on biobank (0.933 against 0.717) and home
  robot (0.838 against 0.750); the two are close on cloud incident (0.963 against 0.944).

---

## Fig. 3, `figs/fig3_store.pdf`

### a, both routes given the same pooled facts (V-106, new)

- x: "pooled facts given to both routes", logarithmic, ticks 8, 64, 128, 512 and "800 (all)" (written one line lower
  so that it does not touch 512). y: "exact match", 0 to 1.
- **The last scale was requested as 1,024 and is drawn at 800.** The pool holds 800 facts (`figdata/constants.json`
  `pool`). The script checks this against V-106's own prompt lengths, which grow by 14.35 tokens per fact from 8 to
  512 facts: the 11,612-token prompt of the last scale corresponds to 800 facts.
- Grey 7 pt notes give the mean prompt length at the two ends: "254 tokens" and "11,612 tokens".

| Legend name | Colour, marker | 8 | 64 | 128 | 512 | 800 (all) | n and seeds | Source |
|---|---|---|---|---|---|---|---|---|
| prompting (Qwen3-8B) | blue `#0072B2`, triangles | 0.9500 | 0.9083 | 0.8500 | 0.5833 | 0.5417 | n = 120 questions per scale, single run, no dots | `v106_logs/v106.json` `prompt[].acc` |
| mean prompt length (tokens) |  | 253.5 | 1,049.9 | 1,972.6 | 7,482.0 | 11,612.1 |  | `prompt[].tokens` |
| HRR store, d = 2048, at most 64 facts per store | vermilion `#D55E00`, circles, 1.6 pt line | 0.9361 (0.9333, 0.9417, 0.9333) | 0.9333 (0.9333, 0.9333, 0.9333) | 0.9194 (0.9250, 0.9083, 0.9250) | 0.9194 (0.9167, 0.9250, 0.9167) | 0.9083 (0.9250, 0.9000, 0.9000) | 120 questions, three codebook seeds (0, 1, 2), dots | `store[].acc`, `store[].acc_seeds` |
| stores (facts per store) |  | 1 (8) | 1 (64) | 2 (64) | 8 (64) | 16 (50) |  | `store[].shards` |

- The LoRA arm of `v106.json` (0.917, 0.825, 0.692, 0.550, 0.425) is **not drawn**, as briefed.
- The store is sharded at 64 facts per store; at the last scale the 800 facts are spread over 16 stores, 50 each. The
  legend therefore says "at most 64 facts per store".
- `v106.json` `config`: model Qwen/Qwen3-8B, `n_eval` 120, `n_pool` 400, `dim` 2048, `cb_seeds` [0, 1, 2], prereg `82da296`. Whether any prompt was
  truncated is not recorded in the artifact.

### b, one store: readout against the number of superposed facts (V-151, HRR)

- x: "facts superposed in one store", logarithmic; labelled ticks 8, 32, 128, 512 and "800 (all)", unlabelled minor
  ticks at 16, 64, 256. y: "readout exact match", 0 to 1. Dashed black line: "0.70 criterion".
- One curve per dimension, a single hue from light to dark; **d = 2048, the dimension of panels a and c, carries the
  store's own vermilion.** Each point is the mean over 200 query instances and three capacity seeds (0, 1, 2);
  per-seed values are not banked, so there are no dots.
- The grid's last value, 1,024, is drawn at 800 (the pool); the other fourteen grid values (8 to 724) are real.

| d | Colour | k = 8 | k = 11 | k = 16 | k = 23 | k = 32 | k = 45 | k = 64 | k = 91 | k = 128 | k = 181 | k = 256 | k = 362 | k = 512 | k = 724 | k = 800 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | `#efc29e` | 0.9450 | 0.9467 | 0.9367 | 0.9100 | 0.8450 | 0.6583 | 0.4033 | 0.1683 | 0.0850 | 0.0283 | 0.0067 | 0.0100 | 0.0050 | 0.0000 | 0.0017 |
| 1024 | `#e2904f` | 0.9450 | 0.9433 | 0.9417 | 0.9433 | 0.9350 | 0.9050 | 0.7917 | 0.5550 | 0.3033 | 0.1200 | 0.0467 | 0.0183 | 0.0067 | 0.0033 | 0.0033 |
| 2048 | `#d55e00` | 0.9450 | 0.9400 | 0.9450 | 0.9400 | 0.9383 | 0.9400 | 0.9383 | 0.9017 | 0.7433 | 0.5150 | 0.2450 | 0.1217 | 0.0400 | 0.0083 | 0.0067 |
| 4096 | `#843a00` | 0.9467 | 0.9450 | 0.9450 | 0.9383 | 0.9400 | 0.9417 | 0.9383 | 0.9350 | 0.9283 | 0.8850 | 0.7317 | 0.4017 | 0.1583 | 0.0533 | 0.0317 |

Source: `v151_logs/v151.json` `capacity.HRR[d][k]`.

### c, sharding at d = 2048 (V-102)

- x: "stores the facts are spread over", 1, 2, 4, 8, 16 (logarithmic). y: "readout exact match, d = 2048", 0 to 1.
  Dashed black line: "0.70 criterion".
- One curve per pool size, the same light-to-dark ramp; faint dots are the three seeds of each cell.
- The curve requested as K = 1,024 is labelled "K = 800 (all)"; its 16-store point holds 50 facts per store.

| Legend name | Colour | Mean at 1, 2, 4, 8, 16 stores | Per seed | Source |
|---|---|---|---|---|
| K = 128 | `#efc29e` | 0.7278, 0.9194, 0.9278, 0.9306, 0.9278 | 0.7000, 0.7750, 0.7083; 0.9250, 0.9083, 0.9250; 0.9250, 0.9250, 0.9333; 0.9333, 0.9250, 0.9333; 0.9250, 0.9250, 0.9333 | `v102_logs/v102_scored.json grid['d2048_K128_S*']` |
| K = 256 | `#e2904f` | 0.2194, 0.6639, 0.9056, 0.9250, 0.9250 | 0.2250, 0.2583, 0.1750; 0.6750, 0.7250, 0.5917; 0.9083, 0.9000, 0.9083; 0.9250, 0.9167, 0.9333; 0.9250, 0.9250, 0.9250 | `v102_logs/v102_scored.json grid['d2048_K256_S*']` |
| K = 512 | `#d55e00` | 0.0389, 0.1889, 0.6472, 0.9194, 0.9222 | 0.0500, 0.0417, 0.0250; 0.1417, 0.2333, 0.1917; 0.6583, 0.5917, 0.6917; 0.9167, 0.9250, 0.9167; 0.9250, 0.9250, 0.9167 | `v102_logs/v102_scored.json grid['d2048_K512_S*']` |
| K = 800 (all) | `#843a00` | 0.0083, 0.0444, 0.2556, 0.7222, 0.9083 | 0.0167, 0.0083, 0.0000; 0.0417, 0.0333, 0.0583; 0.2917, 0.2417, 0.2333; 0.7000, 0.7417, 0.7250; 0.9250, 0.9000, 0.9000 | `v102_logs/v102_scored.json grid['d2048_K1024_S*']` |

n and seeds: three seeds per cell (0, 1, 2) and 120 query instances per seed, the defaults of `v102.py`, consistent
with every value being a multiple of 1/120; neither is recorded in the artifact (as noted for v52).

---

## Fig. 4, `figs/fig4_mechanism.pdf`

Layout: a (top left) and c (bottom left) are stacked; **b, the key panel, fills the right half** (73 x 72 mm of axes,
against 47 x 43 mm in v54).

### a, three arenas, five arms

- x: three groups, "A: 3 hops, 16 pairs per level", "B: 1 hop, 64 bindings", "C: 2 hops, 32 bindings per level".
  y: "chain accuracy", 0 to 1.05. Bars are means of three seeds, dots are the seeds.
- The continuous handover is not in this panel any more (it is a marker in b).

| Legend name | Colour | A: mean (per seed) | B: mean (per seed) | C: mean (per seed) | Source |
|---|---|---|---|---|---|
| frozen-state probe | grey `#7F7F7F` | 0.5090 (0.5080, 0.5110, 0.5080) | 0.8147 (0.7940, 0.8180, 0.8320) | 0.5153 (0.5120, 0.5300, 0.5040) | A v1_logs/frozen_llm.json acc; B v11_logs/v18_spanfix.json arm floor; C v13_logs/v13_fixed.json arm floor |
| projection trained through the binding loss | pink `#CC79A7` | 0.5010 (0.5110, 0.4990, 0.4930) | 0.5073 (0.5020, 0.4980, 0.5220) | 0.4833 (0.4720, 0.5020, 0.4760) | A v8_logs/v8.json acc; B v18_spanfix arm spans; C v13_fixed arm spans |
| LoRA fine-tune | orange `#E69F00` | 0.5053 (0.4930, 0.5080, 0.5150) | 0.7950 (0.9800, 0.8950, 0.5100) | 0.5158 (0.5150, 0.5175, 0.5150) | A v1_logs/lora.json acc (rank 16); B, C v15_logs/v15.json (rank 16) |
| handover | vermilion `#D55E00` | 0.9137 (0.9150, 0.9130, 0.9130) | 0.9960 (0.9900, 1.0000, 0.9980) | 0.9340 (0.9320, 0.9320, 0.9380) | A v9_logs/v9_gpu1_rerun_partial.json arm extract_then_bind; B v18_spanfix arm probe_argmax; C v13_fixed arm probe_argmax |
| exact codebook rows | black `#000000` | 0.9183 (0.9180, 0.9190, 0.9180) | 0.9960 (0.9900, 1.0000, 0.9980) | 0.9340 (0.9320, 0.9320, 0.9380) | A v4_logs/v4_perocc.json epsilon 0; B v18_spanfix arm parser; C v13_fixed arm parser |

n and seeds (from the v52 notes, which the artifacts only partly record): three seeds per bar (42, 123, 456).
Arena A, probe and LoRA: V-1 protocol, 2,000 train / 1,000 eval. Arena C: train arena 800, test arena 500. LoRA on B and
C (`v15.py`): rank 16, 600 train / 400 test, best test epoch; on C the three seeds share one data draw. The artifacts
do not record n for the other arms of arenas A and B. On arena B the three LoRA seeds are 0.980, 0.895, 0.510: the
bar's mean (0.795) averages two seeds near the top and one at chance, which the dots show.

### b, imposed cross-talk

- x: "cross-talk (mean cosine between different symbols’ codes)". **The axis is linear from 0 to 0.1 and logarithmic
  from 0.1 to 1** (matplotlib `symlog`, `linthresh = 0.1`); labelled ticks 0, 0.05, 0.1, 0.2, 0.5, 1, minor ticks at
  0.3, 0.4, 0.6, 0.7, 0.8, 0.9. Please say so in the caption. y: "chain accuracy, arena C (fitted rule)", 0.4 to 1.0.
- Black curve with dots: exact codebook rows with cross-talk imposed (V-163), mean of three codebook seeds at each of
  14 levels; faint grey dots are the three seeds, each at its own measured cross-talk. Test arena 600 instances per
  seed, decision threshold fitted on a training arena of 800.
- Dashed grey line at 0.615, labelled "frozen-state probe + 0.10"; dotted grey line at 0.515, labelled "frozen-state
  probe" (the arena-C probe of panel a, 0.5153, as V-95 and V-163 fix it; `figdata/constants.json`).
- Thin dashed vertical line from the x axis to the curve at 0.2486, labelled "tolerance 0.25": the largest imposed
  cross-talk whose mean accuracy still clears 0.615 (`v163_scored.json` `tolerance`).

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

Source: `v163_logs/v163.json` `curve[].crosstalk_mean`, `fitted_mean`, `crosstalk`, `fitted`.

Arms, each drawn at (its measured cross-talk, the accuracy it reached); the small dots of the same colour are seeds:

| Legend name | Marker, colour | Cross-talk: mean (per seed) | Accuracy: mean (per seed) | Source |
|---|---|---|---|---|
| handover | filled circle, vermilion `#D55E00` | 0.0001 (0.0001) | 0.9340 (0.9320, 0.9320, 0.9380) | x geometry_banked.json crosstalk_discrete (one seed); y v13_logs/v13_fixed.json arm probe_argmax (three seeds) |
| continuous handover | filled square, sky blue `#56B4E9` | 0.0315 (0.0250, 0.0441, 0.0255) | 0.9317 (0.9233, 0.9383, 0.9333) | crosstalk_audit2.json fit3.xt_seeds, fit3.acc_seeds |
| projection, before training | open diamond, pink `#CC79A7` | 0.7177 (0.7204, 0.7139, 0.7188) | 0.5160 (0.5140, 0.5000, 0.5340) | v165_logs/v165.json seeds[].init.different, seeds[].init.acc |
| projection trained through the binding loss | filled diamond, pink `#CC79A7` | 0.9884 (0.9879, 0.9895, 0.9878) | 0.4833 (0.4720, 0.5020, 0.4760) | v165_logs/v165.json seeds[].epochs[-1].different, seeds[].acc_best |

Carried over from v52, still true: the handover's cross-talk is one seed (seed 0, 200 test items) and its accuracy
three seeds, so it has no seed dots; the continuous handover is the audit-2 run (seeds 0, 1, 2; 1,500 train / 600 test;
projection fitted for 3 epochs); the projection is V-165 (seeds 42, 123, 456; train arena 800 / test arena 500;
cross-talk after the final epoch, accuracy at the best test epoch, equal to the banked arm of panel a, which the script
asserts). The panel mixes these four harnesses; the caption should say that each arm is plotted at its own measurement.
The open diamond is drawn over the curve, the other markers under it; no curve point is hidden.

### c, bindings per level in arena C

- x: "bindings per level", 2, 4, 8, 16, 24, 32 (logarithmic). y: "chain accuracy, arena C", 0.4 to 1.02. Three seeds per point
  (faint dots). The black line is drawn thick under the vermilion one: from 8 bindings on the two are identical in
  every seed.

| Legend name | Colour, marker | Mean at 2, 4, 8, 16, 24, 32 | Per seed | Source |
|---|---|---|---|---|
| exact codebook rows | black `#000000`, circles, thick | 1.0000, 0.9980, 0.9967, 0.9720, 0.9507, 0.9340 | 1.0000, 1.0000, 1.0000; 0.9980, 0.9980, 0.9980; 0.9940, 0.9960, 1.0000; 0.9700, 0.9720, 0.9740; 0.9500, 0.9540, 0.9480; 0.9320, 0.9320, 0.9380 | `v14_logs/v14_fixed.json [].parser` |
| handover | vermilion `#D55E00`, squares | 0.9887, 0.9947, 0.9967, 0.9720, 0.9507, 0.9340 | 0.9800, 0.9880, 0.9980; 0.9960, 0.9940, 0.9940; 0.9940, 0.9960, 1.0000; 0.9700, 0.9720, 0.9740; 0.9500, 0.9540, 0.9480; 0.9320, 0.9320, 0.9380 | `v14_logs/v14_fixed.json [].probe_argmax` |
| frozen-state probe | grey `#7F7F7F`, triangles | 0.7953, 0.5707, 0.5333, 0.5113, 0.5340, 0.5153 | 0.8060, 0.8080, 0.7720; 0.5760, 0.5760, 0.5600; 0.5240, 0.5420, 0.5340; 0.5100, 0.4940, 0.5300; 0.5340, 0.5460, 0.5220; 0.5120, 0.5300, 0.5040 | `v14_logs/v14_fixed.json [].floor` |

n per point is not recorded in the artifact.

---

## Supplementary Fig. 1, `figs/figS1_sweeps.pdf`

### a, chain depth

- x: "chain depth (each arm trained at that depth)", 1, 2, 3. y: "chain accuracy", 0.4 to 1.02. Three seeds per point.
  Exact codebook rows and the handover are identical in every seed.

| Legend name | Colour, marker | Mean at depth 1, 2, 3 | Per seed | Source |
|---|---|---|---|---|
| exact codebook rows | black `#000000`, circles, thick | 0.9580, 0.9340, 0.8800 | 0.9540, 0.9600, 0.9600; 0.9320, 0.9320, 0.9380; 0.8840, 0.8800, 0.8760 | `v57_logs/v57.json cells[depth].parser` |
| handover | vermilion `#D55E00`, squares | 0.9580, 0.9340, 0.8800 | 0.9540, 0.9600, 0.9600; 0.9320, 0.9320, 0.9380; 0.8840, 0.8800, 0.8760 | `v57_logs/v57.json cells[depth].probe_argmax` |
| LoRA fine-tune | orange `#E69F00`, inverted triangles | 0.5073, 0.5007, 0.5153 | 0.5200, 0.5220, 0.4800; 0.4880, 0.5400, 0.4740; 0.5200, 0.5200, 0.5060 | `v57_logs/v57.json cells[depth].lora_direct` |
| frozen-state probe | grey `#7F7F7F`, triangles | 0.5407, 0.5153, 0.5300 | 0.5400, 0.5280, 0.5540; 0.5120, 0.5300, 0.5040; 0.5280, 0.5180, 0.5440 | `v57_logs/v57.json cells[depth].floor` |

The LoRA here is rank 8, final epoch, 800 train / 500 test (`v57.py`, per the v52 notes), not the rank-16 run of Fig. 5a.

### b, one fitted projection, read continuously or snapped (V-21)

- x: "epochs fitting the projection", five categories 0, 1, 3, 10, 30, equally spaced. y: "chain accuracy", 0.4 to 1.06. Three
  seeds per point. The run is the 256-symbol variant of the two-hop arena (`v21.json`: `hops` 2, `n_ctx` 32,
  `n_sym` 256).

| Legend name / label | Colour, marker | Values at 0, 1, 3, 10, 30 epochs: mean | Per seed | Source |
|---|---|---|---|---|
| read continuously | sky blue `#56B4E9`, circles, thick | 0.4807, 0.6580, 0.7533, 0.8373, 0.8733 | 0.4960, 0.4580, 0.4880; 0.6660, 0.6500, 0.6580; 0.7480, 0.7660, 0.7460; 0.8280, 0.8520, 0.8320; 0.8720, 0.8680, 0.8800 | `v21_logs/v21.json cells[].frozen_cont (hops 2, n_ctx 32, n_sym 256)` |
| snapped to the nearest codebook row | vermilion `#D55E00`, squares | 0.4773, 0.9773, 0.9800, 0.9840, 0.9873 | 0.4880, 0.4560, 0.4880; 0.9760, 0.9740, 0.9820; 0.9800, 0.9800, 0.9800; 0.9860, 0.9800, 0.9860; 0.9880, 0.9840, 0.9900 | `v21_logs/v21.json cells[].proj_argmax (hops 2, n_ctx 32, n_sym 256)` |
| exact codebook rows, dashed horizontal line with a text label | black `#000000` | 0.9920 | 0.9920, 0.9900, 0.9940 | `v21_logs/v21.json parser` |
| frozen-state probe, dashed horizontal line with a text label | grey `#7F7F7F` | 0.5173 | 0.4860, 0.5100, 0.5560 | `v21_logs/v21.json floor` |

The vermilion series is the same fitted projection snapped to its nearest codebook row (`proj_argmax`), not the
handover of Fig. 5a (which uses the identity head); the legend says "snapped to the nearest codebook row" and "read
continuously". The continuous read has the sky blue of the continuous handover of Fig. 5b. n per point is not recorded.

### c, facts held at the 0.70 criterion against dimension (V-151)

- x: "dimension d", 512, 1024, 2048, 4096 (log scale, base 2). y: "facts held at 0.70 readout (interpolated)" (log
  scale, base 2, 24 to 560). k* is the 0.70 crossing of each capacity curve, by linear interpolation in log k between
  the two grid points that bracket it.
- HRR is the main series (vermilion `#D55E00`, circles, 1.9 pt line). MAP (`#5A5A5A`, squares), BSC (`#8A8A8A`,
  triangles) and FHRR (`#B0B0B0`, diamonds) are thin grey lines. Dashed black line: least-squares power law fitted to
  the four HRR points, legend "power law fitted to HRR, exponent 0.890". No dots (k* is computed from the banked means).

| Model | k* at d = 512, 1024, 2048, 4096 | Exponent of k* ~ d^alpha | 95% bootstrap interval |
|---|---|---|---|
| HRR | 41.7, 73.3, 136.7, 264.7 | 0.890 | 0.815 to 0.953 |
| MAP | 43.5, 79.5, 139.1, 247.9 | 0.834 | 0.807 to 0.868 |
| BSC | 29.3, 54.3, 96.0, 177.1 | 0.861 | 0.821 to 0.891 |
| FHRR | 79.7, 139.7, 249.9, 456.9 | 0.840 | 0.810 to 0.870 |

These reproduce the ledger (HRR k* 41.7, 73.3, 136.7, 264.7; exponent 0.890; interval 0.815 to 0.953), which the
script asserts. The interval is the frozen scorer's: 20,000 bootstrap resamples of the four (d, k*) points, seed
20261008, models in the order HRR, MAP, BSC, FHRR on one random stream (`figdata/constants.json`
`capacity_criterion`). None of the four intervals contains 1.0: the growth is sub-linear. HRR and MAP nearly coincide.

---

## Not drawn: bAbI tasks 1 and 2 (Phi-2), for the Supplementary Table

The bAbI bars were removed from the main figures. `make_figures.py` still reads the values (`babi_table()`) and writes
them to `FIGURE_VALUES.json` under `not_drawn_bAbI`:

| Arm | Task 1 (one hop): mean (per seed) | Task 2 (two hops): mean (per seed) | Source |
|---|---|---|---|
| prompting | 0.7920 (0.7920, 0.7920, 0.7920) | 0.3860 (0.3860, 0.3860, 0.3860) | v49_logs/v49.json cells['task1/zeroshot/all'].acc and cells['task2/zeroshot/all'].acc (n = 500 each; one deterministic decoding stored three times) |
| LoRA fine-tune | 0.9993 (0.9980, 1.0000, 1.0000) | 0.9722 (0.9714, 0.9738, 0.9714) | task 1: v50_logs/v50_task1_VOID_locoffset.json arms.lora_direct; task 2: v131_logs/v131b.json run r16_e6_lr0.0001 accs (highest test mean of 12 configurations) |
| LoRA fine-tune, first configuration (not the best of twelve) | - | 0.8008 (0.8500, 0.6952, 0.8571) | v50_logs/v50_task2_BANKED.json arms.lora_direct |
| projection trained through the binding loss | - | 0.5024 (0.4952, 0.5048, 0.5071) | v37_logs/v37b.json arms.spans (task 2 only; task 1 not run) |
| handover | 1.0000 (1.0000, 1.0000, 1.0000) | 1.0000 (1.0000, 1.0000, 1.0000) | task 1: v35_logs/v35.json cells[family none].chain; task 2: v37_logs/v37b.json arms.probe_argmax |
| frozen-state probe | 0.7713 (0.7800, 0.7820, 0.7520) | 0.2452 (0.2524, 0.2357, 0.2476) | task 1: v30_logs/v30.json arms.floor; task 2: v48_logs/v48.json floor_last_token |

Prompting is one deterministic zero-shot decoding stored three times (n = 500 per task; task 2 is scored on all 500,
the other task-2 arms on the 420 expressible questions). LoRA on task 2 is the best of twelve configurations by test
mean (rank 16, 6 epochs, lr 1e-4; 775 train / 420 eval, seeds 42, 123, 456); the first configuration (0.80) is listed
only so that you can see what was dropped. The projection trained through the binding loss was not run on task 1.

---

## Decisions I made, and where the sources disagreed with the brief

**Numbers.** Nothing disagreed numerically. All operational-log numbers were found verbatim. V-106 matches the brief
(prompt 0.950, 0.908, 0.850, 0.583, 0.542; store 0.936, 0.933, 0.919, 0.919, 0.908; 254 and 11,612 tokens). V-151
reproduces the ledger. The one thing not where the brief said: "QA3 = 72.1 at one million" is not in `b18_verdict.txt`
(see Fig. 2b).

**Things to know before writing the captions.**

1. **ARMT per task (Fig. 2b)** comes from the ARMT paper's Table 1 for both lengths, fetched from arXiv, because the
   leaderboard's QA5 row for ARMT is a copy of its average row. Details under the panel. Please confirm against your
   copy of the paper if you have one.
2. **Per-site n (Fig. 3c)** is counted from the scored answers' ids, not read from the verdict text.
3. **Fig. 3a mixes three held-out log sets** and three read-outs of the same reader and memory; Qwen3-32B is the
   budget-forced score throughout.
4. **"800 (all)"** is used in Fig. 4a, 4b and 4c for the scale requested as 1,024. In 4a the store's legend says "at
   most 64 facts per store" because the last point has 50 per store.
5. **Fig. 1b is an illustration.** The seed (0) and the story are mine; I replaced the sixth location "bathroom" by
   "park" (also a bAbI location) so that the tick labels "bedroom" and "bathroom" do not touch.

**Colour decisions** (the brief fixes nine colours; these are the cases it does not cover).

6. **Continuous handover: sky blue `#56B4E9`**, as in v54 (you asked for the old panel b content). The brief gives sky
   blue to Qwen3-32B and other open models, so the two share a colour across figures (Fig. 5b and Supplementary Fig. 1b
   against Figs 2a and 3). They never meet in one figure. If you would rather keep sky blue for the open models only,
   the continuous handover is one entry of `figstyle.C` (`"cont"`).
7. **Fig. 2a systems:** ARMT black, Claude Opus 5.5 pink, GPT-4-0125 grey (all as in v54), GPT-6.1 Sol bluish green
   (was blue; now the same colour as in Fig. 3), Llama-3.1-70B sky blue (an open model; was light grey). Black, pink
   and grey therefore name other things here than in Figs 4 and 5.
8. **Fig. 2c direct arm: prompting blue**, not sky blue: it is the frozen reader itself answering without the memory.
9. **Fig. 2b:** 1M solid, 10M a pale fill with an edge in the system's colour (not hatching).
10. **Fig. 3b segments:** both right light grey `#D9D9D9`, only ours vermilion, only GPT-6.1 Sol bluish green, both
    wrong dark grey `#4D4D4D`.
11. **Fig. 4b and 4c ramps:** one hue from light to dark for the ordered series (d, K), built on the store's vermilion;
    the second-darkest step is the vermilion itself.
12. **System of record: grey-brown `#A08C78`** (the brief left the choice between grey-brown and light grey).

**Layout decisions.**

13. Fig. 5: b fills the right half; a and c are stacked on the left. Fig. 3: a and b on top, c below a, one legend for
    the systems of a and c at the lower right. Fig. 2: a across the top with its legend beside it, b and c below.
14. Legends sit under or beside the axes, never over data, except Fig. 5c (in the empty band between the curves).
15. GPT-4-0125 and Llama-3.1-70B are labelled at the ends of their curves in Fig. 2a, with their 128k values.
16. The frozen-state probe line of Fig. 5b is labelled "frozen-state probe + 0.10" (the brief says "floor + 0.10").
17. Supplementary Fig. 1c prints the exponent as 0.890 in the legend.

## Not done

- No `.tex` or `.bib` file was touched and nothing was committed. `main.tex` and `supplementary.tex` do not yet
  include these files.
- `figdata/INDEX.json` was not extended: the script reads the artifacts directly and uses the index only to assert
  that the values agree with the v54 figure set (`figdata.same`). The `_about` strings of
  `figdata/benchmarks/babilong.json` and `clutrr.json` still name the v54 figure ("main Fig. 4", `fig6_benchmarks`).
- ARMT's standard deviations are recorded, not drawn. No error bars or intervals are drawn for ours in Fig. 2b or for
  any bar of Fig. 3.

## Files

Added in `/path/to/paper`:
`make_figures.py`, `schematic.py`, `figdata.py`, `FIGURE_VALUES.json`, `FIGURE_NOTES.md` (this file),
`figs/fig1_overview.pdf`, `figs/fig2_benchmarks.pdf`, `figs/fig5_oplogs.pdf`, `figs/fig3_store.pdf`,
`figs/fig4_mechanism.pdf`, `figs/figS1_sweeps.pdf`,
`figdata/oplogs.json`, `figdata/constants.json`, `figdata/worked_example.json`,
`figdata/benchmarks/babilong_per_task.json`, `figdata/benchmarks/armt_table1_arxiv2407.04841.txt`,
`figdata/benchmarks/babilong_leaderboard_v0_results.csv`.

Changed: `figstyle.py` (rewritten for the new sizes and the fixed arm colours and names; the v54 file is the one in
`/path/to/paper_v54`).

Rendered pages (200 dpi, from the final PDFs), in
`/path/to/scratch`:
`fig1_overview.png`, `fig2_benchmarks.png`, `fig5_oplogs.png`, `fig3_store.png`, `fig4_mechanism.png`,
`figS1_sweeps.png`.

The two JSON files with transcribed values were built by scratch scripts that cut each value and its line out of the
source by regular expression (`scratchpad/build/build_oplogs.py`, `build_pertask.py`); this file was written by
`scratchpad/build/make_notes.py` from `FIGURE_VALUES.json`. Those three scripts are not in the repository.

