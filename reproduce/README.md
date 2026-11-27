# Reproduce the headline numbers

One command, from this folder (Python 3 with numpy; nothing else, no network, no GPU, a few seconds):

    sh run_all.sh

It prints the numbers below; `expected_output.txt` holds the full output as archived.

| Script | Manuscript claim | Inputs (in `inputs/`) |
|---|---|---|
| `equal_bit.py` | Corrected least squares ahead of the store on every encoder: final design (STORE) −4.06 to −6.53 over the region, strongest variant (VSAPCX) −2.01 to −5.31; online least squares (NLMSC) ahead of the final design −1.51 to −4.79; SDM at training codes ahead of the first design (VSAR) −0.92 to −4.73; final design against the best of seven +1.25 to −3.73; the superseded lead over the best of six +6.77 to +7.10 on the language models | `b60_part_*.json`, `b62_part_*.json`, `b64_part_*.json` (per-condition test accuracies) |
| `device_cells.py` | Cells per stored pair: sparse-block store 32 / 32 / 45 / 54 / 91 / 304 / 1,448; insertable LDPC table 32 / 32 / 38 / 64 / 91 / 304 / 1,218 | `b61_test1.json.gz`, `b61_choice.json`, `b63_test1.json.gz`, `b63_choice.json` (per-store test outputs, development choices) |
| `forward_pass.py` | Arithmetic: published read −3.23 (−4.60 to −1.87 over items; −4.80 to −1.67 with seeds also resampled), best match +3.03 (2.10 to 3.98; 1.80 to 4.27 with seeds) against the discrete module; chains (task R) +0.65 (0.27 to 1.08 with seeds); Welch t intervals over the three seed accuracies as a check | `forward_pass/{vsa,vsaargmax,disc}_s{0,1,2}.jsonl`, `forward_pass/R_{vsa,disc}_s{0,1,2}.jsonl` (per-item test predictions) |

Point estimates are recomputed from the archived outputs with plain numpy; the equal-bit intervals (cluster bootstrap over
entities) need the per-query outputs and are produced by each test's verdict code in `code/`. The dense store's and
the static tables' cell requirements (76 … 1,722 and 19 … 724) come from B38/B59's verdict code, which combines several
search tiers. These scripts check that the reported numbers follow from the archived outputs; they do not re-run the
experiments.
