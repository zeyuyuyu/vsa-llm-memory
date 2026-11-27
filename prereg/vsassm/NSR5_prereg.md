# NSR5 pre-registration: does the frontier model hold up as the record grows?

Date: 2026-11-11. Written after NSR1's frontier addendum (gpt-6.1-sol xhigh reading one day's log ties the pipeline:
0.917 vs 0.930) and before any frontier call on multi-day sites.

## Design

Multi-day sites (nsr/nsr5_sites.py): K logs of one scenario run back to back (day k continues from day k-1's state;
items placed earlier and never touched again stay put; home / ops / biobank re-place the same things every day, so later
days overwrite earlier facts). Sources: retired test, nsr1, nsr2 (30 logs per scenario; test2 never used). Gold = replay of
the concatenated true events (checked: the 1-day merge reproduces the simulator truth on 120/120 logs). Pipeline = replay of
the concatenated reader events (each day read by the shipped reader, reset at day boundaries). Questions: NSR1's impact
questions over the merged site. K = 8: 12 sites, 69 questions, prompts ~41k tokens; K = 4: 28 sites, 161 questions,
~20k tokens; K = 1 reference: nsr1 (230 questions, NSR1 addendum). K = 2 (345 questions) only if the wallet allows.
Frontier arm: gpt-6.1-sol, reasoning_effort xhigh, streamed, NSR1 prompt format (asset list of the merged site, catalog,
all K days with "=== Day k ===" separators). Registered parser; the underscore-normalising parse as a sensitivity analysis.

## Measures and bars

Set-exact accuracy and micro-F1 over answer items (set-exact falls with answer size even at constant per-item accuracy:
pipeline set-exact 0.901 / 0.861 / 0.801 / 0.710 at K = 1/2/4/8 while its K = 8 micro-F1 is 0.942).
- **Z1:** pipeline - frontier set-exact at K = 8 >= 0.10.
- **Z2:** (pipeline - frontier) at K = 8 minus (pipeline - frontier) at K = 1 (nsr1: +0.013) >= 0.10.
- **Z3:** pipeline - frontier micro-F1 at K = 8 >= 0.03.
Reported: K = 4 on every measure; per scenario; unparseable answers; spend.

## Predictions (low confidence)

The frontier model degrades with length and overwrites: set-exact about 0.45 at K = 8 (about 0.6 at K = 4), micro-F1 about
0.85 at K = 8, so Z1, Z2 and Z3 hold. If the frontier model stays near the pipeline (Z1 fails), the pipeline's case at
these scales rests on cost, latency, on-device operation and traceability alone.
