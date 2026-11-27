# B40 amendment 2 (dev only; before any test run): a fixed-depth-3 version of the label-free sub-experiment

Date: 2026-11-19, about 04:40 box time. No test split has been scored by any arm.

The registered label-free sub-experiment (S6) trains E2E-VSA, E2E-TAB and E2E-CONST on depth-2 questions with two
fixed hops. On the depth-2 dev rows the floor arm (one learned vector mixed in, no module information) already reaches
0.995 (E2E-VSA 0.985, E2E-TAB 0.995; dev record), so S6's bar ("learns" = floor + 0.10) cannot be met by construction:
the LLM answers depth-2 questions by itself. The depth-2 version is kept and reported as registered (expected reading:
NO HEADROOM). Added, so that the label-free question can be read at all: the same three arms with THREE fixed hops on
depth-3 questions, where the frozen LLM's first-token dev accuracy is far lower (the frozen LLM's pooled first-token
dev accuracy over depths 1-3 is 0.687 while depth 1 is ~1.0 and depth 2 ~0.9 on training rows, so depth 3 is the
weak cell).

- Data: train_e2e3 = 12,000 depth-3 small rows (seed 1103; sha256 of rows f261a1bfc4de04e1); dev = the 200 depth-3
  dev rows; test = the depth-3 test cells (small and large, 400 rows) of the existing test split.
- Arms: identical to S6's (same code with hops = 3: E2E-VSA three soft unbind-and-clean-up steps; E2E-TAB three
  table hops; E2E-CONST), same dev grid (lr 1e-3, 3e-3; 3,000 steps; dev every 500 steps; best dev step), seeds 0-2,
  the shuffled-pairing check.
- Readings: S6's bars on the depth-3 test cells. Prediction (P14 extended): E2E-VSA does not learn; E2E-TAB learns.

Code change: E2EVSA / E2ETab take a hop count (default 2, behaviour unchanged for 2); b40_run --hops.
