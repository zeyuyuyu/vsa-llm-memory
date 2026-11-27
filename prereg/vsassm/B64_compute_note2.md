# B64 compute note 2 (dev done, choices made; before any test run)

Written 2026-11-23 about 03:29 (machine clock), before the test plan is started.

1. **Dev.** b64_out/plan_dev.jsonl, 126 of 126 jobs, 03:15:06-03:27:11, local, 8 workers; no failure or traceback.
2. **Choices.** b64_run.py choose for all 7 encoders (B36's rule; 504 configurations per cell): results/b64_choice_{enc}.json.
   The assertion that B49's six dev accuracies reproduce B36's BEST of six held in all 84 x 2 cells. Test plan
   b64_out/plan_test.jsonl (350 jobs = 7 encoders x V 2 / 16 x 25 test stores, the chosen configurations at 4 / 8 / 16
   bits per fact, both accountings), md5 d0f39a63...
3. **Dev-only observations (dev rows; no test output):**
   - BEST7's dev family over the 12 R1 cells (ideal): SDMT in 11 / 12 / 12 / 12 (Qwen3-8B / Granite / Llama /
     Mistral; the other cell CM), 10 / 11 / 10 (e5 / bge / MiniLM; others CM or IDX).
   - Dev SDMT minus B36's SDM (ideal, R1 mean): +12.9 / +16.4 / +15.6 / +13.4 (language models), +10.2 / +10.2 / +11.0
     (sentence encoders). Dev SDMT minus the best of six (each cell's dev-best of the six), V = 2 / V = 16: Qwen3-8B
     +8.8 / +8.6, Granite +9.3 / +13.1, Llama +9.1 / +13.0, Mistral +6.4 / +6.0, e5 +4.5 / +2.5, bge +2.9 / +2.7,
     MiniLM +4.0 / +0.7 (dev selections are optimistic for SDMt: 504 configurations).
   - Composition of SDMt's 84 ideal R1 choices: prep whiten 73 / center 11; location set L (labels) 49, LA 27, A 8;
     a = 1: 24, 2: 27, 4: 15, 8: 11, 16: 4, 32: 2, 64: 1; counters at V = 16 onehot 42 of 42, at V = 2 bits 34 of 42;
     q tern 45, cnt 37, clip3 2; share 1: 72, 1/2: 12. **Prediction P8's "location set LA in at least half" already
     fails on the dev choices (27 of 84); its other two parts hold (a <= 16 in 81 of 84; onehot in 42 of 42 at V = 16).**
     P7 holds on the dev choices.
4. **Scheduling change (no effect on results).** b64_run.runplan gained --byenc (jobs sorted by encoder, handed out in
   chunks of 5 so a worker keeps one encoder's setup); every job is computed from registered seeds and its output does
   not depend on the schedule. Used for the test plan only.
5. No cut so far. The test is run once, locally, 8 workers; then compute note 3 before any test output is read.
