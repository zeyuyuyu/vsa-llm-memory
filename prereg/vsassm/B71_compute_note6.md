# B71 compute note 6 (test run once; written before any test output is read)

Date: 2026-11-23, 18:13 (machine clock).

1. **Test.** b71_out/plan_test.jsonl (1,767 jobs, md5 6878b608..., made on both machines from the committed choices of
   8e332d5) ran once on the box, 24 workers, 18:00:34 to about 18:11; 0 failed jobs; every job written once. The test job
   files are copied back as one tar (md5 f86792f9..., 3.5 MB) and unpacked into b71_out/jobs; nothing in them has been
   opened yet.
2. **D2 test.** SDMt with capped locations: b71_out/d2sdm/plan_test.jsonl (1,050 jobs, md5 88591442...) on the box, 24
   workers, from 18:06 (choices committed in e410f08 before it); LAMc's re-chosen configurations: b71_out/d2lam/
   plan_test.jsonl (165 jobs, md5 4c4a6808...) locally, 4 workers, from 18:06. Box worker total during the tests: 48.
3. **No cut.** Every registered part ran: all 7 encoders, (D)'s table and D1 / D2, VSAD's several passes, kind P.
4. **Reproduction check, dev only (compute note 5, item 2).** The local re-run of the 15 dev jobs of the three cells that
   differed gives exactly the box's numbers (e5 (256, 2, 8) Hebbian -0.0066 points; bge (1,024, 2, 4) Hebbian and
   retraining +0.0255; MiniLM (256, 2, 16) Hebbian -0.0815), so the difference is not the machine. All three cells use sign
   features at q = 2, where B56 found that its own re-computation of the store differed from B53's only at exact ties on
   the quantizer lattice (568 queries): the scores are sums of equal-magnitude terms, so exact ties occur, and B53's store
   (per-block float64 accumulation of float32 block products) and b56_run.mf_read (one float32 product) break them
   differently. The registered bar (<= 1e-12 in every cell) fails in 3 of 84 cells by at most 0.08 points (about one dev
   query); B56's retraining choice is re-made in 84 of 84 cells.
5. Analysis code added after the test is run but before its output is read: b71_verdict.audit_panel (the registered dev
   audit's summary table). No arm, choice, bar or rule changed.
