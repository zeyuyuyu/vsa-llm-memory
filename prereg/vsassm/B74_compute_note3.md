# B74 compute note 3 (2026-11-23, about 23:43 machine clock; dev choices, before any test run)

1. **Development runs** (after compute note 2): FX dev on the box 23:35-23:40 (84 jobs, 48 workers, 0 failures) for the
   four language models; locally 23:35-23:37 (68 jobs, 6 workers, 0 failures) for the three sentence encoders. Dev job
   files downloaded as one tarball (md5 ccdd30f3... equal on both ends).
2. **Dev choices** (b74_run.py choose, B32's rule for H's share, B60's rule for L's (f, alpha)):
   results/b74_choice_{enc}.json (48 entries per encoder). Composition (cells of 12 with share 1 for H; for L cells with
   f = 1 / with alpha = 1e4, the Hebbian limit): qwen8 HP 9, HT 11, LP 9 / 1, LT 12 / 0; b46 12, 12, 12 / 0, 12 / 0;
   llama 12, 11, 12 / 3, 12 / 1; mistral 11, 12, 12 / 0, 12 / 1; e5 12, 11, 12 / 3, 12 / 0; bge 8, 12, 9 / 3, 12 / 1;
   minilm 12, 12, 9 / 0, 12 / 0. No dev accuracy beyond these choices was summarised.
3. **(C) dev choices** committed abacb7e (23:35): per cell and draw the B62-rule choice within the draw's 72
   configurations; the draw's choice differs from the full-grid choice in 192-208 of the 240 (cell, draw) pairs per
   encoder (112-131 distinct new (cell, configuration) pairs), so 335 (language models) and 290 (sentence encoders)
   test jobs are needed.
4. **Test plans** (generated from the committed choices): language models b74_out/plan_test_llm.jsonl = FX 126 + label
   932 + NL72 335 = 1,393 jobs (box, 48 workers); sentence encoders plan_test_sent.jsonl = 102 + 649 + 290 = 1,041 jobs
   (local, 6 workers). Each test job runs once.
