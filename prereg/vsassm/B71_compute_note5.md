# B71 compute note 5 (dev choices committed; no test run yet, no test output read)

Date: 2026-11-23, 17:59 (machine clock).

1. **Dev finished on the box** at 17:53 (VSAPR 600, NLMSF 315 + 18 supplementary, VSAD kind P 144, VSAD kind R 171 jobs; 0
   failed jobs in any log). The choices were computed ON THE BOX from the job files (b71_run.py choose, the registered
   rules), after the inputs results/b53_choice_*, b56_choice_*, b60_choicepc_* (md5-identical to the repository's) were
   copied there, and downloaded: results/b71_choice_{enc}.json (md5 qwen8 1d7bfccd, b46 9b02db34, llama feb992da, mistral
   ceca8ab4, e5 e19404a3, bge 0b76683b, minilm 70f7c917). The summary path of compute note 4 (b71_devsum.py) was not
   needed. The audit (b71_verdict.py audit, dev rows only) ran there too: results/b71_audit_{enc}.json.
2. **Checks on the dev rows (before any test run).** Bits audit of all 296,862 dev rows: 0 over budget, 0 not maximal, 0
   formula mismatches (results/b71_check_bits.json). Reproduction (results/b71_check_repro.json): the audit re-chose B56's
   retraining configuration exactly in 84 of 84 R1 cells; the dev accuracies agree to 0 in 81 of 84 cells; in 3 cells, all
   on sentence encoders, they differ: the Hebbian bundle e5 (256, 2, 8) -0.007 points, MiniLM (256, 2, 16) -0.082, and bge
   (1,024, 2, 4) +0.026 for both the Hebbian bundle and the retraining. The registered bar (<= 1e-12) therefore FAILS in 3
   of 84 cells. These dev jobs ran on the box, whereas B53 / B56 ran on this machine (compute note 4 named this risk); the
   15 dev jobs of those three cells are being re-run here (B71_OUT in the scratchpad, same code and plan entries) to test
   whether the difference is the machine. Their result is reported as a check; no choice depends on it.
3. **Test plan.** b71_run.py plan test made on both machines from the committed choices: md5 6878b608... on both (1,767
   jobs). The test runs once on the box, 24 workers, after this commit.
4. **D2.** The SDMt capped dev runner (stopped in note 4) did not restart at 17:53 (the ssh call timed out before the
   start); 80 of 297 dev jobs exist. It is restarted with 24 workers beside the test. D2's choices are committed separately
   before D2's test jobs run. D2 LAMc's re-choice (results/b71_d2_lam_choice.json, committed with this note) uses LAMc's
   existing dev rows only.
