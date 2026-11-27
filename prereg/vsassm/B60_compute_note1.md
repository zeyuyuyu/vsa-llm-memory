# B60 compute note 1 (during dev; no test store built, no test output read)

Written 2026-11-21 about 08:15 (machine clock), before the dev choices are committed and before any test run.

1. **Pre-registration time.** The pre-registration's header says "about 03:45"; it was committed at 03:23:04 and
   stamped at 03:23:06 (commits d54201d, e62d654). The header time was an estimate written before the commit; the
   commit times are the record.
2. **Dev runs (session 1).** Box (~/b60_run, code md5 = the committed b60_run.py; codes md5-checked against B49's):
   plan_dev (252 jobs: 7 encoders x V 2 / 16 x 9 dev stores x 2 preps, every configuration) with 32 workers and
   plan_mapdev (756 jobs: maps 1-3, rp configurations) with 16 workers, both from 03:24:34. When plan_dev finished
   (03:35:28) the map runner was stopped (48 of 756 jobs finished; jobs in flight discarded) and restarted with 40
   workers, and plan_devR2 (126 jobs: V = 1,024) started with 8 workers (48 in all). Every job is computed
   independently from registered seeds, so its output does not depend on how the plan is split into runs. Locally:
   VSAPc's whiten matched-filter grid parts (14 jobs, 2 at a time; the local machine carried another session's load,
   load average 50-90) and the first 3 of 28 VSAPc read-rule dev jobs.
3. **Interruption.** The working session ended at about 03:40 and was resumed at 08:06. Meanwhile all box dev jobs
   finished (252 + 756 + 126; no failure or traceback in any log); the local machine rebooted, which stopped the VSAPc
   read-rule dev queue (3 of 28 finished; the queue was restarted at 08:07 and skips finished files) and erased the
   session scratchpad, including the diagnosis's step5 code and its saved predictions (lsdiag/step5_*.pkl). For the
   registered step5 agreement check, the step5 code (step5_main.py, flex.py, pl.py and the clean-room quantizer and
   training statistics it imports) is reconstructed verbatim from the session transcript, committed as b60_step5.py,
   and re-run on B49's dev and test stores (it computes its own predictions; it reads none of B60's). This is a
   re-run of the diagnosis's implementation, not of its original output files.
4. **No cut of the registered design.** All registered dev runs exist or are running (VSAPc read rules). Plan for the
   test (time budget restarted at 3 hours from 08:06): commit every dev choice (LAMc, BESTc, LAMc per map, R2, VSAPc),
   then run the test once: LAMc's chosen configurations on the test stores for the registered map, the three maps and
   R2 (box), VSAPc's chosen reads (local). The verdict is computed only after every test run has finished. If time runs
   short, the registered priority holds and any cut is written into a further compute note before the verdict is run.
5. Dev-only observation recorded before the test (dev rows, no test output): LAMc's dev accuracy minus LAM's (B49's
   dev choice) over the R1 cells, V = 2 / V = 16: Qwen3-8B +0.63 / +7.03, Granite +0.90 / +8.16, Llama +0.44 / +7.67,
   Mistral +0.46 / +7.82, e5 +0.81 / +5.19, bge -0.03 / +5.11, MiniLM +0.50 / +3.22 points.
