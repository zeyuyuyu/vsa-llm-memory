# B31 note after the test run (2026-11-18, written by the parent session; the B31 agent ended in a session restart)

- The local PACE bridge (b31_pace_bridge.sh) stopped when the session restarted. By then every final had been
  delivered: the orchestrator logged "all finals present" at 13:46:40 UTC. md5 checksums show the 12 PACE
  lora_final outputs and the one lora_tune output are byte-identical on the box.
- The verdict (b31_verdict.py, as registered) was run on the box at 13:58 UTC. Its ledger counts only the A800
  (34.82 GPU-h). PACE ran 13 B31 jobs (sacct: 405.7 job-minutes, about 6.8 GPU-h, free tier), for a total of about
  41.6 GPU-h, below the cap of 60 (raised to 90).
- The second implementation (b31_indep.py) agrees per instance in every cell. Every accuracy recomputed from the
  saved outputs equals the logged value.
