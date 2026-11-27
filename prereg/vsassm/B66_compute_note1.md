# B66 compute note 1 (during dev; no dev or test output of a NEW arm read except as listed)

Date: 2026-11-23, 13:41 (machine clock). Prereg committed 13:38 (012ac63), OpenTimestamps proof committed (d96e91d).

- Placement: local (16 cores): b66_cr.py dev phase, Qwen3-8B and Llama, 4 processes each x 2 threads, nice 10 (started
  13:39); LSfc dev plan (297 jobs, 3 workers, one thread each; started 13:39). Box (~/b66_run, code md5 as listed by
  `b66_box.sh setup`; panel codes md5-checked, all 9 ok): SDMTm dev (maps 1-3, 126 jobs each, 10 workers per map) and
  VSARm dev (maps 0-3 x 7 encoders x 6 parts, 16 parallel processes), started 13:41.
- A2 pipeline side (B64 code on the clean room's grid): the dev choices were computed from B64's existing dev rows
  (b66_run.py crgrid, all 7 encoders) and are committed with this note, before its test plan is made or run. Composition
  read (dev only): on Qwen3-8B the chosen a is 1 in 6, 4 in 5, 16 in 1 of 12 ideal R1 cells, and BEST7CR = SDMTCR in
  10 of 12; on Llama a = 1 in 7, 4 in 5, BEST7CR = SDMTCR in 12 of 12.
- b66_verdict.py (A verdict) written during dev, not yet run on anything.
