# B28 smoke record (no amendment)

Date: 2026-11-17, about 19:40 UTC. Written after the smoke runs and while the stage-1 gates (encoding comm only)
and the encoding-independent fine-tune reruns were running, and before any non-comm encoding was run on a test
split. Nothing in the code, bars, predictions or readings changes; this file only discloses what the smoke runs
showed, because they score the encodings on training and dev rows.

## What ran (box /path/to/work/b28, logs/smoke.*; GPU 2, 19:27-19:3x UTC)

- b28_babi.py --smoke (seed 42, first 100 TRAINING rows; naming head trained on the full training split):
  - task 4: comm .620 (alias 60 rows: .367; non-alias 40: 1.000), perm 1.000, roles 1.000, rtag .730, shift .620;
    naming head identical to exact rows under every encoding.
  - task 15: comm .800 (alias 47: .574; non-alias 53: 1.000), perm 1.000, roles 1.000, rtag .780, shift .800.
  - task 2: every encoding .940 on the 100 parsed training rows (1.000 on the 94 expressible).
- b28_arenaC.py --split dev (B27 dev split v13.Arena(tok, 500, 777), seed 42): exact rows comm .906 (retrieval
  correct on 150 alias instances .460, on the other 350 1.000), perm 1.000.
- b28_lora2.py --smoke: 64 training rows, one epoch, scored on 32 training rows (.3125); code path check only.

## Bearing on the registered predictions (not changed)

- The structural alias flag marks 150 of 500 dev instances of arena C (0.30), above the registered range for the
  test split (P9: [0.10, 0.25]); the flag counts both hops, whereas V-163's 228 of 1,500 counted first-hop errors.
  P9 stays as registered and will be scored as registered.
- Code paths of all five encodings, both arms and all three tasks run without error; comm = shift per question in
  the smoke rows, as derived.
