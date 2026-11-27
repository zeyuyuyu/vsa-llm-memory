# B47 note after the test (2026-11-20 ~03:10 UTC; written after every registered output was read)

1. **Exploratory 2 decides the answer-token arms.** With the label "the last number before 'Answer:' is even" the full
   fine-tune (fullgen) reaches dev 1.000 (training subset 1.000), so its answer-token pipeline learns and its
   chance-level task results (test 0.520 at 800, 0.487 at 8,000) are a property of the task. The LoRA answer-token arm
   (gen, rank 8, lr 1e-4) stays at 0.558 (majority 0.544), below the registered 0.70, so **its task results are
   withdrawn** (exploratory 2's registered reading): a LoRA run through the bf16 LM head did not learn even a property
   of the adjacent token. The likely cause, not tested: its yes/no logits are computed in bf16 at magnitudes where one
   bf16 step is 0.0625-0.125 (the saved margins are 0.000 or 0.125 on most inputs), so small input-dependent changes are
   invisible to the loss; the full fine-tune can move the weights far enough to cross that resolution.
2. **Registered F2 against the arms that remain interpretable.** At 800 the registered B+ was fullgen anyway (0.520 >
   gen 0.476): margin +0.480, H-nest [+0.451, +0.511] -> STANDS. At 8,000 the registered B+ was gen (0.501); with gen
   withdrawn, against fullgen (0.487): +0.513, H-nest [+0.460, +0.567], H-cross [+0.454, +0.573], t-seed [+0.386,
   +0.640], per seed +0.566 / +0.464 / +0.510 -> STANDS. The reading does not change.
3. **Seeds 42 vs 123 / 456 recipes.** Seed-42 states were trained with micro-batch 8 on PACE H200s; seeds 123 of gen 800,
   gen 8,000 and scratch 800 ran on the box with micro-batch 4 (batch 8 in both); test scoring of seed 42 ran on an A100.
   Dev rechecks of the near-chance answer-token models differ from the logged dev by up to 0.056 because their margins
   are bf16 ties (median |margin| 0.000); the scratchpad and classifier rechecks agree within 0.002.
4. **A800 claim time.** Worker m2 held GPU 0 from 17:58 to 02:40 UTC but computed only until 19:28 UTC (its queue had been
   emptied when three runs moved back to PACE, and the queue was closed only after the session resumed): 7.2 h of idle
   claim, counted in the 23.6 GPU-hour total.
