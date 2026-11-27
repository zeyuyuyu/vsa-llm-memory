# V-153 pre-registration — replication of the load-bearing claims on fresh data and fresh seeds, plus two asymmetries the code audit found

Written and committed BEFORE any V-153 run. Motivated by the advisor's request that every conclusion be
checked by repeated experiment before submission, and by three things found while reading the producing code
this session (none of which any of the 32 checkers can see):

1. **Arena data never varied across "seeds".** `v13.run` and `v15.get_rows` build arena C as
   `Arena(tok, n_tr, 42), Arena(tok, n_te, 4242)` regardless of the seed argument. The three reported seeds
   varied model initialisation only; train and test data were identical in all three. Every arena-C number
   in the manuscript is therefore one draw of data.
2. **The handover is given the query and the candidate as gold ids.** In `v13.BindArm.sims`, the context
   bindings for `probe_argmax` are read from frozen hidden states through the identity head
   (`self.sym[int(self.id_head(v).argmax())]`, which never reads the gold id), but the query role and the
   candidate answer use `q_vec(sid)`, i.e. the generator's gold ids. The rows carry `q_span` and `cand_span`
   for exactly this purpose and the arm does not use them. A LoRA reading raw text gets no such help.
   `probe_argmax` equals `parser` to the digit, per seed, in three experiments (arena C .932/.932/.938;
   task 4 .674/.702/.696; task 2 1.000 x3) — consistent with a perfect identity head, but the head's
   accuracy was never logged, so that is an inference, not a measurement.
3. **Best-epoch-on-test.** `v13.run` and `v15.run` return `best = max(best, test_acc)` over epochs, i.e. the
   reported score is selected on test, for every arm on both sides.

## Runs (producing code verbatim except where stated; seeds 1001 / 2002 / 3003, none previously used)

- **R1 arena C mechanism**: `v153_v13.py` = `v13.py` + default-off `--fresh_data` (train data seed
  100000+seed, test 200000+seed, so each replicate is an independent draw of data AND initialisation), + a new
  arm `probe_full` identical to `probe_argmax` except the query role and candidate are read through the
  identity head from `q_span` / `cand_span`, + logging of identity-head top-1 accuracy on test spans
  (context / query / candidate) and of final-epoch accuracy beside best-epoch. Arms: floor, parser,
  consistent, spans, probe_argmax, probe_full, frozen_cont. Budget unchanged.
- **R2 arena C LoRA**: `v153_v15.py` = `v15.py` + `--fresh_data` (same derived seeds as R1) + `--log_train`
  (training-set accuracy after every epoch). Arena C only, registered 2-epoch budget.
- **R3 bAbI task 2, two hops**: `v37b.py` verbatim, `--seeds 1001 2002 3003`. bAbI is a fixed dataset, so
  seeds vary initialisation only, which is the correct replication for a fixed benchmark.
- **R4 bAbI task 4, the out-of-sample prediction**: `v58.py` verbatim, `--seeds 1001 2002 3003`.

## Banked values (read from the artifacts this session, not from the manuscript)

    arena C  v13_logs/v13_fixed.json   floor .5153  parser .9340  consistent .9293  spans .4833
                                       probe_argmax .9340  frozen_cont .9240
    arena C  v15_logs/v15.json (C)     lora .5158  (.515 / .5175 / .515)
    task 2   v37_logs/v37b.json        floor .2032  parser 1.0000  probe_argmax 1.0000  spans .5024
    task 4   v58_logs/v58.json         floor .6147  parser .6907  probe_argmax .6907  lora_direct 1.0000

## Bars

**Q --- replication.** For every banked (run, arm), the fresh-seed mean of the SAME quantity (best-epoch,
as banked) lies within **0.04** of the banked mean. Reported per arm; the bar is that every LOAD-BEARING arm
passes: arena C floor, parser, spans, probe_argmax, lora; task 2 probe_argmax; task 4 probe_argmax,
lora_direct.

**C --- the manuscript's claims, on fresh data, per seed (3/3 required):**
- C1 interface: arena C `spans <= floor + 0.05`.
- C2 handover works: arena C `probe_argmax >= parser - 0.05`.
- C3 in-context binding advantage: arena C `probe_argmax - lora >= 0.30`.
- C4 two hops: task 2 `probe_argmax >= 0.98`.
- C5 exactness condition: task 4 `lora_direct - probe_argmax >= 0.20`.

**N --- the two asymmetries:**
- N1 is the gold-query asymmetry harmless? arena C `|mean(probe_full) - mean(probe_argmax)| <= 0.02` AND
  `probe_full >= parser - 0.05` in 3/3.
- N2 does LoRA learn arena C at all? LoRA's final-epoch **training** accuracy `<= 0.60` in 3/3 reads as "does
  not learn the task"; `>= 0.90` in 3/3 reads as "fits training, fails test". Anything between is reported
  as partial.
- N3 descriptive: identity-head top-1 on test context / query / candidate spans; best-minus-final epoch gap
  for every arm on both sides.

## What I expect (recorded separately from the bars)

- Q and C1--C5 hold. These are the claims; fresh data is the test they have not had.
- N1 holds, but genuinely uncertain: the head is trained only on context spans, and the query appears in a
  different textual frame (`Q [R..] {..} ?`), so reading it is a real generalisation test for the head.
- N2: I expect training accuracy at or near chance (<= 0.60), because on this family LoRA's recorded training
  loss stayed at 0.72--0.78 against chance 0.693. If so, the arena-C result is "a low-rank update does not
  learn in-context binding", not "it does not generalise to unseen combinations" (arena C has none; audit
  95ef459).
- Best-vs-final gaps small (two epochs).

## Scope and protocol

Arena C and bAbI tasks 2 and 4 only. Arena B, task 15, 2WikiMultihopQA and MuSiQue are not in this
registration; their code is under audit and they get their own replication once the audit reports. Scorer
`v153_score.py` frozen and committed with this file, run verbatim. PYTHONHASHSEED=0.
