# V-152 pre-registration — the fine-tune comparator selected honestly, on a validation split that existed all along

Written and committed BEFORE the V-152 sweep and before any V-152 number exists.

## The defect this run addresses

V-131 Part B swept twelve LoRA configurations at three seeds and reported the best, **selected on the test
set**. Both the runner and the manuscript disclose it, and the manuscript's wording is blunt: *"no honest
procedure would give a method its best configuration chosen on the data it is scored on."* The runner's
stated reason:

> Selection is on the test set, deliberately and disclosed: this harness has no validation split, inventing
> one would change the training set and break comparability, and selecting the OPPONENT's configuration on
> the set it is scored on is generous to the opponent.

**The premise is false, and that is why this run exists.** `v5.load_babi` takes `(task, n_train, n_val,
n_test)` and returns `{"train", "val", "test"}`; `v50.build_rows` calls it with `n_val = 200` and then
returns only train and test. **The validation split was loaded and discarded on every run.** Using it costs
nothing: the training set and the evaluation set stay byte-identical.

Measured before writing this file (counts only, no accuracy): task 2 gives **775 expressible train**
(V-131's exact number), **80 expressible val**, and the evaluation arm is untouched at 420. Passage overlap
is **0 between all three pairs**, verified rather than assumed.

## Design

Twelve configurations (rank 8/16/32 x epochs 2/6 x lr 1e-4/3e-4) at seeds 42/123/456 — V-131 Part B's grid
exactly. Runner is `v152_v50.py`, which is `v50.py` plus a **default-off** `--eval_val` flag; with the flag
absent the executing code is `v50.py`'s, and the diff removes three lines, each replaced by a superset of
itself. Training is untouched; the validation pass runs after it, on the same trained model, through the
same batching and the same answer-candidate restriction. `PYTHONHASHSEED=0`. Device `cuda:0`.

Each cell yields **both** a validation accuracy (80 rows) and a test accuracy (420 rows).

## Bars

**T0 --- validity gate.** The test accuracies reproduce V-131 Part B. At least 10 of 12 configurations have a
test mean within **0.02** of their banked V-131b mean, and the banked best configuration `r16_e6_lr1e-4` is
within **0.01** of its banked 0.9722. Training is supposed to be identical; if it is not, the `--eval_val`
patch changed something and nothing else is read.

**T1 --- does honest selection change the comparator?** With the configuration chosen as the argmax of the
**validation** mean over three seeds, that configuration's **test** mean is at least **0.9422**, i.e. within
three points of the test-selected 0.9722. Holds iff V-131's disclosed handicap cost the opponent nothing
material.

**T2 --- does the validation set pick the same configuration?** `argmax_val == argmax_test`, where
`argmax_test` is V-131b's `r16_e6_lr1e-4`.

**T3 --- the margin, stated both ways.** With the handover banked at 1.000, report the margin against the
val-selected comparator and against the test-selected one, and whether each clears the manuscript's
registered ten-point bar. Descriptive on the numbers, with one bar: the val-selected margin is **>= the
test-selected margin** (it must be, since test-selection maximises the opponent; a violation means the
selection code is wrong).

**T4 --- descriptive, no bar.** Whether 80 validation rows can discriminate at all: the standard error at
the observed validation maximum, how many configurations sit within one and within two of those of the
maximum, and the rank correlation between validation and test means across the twelve.

## What I expect (recorded before the sweep, separately from the bars)

- T0 passes; the training path is unchanged.
- **T1 holds.** On test the best configuration leads the second by about nine tenths of a point and the
  top four are all above 0.95, so even a mis-selection among the plausible ones costs little.
- **T2 is the one I am least sure of, and it is the point of the run.** Eighty rows put the standard error
  near a 0.97 accuracy at about 0.019, so configurations within four points of each other are not separable
  on this validation set, and the top four on test are inside that band. The validation argmax could easily
  land elsewhere. **If T2 fails while T1 holds, the honest answer is "a real tuning procedure would not have
  found the best configuration, and it would not have mattered" — which is a stronger statement for the
  paper than the one it currently makes, and an honest one.**
- T3's bar holds by construction unless the code is wrong; the numbers are the point.
- **Direction, stated in advance because it favours us.** Test-selection maximises the opponent, so an
  honest procedure can only lower the comparator and only widen our margin. A result that flatters us,
  obtained by changing the method, needs the old number kept beside it: whatever this run returns, the
  manuscript should report both — the margin against an honestly tuned opponent and the margin against one
  whose configuration was chosen on the data it was scored on.

## Scope

bAbI task 2, Phi-2 frozen except the LoRA adapter, V-50's instance set and answer-candidate restriction.
Nothing here is about the arenas, the held-out-combination split, or analogical mapping. The handover's
1.000 is banked and is not re-run.

## Protocol

`v152_score.py` frozen and committed with this file before the sweep runs, and run verbatim.
