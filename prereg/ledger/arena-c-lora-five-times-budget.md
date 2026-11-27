# Ledger excerpt: LoRA fine-tune on arena C at five times the budget

Supplementary Table `tab:prereg`, row "Arena C (LoRA fine-tune at five times the budget)" (marked "ledger" in the Supplementary Information).

**Source.** `ledger.md` of the experiment repository `ngcls` (private; see `README.md` in this folder), at commit `0732e13` (`0732e13ef2322f308a99dcac3bf0bd2e25a0da10`, tag `archive-2026-11-11`); the copy read has MD5 `6843647bad4782ac58fcd341448bf95c`, identical to that commit's working tree. Line numbers refer to that file. For each part, the commit is the one that added those lines, from `git blame` and checked with `git log -S` on the heading; dates are commit author dates in UTC+08:00 (the machine's clock). Parts not from `ledger.md` say so.

| # | Part | Ledger lines | Heading date | Added in commit | Commit date |
|---|---|---|---|---|---|
| 1 | Context: V-15 registration (matched-budget design that V-15b extends) | 9032-9054 | none in heading | `e293493` | 2026-08-20T12:42:55+08:00 |
| 2 | PRE-REGISTRATION: V-15b | 9173-9211 | 2026-08-20 | `04282ea` | 2026-08-20T15:34:48+08:00 |
| 3 | Verdict: V-15b | 9500-9526 | 2026-08-20 | `b3f9b81` | 2026-08-20T21:01:42+08:00 |

**Depositor's notes** (written when this excerpt was deposited, 2026-11-21; not part of the ledger).

- **Row in the paper.** "pre-specified extension"; outcome "all seeds between 0.48 and 0.54"; counted in the paper as an extension without a bar or prediction.
- **Matching registration.** V-15b (part 2): epochs 2 -> 10, everything else as V-15 (part 1, the matched-budget design it extends). Verdict (part 3): best-epoch accuracies 0.5375, 0.52, 0.515; per-epoch values 0.48-0.5375. The numbers match the row.
- **Discrepancy: it had bars and a prediction.** V-15b registered (i) "COLLAPSE EXPLAINED BY BUDGET: acc >= parser - 10pp (.8340) in >= 2/3 seeds", (ii) "COLLAPSE NOT EXPLAINED BY BUDGET AT 5x: acc <= floor + .10H (.5572) in 3/3 seeds", (iii) intermediate, and predicted (ii). Bar (ii) held in 3/3 and the prediction held. The row can state the bar and its outcome instead of "pre-specified extension"; counted that way it is a test.
- **Note.** Accuracies are best epoch on the test set, as in the arena scripts. The constants (floor 0.5153, parser 0.9340) are V-13's; the V-13R re-run reproduced both exactly.

---

## Part 1: Context: V-15 registration (matched-budget design that V-15b extends) (ledger.md lines 9032-9054)

SHA-256 of the lines below, each followed by a newline: `41aeb979357c8c6687f4d9f7b2b5db09aaf31d4e163d88d747e55934f35de56a`.

````text
## V-15: ANSWER "WHY VSA AT ALL" WITH A MEASUREMENT.
LoRA beats every frozen-feature arm on the tasks measured so far (.988 and 1.000 on V-3's admitted tasks),
so a referee is entitled to ask what the symbolic route buys. Positioning cannot answer that; a measurement
can. The candidate is the property VSA is theoretically for and LoRA has no reason to have: generalisation
to symbol COMBINATIONS never seen in training.
DESIGN: arena B (role binding) has exactly that split by construction -- training never contains a held-out
(role, filler) pair and evaluation contains only held-out pairs. The symbolic parser arm scores .9960 there
WITHOUT TRAINING ON PAIR IDENTITY AT ALL. Run LoRA r=16 on the same split, same budget, 3 seeds, and also
on arena C.
BARS:
  SYMBOLIC ADVANTAGE ESTABLISHED iff parser - LoRA >= 10pp on the held-out split in 3/3 seeds, on at least
    one arena. Then the paper can say the symbolic route generalises to unseen combinations where
    parameter-efficient fine-tuning does not, which is a reason to want it that does not depend on accuracy
    on seen combinations.
  NO ADVANTAGE iff LoRA >= parser - 2pp in 3/3 on both arenas. Then LoRA does compositional generalisation
    too, the honest answer to "why VSA" is "on this evidence, no reason", and the manuscript will say that
    in those words.
  INTERMEDIATE otherwise.
REGISTERED EXPECTATION: I expect a symbolic advantage on the held-out split, because LoRA must learn pair
identity from data that excludes the test pairs while the parser arm computes the answer algebraically. But
I have been wrong about LoRA before -- it solved V-3's admitted task at .988 when I expected frozen features
to cap it -- so I am recording the NO ADVANTAGE branch as a live possibility and its consequence in advance.
GPU: 0 or 2 only; GPU 1 remains reserved.
````

## Part 2: PRE-REGISTRATION: V-15b (ledger.md lines 9173-9211)

SHA-256 of the lines below, each followed by a newline: `723f6b32362945002c3c603b9d9b59216909110834b7a8b2cb2a4b8a1276b204`.

````text
## 2026-08-20 ~07:35Z — V-15b PRE-REGISTERED (this entry + scorer v15b_score.py committed BEFORE launch).

PURPOSE: falsify our own headline. V-15 (f8a3415) established the symbolic advantage on arena C but
disclosed, in the verdict itself, that LoRA's TOTAL collapse (.5158 vs floor .5153, all 3 seeds, both
epochs at chance) is compatible with an optimisation/budget failure rather than a compositional one —
the registered budget was 2 epochs. That caveat now sits next to the manuscript's central why-VSA claim
(sec:whyvsa, local 05fa08b). Per discipline 2 this is the fresh registration NAMING the confound:
TRAINING BUDGET (epochs). Nothing else changes. Motivation is adversarial to our own result: better we
kill the claim here than a reviewer does.

DESIGN: v15.py UNMODIFIED. Arena C only; epochs 2 -> 10 (5x the registered budget); same seeds
42/123/456; same lr 2e-4, bs 2, n_train 600, n_test 400; cuda:0 (GPU 1 reserved); out v15b_logs/.
lr is deliberately NOT swept: this registration names budget as the sole axis; an lr/optimiser
hypothesis would need its own registration. No arena-B arm: B's question is seed reliability, not
budget (sd42 already solved it at 2 epochs).

CONSTANTS (re-copied from f8a3415 / 5d496c7, not from memory): floor .5153, parser .9340,
H = .4187.

REGISTERED BARS (scorer v15b_score.py, committed with this entry; acc = best-epoch test acc per seed):
 (i)  COLLAPSE EXPLAINED BY BUDGET: acc >= parser - 10pp (.8340) in >= 2/3 seeds.
      CONSEQUENCE: the why-VSA headline is RETRACTED AS STATED; sec:whyvsa must be recast from
      "learned routes fail at matched budget" to "learned routes are slower" (the advantage is
      sample/compute efficiency only), per rule 5.
 (ii) COLLAPSE NOT EXPLAINED BY BUDGET AT 5x: acc <= floor + .10H (.5572) in 3/3 seeds.
      CONSEQUENCE: the caveat is TIGHTENED to "persists at 5x budget"; the matched-budget claim
      stands with stronger footing. This does NOT license "no budget ever suffices" — the
      manuscript sentence must say "at up to 5x the matched budget".
 (iii) INTERMEDIATE: anything else. Report verbatim; manuscript states partial budget-dependence.

FALSIFIABLE PREDICTION (recorded so it can be wrong): I expect (ii) — in V-15 both epochs sat at
chance for all seeds with no visible learning signal. But V-3 taught me LoRA surprises on this
apparatus (.988 where I expected a frozen-feature cap), and arena B's sd42 shows the split is
learnable in principle; (i) is a live branch and its consequence — retraction — is accepted in
advance.

BUDGET: ~75 min/seed sequential (886s at 2 epochs -> ~4400s at 10), ~3.7h total, single GPU (cuda:0).
GPU 1 untouched. Launched nohup immediately after the prereg commit; verdict in a later tick by
running the committed scorer on v15b_logs/v15b.json — no bar may change after this entry.
````

## Part 3: Verdict: V-15b (ledger.md lines 9500-9526)

SHA-256 of the lines below, each followed by a newline: `a171fd3e83e60b60152c22a35baed47db4a3243b57b0cad412350e89f4c7f787`.

````text
## 2026-08-20 ~13:05Z tick — V-15b VERDICT: bar (ii) 3/3 — LoRA's arena-C collapse is NOT explained by
## training budget at 5x. The V-15 caveat tightens; the symbolic-advantage claim stands as scoped.

Scored run: v15b_logs/v15b.json (pid 1612036, completed ~20:56 server time, 3/3 seeds logged). Scorer =
v15b_score.py, committed at prereg 04282ea BEFORE launch, run verbatim, output reproduced exactly:

  V-15b accs (best-epoch, seeds 42/123/456): [0.5375, 0.52, 0.515]
  bar (i)  acc >= 0.8340: 0/3 (need >= 2)
  bar (ii) acc <= 0.5572: 3/3 (need 3/3)
  V-15b VERDICT: (ii) COLLAPSE NOT EXPLAINED BY BUDGET AT 5x — tighten caveat; claim stands

The registered prediction on record (04282ea) was bar (ii); for once the expectation was right. Per-epoch
trajectories (v15b.log) show no late learning: all three seeds oscillate .48–.5375 across all 10 epochs —
10.5M trainable LoRA params never leave chance on arena C while the 2-parameter symbolic parser sits at
.9340. V-15's disclosed caveat ("may be optimisation/budget, 2 epochs") is REPLACED by the measured
statement: the collapse persists at 5x budget (10 epochs, ~110 min/seed); budget is ruled out at this
scale. What is NOT ruled out (honesty): lr/schedule choices, LoRA rank/placement, and arena-C's specific
format — none of these may be probed without a fresh prereg naming the confound.

MANUSCRIPT CONSEQUENCE — QUEUED, not applied this tick, by design: the sec:whyvsa caveat tighten lands in
the SAME single writeup pass already registered at the V-13R verdict (arena-C row restoration + v14_fixed
outcome + this caveat all touch tab:threearena / remedy paragraph / sec:whyvsa). One pass, when v14_fixed
lands, instead of three overlapping edits to both builds.

STILL IN FLIGHT (ps-verified under /tmp/ngcls_tick.lock before this entry): v14_fixed pid 2359259 cuda:0
(V-14 sweep on the daa512c-fixed arm; original V-14 bars apply verbatim at its verdict). GPU 1 untouched.
Nothing new is registered by this entry.
````
