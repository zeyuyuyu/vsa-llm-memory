# Ledger excerpt: Imposed code coherence on exact codes (tolerance definition)

Supplementary Table `tab:prereg`, row "Code coherence" (marked "ledger" in the Supplementary Information).

**Source.** `ledger.md` of the experiment repository `ngcls` (private; see `README.md` in this folder), at commit `0732e13` (`0732e13ef2322f308a99dcac3bf0bd2e25a0da10`, tag `archive-2026-11-11`); the copy read has MD5 `6843647bad4782ac58fcd341448bf95c`, identical to that commit's working tree. Line numbers refer to that file. For each part, the commit is the one that added those lines, from `git blame` and checked with `git log -S` on the heading; dates are commit author dates in UTC+08:00 (the machine's clock). Parts not from `ledger.md` say so.

| # | Part | Ledger lines | Heading date | Added in commit | Commit date |
|---|---|---|---|---|---|
| 1 | PRE-REGISTRATION: V-95 (tolerance first defined, prediction P3) | 16530-16581 | none in heading | `477dcbb` | 2026-09-01T03:31:42+08:00 |
| 2 | Verdict: V-95 result | 16754-16779 | none in heading | `c89fa3c` | 2026-09-01T03:53:46+08:00 |
| 3 | PRE-REGISTRATION: V-163 (same definition under the fitted rule; gates, bars, predictions) | 26691-26752 | 2026-11-11 | `528d3ea` | 2026-11-11T05:14:32+08:00 |
| 4 | Verdict: V-163 | 26785-26826 | 2026-11-11 | `2475c2a` | 2026-11-11T05:34:02+08:00 |

**Depositor's notes** (written when this excerpt was deposited, 2026-11-21; not part of the ledger).

- **Row in the paper.** "tolerance defined in advance as the largest code coherence at which accuracy exceeds the floor by ten points"; outcome "0.25 (0.15 under a fixed decision rule)"; departures "--". Counted in the paper as a definition, without a bar or prediction.
- **Matching registrations.** The definition was first registered in V-95 (part 1, prediction P3: "the largest c at which accuracy still clears floor + 0.10 = 0.615"); V-95 measured 0.0989 under the fixed rule 20 cos - 4 > 0 (part 2). V-163 (part 3) re-registered the same definition ("defined exactly as V-95 defined it") under the fitted rule that the arms use; its verdict (part 4) gives 0.2486 (imposed level 0.25, accuracy 0.620). The definition and the 0.25 match the paper.
- **Differences from the row.** (1) V-163 is not only a definition: it registered gates G0 and G1, bars B1-B3 and predictions (tolerance in [0.20, 0.30)); all held. V-95 registered predictions P1-P3; all hit. (2) "0.15 under a fixed decision rule" is not stated in the ledger. It follows from the V-163 curve (fixed rule 0.8306 at 0.15, 0.6044 at 0.20, against the 0.615 criterion); the fixed-rule tolerance the ledger states is V-95's 0.0989, on a grid without 0.15. (3) V-163 says it "is not blind": a read-only re-simulation on 2026-11-05 (fitted-rule accuracy 0.693 at 0.2 and below 0.615 at 0.3) preceded the registration. The row's last column ("--") does not record this prior look.

---

## Part 1: PRE-REGISTRATION: V-95 (tolerance first defined, prediction P3) (ledger.md lines 16530-16581)

SHA-256 of the lines below, each followed by a newline: `6cd3a7c93bca07f8dc4ff2620092040591808a2a3bef9252f66f094fba74f060`.

````text
## V-95 prereg: where is the CROSS-TALK THRESHOLD for two-hop binding? Before any code.

**What V-94 left open.** The geometry of every route is now measured -- supplied codebook -0.0001, a perfect
classifier's continuous logits 0.3904, raw span-pooled states 0.9149 -- and the codebook route scores 0.9244
while every continuous route sits at arena C's floor of 0.515. What is missing is the FUNCTION between those
two facts: how much cross-talk can two-hop binding tolerate before it fails? Nobody in this literature reports
that number, and it is the difference between "continuous codes are hopeless" and "continuous codes need to
reach cross-talk X, and the best available is 0.39".

**Design, and it needs no training at all.** Build a synthetic codebook with cross-talk dialled to a target:
```
    v_s = sqrt(1 - c) * sym_s  +  sqrt(c) * u        (then L2-normalised)
```
where `sym_s` are the arena's orthonormal random rows and `u` is one fixed common direction, mimicking the
narrow cone that makes LLM states anisotropic. For orthonormal `sym`, the cosine between distinct `v_s`
becomes approximately `c`, so the sweep addresses cross-talk directly. Substituting this codebook into the
`parser` arm and evaluating -- no gradient steps anywhere -- isolates cross-talk as the only variable, with
binding, unbinding, cleanup and chaining untouched.

**Sweep**: c in {0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}, three codebook seeds each, and the
REALISED cross-talk measured rather than assumed. Arena C, N_CTX=32, HOPS=2, floor 0.515.

**Registered predictions.**
- P1 (gate): at c = 0.0 the arm reproduces the codebook result, 0.9244 +/- 0.03. If it does not, the sweep
  harness is not the one that produced that number and nothing is read from the rest.
- P2: accuracy is monotonically non-increasing in realised cross-talk, allowing seed noise of 0.03. A
  non-monotone curve would mean cross-talk is not the operative variable and the whole V-93/V-94 story needs
  re-examining.
- P3: **the threshold -- the largest c at which accuracy still clears floor + 0.10 = 0.615 -- lies strictly
  below 0.39.** This is the sharp claim: if the threshold is below 0.39, then even a PERFECT continuous
  classifier's codes are on the wrong side of it, and the discrete lookup is not a convenience but a
  requirement. If the threshold is above 0.39, continuous codes are viable and the manuscript's handover is
  a stronger claim than the geometry warrants -- which would be a problem for our own paper, found by us.

**Committed readings.**
- P3 holds: the manuscript's handover acquires a quantitative justification. "Identity must be supplied"
  becomes "two-hop binding over 32 superposed pairs requires cross-talk below T, the best continuous route
  available reaches 0.39, and only a discrete lookup reaches 0". That is the sharpest form this project's
  central claim has had.
- **P3 fails**: continuous codes at 0.39 would suffice, so the failure of every continuous arm is NOT
  explained by cross-talk and V-93/V-94's mechanism is wrong. I would then have to retract the isotropy story
  as I retracted the alignment story in V-92, and report that the reason continuous routes fail remains
  unknown.
- P2 fails: cross-talk is not the operative variable at all and both preceding entries need re-examination.

**Why this is worth the run despite being cheap.** Every other experiment in this arc measured a property of
a trained arm and inferred a mechanism. This one manipulates the suspected cause directly, with nothing
trained, so it is the first genuinely causal test in the sequence rather than another correlation.

**I am blind on all three.** No cross-talk sweep has been run here.

**Hard stop: 1 calendar day.** Compute: the A800 box. Nothing enters the 9/16 submission.
````

## Part 2: Verdict: V-95 result (ledger.md lines 16754-16779)

SHA-256 of the lines below, each followed by a newline: `2db4a02c79d5c44c9e755c0d3af6dee29a334839d74460cc42175b33d7d67b37`.

````text
### V-95 RESULT (prereg 477dcbb). **The cross-talk threshold is 0.0989.** P1, P2, P3 all HIT.

```
  realised crosstalk   accuracy
      0.0001            0.9244     <- reproduces the manuscript's codebook arm exactly
      0.0492            0.9178
      0.0989            0.9033     <- last point clearing floor+.10 = 0.615
      0.1985            0.7194
      0.2988            0.5794
      0.3987 .. 0.8999  0.4989     <- identical to four decimals at every level
```

**P1 HIT**: c=0 gives 0.9244 against the banked 0.9244. **P2 HIT**: monotone. **P3 HIT**: the threshold,
0.0989, is four times below the 0.3904 a perfect continuous classifier achieves.

**Reading the flat tail honestly.** From cross-talk 0.40 upward the accuracy is 0.4989 with per-seed values
identical to four decimals at every level, which means the pipeline has degenerated to a constant output.
Those points carry no information beyond "dead past 0.4", and I am not presenting them as six independent
measurements.

**What the number buys.** Two-hop HRR binding over 32 superposed pairs tolerates cross-talk up to about 0.10
and dies by 0.40. The best geometry any continuous map of transformer states achieves is 0.3904 -- measured
from a head that identifies symbols at 1.0000. So the discrete lookup is not a convenience or an
implementation choice: **the continuous route is on the wrong side of a threshold that was measured
independently of it.** This is the first causal manipulation in the V-89..V-95 arc -- cross-talk was dialled
directly with nothing trained -- rather than a property measured on a trained arm and a mechanism inferred.
````

## Part 3: PRE-REGISTRATION: V-163 (same definition under the fitted rule; gates, bars, predictions) (ledger.md lines 26691-26752)

SHA-256 of the lines below, each followed by a newline: `95cd9791703169d6f4765b1507b1ff4c18ae7cd55c1cc53e406c34636e7973f0`.

````text
## V-163 prereg (2026-11-11): the V-95 cross-talk sweep under the decision rule the Table arms use, and what limits exact rows in arena C. Before any code.

Live session, user-directed (review of the npj manuscript before submission; user: "你来做决定吧，按照你的决定改").

**Why.** Two defects found in the code walkthrough of the manuscript (2026-11-05):
1. V-95 scored the imposed cross-talk sweep with the arm's untrained logit, 20·cos − 4 > 0, that is, positive iff
   cos > 0.2. Imposed cross-talk c lifts every pairwise cosine to about c, so from c ≥ 0.2 on every instance is called
   positive and the accuracy is the positive rate: the flat 0.4989 from c = 0.3 to 0.9 is that rate, not a measurement
   of the algebra. The arms the manuscript plots on the same axis (exact rows, handover, continuous projection) are
   scored by v13.run()'s rule, a threshold fitted in closed form on training cosines. "Tolerance ≈ 0.10" therefore
   mixes the algebra's degradation with a fixed threshold.
2. Methods says of the exact-row arm in arena C that "superposition interference is its only remaining error source".
   The arena shares one codebook between roles and fillers and circular convolution commutes, so unbinding
   Σ_j b_j ⊛ a_j by the query q also returns a_j for any pair whose filler is q. If that a_j is itself a candidate
   filler, clean-up can snap to it. That would be an alias of the arena's design, not superposition noise.

**Prior looks, disclosed: this run is not blind.** A read-only CPU re-simulation on 2026-11-05 (index-level copy of
the arena, not banked; paper repo deck/codecards/sim/) gave: fitted-rule accuracy 0.693 at c = 0.2 and below 0.615
at c = 0.3; 76 of 500 test instances with the alias condition, first-hop error 0.49–0.58 on them and 0.0000 on the
rest. AUROC was not looked at. This entry banks the measurement with v13's and v95's own code and fixes in advance
what the manuscript says under each outcome.

**Design.** No language model: the parser arm reads no hidden states, so H is a dummy; arena, chain and cone codebook
are imported from v13.py and v95.py. No training anywhere.
- Part A, sweep: c ∈ {0, .05, .1, .15, .2, .25, .3, .35, .4, .5, .6, .7, .8, .9}; codebook seeds 0, 1, 2; test arena
  v13.Arena(tok, 600, 5000 + s) as in V-95; training arena v13.Arena(tok, 800, 42) for the fitted threshold, as in
  v13.run(). Three readings of the same final cosines: `fixed` (V-95's rule), `fitted` (v13.run().acc(), threshold
  from the training arena under the same codebook), `auroc` (no decision rule).
- Part B, ceiling, at c = 0 on the Table configuration (codebook seeds 42, 123, 456; test arena
  v13.Arena(tok, 500, 4242)): first-hop clean-up with the true key, split by the alias condition (another pair of the
  level has the query symbol as its filler, and that pair's role symbol is among the level's candidate fillers and
  is not the true answer).

**Gates (if either fails, nothing is read and the manuscript only states V-95's rule in its caption).**
- G0: `fixed` reproduces v95_logs/v95_scored.json per seed at each of V-95's eleven c, within 1e-4.
- G1: `fitted` at c = 0 on the Table configuration reproduces the parser arm of v13_logs/v13_fixed.json
  (0.932, 0.932, 0.938), within 1e-4.

**Reported whatever the outcome.** The whole curve under the three readings; the tolerance under `fitted`, defined
exactly as V-95 defined it (largest tested cross-talk whose mean accuracy is ≥ 0.615 = floor + 0.10).

**Bars.**
- B1: the tolerance under `fitted` is below 0.39 (V-95's P3, a perfect classifier's continuous logits).
- B2: at every c ≥ 0.7, `fitted` ≤ 0.565 (floor + 0.05) and `auroc` ≤ 0.60.
- B3: pooled over the three Table seeds, at least 90% of first-hop clean-up errors fall on alias-condition
  instances, and the first-hop error rate on the other instances is ≤ 0.01.

**What the manuscript says under each outcome.**
- Gates pass: the cross-talk panel is redrawn with the `fitted` curve (the rule of every arm plotted on it); the
  tolerance quoted is the one measured here; V-95's fixed-rule numbers stay in the Supplementary with the rule named.
- B1 holds: "a continuous route at 0.39 is on the wrong side of the tolerance" stays. Fails: withdrawn.
- B2 holds: "imposed cross-talk at the level of the failing arms is sufficient to put the chain at the floor" stays.
  Fails: narrowed to the measured values; if `auroc` stays high while `fitted` is at the floor, that is reported as a
  threshold-transfer failure, not as lost information.
- B3 holds: the Methods clause is replaced by the alias explanation, stated as a property of this arena's shared
  codebook. Fails: the clause is deleted and no explanation is offered.

**Predictions on the record.** G0 and G1 hold. Tolerance in [0.20, 0.30). B1 holds. B2 holds for `fitted`; low
confidence for `auroc` (unseen). B3 holds.

**Out of scope.** No new arm, no change to any banked number of V-13 or V-95, no claim about other shapes of
cross-talk than the single common direction of V-95.
````

## Part 4: Verdict: V-163 (ledger.md lines 26785-26826)

SHA-256 of the lines below, each followed by a newline: `2f5bfdd0e0ca6c2862647f8beffa069a48c3207c90aa39a40be977c970136671`.

````text
## V-163 CLOSED (2026-11-11): under the rule the Table arms use, the cross-talk tolerance is 0.25, not 0.10; arena C's exact-row ceiling is a commutativity alias. G0 PASS | G1 PASS | B1 HOLDS | B2 HOLDS | B3 HOLDS

Prereg 528d3ea, code and frozen scorer a62631a (before the run). CPU only, no language model. Scorer output verbatim
in v163_logs/v163_score.out; artifact v163_logs/v163.json.

- **G0.** The fixed rule reproduces V-95 at all 33 per-seed values, worst difference 0.000000. **G1.** The fitted rule at
  c = 0 on the Table configuration reproduces the banked parser arm (0.932, 0.932, 0.938), worst difference 0.000000.
- **Curve (mean of three codebook seeds; fixed / fitted / AUROC).** c = 0: .9244 / .9222 / .918. 0.05: .9178 / .9167 /
  .917. 0.10: .9033 / .9022 / .907. 0.15: .8306 / .8317 / .832. 0.20: .6044 / .7017 / .718. 0.25: .5006 / .6200 / .631.
  0.30: .4989 / .5706 / .564. 0.35: .4989 / .5461 / .545. 0.40: .4989 / .5356 / .526. 0.50-0.90: .4989 / .50-.53 /
  .49-.52. Monotone within 0.03.
- **Tolerance under the fitted rule** (largest tested cross-talk with mean accuracy ≥ 0.615): **0.2486** (c = 0.25,
  accuracy 0.620); the next tested point, c = 0.30, is at 0.571. V-95's 0.0989 was the fixed rule's.
- **B1 HOLDS:** 0.2486 < 0.39. **B2 HOLDS:** at c ≥ 0.7 the fitted rule gives at most 0.526 and AUROC at most 0.503.
- **B3 HOLDS:** at c = 0 on the Table configuration, 115 first-hop clean-up errors over three codebook seeds, all 115 on
  the 228 alias-condition instances (76 of 500 per seed; error rate 0.504 there) and all 115 landing on the aliased role
  symbol; 0 errors on the other 1,272 instances.

**Predictions against outcome.** G0, G1, B1, B2, B3 as predicted. Tolerance predicted in [0.20, 0.30): 0.2486. AUROC
(unseen before the run) falls with the fitted accuracy, to chance by c ≈ 0.5.

**Reading.** (1) Imposed cross-talk degrades the two-hop chain gradually between 0.10 and 0.30 and leaves it at chance
from about 0.35 on; the decision-rule-free AUROC falls in step, so the loss is of information, not of threshold
transfer. The causal statement stands; its number changes. The sharp fall between 0.10 and 0.20 in V-95 was partly the
fixed threshold: once every pairwise cosine exceeds 0.2 the fixed rule calls every instance positive. (2) A continuous
route at 0.39 is still on the wrong side. (3) With exact rows the first hop fails only when another pair of the level
has the query symbol as its filler and that pair's role symbol is a candidate filler; then clean-up is a coin flip
between the true filler and the aliased role (0.504). The 0.934 ceiling of the exact-row arm in arena C is therefore
set by the arena sharing one codebook between roles and fillers under a commutative binding, not by superposition
noise at 32 bindings and d = 2,048.

**Consequences for the manuscript (as registered).** The cross-talk panel is redrawn with the fitted curve and the
tolerance quoted as about 0.25 under the rule of the arms plotted on it; V-95's fixed-rule numbers go to the
Supplementary with the rule named. The Methods clause "superposition interference is its only remaining error source"
is replaced by the alias explanation. The arm labelled "trained through the binding loss" at 0.699 in that panel is
V-93/audit-2's untrained projection (fit0); the trained arm's different-symbol cosine is V-89's 0.9936 and the label
is corrected.

**Literature.** Consistent with the classical account, no novelty claimed: circular convolution is commutative, so
role-filler pairs drawn from one codebook cannot be told apart by order without a permutation or separate role codes
(Plate 1995, 2003; Kanerva 2009); near-orthogonality of the codebook is the stated precondition of HRR retrieval.
Not found and not searched further: a published tolerance figure for common-direction cross-talk at this load.
````
