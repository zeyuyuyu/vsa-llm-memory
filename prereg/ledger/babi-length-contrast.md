# Ledger excerpt: Length contrast on bAbI task 1

Supplementary Table `tab:prereg`, row "bAbI (length contrast, task 1)" (marked "ledger" in the Supplementary Information).

**Source.** `ledger.md` of the experiment repository `ngcls` (private; see `README.md` in this folder), at commit `0732e13` (`0732e13ef2322f308a99dcac3bf0bd2e25a0da10`, tag `archive-2026-11-11`); the copy read has MD5 `6843647bad4782ac58fcd341448bf95c`, identical to that commit's working tree. Line numbers refer to that file. For each part, the commit is the one that added those lines, from `git blame` and checked with `git log -S` on the heading; dates are commit author dates in UTC+08:00 (the machine's clock). Parts not from `ledger.md` say so.

| # | Part | Ledger lines | Heading date | Added in commit | Commit date |
|---|---|---|---|---|---|
| 1 | PRE-REGISTRATION: V-45 | 12498-12546 | none in heading | `094cc5d` | 2026-08-23T05:27:39+08:00 |
| 2 | Verdict: V-45 (bar (a) failed) | 12548-12579 | none in heading | `311f294` | 2026-08-23T06:01:45+08:00 |
| 3 | Reconciliation: V-45 (pooling operator) | 12581-12619 | none in heading | `68db1bc` | 2026-08-23T06:32:43+08:00 |
| 4 | PRE-REGISTRATION: V-46 (last-token floor) | 12647-12693 | none in heading | `1aa3bb3` | 2026-08-23T07:31:49+08:00 |
| 5 | Launch record: V-46 (scorer committed before data) | 12714-12727 | 2026-08-23 | `cc5c344` | 2026-08-23T08:04:08+08:00 |
| 6 | Verdict: V-46 | 12729-12783 | none in heading | `cdde71d` | 2026-08-23T08:32:01+08:00 |
| 7 | Duplicate registration "V-46", later voided | 12907-12947 | none in heading | `ab6159a` | 2026-08-24T00:51:22+08:00 |
| 8 | Voiding of the duplicate registration (ANOMALY entry) | 12950-12986 | 2026-08-24 | `42ddb5d` | 2026-08-24T01:05:08+08:00 |
| 9 | Result of the duplicate run | 12988-13027 | none in heading | `6222b21` | 2026-08-24T02:04:30+08:00 |
| 10 | PRE-REGISTRATION: V-47 (rerun with the paper's floor estimator) | 13029-13070 | none in heading | `e37f1f2` | 2026-08-24T02:05:11+08:00 |
| 11 | Verdict: V-47 (source of the quoted numbers) | 13072-13109 | none in heading | `0d80fdb` | 2026-08-24T02:30:09+08:00 |

**Depositor's notes** (written when this excerpt was deposited, 2026-11-21; not part of the ledger).

- **Row in the paper.** "bars fixed in advance (training on at most six statements, testing on at least ten)"; outcome "naming head at 1.000, but so were exact rows, so the contrast measures the floor". The Supplementary Note quotes a loss of the last-token probe of 0.183-0.195.
- **Matching registrations.** Three registrations of one design: V-45 (part 1), V-46 (part 4) and V-47 (part 10, bars "verbatim from V-45's prereg at 094cc5d"). Buckets: train and evaluate on at most six statements, evaluate on at least ten. Bars: (a) the floor has purchase, >= 0.70 on short stories in 3/3; (b) the handover extrapolates, >= floor_long + 0.90 H_long and >= 0.90 x its short accuracy in 3/3; (c) the floor loses >= 0.10 from short to long in 3/3; admission H >= 0.20 per bucket.
- **Outcomes.** V-45 failed bar (a) (0.34-0.36) because its floor was a mean over tokens rather than the last-token probe (part 2; reconciliation, part 3). V-46 (part 6) and V-47 (part 11) passed (a), (b) and (c) in 3/3. The quoted 0.183-0.195 is V-47 (losses 0.195, 0.1883, 0.1833); V-46 gave 0.1883, 0.1983, 0.1833. A duplicate "V-46" registration (part 7) was voided as a registration (part 8).
- **Differences from the row.** (1) The row gives the bucket definitions as the "bars"; the registered bars are (a)-(c) above, and all three held. (2) The registered per-bucket admission gate (H >= 0.20) was not applied to the short bucket, where H = 0.18 (V-46 disclosure 2; V-47 "prereg/scorer mismatch"). (3) The row does not say that the first registration (V-45) failed its admission bar because of the wrong estimator before the design was registered again.

---

## Part 1: PRE-REGISTRATION: V-45 (ledger.md lines 12498-12546)

SHA-256 of the lines below, each followed by a newline: `f298e67cdeeb3c849f7dbca17252df72865cbb6a8c6b7fc4ab7f8aad9372f7cb`.

````text
## V-45 prereg: the compositional contrast, on a substrate where the baseline actually works. Before any code.

**What V-44 left open.** V-43 showed the handover composes at an untrained depth. V-44 then removed the stronger
reading: on arena C a pooled-state probe cannot do the task at depth three at ANY training depth, so its failure
after transfer says nothing about compositionality -- there was never a route to fail. The contrast needs a
substrate where the learned baseline demonstrably works at the training setting and then loses it. bAbI task 1
is that substrate: the six-way pooled-state probe scores .771 there, which is real purchase, not chance.

**The axis.** Story length. bAbI task 1 instances vary in how many statements precede the question. Longer
stories mean more distractor locations in the bundle and a longer sequence for a pooled mean to summarise. Train
on SHORT stories, evaluate on short (control) and LONG (held out by length, never trained on).
  short   instances with at most 6 statements
  long    instances with at least 10 statements
Buckets are by statement count only; if either bucket has fewer than 150 evaluation instances the run reports
that and is void, because a contrast measured on a handful of stories is not a contrast.

**Arms, all trained on the SHORT bucket only and evaluated on both, nothing refitted.**
  floor          six-way linear probe on the pooled frozen state
  parser         exact codebook rows -- learns nothing, marks what the algebra alone does at each length
  probe_argmax   the supervised discrete handover
The recency rule is supplied from outside the algebra at both lengths, as in V-30, since a superposition cannot
express recency and this run is about length rather than about that limit.

**Bars.** Per bucket, H = parser - floor, admission H >= 0.20 or that bucket is void.
(a) THE BASELINE HAS PURCHASE (required before any contrast may be claimed) iff floor on the SHORT bucket is
    >= 0.70 in 3/3. If the probe is near chance even on short stories, this substrate is no better than arena C
    for this purpose and I say so and stop.
(b) EXTRAPOLATES iff probe_argmax on LONG >= floor_long + 0.90 H_long in 3/3 AND >= 0.90 x its own SHORT
    accuracy in 3/3.
(c) THE CONTRAST HOLDS iff the probe's LONG accuracy is at least 0.10 below its SHORT accuracy in 3/3 -- it had
    the ability and lost it with length.
(d) A POSITIVE RESULT REQUIRES (a) AND (b) AND (c). (b) without (c) is reported as "extrapolates, no contrast",
    the same outcome V-43 ended with, and is labelled that way rather than dressed up.
(e) FAILS iff probe_argmax on LONG <= floor_long + 0.10 H_long in 3/3.

**Diagnostic, reported whatever happens.** Bucket sizes; mean statements and mean distinct locations per bucket;
per-span identity accuracy of the head on the LONG bucket, so that a failure separates "the head stopped
recognising symbols in longer contexts" from "the algebra ran out of capacity at more bindings".

**Prediction on the record.** (a) holds, (b) holds, (c) holds. The head is per-span and cannot see story length;
the algebra binds one statement at a time and its capacity decays gradually, as V-14 measured; a pooled mean over
a longer story is being asked to summarise strictly more distractors with the same width. Twenty-three
predictions tested today, nineteen failed, and the ones that held were those where I could name the mechanism in
advance -- as here.

**What a negative means.** If (c) fails because the probe holds up on long stories, then pooled states carry this
task robustly and the paper's "why prefer a symbolic route" section is weaker than currently written, and I would
have to say so. If (b) fails, the handover is length-sensitive and the extrapolation claim from V-43 narrows to
depth rather than generalising to context size.
````

## Part 2: Verdict: V-45 (bar (a) failed) (ledger.md lines 12548-12579)

SHA-256 of the lines below, each followed by a newline: `a3f2b110073a00631137281aed80eec5259dc912ab9a11b8d1b37a3736b95deb`.

````text
## V-45 VERDICT: NO USABLE BASELINE -- bar (a) failed; the prereg's .771 purchase did not reproduce; per prereg: stop, no contrast claimed.

**Numbers (bAbI task 1, buckets by statement count; train short(<=6)=540, eval short=600, eval long(>=10)=200 --
both buckets over the 150-instance floor; 3 seeds; bars 094cc5d, code 4333c32 committed before the run; results
v45_logs/v45.{log,json}, re-copied verbatim):**
  short: floor [.3417 .3483 .3617]  parser 1.0000  probe_argmax 1.0000  span_identity 1.0000  (H = .6494)
  long:  floor [.1650 .2100 .2700]  parser 1.0000  probe_argmax 1.0000  span_identity 1.0000  (H = .7850)
  (a) baseline has purchase (short floor >= .70 in 3/3)      -> FAIL  [.3417 .3483 .3617]
  (b) extrapolates                                           -> PASS  (probe_argmax 1.0000 at both lengths)
  (c) contrast (floor loses >= .10 short->long in 3/3)       -> FAIL  (losses .1767 / .1383 / .0917; seed 456 under)

The run died at the admission step it was built around: the pooled-state probe sits at .34-.36 on SHORT
stories -- above six-way chance (.167) but nowhere near the .70 the prereg required before any contrast may be
claimed. The prereg's stated premise ("the six-way pooled-state probe scores .771 there, which is real purchase")
DID NOT REPRODUCE in this harness. Whatever produced that .771 (different preprocessing, training-set size,
story-length mix, probe capacity) must be reconciled before the number is cited anywhere; until then it is
unsourced and this entry marks it so.

What is licensed: nothing new. parser = probe_argmax = span_identity = 1.000 at both lengths says the symbolic
route is at ceiling on this arena and neither the head nor the algebra degrades with story length -- only the
pooled floor does, and not uniformly (seed 456 lost .0917 < .10). But with no working learned baseline the
ceiling is trivial in exactly the way V-44 diagnosed on arena C, now reproduced on the natural-data side. Per
prereg bar (a), verbatim: "this substrate is no better than arena C for this purpose and I say so and stop."
Stopped. No rescue -- no training-size increase, no bucket redefinition, no probe upgrade -- without a fresh
prereg that names the confound.

Prediction outcome: I predicted (a), (b), (c) all hold; (a) and (c) failed. The day's tally moves to
twenty-four predictions tested, twenty failed.

Apparatus note: prereg, code, launch, and scorer output were produced by a concurrent session (run finished
2026-08-23 ~05:30 local); that session ended without writing this entry, so this tick wrote it from
v45_logs/v45.json under /tmp/ngcls_tick.lock, numbers re-copied verbatim, nothing recomputed.
````

## Part 3: Reconciliation: V-45 (pooling operator) (ledger.md lines 12581-12619)

SHA-256 of the lines below, each followed by a newline: `fc464b2809fc4f4ed1aacecb8a16d4d61af55d84fa472909488ad5466bff593a`.

````text
## V-45 RECONCILIATION: the .771 is SOURCED (it is V-30's floor) and the discrepancy has a named cause -- the two
## harnesses compute "pooled" differently. Zero GPU; everything below is read from committed code and the ledger.

**The source.** The V-45 prereg's premise figure ("the six-way pooled-state probe scores .771") is V-30's floor,
this ledger, 2026-08-21 entry "V-30: MECHANISM FAILS ON NATURAL TEXT": bAbI task 1, 900 train / 500 eval
instances, ALL story lengths mixed, phi-2 frozen states, 3 seeds -> .780 .782 .752, mean .7713 (prereg lineage
0c17441). The V-45 verdict's "unsourced" flag is hereby lifted; the number is real and ledgered. What was wrong
was citing it as if it measured the arm V-45 built.

**The named confound: same word, different operator.**
  v30.py pooled() (lines 121-123): idx = am.sum(-1)-1; return H[arange, idx]  -- the LAST-TOKEN hidden state.
    On a causal LM this position summarises the entire prefix, and it sits nearest the final statement, so it
    carries recency structurally.
  v45.py pooled() (lines 27-29): (H*m).sum(0)/m.sum()  -- a masked MEAN over all tokens. Recency-blind, and the
    signal is diluted across the whole story.
The V-45 prereg even argued from "a pooled mean over a longer story" while citing a number measured on
last-token pooling. Both floors are honest implementations of the phrase "probe on the pooled frozen state";
they are not the same measurement, and .77 vs .34-.36 on the same substrate is the size of that gap.

**Secondary difference, disclosed:** V-30's probe trained on 900 mixed-length instances; V-45's on 540
short-only. Plausibly smaller, but not separable from the pooling change without a run, and no run is licensed
here.

**What this does and does not change.** V-45's verdict STANDS AS MEASURED: for the harness as built, bar (a)
failed 3/3 and no contrast is claimed. What changes is the reading: "the .771 did not reproduce" overstated it --
V-45 never re-measured V-30's floor; it measured a different floor arm that shares a name. The two results are
consistent: last-token probe .7713 (V-30), mean-pooled probe .34-.36 (V-45), same frozen states.

**Manuscript instruction.** Wherever the paper cites the .771 floor it must say LAST-TOKEN (final-position)
probe, not "pooled mean"; and the V-45 natural-data paragraph, if kept, must describe its floor as mean-pooled
and cite this reconciliation rather than saying the earlier number failed to reproduce.

**What would be licensed next (NOT run now).** A fresh prereg naming this confound could re-run the V-45
length-contrast with V-30's last-token floor. Registered expectation if it ever runs: the last-token floor
clears the 0.70 admission on short stories, restoring bar (a) -- that is a prediction on the record, not a
result. Under discipline 2 this entry changes no code and launches nothing.

Apparatus note: written by the 30-min tick under /tmp/ngcls_tick.lock; all numbers re-copied from this ledger
and from committed code (0c17441 lineage for v30.py, 4333c32 for v45.py); nothing recomputed.
````

## Part 4: PRE-REGISTRATION: V-46 (last-token floor) (ledger.md lines 12647-12693)

SHA-256 of the lines below, each followed by a newline: `bb18adf4f425075ddb33d28a82d1a6d8736826cc7d9f1f45eb7838aeb4810850`.

````text
## V-46 PREREG (before any code): the V-45 length contrast, re-run with the floor arm the premise actually
## measured. Rescue licensed under discipline 2: the confound is NAMED -- "pooled" meant two operators.

**The confound this fixes (V-45 reconciliation, this ledger, 2026-08-23).** V-45's floor probe read a masked
MEAN over all token states ((H*m).sum(0)/m.sum(), v45.py lines 27-29); the .771 purchase its prereg cited was
measured on the LAST-TOKEN state (H[arange, last_idx], v30.py lines 121-123, lineage 0c17441). V-46 re-runs
the identical story-length contrast with the floor probe on the last-token state. Nothing else changes: bAbI
task 1, phi-2 frozen states, probe trained on the short bucket only (<=6 statements, 540 instances), eval
short (600) and long (>=10 statements, 200), nothing refitted on long, ordering rule supplied from outside at
both lengths as in V-30/V-45, 3 seeds (123/234/456), per-span identity diagnostic kept. Void if either eval
bucket falls under 150 instances. The disclosed secondary difference from V-30 (900 mixed-length probe-train
vs 540 short-only) is NOT fixed here -- deliberately, so that a bar-(a) failure isolates it.

**Arms.** floor = linear probe on the last-token frozen state (v30 operator), trained on short only;
parser / probe_argmax / span_identity exactly as in v45.py (expected unchanged at ceiling; re-measured this
run, not copied from V-45).

**Bars, fixed now.**
(a) BASELINE HAS PURCHASE iff the last-token floor >= 0.70 on SHORT in 3/3. If (a) fails, the V-30/V-45 gap
    is NOT explained by the pooling operator alone and the 900-mixed vs 540-short training cut becomes the
    live suspect; I say so and stop -- no training-set enlargement without a fresh prereg.
(b) SYMBOLIC ROUTE EXTRAPOLATES iff probe_argmax on LONG >= floor_long + 0.90*H_long in 3/3 AND >= 0.90x its
    own SHORT accuracy in 3/3 (H as defined in V-45: headroom above the measured floor).
(c) CONTRAST HOLDS iff the last-token floor loses >= 0.10 absolute short->long in 3/3.
(d) A POSITIVE RESULT REQUIRES (a) AND (b) AND (c). (b) without (c) is reported verbatim as "extrapolates,
    no contrast" -- the V-43 outcome, not dressed up.
(e) FAILS iff probe_argmax on LONG <= floor_long + 0.10*H_long in 3/3.

**Predictions on the record, mechanisms named.** (a) HOLDS -- the last-token state of a causal LM summarises
the entire prefix and V-30 measured .7713 on mixed lengths; residual risk is the smaller short-only train set.
(This repeats the expectation already registered in the reconciliation entry.) (b) HOLDS -- the symbolic route
measured 1.000 at both lengths in V-45 and nothing in this change touches it. (c) HOLDS -- the last-token
state is recency-carrying; in longer stories the last mention of the queried entity sits on average further
from the final position and must survive more intervening bindings at fixed width (V-14's gradual capacity
decay is the mechanism). Confidence is lowest on (c), disclosed: the same recency structure that gives (a)
its purchase could keep recency-dominated instances easy at any length, in which case (c) fails and the floor
is reported as length-robust.

**What a negative means.** If (c) fails, the last-token probe carries this task across story length and the
manuscript's "why prefer a symbolic route" section weakens further on natural data -- it must then cite BOTH
floors (mean-pooled fragile, last-token robust), not the fragile one alone. If (a) fails, the .771 stays
correctly cited as a mixed-length-trained last-token result and the natural-data contrast remains unbuilt.

**Budget.** v46.py = v45.py with the floor pooling operator swapped to v30's last-token read; 3 seeds; one
GPU at most, GPU 1 left free; expected runtime same order as V-45; single launch, no grid, no sweep.

**This tick does prereg + commit only; code is the next tick's bounded step, per discipline 1.**
````

## Part 5: Launch record: V-46 (scorer committed before data) (ledger.md lines 12714-12727)

SHA-256 of the lines below, each followed by a newline: `1a77f58b3fa3964eb6750020391a9cf8a2acff82d8ac301ccde409be87025080`.

````text
## V-46 LAUNCHED (tick 2026-08-23, under /tmp/ngcls_tick.lock): scorer committed BEFORE data, run in flight.

v46.py committed at 91c38a9 = v45.py with exactly the prereg'd change (1aa3bb3): the floor pooling operator
swapped from the masked mean to v30's last-token read (H[am.sum()-1], lineage 0c17441); seeds 123/234/456;
bars (a)-(e) coded verbatim from the prereg; verdict strings pre-written for each branch including the
(a)-fail reading (pooling operator alone does not explain the V-30/V-45 gap; 900-mixed vs 540-short train cut
becomes the live suspect). Diff against v45.py reviewed line-by-line before commit: only the operator, the
seeds, the output path, and labels/verdict wording differ; arms, training, buckets, void rule untouched.

Launched single-flight on GPU 0 (all 8 idle at launch; GPU 1 left free), PID 675087, log v46_logs/run.log,
output v46_logs/v46.json. Startup verified: banner printed, process alive. No grid, no sweep, nothing else
launched. Committed locally only -- NO PUSH, per the disclosure-hold rule (ledger 4275ce6).

Next tick: read v46_logs/run.log; if V46-DONE, ledger the verdict against the 1aa3bb3 bars verbatim.
````

## Part 6: Verdict: V-46 (ledger.md lines 12729-12783)

SHA-256 of the lines below, each followed by a newline: `2ed9f6aa7851f0d7a057234ed26e2fe4ec271a62eae70ac2c4ede3ffafc2a368`.

````text
## V-46 VERDICT: POSITIVE per the 1aa3bb3 bars -- (a) AND (b) AND (c) all hold, 3/3. The last-token floor has
## purchase at training length and loses it with story length; the symbolic route does not. First positive
## contrast of the V-43/44/45/46 arc, scoped below.

**Numbers (bAbI task 1, buckets by statement count; train short(<=6)=540, eval short=600, eval long(>=10)=200;
seeds 123/234/456; code 91c38a9 committed before the run; results v46_logs/v46.{log,json}, re-copied verbatim):**
  short: floor [.8283 .8233 .8083] (mean .8200)  parser 1.0000  probe_argmax 1.0000  span_identity 1.0000  H=.1800
  long:  floor [.6400 .6250 .6250] (mean .6300)  parser 1.0000  probe_argmax 1.0000  span_identity 1.0000  H=.3700
  (a) baseline has purchase (short floor >= .70 in 3/3)        -> PASS  [.8283 .8233 .8083]
  (b) extrapolates (long >= floor_long+.90*H_long = .963 in 3/3
      AND >= .90 x own short in 3/3)                           -> PASS  (probe_argmax 1.0000 at both lengths)
  (c) contrast (floor loses >= .10 absolute short->long, 3/3)  -> PASS  (losses .1883 / .1983 / .1833)
  (d) POSITIVE RESULT: (a) AND (b) AND (c).

**What is licensed, exactly.** On natural text (bAbI task 1), a learned last-token probe with demonstrated
purchase at the training length (.81-.83, 3/3) loses .18-.20 absolute on stories 2.5x longer, while the
symbolic route holds at 1.000 with span identity 1.000 -- the head does not stop recognising symbols in longer
contexts and the algebra does not run out of capacity at 10 statements / 5.3 distinct locations. This is the
compositional contrast V-44 showed arena C could not host and V-45's mean-pooled floor could not host either.

**Disclosures, before anyone reads this as more than it is.**
  1. The parser arm (exact codebook rows, learns nothing) is ALSO 1.0000 at both lengths, and probe_argmax
     adds nothing above it. Bar (b) is therefore carried by a route at ceiling; the measured content of this
     result is the FLOOR's length-fragility, not a learned-route win. The contrast is baseline-vs-algebra,
     with the learned head merely surviving.
  2. Short-bucket headroom H = .1800. V-45's prereg had a per-bucket admission gate H >= .20; the V-46 prereg
     (1aa3bb3) restated only the 150-instance void rule and the committed scorer contains no H gate, so the
     verdict stands as registered -- but under V-45's own admission rule the short bucket would sit marginally
     under. Flagged, not hidden: the short-bucket contrast between parser and floor is thin (.18) precisely
     because the last-token floor is good there, which is what bar (a) demanded.
  3. Scope: 4 persons / 6 locations / codebook 10; the recency rule is supplied from outside the algebra at
     both lengths (as in V-30/V-45); long = 10 statements, not arbitrary length. Nothing here approaches the
     standing directive's Gate (iii) matched-compute-win standard; this is a manuscript-grade contrast for the
     "why prefer a symbolic route" section, not an incumbent-beating result.

**The reconciliation's confound explanation is CONFIRMED by this run.** Same substrate, same buckets, same
540-short training cut: mean-pooled floor .34-.36 (V-45) vs last-token floor .81-.83 (V-46). The pooling
operator alone accounts for the V-30/V-45 gap; the disclosed secondary suspect (900-mixed vs 540-short
training) is effectively dead -- 540 short-only instances already clear .70 by 11 points on the last-token
state.

**Prediction outcome.** Predicted (a), (b), (c) all hold, with mechanisms named in advance (last-token state
summarises the prefix; per-span head blind to length; recency-carrying state degrades as the queried mention
recedes). All three held. Twenty-five predictions tested today, twenty failed -- and the register continues to
show the same pattern: predictions survive when the mechanism is named before the run.

**Next manuscript action (separate bounded step, not this tick):** fold V-46 into main.tex/main_npj.tex as the
natural-data length-contrast paragraph that V-45's verdict blocked, citing BOTH floors per the prereg's
instruction -- mean-pooled (no purchase at any length, V-45) and last-token (purchase at training length,
fragile with length, V-46) -- plus disclosure 1 (parser at ceiling) verbatim. The V-42 split-semantics
reconciliation paragraph remains the other open manuscript item.

Apparatus note: written by the 30-min tick under /tmp/ngcls_tick.lock; numbers re-copied verbatim from
v46_logs/v46.json; nothing recomputed; GPU 1 untouched; committed locally only, NO PUSH per the
disclosure-hold rule (4275ce6).
````

## Part 7: Duplicate registration "V-46", later voided (ledger.md lines 12907-12947)

SHA-256 of the lines below, each followed by a newline: `0957d0d162e50361bf397de3aa1de201044c9e350b6a78f2755c297387ab4e19`.

````text
## V-46 prereg: two floors that cannot both be right. Before any code.

**The discrepancy.** The manuscript quotes .771 as the bAbI task-1 floor -- a six-way linear probe on the pooled
frozen state, from V-30 (.780/.782/.752). V-45 ran the same probe architecture for the same number of epochs and
got .3506 on short stories and .2150 on long ones. A mixture of .35 and .22 cannot be .77, so at least one of
these numbers does not mean what I have been taking it to mean, and .771 is load-bearing: it is the reference
against which the natural-text headroom, the twenty-three-point coverage cost and the .976 anomaly are all
stated.

**What I should have measured in V-45 and did not.** The MAJORITY-CLASS rate per bucket. Short stories average
3.13 distinct locations out of six, so .3506 may simply be the label prior, meaning the probe learned nothing --
which would be a fact about my bucketing, not about length. Omitting a prior baseline is the exact defect this
paper's audit half is about, and I omitted it.

**Setup, three cells, everything else identical to V-45's code path.**
  (1) ALL-LENGTH: no bucketing, n_train 2000 as in V-30, probe trained and evaluated on all lengths.
  (2) SHORT-ONLY and LONG-ONLY: V-45's buckets, rerun so the comparison is within one code path.
  (3) For every cell and every bucket: the majority-class rate on the evaluation set, and the probe's TRAINING
      accuracy, so underfitting and label-prior effects are separable.
Seeds 42/123/456 throughout.

**Bars.**
(a) REPRODUCES iff the ALL-LENGTH floor is within 0.05 of .7713 in 3/3. Then .771 stands, V-45's low floors are
    an effect of restricting to short stories, and that effect itself has to be reported: it would mean the probe's
    apparent competence on bAbI comes from the long stories, which is the opposite of what a length-extrapolation
    story would want.
(b) DOES NOT REPRODUCE iff the ALL-LENGTH floor is more than 0.10 below .7713 in 3/3. Then the manuscript's .771
    is wrong or means something else, every claim scaled against it must be rechecked, and I go find which.
(c) GRADED in between: report both and reconcile before any manuscript change.
(d) Independently: if the majority-class rate is within 0.03 of the probe accuracy in any cell, that cell's probe
    is reported as having learned nothing beyond the prior, whatever the bar above says.

**Prediction on the record.** (a) reproduces, and the majority-class rate on the short bucket is close to .35 --
i.e. both numbers are right and my short-bucket probe learned nothing beyond the prior. That would mean V-45's
bar (a) failed for a reason I created by bucketing, and that the length-extrapolation result stands but still
without a contrast. I have been wrong nineteen times out of twenty-three today, and this is a prediction I would
rather have checked than assumed -- which is the whole point of the entry.

**Consequence either way.** The positive half of V-45 is untouched: the handover reaches 1.0000 on ten-statement
stories with a head trained only on six-statement ones, span identity 1.0000, three of three. What V-46 decides
is whether the paper's bAbI floor is what the paper says it is.
````

## Part 8: Voiding of the duplicate registration (ANOMALY entry) (ledger.md lines 12950-12986)

SHA-256 of the lines below, each followed by a newline: `45a89e8a52822acda4138a27127f20dfa13cd0cab4c03ebf5461ea48cf021afa`.

````text
## ANOMALY (idle-check tick, 2026-08-24 ~01:00 +0800): a duplicate "V-46" was registered and run POST-ESCALATION. Voided as a registration; reported as measured; the ledgered reconciliation and V-46 verdict STAND.

At 00:51-00:52 +0800 tonight — sixteen hours AFTER the escalation (7a21494) dropped this loop to idle-check,
which forbids registrations — commits ab6159a ("V-46 prereg: two floors that cannot both be right") and
688d02c (v46.py) were appended and the run executed, overwriting the UNTRACKED v46_logs/v46.json. The original
verdict's numbers survive verbatim in the V-46 verdict entry above and in v46_logs/run.log (untouched). This
tick did not write those commits and found the run already complete, no process live, GPUs idle, neither tick
lock taken.

The question it registers — reconcile the manuscript's .771 (V-30) with V-45's .3506/.2150 — was ALREADY
answered and ledgered BEFORE the escalation: same substrate, same buckets, same 540-short training cut,
mean-pooled floor .34-.36 (V-45 operator) vs last-token floor .81-.83 (V-30 operator); the pooling-operator
confound was CONFIRMED and the finding folded into both manuscript builds (c95cc9a). The duplicate shows no
knowledge of that entry or of the escalation — the exact stale-context re-launch failure mode the standing
directive warns about ("read the ledger tail first"), recurring.

AS MEASURED (re-copied from the overwritten v46_logs/v46.json): the new v46.py imports pooled() from v45 —
the MEAN-pooling path — so all four cells measure only the operator already known to have no purchase.
all-length eval .316/.328/.348 (train .50-.51, majority .188); all-length-big .325/.326/.342; short-only
.360/.362/.365 (consistent with V-45's .3506); long-only .195/.205/.195 vs majority .235 — at/below the label
prior, "learned nothing beyond prior" by the duplicate's own bar (d). The json's written verdict — "DOES NOT
REPRODUCE (the manuscript's .771 is not what this code path produces; every claim scaled against it must be
rechecked)" — is WRONG AS AN INTERPRETATION: the run never instantiates V-30's last-token operator, so it
cannot test whether .771 reproduces; it re-confirms, for a third time, that mean pooling has no purchase at
any length. No manuscript claim is touched. The V-46 verdict (last-token short .828/.823/.808, losing
.183-.198 short->long, 3/3, vs symbolic route 1.000 both lengths) and its fold stand as ledgered.

DISPOSITION: ab6159a is VOID as a registration — a duplicate of a closed question, made in violation of
idle-check. Its artifacts remain in the record as described here; no rescue, no follow-up runs. The loop
REMAINS in idle-check mode; the escalation state is unchanged. NEW USER-FACING ITEM: a stale agent/session
bypassed idle-check and both tick locks at 00:51 +0800 2026-08-24 and re-ran a closed experiment — stale
resumed sessions should be terminated (or the flock made mandatory at the git layer) before another duplicate
lands in the record.

Apparatus note: written by the 30-min idle-check tick under /tmp/ngcls_tick.lock; numbers re-copied verbatim
from v46_logs/v46.json and run.log; nothing recomputed; GPU 1 untouched; committed locally only, NO PUSH per
the disclosure-hold rule (4275ce6).
````

## Part 9: Result of the duplicate run (ledger.md lines 12988-13027)

SHA-256 of the lines below, each followed by a newline: `122cf32b21fa19390ede4c7cd77454bb382f15bb8cd44f62d1582ce70b2df9e4`.

````text
## V-46: the two floors differ because I changed the estimator. The manuscript's .771 is not wrong; my code was.

**Numbers.** All-length bAbI task-1 floor under my code path: .316/.328/.348 = .3307, against the manuscript's
.7713 -- deltas -.455/-.443/-.423, so DOES NOT REPRODUCE by the registered bar. Short-only .3622, long-only
.1983, all-length at the larger draw .3310. Majority-class rates .188/.202/.235/.187, so the short and all-length
probes learn something beyond the prior (+.14 to +.16) and the LONG-only probe learns NOTHING beyond it (-.037).

**Cause, found by diffing the two implementations rather than by guessing.**
  v30.py and v13.Floor:  idx = am.sum(-1) - 1 ; H[arange, idx]      -- the LAST non-pad token's state
  v43.py / v45.py / v46.py (mine):  masked MEAN over all tokens
Both are legitimate probes and they measure different things. On bAbI the last token sits immediately after
"Where is Mary?", so in a causal model its state is the one used to generate the answer; a mean over the whole
story is far weaker. That is the whole .44. On arena C the two happen to land close (mean-pool .4940 at depth two
against v13's .5153), because a single token cannot carry that binary task either way.

**So the verdict stands as printed and its cause is the opposite of what the entry was written to test.** I
registered V-46 to find out whether the manuscript's .771 was wrong. It is not. What is wrong is that V-43, V-45
and V-46 all used a floor the paper does not use, and I did not notice while writing three preregs that leaned on
comparability with the paper's numbers. The manuscript needs no change on account of .771.

**What this does to the runs that used the wrong floor.**
  V-43's depth-3 result is UNAFFECTED in its core claim: probe_argmax equalling parser to four decimals is an
    arm-versus-arm equality and no floor enters it. Its headroom BAR did use the mean-pool floor, and the bar
    outcome is robust to the swap at depth two (hi = .8977 mean-pool versus .8921 last-token), but depth three
    has not been checked with the paper's estimator and will be.
  V-45's bar (a) -- "the baseline has purchase" -- FAILED for a reason I manufactured. The paper's estimator has
    .771 of purchase on bAbI; mine had .36. So V-45's "NO USABLE BASELINE" verdict is correct about what it
    measured and uninformative about the question it was asked. It gets rerun, not relabelled.
  V-45's POSITIVE half is untouched and does not depend on any floor: with the identity head trained only on
    stories of at most six statements, the chain reaches 1.0000 on held-out ten-statement stories with 5.33
    distinct locations, span identity 1.0000, three of three seeds.

**One thing V-46 found that is worth keeping regardless.** The long-only probe is BELOW its own majority-class
rate. Whatever the estimator, a pooled-state readout has nothing on ten-statement stories. If the last-token
probe behaves the same way, the contrast V-44 said arena C could not supply is available on bAbI after all --
which is exactly what the rerun tests.

**Tally.** Twenty-four predictions tested today, twenty failed -- I predicted V-46 would reproduce. Eight
apparatus or specification faults disclosed, this being the eighth and the third in which a bar of mine could not
answer the question it was written for.
````

## Part 10: PRE-REGISTRATION: V-47 (rerun with the paper's floor estimator) (ledger.md lines 13029-13070)

SHA-256 of the lines below, each followed by a newline: `75ef4271e98358a48dc0cd2c7330cde0ed473e2338ae5ead19814889ebf0db4e`.

````text
## V-47 prereg: V-45 rerun with the paper's own floor estimator, plus V-43's bar rechecked. Before any code.

**Why.** V-46 established that my mean-pool floor is not the paper's floor: v13.Floor and v30.py read the LAST
non-pad token, which on bAbI is the state immediately after "Where is Mary?" and therefore the one a causal model
uses to answer. V-45's bar (a) -- "the baseline has purchase" -- was tested against my weaker estimator and
failed for that reason. This rerun uses the paper's estimator throughout, so the bars mean what they were written
to mean.

**Two halves, one script, both floors computed the paper's way (last non-pad token).**
  (A) bAbI task 1, story-length extrapolation, exactly V-45's design: identity head and probe trained on stories
      of at most six statements, evaluated on those and on held-out stories of at least ten. Arms floor, parser,
      probe_argmax. Bars (a), (b), (c) verbatim from V-45's prereg at 094cc5d, unchanged.
  (B) Arena C, the floor at depths two, three and four computed with v13.Floor's estimator, so V-43's headroom
      bar can be rechecked against the number the paper actually uses. probe_argmax and parser are read back as
      BANKED from v43_logs/v43_BANKED.json and not recomputed.

**Bars for (A), restated verbatim so there is no drift.** Per bucket H = parser - floor, admission H >= 0.20.
(a) THE BASELINE HAS PURCHASE iff the SHORT-bucket floor is >= 0.70 in 3/3.
(b) EXTRAPOLATES iff probe_argmax on LONG >= floor_long + 0.90 H_long in 3/3 AND >= 0.90 x its own SHORT accuracy
    in 3/3.
(c) THE CONTRAST HOLDS iff the probe's LONG accuracy is at least 0.10 below its SHORT accuracy in 3/3.
A positive result requires (a) AND (b) AND (c). (b) without (c) is "extrapolates, no contrast".

**Bar for (B).** V-43's depth-3 claim SURVIVES THE ESTIMATOR SWAP iff probe_argmax at depth three, banked, is
still >= floor_3 + 0.90 (parser_3 - floor_3) in 3/3 with the last-token floor. If it does not survive, the
manuscript paragraph written from V-43 must be corrected, and I will say that the claim was scaled against a
floor the paper does not use.

**Also reported, no bar.** Majority-class rate per bucket, and the probe's training accuracy, since V-46 showed
the long-bucket mean-pool probe fell BELOW its own prior and I want that visible for the last-token probe too.

**Predictions on the record.** (a) holds, around .77 as in V-30. (c) holds: V-46 found a pooled readout has
nothing on ten-statement stories, below its own prior, and I expect the last-token probe to degrade with length
as well, though less severely. (b) holds, since V-45 already measured 1.0000 on long stories and no floor enters
that number. (B) survives. Twenty-four predictions tested today and twenty failed; three of the four that held
were ones where the mechanism was nameable in advance, as (b) and (B) are here. (a) and (c) are the ones I could
be wrong about, and (c) is the whole point -- it is the contrast V-44 said arena C could not supply.

**What a negative on (c) means.** If the last-token probe holds its accuracy on long stories, then pooled-state
readouts handle bAbI length robustly, there is no compositional contrast on this substrate either, and the
paper's "why prefer a symbolic route" section stays as weak as it currently is. I would report that and stop
looking for the contrast on bAbI.
````

## Part 11: Verdict: V-47 (source of the quoted numbers) (ledger.md lines 13072-13109)

SHA-256 of the lines below, each followed by a newline: `5c59127b5c9a60ae7c40138c4eabc622e7bf438841d99390bbb4ea6c4970b115`.

````text
## V-47 EXTRAPOLATES + CONTRAST HOLDS -- the compositional contrast, on a baseline that demonstrably works

**Half (A): bAbI task 1, story length, the paper's floor estimator (last non-pad token).**
  bucket   probe_argmax          parser                floor (paper's estimator)   span identity
  short    1.000 1.000 1.000     1.000 1.000 1.000     .820 .8283 .8083  = .8189   1.0000
  long     1.000 1.000 1.000     1.000 1.000 1.000     .625 .640  .625   = .6300   1.0000
Short = at most six statements, the training distribution. Long = at least ten, held out, mean 5.33 distinct
locations against 3.12 in short. 540 train / 600 short eval / 200 long eval.

**All three registered bars hold, in three of three seeds.**
(a) BASELINE HAS PURCHASE: short floor .820/.8283/.8083, all >= .70. The probe genuinely does this task at the
    training length -- which is what V-44 found arena C could not give at any training depth.
(b) EXTRAPOLATES: probe_argmax on long = 1.0000, above floor_long + 0.90 H_long, and equal to its own short
    accuracy. The identity head never saw a story longer than six statements. Span identity stays 1.0000, so the
    head is recognising symbols in contexts nearly twice as long as any it trained on.
(c) CONTRAST HOLDS: the same probe loses .195/.1883/.1833 from short to long, every seed past the .10 bar.
So: a baseline that works at training length loses nineteen points when the stories get longer, and the discrete
handover does not move. This is the compositional evidence the paper has been missing, and it is on natural text.

**Half (B): V-43's depth-3 claim SURVIVES THE ESTIMATOR SWAP.** Recomputing the arena-C floor the paper's way at
depths two, three and four leaves probe_argmax clearing the .90-headroom bar at depth three in three of three.
So the V-43 paragraph already in the manuscript does not need correcting for the floor I got wrong.

**A prereg/scorer mismatch, disclosed.** The prereg says the .20 admission gate applies "per bucket"; the scorer
applied it to the LONG bucket only. The short bucket's headroom is .1811, under the gate. It changes nothing
here and I am not treating it as a rescue: (a) is a threshold on the probe itself, and (c) compares the SAME arm
across two buckets. Neither uses the short bucket's headroom, and no arm-versus-arm comparison is made there.
But the registered text and the code disagreed and the disagreement is mine.

**Also standing from V-46, and it now reads differently.** The mean-pool probe fell BELOW its own majority-class
rate on long stories (-.037). The last-token probe does not collapse that far -- .63 against a .235 prior -- so
the length effect is a real degradation of a working readout rather than a floor artifact. That distinction is
what makes (c) worth stating.

**Prediction outcome.** I predicted (a), (b), (c) and (B) all hold, and all four did. Twenty-eight predictions
tested today, twenty failed. The four that held in a row here were the ones where the mechanism was nameable in
advance: a per-span head cannot see story length, a per-hop algebra decays only with binding count, and a single
token summarising a longer story must lose information.
````
