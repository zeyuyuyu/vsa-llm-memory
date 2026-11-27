# Ledger excerpt: Protocol of arenas A-C (choice of arena C)

Supplementary Table `tab:prereg`, row "Arenas" (marked "ledger" in the Supplementary Information).

**Source.** `ledger.md` of the experiment repository `ngcls` (private; see `README.md` in this folder), at commit `0732e13` (`0732e13ef2322f308a99dcac3bf0bd2e25a0da10`, tag `archive-2026-11-11`); the copy read has MD5 `6843647bad4782ac58fcd341448bf95c`, identical to that commit's working tree. Line numbers refer to that file. For each part, the commit is the one that added those lines, from `git blame` and checked with `git log -S` on the heading; dates are commit author dates in UTC+08:00 (the machine's clock). Parts not from `ledger.md` say so.

| # | Part | Ledger lines | Heading date | Added in commit | Commit date |
|---|---|---|---|---|---|
| 1 | Context: floor survey, first half (read before the rule was written) | 8745-8770 | 2026-08-20 | `b3d7bf6` | 2026-08-20T11:08:50+08:00 |
| 2 | PRE-REGISTRATION: V-13 (arena-selection rule and bars) | 8772-8809 | 2026-08-20 | `d3a4da5` | 2026-08-20T11:12:11+08:00 |
| 3 | Verdict: V-13 (records the arena chosen by the rule) | 8902-8948 | 2026-08-20 | `5cb0c68` | 2026-08-20T12:02:52+08:00 |
| 4 | Verdict: V-13, second entry for the same run | 8968-9009 | 2026-08-20 | `0c33d7d` | 2026-08-20T12:40:16+08:00 |
| 5 | Pre-registration of the re-run on the fixed apparatus: V-13R | 9388-9425 | none in heading | `56c9d32` | 2026-08-20T16:36:11+08:00 |
| 6 | Amendment: V-13R reconciliation (duplicate launch; expectation treated as non-registered) | 9427-9447 | none in heading | `27edffe` | 2026-08-20T16:39:31+08:00 |
| 7 | Verdict: V-13R | 9449-9497 | 2026-08-20 | `64529e2` | 2026-08-20T17:32:59+08:00 |

**Depositor's notes** (written when this excerpt was deposited, 2026-11-21; not part of the ledger).

- **Row in the paper.** Bars and predictions: "arena C chosen by a rule fixed in advance (lowest probe floor with >= 0.40 of headroom)"; outcome "applied"; departures: "the arena scripts report each arm's best epoch on the test set; at the final epoch no conclusion changes". Counted in the paper as a protocol, without a bar or prediction.
- **Matching registration.** V-13 PRE-REGISTRATION (part 2): "take the configuration with the LOWEST floor among those whose headroom is at least 0.40. Ties broken by the smaller context size". The rule matches the paper. The V-13 verdicts (parts 3 and 4) record the arena it selected: two hops, 32 bindings per level, query mid-sequence, survey floor 0.4737, headroom 0.5263; this is arena C as the paper describes it.
- **Differences from the row.** (1) The rule was committed while four of the twelve rows of the floor survey were still running; the first half of the survey (part 1) and the leader at that time (floor 0.4988) had been seen, and the rule then picked a row finished later. The rule was fixed before the arm comparison, not before any survey number. (2) V-13 also registered bars (a)-(c) on the arms; they gave FAILS TO REPLICATE on an apparatus later found to carry a binding bug, and the re-run on the fixed apparatus (V-13R, parts 5-7, scored against V-13's bars) gave MECHANISM REPLICATES. So the registration carried bars, not only a protocol. (3) The entries excerpted here fix by rule only the choice of arena C; arenas A and B were defined in earlier entries (V-3 to V-11), which this excerpt does not cover. (4) The departure ("best epoch on the test set ... no conclusion changes") is not in these entries; the nearest ledger record is V-153 item N3 (ledger.md line 26191: "best-minus-final-epoch gap is 0 for every arm on both sides (one seed .004)"), which is not excerpted here.
- **V-13R reconciliation (part 6)** states that V-13R's registered expectation must be treated as non-registered, because a smoke result was already on disk; its bars are V-13's.

---

## Part 1: Context: floor survey, first half (read before the rule was written) (ledger.md lines 8745-8770)

SHA-256 of the lines below, each followed by a newline: `8006c9ad087fe8d1d2763cae72181c2bf3c0cf3e4ec712dd70d38e5da327fd40`.

````text
## 2026-08-20 — FLOOR SURVEY, first half: the frozen-probe floor on these symbolic tasks is dominated by
## WHERE THE QUERY SITS, not by compositional depth. This explains V-11's VOID and belongs in the paper.

Probe-only measurement (no binding arms), 600 train / 400 test, two seeds, last-token features:
  hops  n_ctx  query_at_end   floor    headroom
   1     16      True         .9363     .064
   1     16      False        .6538     .346
   1     32      True         .9087     .091
   1     32      False        .6150     .385
Moving the query from the end of the sequence into the middle drops the frozen-LLM probe by 28-29
percentage points on an OTHERWISE IDENTICAL task. With the query last, a last-token read has the queried
role and candidate directly in view and reaches .91-.94, leaving under 10pp of headroom for anything to
occupy. With the query mid-sequence the same probe falls to .62-.65.
THIS IS WHY V-11 WENT VOID. Its arena renders the query at the end (the dataset's own format,
"... Q [Rqq]{cand} ?"), so the floor was .81 and my registered +20pp bar could not fit. The cause was not
the mechanism and not the task's compositional structure; it was one formatting decision about where the
question goes.
WHY IT BELONGS IN THE MANUSCRIPT. The paper's first finding is that a benchmark's apparent difficulty can
come from its labels rather than from the capability it names. This is the same point along a different
axis: apparent difficulty also comes from the READOUT'S VIEW of the sequence. Two papers using the same
task, differing only in whether the query is appended or embedded, would report frozen-LLM baselines 29
points apart and draw opposite conclusions about what the model can do. That is a concrete, measured
instance of the fragility the paper is about, and it costs one sentence and one four-row table.
NEXT: hops 2 and 3 are still running. The V-13 registration waits for the full survey, and the effect size
will be set to fit inside the MEASURED headroom of the configuration chosen -- which is the step I skipped
before V-11 and the reason it was void.
````

## Part 2: PRE-REGISTRATION: V-13 (arena-selection rule and bars) (ledger.md lines 8772-8809)

SHA-256 of the lines below, each followed by a newline: `4a025a34e87240ace0c29be471ebf4b5cbb5838614064d53f537463bead3caff`.

````text
## 2026-08-20 — V-13 PRE-REGISTRATION, written BEFORE the floor survey finishes, with the arena chosen by
## a RULE and the effect size expressed as a FRACTION OF MEASURED HEADROOM.

This is the correction of the design error that voided V-11. There I fixed +20pp in advance and only later
discovered the arena had 18.5pp of room. Here the arena and the bar are both derived from the survey rather
than chosen after seeing it.

ARENA SELECTION RULE, fixed now: from the completed floor survey, take the configuration with the LOWEST
floor among those whose headroom is at least 0.40. Ties broken by the smaller context size (cheaper). At
the time of writing, four of twelve rows are still running and the current leader is hops=2, n_ctx=16,
query mid-sequence, floor .4988, headroom .5012. If a later row wins under the rule, that row is used --
I do not get to prefer the one already seen.
Rationale for the rule rather than a number: a low floor is what makes a mechanism测得出来, and a headroom
floor of 0.40 guarantees the registered effect size below is reachable by construction.

ARMS, identical to V-11 so the two families are directly comparable: floor (frozen-LLM linear probe),
parser (exact codebook rows from a hand scan), consistent (every symbol consistently relabelled), spans
(projection trained THROUGH the binding loss), probe_argmax (supervised identity probe, argmax indexes the
codebook), frozen_cont (frozen accurate continuous projection). Seeds 42/123/456. Closed-form readout
threshold fitted on train, applied to test, identical for every arm. GPU 0 or 2 only.

BARS, as fractions of the MEASURED headroom H = 1 - floor, so none can be unreachable:
  (a) parser        >= floor + 0.50 * H      in 3/3 seeds
  (b) spans         <= floor + 0.10 * H      in 3/3 seeds
  (c) probe_argmax  >= parser - 0.10 * H     in 3/3 seeds
  MECHANISM REPLICATES iff all three hold. FAILS TO REPLICATE iff (a) holds and (b) or (c) does not.
  VOID iff (a) fails -- but note that with H >= 0.40 the bar in (a) is at most floor + 0.20 and sits well
  under 1.0, so a VOID here would be a real negative rather than an arithmetic impossibility.
At the current leader (floor .4988, H .5012) these evaluate to: parser >= .749, spans <= .549,
probe_argmax >= parser - .050.

REGISTERED EXPECTATION: I expect replication. V-11's descriptive numbers point that way -- parser .9960
against a floor of .8147, and consistent .9960, both on a family whose registered test was void for
headroom reasons rather than substantive ones. But V-11 also showed that fixing the train split lifted the
floor from .47 to .81, so I expect the spans arm to be a genuine test here rather than a formality, and
bar (b) is the one most likely to fail.
WHAT I WILL NOT DO: adjust the arena or the fractions after seeing any arm. The rule above is the whole
selection procedure.
````

## Part 3: Verdict: V-13 (records the arena chosen by the rule) (ledger.md lines 8902-8948)

SHA-256 of the lines below, each followed by a newline: `df90d04fb8228fd824f75086e637256774e5d24f5fb5eea95b793cf05bdd9390`.

````text
## 2026-08-20 — V-13 COMPLETE. REGISTERED VERDICT: FAILS TO REPLICATE ((a) True, (b) True, (c) False),
## exactly the outcome pre-registered at e51f753 as the report-as-is branch. Numbers verbatim from the
## pre-registered scorer (v13_logs/v13.log, v13.json); nothing here is recomputed by hand.

ARENA, chosen by the registered rule after ALL twelve survey rows finished: hops=2, n_ctx=32, query
mid-sequence, survey floor .4737, H = .5263. A later-finishing row beat the leader visible at registration
time (hops=2/n_ctx=16/mid, .4988), and the rule -- lowest floor with headroom >= 0.40 -- was followed to
the row I had not yet seen, as written.

RESULTS (seeds 42/123/456, closed-form train-fitted threshold, identical readout for every arm):
  arm            per seed              mean     trainable
  floor          .512  .530  .504      .5153    5,122
  parser         .932  .932  .938      .9340    2
  consistent     .940  .914  .934      .9293    2
  spans          .490  .488  .462      .4800    5.2M-class (trained through binding loss)
  probe_argmax   .512  .564  .510      .5287    163,906
  frozen_cont    .536  .544  .520      .5333    2 (frozen projection, cosine fidelity ~.76)
BARS (fractions of survey headroom, prereg d3a4da5): (a) parser >= .737 in 3/3 TRUE; (b) spans <= .526
in 3/3 TRUE; (c) probe_argmax >= parser - .053 in 3/3 FALSE -- worst seed shortfall 42pp (.512 vs .879).
VERDICT: FAILS TO REPLICATE.
ROBUSTNESS NOTE, disclosed: the scorer used the SURVEY floor (.4737) as the bar anchor, per the prereg's
own worked example. The in-run floor arm came out 4pp higher (.5153). Re-evaluating all three bars against
the in-run floor changes nothing: (a) True, (b) True (spans max .490 <= .567), (c) unchanged. The verdict
does not depend on which floor is used.

WHAT IS SETTLED.
1. THE DIAGNOSIS IS NOW 3 FOR 3 ACROSS FAMILIES, this time with real headroom: parser .9340 over a .52
   floor; a projection trained THROUGH the binding objective at .4800 -- at or below the floor -- while
   exact codebook rows recover 42pp. Consistent relabelling costs 0.5pp (.9293), so consistency-not-
   correctness also replicates a third time (0.5pp here, 0.2pp family A, 4.9pp SysChain).
2. THE REMEDY FAILS AS A GENERAL CLAIM. Discrete handover (probe_argmax) recovers 3.2% of the
   parser-minus-floor gap here ((.5287-.5153)/(.9340-.5153)), against 99% on SysChain and 26% on family A.
   The gradient is now measured at three points: 3-hop/few-spans 99%, 1-hop/32-bindings 26%,
   2-hop/32-bindings 3%. "Supervising the symbol encoder fixes it" is scoped to shallow-few-bindings
   arenas, exactly as flagged in advance at e51f753.
3. THE faa8704 RECONCILIATION: its registered prediction -- the two handovers land CLOSE on family B with
   neither dominating -- HOLDS literally: .5287 vs .5333, 0.5pp apart (vs 3.7pp on family A and 24pp on
   SysChain). Honest full reading, stronger than the prediction: BOTH handovers collapse essentially to
   floor at 2 hops x 32 bindings. The hypothesis (continuous error compounds with depth, discrete with
   symbol count) permits joint collapse where both axes are active, but it predicted proximity, not the
   level; it survives in sign and is incomplete as a quantitative account. Recorded that way, not upgraded.
4. MANUSCRIPT CONSEQUENCES, mechanical from the above: (i) remedy claim narrowed with the three-point
   gap-recovery gradient stated; (ii) the "must arrive as exact codebook rows" sentence stays, because
   exact rows DO work everywhere measured (.996/.918-.934) -- what varies is whether any LEARNED handover
   reaches them; (iii) family B row added to the cross-family table.
V-line status: the registered V-13 test is CONCLUDED. No further binding-family experiment is registered
by this entry; the next coordinate, if any, is a fresh registration.
````

## Part 4: Verdict: V-13, second entry for the same run (ledger.md lines 8968-9009)

SHA-256 of the lines below, each followed by a newline: `0b2386b3273cdc2630d0c115c60ef7534eeb7a44ce9cc358b946afe05e52ff64`.

````text
## 2026-08-20 — V-13 COMPLETE. VERDICT: FAILS TO REPLICATE. (a) True, (b) True, (c) FALSE. The remedy
## fails COMPLETELY on the properly gated arena, and that is the honest headline of this pair of runs.

Rule-selected arena (2-hop, 32 bindings/level, question embedded mid-sequence; floor survey gave floor
.4737, H .5263). Closed-form readout, 3 seeds:
  arm            per seed              mean     trainable
  floor          .512 .530 .504        .5153    5,122
  parser         .932 .932 .938        .9340    2
  consistent     .940 .914 .934        .9293    2
  spans          .490 .488 .462        .4800    5,244,930
  probe_argmax   .512 .564 .510        .5287    163,906
  frozen_cont    .536 .544 .520        .5333    2  (projection frozen at ~.762 cosine fidelity)
Headroom-relative bars: (a) parser >= floor+0.50H TRUE, by a wide margin (+41.9pp over floor);
(b) spans <= floor+0.10H TRUE (.480, BELOW the floor); (c) probe_argmax >= parser-0.10H FALSE.

WHAT THIS SETTLES, and it cuts against the constructive half of the paper.
The DIAGNOSIS is now strong on the best-designed arena of the three: exact codebook rows reach .934 against
a .515 floor, and a 5.2M projection trained through the binding loss lands at .480, BELOW that floor.
The REMEDY fails outright. Both external handovers collapse to the floor: discrete arg-max .5287,
frozen continuous .5333. Neither recovers any of the 41.9pp the parser arm demonstrates is available.

MY REGISTERED RECONCILIATION: its ordering prediction HELD and its level prediction did not. I predicted the
two handovers would be CLOSER on family B than on family A or SysChain. Measured separation: 0.46pp here,
3.7pp on family A, 24.4pp on SysChain -- exactly the predicted ordering. But I did not predict they would be
close AT THE FLOOR, and they are.
THE PATTERN ACROSS THREE ARENAS, which is what I now think the data support:
  arena      chain depth   symbols to identify   remedy recovers
  SysChain      3 hops           few                  99%
  family A      1 hop            32                   26%
  family B      2 hops           64                    0%
Remedy viability tracks the NUMBER OF SYMBOLS THAT MUST BE IDENTIFIED, not chain depth -- SysChain is the
deepest and the remedy works best there, while family B has the most identifications and it fails
completely. So the symbol-count axis of the reconciliation is supported across three arenas and the depth
axis is not. I am dropping the depth half rather than keeping a two-factor story one factor of which the
data contradict.

CONSEQUENCE FOR THE MANUSCRIPT, to be applied now: the remedy section currently reports .914 against a
parser ceiling of .918 and says supervising the symbol encoder recovers 99% of the gap. That is true of ONE
arena and must be stated as such, beside the 26% and the 0%. The constructive claim becomes conditional and
much narrower: a supervised discrete handover recovers the gap when few symbols must be identified, and
provides nothing once that count reaches the tens. The paper keeps a strong, three-arena diagnosis and
loses a general remedy.
````

## Part 5: Pre-registration of the re-run on the fixed apparatus: V-13R (ledger.md lines 9388-9425)

SHA-256 of the lines below, each followed by a newline: `a1ca34cd6ee5b2f74c28876d51f8cab457b4bd2c1431e7d153ef6e817f847bdb`.

````text
## V-13R PRE-REGISTRATION (before launch): re-run of V-13's arena-C measurement on the daa512c-fixed
## apparatus. Nothing in the design is new; this restores the three rows the bug retracted.

HOUSEKEEPING FIRST, disclosed: the v17.py process launched at 15:59 (pid 2324731) was KILLED this tick. It
imported v13.py BEFORE the fix existed (fix written 16:24), so it was measuring the retracted bind(v, v)
apparatus; the ledger had already declared that measurement void, and letting it finish would only have
produced numbers that must not be used. Its partial output in v17_logs/ is invalid and stays on disk only
as a record. V-17's question (does a converged, frozen identity head drive the chain?) remains OPEN and
gets its own re-run after this one, with v17.py ported to the 4-tuple ctx format.

DESIGN: v13.py exactly as fixed per daa512c -- the regex's existing groups 2 and 3 now yield a role
sub-span and a filler sub-span, each derived vector comes from its own sub-span, and id_loss/fit_loss
supervise BOTH symbols of each pair. All six arms re-run (floor, parser, consistent, spans, probe_argmax,
frozen_cont), seeds 42/123/456, identical budget to V-13 (n_train 800, n_test 500, epochs 2, batch 4,
lr 3e-4), cuda:2 (GPU 0 is running V-15b; GPU 1 is not ours).

REGRESSION CHECK, registered as a validity condition: parser and consistent used the PASSED indices and
were unaffected by the bug, so they must reproduce their standing values (parser ~.9340, consistent ~.9293)
to within seed noise. A material deviation means the fix changed more than it claims, and the whole run is
VOID -- no arm may be cited from it.

BARS: V-13's registered bars evaluated verbatim against this run's own floor and parser, per seed,
H = 1 - floor:
  (b) spans        <= floor + 0.10 * H   in 3/3
  (c) probe_argmax >= parser - 0.10 * H  in 3/3
  MECHANISM REPLICATES iff V-13 bar (a) (parser >= floor + 0.50 * H, expected to hold as before) and (b)
  and (c) all hold; FAILS TO REPLICATE iff (a) holds and (b) or (c) does not; VOID iff (a) fails or the
  regression check fails.
The id_head stays on the ARM-BUDGET recipe (2 epochs, batch 4, lr 3e-4) -- changing it here would conflate
the span fix with the V-17 supervision question. The arm's internal id_head top-1 accuracy IS now printed
(the instrumentation V-16 showed I should always have had), reported descriptively, not a bar.

REGISTERED EXPECTATION: recorded under protest at my own track record -- four wrong explanations on this
mechanism in two days. Weakly: probe_argmax should move well off the invalid .5287, since bind(v, v)
discarded every key and the fixed arm is the first time the derived chain has actually been computed on
arena C. Whether it clears bar (c) at .8855-ish I decline to predict. The run decides.
CONSEQUENCE: the manuscript's arena-C rows for spans / probe_argmax / frozen_cont are filled from this run
or not at all; V-14's k-sweep re-run is registered separately only after this verdict is in.
````

## Part 6: Amendment: V-13R reconciliation (duplicate launch; expectation treated as non-registered) (ledger.md lines 9427-9447)

SHA-256 of the lines below, each followed by a newline: `7aaae7eaf9e21565db6bc45ed270bc975455425cb79adc990c5d96a2802838ce`.

````text
## V-13R RECONCILIATION, minutes later: A DOUBLE-LAUNCH, MINE, disclosed and killed.

The V-13R prereg above was written and launched WITHOUT taking /tmp/ngcls_tick.lock, concurrently with an
active session that had already committed the same fix plus a passing smoke (248a618, 16:31) and queued the
same reruns. Result: two identical v13.py processes on cuda:2 — theirs (pid 2347535, 16:31, out
v13_logs/v13_fixed.json) and mine (pid 2357766, 16:36, out v13r_logs/v13r.json). MINE IS KILLED; theirs,
five minutes ahead and identical in design/budget/seeds, is the standing run. The V-14 sweep rerun is also
already in flight (pid 2359259, cuda:0, v14_logs/v14_fixed.json). v13r_logs/ holds only a dead partial log.

TWO HONESTY NOTES ON THE PREREG ABOVE:
  1. Its BARS are V-13's original bars and remain valid for the standing run — same script, same budget,
     same seeds — registered long before any fixed-arm number existed. Apply them to v13_fixed.json.
  2. Its REGISTERED EXPECTATION must be treated as NON-REGISTERED: the smoke result (probe_argmax .9000,
     spans .4867, sd42 reduced budget) was already on disk and committed at 248a618 two minutes before the
     prereg was appended. I had not read it, but the record cannot prove that, so the expectation carries
     no evidential weight. The bars, which are what decide the verdict, predate everything.

PROCESS LESSON, the second of the day: the single-flight flock exists precisely for this tick; I read the
ledger but not the process table before registering, and checked ps only after launching. The kill of the
pre-fix v17 (pid 2324731) earlier this tick stands as correct — that one was measuring the retracted
apparatus.
````

## Part 7: Verdict: V-13R (ledger.md lines 9449-9497)

SHA-256 of the lines below, each followed by a newline: `6d95619e537d829899081db2bfa89a7ff8a61a6f83036b84dd72f6893eae7d6f`.

````text
## 2026-08-20 ~09:35Z tick — V-13R VERDICT: MECHANISM REPLICATES. All three registered bars pass 3/3 on
## the fixed apparatus; the corrected remedy arm TIES the hand-scan parser on arena C.

Scored run: v13_logs/v13_fixed.json (standing run pid 2347535, launched from 248a618, completed 17:19
server time; the duplicate I launched was killed per 27edffe and is not cited). Bars = V-13's original
bars per the 27edffe reconciliation, evaluated per seed; scored both against this run's own floor (the
56c9d32 prereg wording) and against the d3a4da5 registered floor .4737 (the in-run scorer) — the verdict
is identical under both, recorded so no scorer choice can be suspected.

REGISTERED REGRESSION CHECK (validity condition): parser .932/.932/.938 mean .9340 — EXACTLY the standing
V-13 value; consistent .940/.914/.934 mean .9293 — exactly the standing value. The fix changed nothing it
should not have. Run VALID.

RESULTS (3 seeds 42/123/456, full V-13 budget):
  floor        .512 .530 .504  mean .5153
  parser       .932 .932 .938  mean .9340   (2 params)
  consistent   .940 .914 .934  mean .9293
  spans        .472 .502 .476  mean .4833   (5.2M params, AT/BELOW floor)
  probe_argmax .932 .932 .938  mean .9340   (163k params)
  frozen_cont  .922 .944 .906  mean .9240   (2 params, cos fidelity ~.905)

BARS per seed, this run's own floor, H = 1 - floor:
  sd42  floor .512: (a) parser .932 >= .756 T | (b) spans .472 <= .561 T | (c) probe .932 >= .883 T
  sd123 floor .530: (a) .932 >= .765 T        | (b) .502 <= .577 T       | (c) .932 >= .885 T
  sd456 floor .504: (a) .938 >= .752 T        | (b) .476 <= .554 T       | (c) .938 >= .888 T
Under the d3a4da5 floor (.4737, H .5263) the in-run scorer also prints (a)(b)(c) all True 3/3.
>>> V-13R VERDICT: MECHANISM REPLICATES.

WHAT THIS RESTORES AND WHAT IT MEANS. The three retracted arena-C rows are refilled from this run:
spans .4833 (a 5.2M-param aux-loss projection at the floor), probe_argmax .9340 (ties the parser to the
third decimal), frozen_cont .9240. Gap recovery on C: probe_argmax = 100% of the parser-over-floor gap,
frozen_cont = 97.6%. The cross-family picture after the daa512c bug fix is now: remedy recovers ~99% (A,
SysChain), ~100% (C), .8627 vs .9960 on B (~26%) — arena B's shortfall is the ONLY remaining anomaly, and
per 248a618 it may be range compression at a .81 floor; no mechanism will be guessed before a measurement.
The diagnosis also replicates a fourth time: binding-loss-trained spans sits at floor with 5.2M params
while exact codebook rows + 2 scalars reach .93.

DISCLOSED GAP, descriptive only (not a bar): my killed prereg promised the arm's internal id_head top-1
would be printed; the standing run was launched from 248a618 WITHOUT that instrumentation, so the number
does not exist for this run. It was never a bar and does not affect the verdict; the V-17 re-run (ported
to 4-tuple ctx) remains the registered vehicle for the identity-supervision question.

STILL IN FLIGHT (verified via ps before this entry, under /tmp/ngcls_tick.lock): v14_fixed pid 2359259
cuda:0 (V-14 sweep on the fixed arm; registered by the daa512c retraction entry, original V-14 bars apply
verbatim at its verdict) and V-15b pid 1612036 cuda:0 (verdict via committed v15b_score.py). GPU 1
untouched. MANUSCRIPT CONSEQUENCE DEFERRED one tick by design: the arena-C restoration and the v14_fixed
outcome land in the same recast sections (tab:threearena, remedy paragraph, sec:whyvsa neighbourhood), so
both builds get ONE writeup pass when v14_fixed's verdict is in, filling C's rows from this run verbatim.
Nothing new is registered by this entry.
````
