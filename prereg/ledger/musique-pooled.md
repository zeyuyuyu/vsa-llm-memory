# Ledger excerpt: Store under pooled facts on MuSiQue

Supplementary Table `tab:prereg`, row "MuSiQue (store under pooled facts)" (marked "ledger" in the Supplementary Information).

**Source.** `ledger.md` of the experiment repository `ngcls` (private; see `README.md` in this folder), at commit `0732e13` (`0732e13ef2322f308a99dcac3bf0bd2e25a0da10`, tag `archive-2026-11-11`); the copy read has MD5 `6843647bad4782ac58fcd341448bf95c`, identical to that commit's working tree. Line numbers refer to that file. For each part, the commit is the one that added those lines, from `git blame` and checked with `git log -S` on the heading; dates are commit author dates in UTC+08:00 (the machine's clock). Parts not from `ledger.md` say so.

| # | Part | Ledger lines | Heading date | Added in commit | Commit date |
|---|---|---|---|---|---|
| 1 | PRE-REGISTRATION: V-79 (subset rule, P1-P4) | 14536-14585 | none in heading | `1fd4489` | 2026-08-27T21:43:40+08:00 |
| 2 | Deviations: V-79, registered before the scored run | 14587-14609 | none in heading | `2e3499e` | 2026-08-27T21:49:17+08:00 |
| 3 | Verdict: V-79 result | 14624-14680 | none in heading | `93c249b` | 2026-08-27T22:11:01+08:00 |
| 4 | Correction to the V-79 result | 14682-14707 | none in heading | `db8dc6e` | 2026-08-27T22:13:37+08:00 |
| 5 | PRE-REGISTRATION: V-79a (chain-depth prediction) | 14709-14720 | none in heading | `db8dc6e` | 2026-08-27T22:13:37+08:00 |
| 6 | Verdict: V-79a | 14722-14758 | none in heading | `7251842` | 2026-08-27T22:17:18+08:00 |
| 7 | PRE-REGISTRATION: V-131 (P3, untruncated MuSiQue single pass) | not in ledger.md: `v131_prereg.md` | 2026-09-14 | `6a2a52d` | 2026-09-14T16:50:46+08:00 |
| 8 | Verdict: V-131 (P3 holds) | 23771-23826 | 2026-09-14 | `ef1be8c` | 2026-09-14T20:56:41+08:00 |

**Depositor's notes** (written when this excerpt was deposited, 2026-11-21; not part of the ledger).

- **Row in the paper.** "subset fixed by a rule before the run; prediction: chain depth explains the pooling loss"; outcome "prediction failed". The Supplementary Note also quotes an untruncated Qwen3-8B single pass (0.288) and an untruncated pooled prompt (0.579).
- **Matching registrations.** The subset rule is in V-79 (part 1: "An instance is EXCLUDED if any hop question lacks `>>`"); the chain-depth prediction is V-79a (part 5: "at +150, the 2-hop cell exceeds the 4-hop cell by >= .20"), which missed with the sign inverted (part 6: 2-hop 0.630, 4-hop 0.859). The Qwen3-8B numbers come from V-131 Part C (part 7, a separate file `v131_prereg.md`, not ledger.md), whose P3 (untruncated single pass trails the store by >= 0.20) held (part 8: 0.2877 against 0.9481).
- **Discrepancy: the depth prediction was registered after the pooled result was read.** V-79a is a diagnostic on banked outputs, written after V-79's result (part 3) and its correction (part 4); its registration and its result were committed 3 min 41 s apart (`db8dc6e`, 22:13:37; `7251842`, 22:17:18). It was blind only to the per-hop split.
- **Discrepancy: V-79's own registered pooling prediction failed and is not in the row.** P3: "at +150 real distractor triples, hrr_gold stays >= .70 while prompt_ev falls >= .20 from its gold-only value"; the store reached 0.6557, so P3 missed. P1, P2 and P4 hit.
- **Sample size.** V-79 registered n = 300; the subset rule left 212 of 2,417 validation instances (part 3, scope limit 2). Deviations before the scored run (part 2) changed the paragraph budget and added an unregistered arm.
- **Amendment to V-131.** See `babi-naming-head-vs-lora.md`, part 5 (V-132): only the validity gate changed; it narrowed the void to Part A, so Part C was read.

---

## Part 1: PRE-REGISTRATION: V-79 (subset rule, P1-P4) (ledger.md lines 14536-14585)

SHA-256 of the lines below, each followed by a newline: `51ff2b12a8667e64747e71f8850865bd80f87cf739a21a85d0413cc9626abb41`.

````text
## V-79 prereg: does the 2Wiki store result replicate on a SECOND mainstream benchmark (MuSiQue)? Before any code.

**Why this, now.** The manuscript's headline rests on one mainstream corpus (V-65/V-67, 2WikiMultihopQA).
A referee's first question is external validity: one dataset, one result. The second exposure is that the
large-pool separation (prompt collapses at 150 facts, store holds) is measured on SYNTHETIC pools. V-79
attacks both with real data.

**Why MuSiQue and not HotpotQA.** The store arm consumes gold (subject, relation, object) triples. 2Wiki
ships them as `evidences`. HotpotQA ships only supporting-fact SENTENCE indices, no relation chain, so the
port would have to extract triples itself -- and that is exactly the no-gold regime already refuted twice
(V-66 phi-2, V-68 Qwen-7B: the gain REQUIRES provided evidence). Running HotpotQA would therefore re-measure
a known negative and I decline to spend a run on it. MuSiQue (Trivedi et al., TACL 2022) ships
`question_decomposition`, whose hop questions are literally `"Green >> performer"` with `answer`
`"Steve Hillage"` -- a subject, a relation and an object. It is also harder than 2Wiki and runs to 4 hops.
Probed loadable this session (infra probe only, no reading taken): `dgslibisey/MuSiQue`, validation split.

**Arms (frozen phi-2 throughout, identical to V-65).**
- `prompt`: the instance's 20 paragraphs in context under a token budget, few-shot, greedy.
- `prompt_ev`: the same gold decomposition verbalised as `subject | relation | object` lines in context.
- `hrr_gold`: HRR store over the gold decomposition triples, d=4096, string-identity codes, chain followed
  in decomposition order (hop k's object becomes hop k+1's subject), nearest-neighbour cleanup.

**Triple construction, declared now.** For each hop, subject = text left of `>>`, relation = text right of
`>>`, object = the hop's `answer`; any `#k` in a subject is replaced by hop k's answer. An instance is
EXCLUDED if any hop question lacks `>>`. The exclusion rate is reported, not hidden.

**Metric.** Exact match after V-65's `norm`, on the covered slice = instances whose final gold answer equals
the object of the last hop. n=300, seeds {42,123,456}.

**Pool factor (the real-data attack on the synthetic 150-fact panel).** For `prompt_ev` and `hrr_gold`,
pool in {gold-only, +50, +150} where the added facts are REAL triples drawn from other MuSiQue instances,
never synthetic filler.

**Registered predictions -- each scored HIT or MISS, no reinterpretation.**
- P1: `prompt` < `prompt_ev` < `hrr_gold` in 3/3 seeds.
- P2: `hrr_gold` - `prompt` >= .20 on the covered slice.
- P3: at +150 real distractor triples, `hrr_gold` stays >= .70 while `prompt_ev` falls >= .20 from its
  gold-only value.
- P4 (usability gate): coverage >= .60. If coverage < .60 the covered slice is too thin and I declare the
  port UNUSABLE rather than read a number off it.

**Negative readings, declared in advance.**
- If `hrr_gold` <= `prompt_ev`: the algebra adds nothing over in-context evidence on MuSiQue, the 2Wiki gain
  does NOT generalise, and that scope limit goes into the manuscript.
- If P3 fails: the large-pool separation is specific to 2Wiki/synthetic pools and the manuscript's
  large-pool sentence must be narrowed to say so.
- Condition guard, as in V-65: if `prompt` >= .85 the corpus is too easy here and no gain is claimable.

**What this cannot show.** MuSiQue supplies gold decomposition, so V-79 tests external validity of the
PROVIDED-EVIDENCE result only. It says nothing about self-extraction, which stays refuted.
````

## Part 2: Deviations: V-79, registered before the scored run (ledger.md lines 14587-14609)

SHA-256 of the lines below, each followed by a newline: `12d985921f1a50814ed6ff92777e4d9f52ce131f408ad0fbb1da3b34af935dd5`.

````text
### V-79 deviations, registered BEFORE the scored run (smoke test n=8 only, no reading taken)

The n=8 smoke run was infra validation, not a measurement, and no number from it is transcribed. It exposed
two things worth fixing first, and both cut AGAINST our own claim, so neither can flatter the result:

1. **Paragraph budget 1200 -> 1500 (`PARA_BUDGET`).** V-65's budget was tuned to 2Wiki, which ships
   sentences; MuSiQue ships full paragraphs, and at 1200 tokens the paragraph arm was truncated on 8/8
   instances. A wider window can only HELP the baseline.
2. **New arm `prompt_sup`: the gold supporting paragraphs only, natural text, untruncated.** This is the
   strongest natural-language baseline available and it exists to answer, in advance, the referee question
   "does the store only win because you starved the paragraph arm?". It can only hurt our claim.
   It is explicitly NOT one of P1-P4 and carries no registered threshold; the scorer labels it
   "ADDED (not registered)".
3. Diagnostic only: the store arm was printing `gold-kept 0.000`, a counter it never increments. Now prints
   `n/a`, so no later reader mistakes it for a measurement.

**P1, P2, P3, P4 and all three negative readings are UNCHANGED from prereg 1fd4489.** Note that P1 is
registered against `prompt`, not against `prompt_sup`; if the store beats `prompt` but loses to
`prompt_sup`, P1 passes and the ADDED line fails, and both get reported.

Corpus facts observed while validating the loader (structural, not outcomes): the `>>` exclusion rule cost
0 of 8 instances, hop distribution was all 2-hop in the first 8, and coverage was 8/8. The full run reports
these over n=300.
````

## Part 3: Verdict: V-79 result (ledger.md lines 14624-14680)

SHA-256 of the lines below, each followed by a newline: `712adbd83412a60bedf1fa973dad76e58c601c6691c5b285ff25a3212b3ddc03`.

````text
### V-79 RESULT (prereg 1fd4489, deviations 2e3499e). Scorer output transcribed verbatim first.

```
  prompt           0.1651   (trunc .887 -- MuSiQue's 20 paragraphs exceed phi-2's window)
  prompt_sup       0.4009   (gold supporting paragraphs only, untruncated; ADDED, not registered)
  prompt_ev_p0     0.8994   prompt_ev_p50 0.3475   prompt_ev_p150 0.1871
  hrr_gold_p0      0.9481   hrr_gold_p50  0.9324   hrr_gold_p150  0.6557
  >>> V-79 VERDICT: P1 ordering 3/3: True | P2 gain>=.20: True (0.7830) | P3 pool separation: False
  (hrr@150=0.6557, prompt_ev drop=0.7123) | P4 coverage>=.60: True (1.0000) ; NEGATIVE READING APPLIES:
  the large-pool separation does not reproduce with real distractor triples, so the manuscript's
  large-pool sentence must be narrowed ; ADDED (not registered): strongest natural-language baseline
  prompt_sup=0.4009 vs hrr_gold_p0=0.9481 -> store beats it
```

**Registered predictions: P1 HIT, P2 HIT, P3 MISSED, P4 HIT.**

**P3 MISSED, and I am not reinterpreting it.** I registered P3 as a conjunction: the store must stay
>= .70 at +150 real distractor triples AND `prompt_ev` must fall >= .20. The store landed at .6557, below
my own bar. P3 is MISSED and stands as MISSED.

**But the auto-fired negative reading's WORDING over-claims, and that has to be said rather than edited
into agreement.** Its text is "the large-pool separation does not reproduce". That is not what the numbers
show. At +150 real triples `prompt_ev` fell 71 points to .1871 while the store fell to .6557 -- a 47-point
separation, wider in relative terms than anything in the synthetic panel. What failed is the store's
ABSOLUTE retention against my .70 bar, not the separation. I am leaving the scorer and the registered
threshold untouched; this paragraph is my reading of a threshold I set too optimistically, and the
manuscript will be narrowed on the specific point the data actually support.

**What the manuscript gains (external validity, the referee's first question).** The headline replicates on
a second mainstream multi-hop benchmark: prompt .1651 -> store .9481, a 78.3-point gain, ordering strict in
3/3 seeds. It also survives the strongest natural-language baseline I could build against it: gold
supporting paragraphs, untruncated, score .4009, so the store beats it by 55 points and the margin cannot
be blamed on having starved the paragraph arm.

**What the manuscript must now concede.** Real distractor triples interfere far more than synthetic ones.
On 2Wiki with synthetic 150-fact pools the store was "nearly unmoved" (within five points). Here, with 150
REAL triples drawn from other MuSiQue instances, it drops .948 -> .656, about 29 points. The mechanism is
already in the paper: real triples share entity and relation strings with the gold triples (hub entities,
repeated relations such as "performer", "spouse"), and shared roles are exactly the reuse condition this
paper measures as harmful (-15/-31). So the finding is mechanistically consistent with our own results, and
the "barely degrades" phrasing must be scoped to synthetic pools. This is a genuine narrowing, taken from
a registered negative reading, not a softening.

**Honest scope limits of V-79, to be stated in the paper and not buried.**
1. The registered `>>` exclusion rule cost 2205 of 2417 validation instances (0.912). V-79 measures
   MuSiQue's STRUCTURED-DECOMPOSITION SUBSET, 212 instances, 8.8% of the split -- not MuSiQue whole. This
   is the slice where the paper's own precondition (role-structured evidence is provided) actually holds,
   which is consistent, but it is a subset and will be labelled as one.
2. n=212, not the registered n=300: the validation split ran out of `>>`-form instances. Forced by the
   corpus, not chosen.
3. Coverage was 1.000 by near-construction (MuSiQue's last decomposition hop carries the final answer), so
   coverage is not evidence of anything here, unlike on 2Wiki.
4. `prompt_ev` at +150 had gold facts survive the window only 46-51% of the time (trunc 1.000). The prompt
   collapse is therefore partly a window-capacity effect -- which is the paper's argument, now on real
   data, but it should be reported as the mechanism rather than left implicit.
5. Hop distribution 2-hop 181, 3-hop 5, 4-hop 26; the 3- and 4-hop cells are too thin to read separately
   and no per-hop claim is made.
````

## Part 4: Correction to the V-79 result (ledger.md lines 14682-14707)

SHA-256 of the lines below, each followed by a newline: `b2c25f18b9854215140bb6ef54ca820463f256b8a020ad4411b82010e9abaeff`.

````text
### CORRECTION to the V-79 entry above, before any of it reached the manuscript

The V-79 entry explained the store's .948 -> .656 drop at +150 as "real distractor triples interfere far
more than synthetic ones", and contrasted it with 2Wiki's "synthetic 150-fact pools". **That explanation is
wrong on its facts and I withdraw it.** I checked the two constructions instead of trusting my own sentence:

1. The 2Wiki large-pool panel was never synthetic. The manuscript says so plainly: "Pooling each instance's
   gold triples with fifty other instances' (102--152 facts per store)". Those are REAL 2Wiki gold triples
   from other instances -- the same construction as V-79's distractor pool. So the contrast I drew does not
   exist.
2. My second hypothesis was also wrong. I suspected 2Wiki's cleanup restricted candidates to gold entities,
   which would have made its large-pool number an artifact. It does not: `v72b.py:60` builds candidates from
   the whole pooled store including distractors (`ents = sorted({x for tr in store for x in (tr[0], tr[2])})`),
   and its docstring says "candidates range over the pool". The two protocols agree on this point.

So the .953-vs-.656 gap is a real cross-corpus difference whose cause I have NOT measured. Candidates, in
the order I think likely, none of them asserted:
- **Chain depth.** 2Wiki compositional is uniformly 2-hop. V-79's slice is 2-hop 181, 3-hop 5, 4-hop 26, and
  a 4-hop chain needs four consecutive cleanups to survive, so errors compound.
- **Relation-vocabulary collisions.** MuSiQue's relations repeat heavily across instances ("performer",
  "spouse", "director"), so a 150-triple real pool can contain distractors sharing a relation with the gold
  chain, which is the reuse condition this paper already measures as harmful (-15/-31).
- **Proximity to the capacity shoulder.** At ~152 bindings and d=4096 we sit on the shoulder of the V-73
  curve, where small differences in pool composition move the readout.

**Nothing about this goes into the manuscript until it is measured.** V-79a below is that measurement.
````

## Part 5: PRE-REGISTRATION: V-79a (chain-depth prediction) (ledger.md lines 14709-14720)

SHA-256 of the lines below, each followed by a newline: `2b57b543cfd378dcc64469a710605a3ff17f73c0aefacdf936ea3740fa2a41e7`.

````text
### V-79a prereg: what drives the store's drop at +150 on MuSiQue? Diagnostic, before any code.

Decomposition of an already-scored result, no LLM, CPU only, same banked instances and same seeds. It adds
no new claim about VSA's value; it exists so the manuscript can state a CAUSE or explicitly decline to.

Reported: `hrr_gold` exact match at pools {0, 50, 150} split by hop count (2 / 3 / 4), and, per instance,
the number of distractor triples whose relation string also appears in the gold chain.

**Registered prediction (one, with teeth): P1 -- at +150, the 2-hop cell exceeds the 4-hop cell by >= .20.**
If P1 HITS, chain depth is the dominant factor and the manuscript says so. If P1 MISSES, depth is not the
story, the drop is attributed to pool composition, and the manuscript reports the drop WITHOUT a mechanism
rather than inventing one. The 3-hop cell (n=5) is too thin to read and no claim is made from it.
````

## Part 6: Verdict: V-79a (ledger.md lines 14722-14758)

SHA-256 of the lines below, each followed by a newline: `6ba1ec24386278c4d9e032c9f29db311dc7da2ab52b2479f20ed94d163463f11`.

````text
### V-79a RESULT. Verbatim.

```
  pool+0   overall=0.9481 | 2h n=181 em=0.9521 3h n=5 em=0.6667 4h n=26 em=0.9744 | mean distractors sharing a gold relation=0.0
  pool+50  overall=0.9324 | 2h n=181 em=0.9374 3h n=5 em=0.6000 4h n=26 em=0.9615 | mean distractors sharing a gold relation=5.6
  pool+150 overall=0.6557 | 2h n=181 em=0.6298 3h n=5 em=0.5333 4h n=26 em=0.8590 | mean distractors sharing a gold relation=16.6
  >>> V-79a VERDICT: P1 2-hop exceeds 4-hop by >=.20 at +150: False (gap -0.2291; 2h=0.6298 n=181,
  4h=0.8590 n=26) -> DEPTH IS NOT THE STORY: the manuscript reports the drop WITHOUT a mechanism
```

**P1 MISSED, and the direction is inverted from my prediction.** I predicted the 2-hop cell would beat the
4-hop cell by >= .20 at +150. It LOST to it by .229: 4-hop instances held at .859 while 2-hop fell to .630.
Chain depth does not explain the drop, and whatever is happening runs the other way.

**The registered consequence applies as written: the manuscript reports the +150 drop with NO mechanism.**
I am not launching further runs to hunt for one. The prereg fixed that consequence in advance precisely so
that a missed prediction could not become a licence to keep fishing until something fit.

**One descriptive statistic is bankable, and only as descriptive.** The mean number of distractor triples
sharing a relation string with the gold chain rises 0.0 -> 5.6 -> 16.6 across pools {0, 50, 150}, tracking
the readout's 0.948 -> 0.932 -> 0.656. It is fully confounded with pool size -- this design cannot separate
"more colliding relations" from "more bindings" -- so it may be reported as an observation and must not be
offered as the cause.

**Why I am not chasing the inverted 4-hop cell either.** It rests on 26 instances against 181, it is the
kind of cell that flips on resampling, and I have no registered prediction about it. It goes in the ledger
as an oddity for a future session, not in the paper.

**Net effect on the manuscript, which is now settled:**
- ADD: the headline replicates on a second mainstream benchmark (prompt .1651 -> store .9481, +78.3 points,
  strict ordering 3/3 seeds), and survives the strongest natural-language baseline available
  (untruncated gold supporting paragraphs, .4009) by 55 points.
- NARROW: "nearly unmoved" / "barely degrades" at large pools is a 2Wiki result and gets scoped to 2Wiki.
  On MuSiQue's slice a ~152-binding pool costs the store 29 points (.948 -> .656) while the prompt route
  falls to .187, so the SEPARATION holds at 47 points but the store's near-invariance does not travel.
- STATE AS LIMITS: 212 instances, 8.8% of the validation split, MuSiQue's structured-decomposition subset;
  cause of the drop unmeasured with chain depth explicitly ruled out.
````

## Part 7: PRE-REGISTRATION: V-131 (P3, untruncated MuSiQue single pass) (`v131_prereg.md` as committed in `6a2a52d`)

Redaction: the name of the venue of a separate manuscript of ours, then under review, is replaced by "[venue redacted]" in two places. Nothing else is changed. The file is unchanged between this commit and the repository HEAD.

SHA-256 of the lines below, each followed by a newline: `99580c4e313241ae2858a634695606fda0d13a61eca05cf13d5e52e5b0a56a2e` (after the redaction stated above).

````text

## 2026-09-14 — V-131 PREREG: the three softest points in the npj manuscript, measured instead of conceded (user-directed: "那你补实验吧"; BLIND on P1–P4)

Four audit passes over the npj manuscript on 2026-09-11 left every reported number traceable and every
sentence evidenced, and named three weaknesses that **no amount of rewriting can fix** — each is a property
of the experiment, not of the prose. This run measures all three. Every part is local GPU; **zero API spend**.

**The three attacks, stated as a reviewer would state them.**
1. *"Your fine-tune was never tuned."* The bAbI task-2 comparator is LoRA at rank 8, two epochs, one learning
   rate. The manuscript itself concedes "a larger or longer one may close part of the gap". A twenty-point
   claim resting on an untuned opponent is the single most attackable number in the paper.
2. *"Your residual-halving claim is inside the noise."* $0.974$ against $0.950$ on $n=300$ is about seven
   discordant rows. The manuscript states it as a fact ("halves the residual error, $5.0\%$ to $2.6\%$") with
   no interval, because **the per-row hits were never banked** — V-125 step 5 recorded exactly this gap as
   `"phi2_v65": "per-seed means only, ABSENT"`.
3. *"Your MuSiQue baseline was truncated, not beaten."* The single pass exceeds Phi-2's window on $0.89$ of
   instances and every $+150$ pooled prompt is cut with only $46$--$51\%$ of gold facts surviving. The
   2026-09-11 audit corrected the *attribution* but left the *baseline* weak.

**Parts.** Everything imports the code path it re-measures; nothing is re-implemented.
- **A — per-row banking (`v65.py --dump_rows`).** A purely additive flag: `hits` is already computed per row
  and only its mean was stored; the flag stores the vector too. No computation changes, which P0 verifies.
  Same $300$ compositional rows, same three seeds, same greedy decoding, Phi-2 frozen.
- **B — the tuned opponent (`v131b.py`).** bAbI task 2, `v50.py --arms lora_direct` invoked as a subprocess so
  the instance set, ordering rule and expressible-subset filter are byte-identical to the manuscript's arm.
  Grid: rank $\in \{8, 16, 32\}$ $\times$ epochs $\in \{2, 6\}$ $\times$ lr $\in \{1\times10^{-4},
  3\times10^{-4}\}$, twelve configurations, three seeds each — $36$ runs.
  **Selection is on the test set, deliberately.** There is no validation split in this harness, and inventing
  one would change the training set and break comparability with the manuscript's arm. Selecting the
  comparator's configuration on the very set it is scored on is maximally generous *to the opponent*, so a
  surviving gap survives a fortiori. The generosity is registered here so it is read as a deliberate handicap
  on us and not as leakage we failed to notice.
- **C — the untruncated baseline (`v131c.py`).** MuSiQue, the same row filter as `v79.py` (gated on
  reproducing its banked $n$, hop distribution and coverage), read by **Qwen3-8B at a $32$k window** instead
  of Phi-2 at $1{,}900$ tokens. Two arms: `single_pass` (all twenty paragraphs, supporting first) and
  `prompt_ev_p150` (gold $+150$ real distractor triples). Truncation rate and gold-kept rate are recorded per
  arm; the point of the run is that both should now be $0$ and $1$.

**Predictions.**
- **P0 (validity gate, Part A):** the re-run reproduces V-65's banked per-seed EMs — `prompt` $0.2533$,
  `prompt_ev` $0.9500$, `hrr_gold` $0.9700/0.9767/0.9767$ — within $\pm.01$. Same code, same seeds,
  deterministic decoding, so a miss means the path moved and **nothing is read**.
- **P1 (PRIMARY, Part B):** at its **best of $36$ runs**, the LoRA fine-tune still trails the supervised
  handover's $1.000$ by $\ge 10$ points, in $3/3$ seeds of that configuration.
- **P2 (Part A):** the paired bootstrap interval ($20{,}000$ resamples, seeded) for
  `hrr_gold` $-$ `prompt_ev` on the $300$ compositional rows **excludes zero**.
- **P3 (Part C):** with no truncation, the MuSiQue single pass still trails the store's $0.948$ by
  $\ge 20$ points.
- **P4 (descriptive, no bar, Part A):** accuracies of all three arms on V-125's $204$ answerable rows
  (`v125_answerable.json`), and the error-removal figure there. This closes the gap V-125 recorded as absent.

**My expected directions, stated separately from the bars and against my own manuscript.**
I expect **P1 to HOLD but by less than the manuscript's margin**: tuning should lift LoRA from $0.801$, and I
would not be surprised by $0.85$--$0.92$, which still clears $10$ points. I expect **P2 to FAIL** — $2.4$
points on $300$ rows is roughly seven rows, and I do not believe that interval excludes zero. I expect **P3 to
HOLD**: the single pass should rise substantially from $0.165$ once nothing is cut, but not to $0.75$.
**I am registering P2's expected failure because it retracts a sentence I wrote and defended through four
audit passes.**

**Readings, fixed now.**
- **P1 holds ⇒** the two-hop comparison survives a tuned opponent; the manuscript restates the margin at the
  *tuned* number, not the untuned one, in the same commit that reports this, and the "a larger or longer one
  may close part of the gap" concession is replaced by a measurement.
- **P1 fails ⇒ reported first, and it is the heavier outcome.** The headline bAbI comparison was a tuning
  artifact. "About twenty points at two hops" is struck and replaced by the measured tuned gap, in the
  abstract, the introduction, Figure 1's caption and the Discussion. If the tuned LoRA reaches $1.000$, the
  claim that structure beats an equally supervised fine-tune at two hops is **withdrawn** on natural text and
  survives only on the synthetic arenas, where the comparator is at chance.
- **P2 holds ⇒** "halves the residual error" keeps its wording and gains an interval.
- **P2 fails ⇒** the sentence is restated as *not resolvable at this sample size*: the store and the
  in-prompt evidence arm are indistinguishable on $300$ rows, which is a stronger version of the
  counter-finding the paper already carries, and the [venue redacted] companion's frontier-reader result stops being the
  only evidence for it.
- **P3 holds ⇒** the MuSiQue separation is a property of the method, not of Phi-2's window; the truncation
  disclosure added on 09-11 stays, with the untruncated number beside it.
- **P3 fails ⇒** the $78$-point MuSiQue gain was substantially a context-window artifact. The benchmark is
  restated at the untruncated number and the abstract's MuSiQue figure changes.
- **P0 fails ⇒** nothing is read from any part.

**Discipline.** Prereg + `v131b.py` + `v131c.py` + the `--dump_rows` patch + the frozen `v131_score.py`
committed before the first run. `PYTHONHASHSEED=0`. Per-row dumps banked for every arm of Parts A and C, so
this run cannot repeat the defect it exists to fix. Value-blind completeness: an arm whose scored-row count
differs from its row count is excluded on counts alone. Intervals are paired bootstraps over rows, seeded.
One tick held before scoring; the scorer is run verbatim.

**Scoping.** Parts A and C give every arm the same gold evidence, so they measure *use* of evidence, not
acquisition; the extraction boundary is untouched. Part B changes only the comparator's hyperparameters,
never the handover. Nothing here bears on the [venue redacted] companion except by supplying it a paired interval it can
cite. No API spend, no money decision, no outward-facing act.
````

## Part 8: Verdict: V-131 (P3 holds) (ledger.md lines 23771-23826)

SHA-256 of the lines below, each followed by a newline: `c84af2f60a9ae830ec0ee7cb9eaa43cce004fc01891d702491a361c0ea926fd9`.

````text
## 2026-09-14 — V-131 CLOSED under the V-132 amended gate (both scorers run verbatim, outputs banked). **P1 FAILS and P2 FAILS — two of the manuscript's own sentences are retracted. P3 holds. P4 is descriptive and partly rescues what P2 loses, on a subset.**

**Gates.** The frozen `v131_score.py` was run first, verbatim, and records what it was written to record:
`prompt_ev` and `hrr_gold` reproduce V-65 **to the digit**, `prompt` misses by $0.0133$, gate fails, nothing
read. `v132_score.py` (generated from it by replacing only the P0 block; all four bar lines byte-identical)
then applied G1/G2/G3: G1 passed exactly, G2 passed — the five-seed spread $[0.2400, 0.2767]$ contains both
the re-run's $0.2400$ and the banked $0.2533$ — G3 passed, per-row dumps complete on all nine arm-seed cells.

**P1 (PRIMARY) FAILS — reported first, as registered.** Twelve configurations $\times$ three seeds, all
completed. Best on test (a disclosed handicap in the comparator's favour): rank $16$, six epochs,
lr $10^{-4}$ → $0.9722$ ($0.9714$, $0.9738$, $0.9714$). The handover's lead is $0.0286$, $0.0262$, $0.0286$
— **$2.6$ to $2.9$ points, not the manuscript's twenty.** The re-run of the manuscript's own point (rank $8$,
two epochs, $10^{-4}$) gives $0.7952$ against its banked $0.8008$, so the grid is comparable and the
difference is tuning, not drift. Six of twelve configurations diverge to $0.16$–$0.41$, so the tuned result
is a real optimum and not a monotone effect of capacity.
**What this costs: the two-hop margin was an under-trained opponent.** Under the paper's own registered
ten-point bar — the one it applies to the one-hop case and reports failing there — the two-hop comparison now
fails too. The capability claim on natural text is withdrawn; what survives is narrower and still true: the
handover sits at $1.000$ in $3/3$ with a linear head, while a fine-tune tuned over twelve configurations
still misses about twelve of $420$ instances.

**P2 FAILS.** Paired bootstrap over the $300$ rows: `hrr_gold` $-$ `prompt_ev` $= +0.0244$, $95\%$ CI
$[-0.0044, +0.0544]$, $26$ discordant rows. **The store and the in-prompt evidence arm are not
distinguishable at this sample size**, so "it halves the residual error, $5.0\%$ to $2.6\%$" is restated as
unresolvable. This was my registered expectation and it retracts a sentence I defended through four audit
passes. For contrast, the headline gap is intact and enormous: store $-$ single pass $= +0.7344$, CI
$[+0.6822, +0.7856]$.

**P3 HOLDS.** MuSiQue read by Qwen3-8B at a $32$k window, truncation rate $0.000$ (phi-2: $0.887$): the
single pass rises from $0.1651$ to $0.2877$ and still trails the store's $0.9481$ by $0.6604 \ge 0.20$. The
separation is a property of the method, not of phi-2's window. Two secondary numbers move and must be
reported with it: the gold-paragraph baseline rises $0.4009 \to 0.5283$, and the $+150$ pooled prompt rises
$0.1873 \to 0.5786$ with **every gold fact kept** (phi-2 kept $46$–$51\%$) against the store's $0.6556$ —
**the pooled separation falls from $47$ points to $7.7$.**

**P4 (descriptive, no bar) — and it is the one good surprise.** With phi-2 per-row hits finally banked, the
gap V-125 recorded as `"phi2_v65": per-seed means only, ABSENT` is closed. On its $204$ answerable rows:
single pass $0.2990$, evidence-in-prompt $0.9559$, store $0.9967$. Error removal $96.6\%$ full,
$99.5\%$ answerable. And `hrr_gold` $-$ `prompt_ev` there is $+0.0408$, CI $[+0.0131, +0.0719]$, which
**excludes zero**. So the algebra's residual advantage is measurable exactly where the arms receive matched
evidence, and vanishes into the noise when the $96$ mismatched rows are mixed back in. This is reported as
descriptive: P4 carried no bar, the subset was defined by another study, and it does not reinstate P2.

**Unconditional correction, independent of every bar.** `v65.build_context` orders the supporting sentences
by iterating a `set`, so the manuscript's $0.253$ single pass is one draw from a permutation distribution
measured here at $[0.2400, 0.2767]$. Every sentence derived from it — the $97\%$ error removal, the $+0.72$
gap, the "$70$ of the $72$ points" decomposition — carries that band and must say so.

**What is untouched.** The evaluation audit (label priors, pooling, question placement), the mechanism chain
(exact rows work, task-gradient codes never do, the $0.10$ cross-talk tolerance, injectivity), the arena
results where the fine-tune is at chance by construction, the capacity and sharding sweeps, and the store's
structural inability to name an absent entity. The spine holds; three specific comparisons shrink.

**Discipline.** Prereg 6a2a52d before the first call; amendment e435591 written after the fact and labelled
as such; both scorer outputs banked. Zero API spend. Manuscript edits follow in the same commit that reports
this, per the registered readings.
````
