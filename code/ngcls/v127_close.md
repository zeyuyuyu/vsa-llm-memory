
## 2026-09-10 — V-127 CLOSED: P1 and P1b BOTH FAIL ⇒ the evidence mismatch is NOT a knowledge-base-annotation artifact. My registered mechanism hypothesis is REFUTED, and the finding is stronger than the one I predicted. The granularity control I registered against myself turned out to matter as much as the dataset.

Prereg 66bf24d, calibration 0e39292 (committed before any judge output), frozen scorer
md5 7813b32e34a6fb7573aa39169a10ea88 run verbatim. Two recorded deviations (retries; then the
max_tokens correction of that first mis-diagnosis). n=300 MuSiQue 2-hop, **zero undecidable**.

**P0 PASS**: both judge arms usable — paragraph calibration .912, sentence .900, each 300/300.

| granularity | hop1 absent | hop2 absent | ANSWERABLE |
|-------------|-------------|-------------|------------|
| paragraph (dataset's own annotation, PRIMARY) | 36 (.120) | **59 (.1967) [.1533, .2433]** | 215/300 = .7167 |
| sentence (V-125-comparable) | 57 (.190) | **91 (.3033) [.2533, .3567]** | 168/300 = .5600 |

**P1 FAILS (.1967 vs bar .15). P1b FAILS (.3033).** Per the registered reading this is REPORTED FIRST and
is the stronger result: **multi-hop benchmarks are broadly not evidence-matched.** MuSiQue is the
construction least likely to have this defect — every hop is a verified single-hop question, answerable
from its paired paragraph by design — and it still leaves the second-hop fact absent on one instance in
five at paragraph granularity and nearly one in three at sentence granularity. The V-125 audit therefore
generalises from one dataset to a class, and the paper's caution is raised accordingly.

**My hypothesis is refuted, and I record the two refinements that survive it.**

1. **Provenance explains PART of it, not the phenomenon.** 2Wiki − MuSiQue at paragraph level is
   **+0.0935 [+0.0271, +0.1639]**, which excludes zero: 2Wiki (KB triples over prose) genuinely is worse
   than MuSiQue (composed verified single-hop questions), 28.9% against 19.7%. At sentence granularity the
   difference is **−0.0140 [−0.0869, +0.0574]**, indistinguishable. So annotation provenance is worth
   roughly nine points and is not the cause.
2. **The granularity control I registered AGAINST MY OWN PREDICTION matters as much as the dataset.**
   Same rows, same judge: .1967 at paragraph against .3033 at sentence, a 10.7-point swing produced by
   nothing but how much surrounding text is counted as "the evidence". Had I measured only the
   paragraph level — which is the dataset's own annotation and the natural choice — I would have
   published a number that looks half as bad AND is not comparable to V-125's sentence-level 28.9%.
   This is the single most useful thing the run produced and it exists only because the prereg forced
   the control on me when the coarser unit favoured my prediction.

**Convergent evidence from V-126, same session, independent design.** V-126's P3 built the 2Wiki-vs-
HotpotQA contrast expecting the same provenance effect (KB triples over prose vs human sentence pointers
into it). The oracle-evidence gains came out **identical: +0.1867 on both**. Two independent tests, one
judged and one behavioural, both refute the provenance story.

**Instrument defects recorded (fourth and fifth in this project's API work).**
- **kimi-k3 returns EMPTY CONTENT through CommonStack**: 156 PARSE + 14 "no choices" + one HTTP 500 on
  the paragraph arm, 42/300 scored. Dropped as unusable on this gateway, not as a judge whose answers I
  disliked — it produced almost none.
- **Invisible reasoning tokens consume the answer budget.** My 200-token ceiling truncated the visible
  JSON on the longer paragraph contexts (9/300 in run 2, 1/300 in run 1) and never on the shorter
  sentence contexts. deepseek-v4-flash spends output budget on reasoning this gateway does not return.
  At max_tokens 1024 the arm scored **300/300 using exactly 300 calls and zero retries**, which confirms
  the mechanism. **My first repair (more retries) was the wrong diagnosis and made it worse (291/300 vs
  299/300), because retries cannot fix a systematic ceiling.** This is the ICLR draft's Sec 4.2 finding
  one step further and belongs in the paper: an unreturned reasoning trace does not merely make hop
  attribution uninterpretable, it silently eats the answer budget on the gateway that hides it.

**Standing rules confirmed / added.** The value-blind completeness rule again excluded the arm with the
BEST calibration (.950 at 299/300) and cost the primary until the instrument was repaired — the criterion
was not relaxed, in line with V-124. **New rule: on any gateway that does not return a reasoning field,
set max_tokens well above the visible answer length and treat truncated output as an instrument ceiling,
never as a transient error.**

**Scoping.** No store re-measurement; nothing here bears on acquisition (V-97/V-98) or on the npj
submission. Calls used 300 of cap 900 on the final run; **spend is not observable on this gateway**.
Both V-126 and V-127 now feed the ICLR draft: the "one benchmark" objection is answered for the central
claim (V-126 P1, both datasets) and for the audit (V-127, a second construction).
