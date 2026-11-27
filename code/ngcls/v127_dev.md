
## 2026-09-10 — V-127 DEVIATION (recorded BEFORE the re-run): the completeness rule excluded the PRIMARY arm, so the instrument is repaired and that arm is re-run. Bars, template, rows and the frozen scorer are UNCHANGED.

**What happened.** The frozen scorer (md5 7813b32e34a6fb7573aa39169a10ea88, run verbatim) excluded three of
four judge arms on the value-blind completeness rule:

| arm | scored | calibration | fate |
|-----|--------|-------------|------|
| deepseek-v4-flash / paragraph | 299/300 | **0.950** | EXCLUDED (one row short) |
| deepseek-v4-flash / sentence  | 300/300 | 0.900 | USABLE |
| kimi-k3 / paragraph | 42/300 | n/a | EXCLUDED |
| kimi-k3 / sentence  | 41/300 | n/a | EXCLUDED |

So **P1, the registered PRIMARY, was not computable**, and the scorer printed the registered consequence:
the provenance claim is not evaluated. The arm lost had the *best* calibration of the four. The criterion is
not being relaxed to rescue it — that is what the rule is for.

**Two instrument defects, both recorded.**
1. **kimi-k3 returns EMPTY CONTENT through CommonStack** on this task: 156 PARSE failures and 14 "no
   choices" on the paragraph arm, one HTTP 500. This is the fourth API-instrument defect in this project
   (after the aggregates-only dumps, the provider-dependent `reasoning` field, and the unreliable cost
   field). kimi-k3 is DROPPED from the re-run as unusable on this gateway, not as a judge whose answers
   I disliked — it produced almost none.
2. **My own 1400-call cap was the binding constraint at the end**: `calls_used == 1400 == cap`, and
   kimi/sentence lost 119 rows to `CALL_CAP`. Because CommonStack exposes no balance meter, that cap is
   the only spend control available, so the fix is a larger cap for a single arm, not an uncapped run.

**The repair, which is to the instrument only.** Retries 4 -> 8, backoff 2s -> 4s per attempt, and the
re-run covers ONE arm: `deepseek-v4-flash / paragraph`, whole (all 300 rows), not just the missing row.
Re-running the whole arm is the conservative choice: which row failed is value-blind (it returned a
truncated `{"`), but re-asking only that row would make the arm a mixture of two sampling occasions.
The complete `deepseek / sentence` arm is **carried forward byte-unchanged** by `--merge_into`, so the
already-read sentence result cannot move.

**Unchanged:** the prereg (66bf24d), all bars (P1 < 0.15 paragraph, P1b < 0.15 sentence, provenance claim
needs BOTH), the judge template, the row set and its shas (paragraph f8ff99b5329837e4da096ec2, sentence
cabb6de9587df870598828da), the 40-row calibration committed at 0e39292, and the frozen scorer.

**What I already know and am not allowed to forget when P1 lands.** The sentence arm read
**0.3033 [0.2533, 0.3567]** against 2Wiki's banked 0.2892, a difference of **-0.0137 [-0.0862, +0.0584]** —
indistinguishable from zero. My registered expectation was that MuSiQue would be far cleaner. That
expectation is losing, and the paragraph re-run is being launched to complete a PRIMARY that will most
likely also refute it, not to rescue it. Recorded here, before the re-run, so the direction cannot be
claimed as hindsight either way.
