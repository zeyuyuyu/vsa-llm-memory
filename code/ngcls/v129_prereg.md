
## 2026-09-10 — V-129 PREREG: scale the EXTRACTOR, not the reader. Does a frontier model close the acquisition gap that an 8B fine-tune could not? (user-directed: "可以，这个可以做"; BLIND on P1–P4)

**Why this rung and not the one I first proposed.** The user approved work on the acquisition bridge. Two of
the three levers I sketched are already dead on this record and I am not re-running them: the
**redundancy family is closed permanently** at all three loci (V-99/107/109/110/122, five gain-price
cancellations), and **chain-weighted supervision is ruled out by V-98 P3** — chain recall equals overall
recall to within $-.006$ to $+.013$, so the chain is not the weak link.

**And "raise recall" cannot win, by arithmetic.** V-98 measured recall $.6589 \Rightarrow$ end-to-end
$.4678$, with EM sitting just above recall$^2$. At the project's own $.798$ recall target the route reaches
$\approx .637$ plus a small offset, call it $.66$, against fine-tune baselines of $.6367$ (V-98's reference)
and $.7333$ (V-99 seed 42). So incremental recall buys parity at best. Chasing it is not worth GPU.

**The unmeasured coordinate.** V-98's frozen-extractor figure of $.0500$ is **Qwen3-8B frozen**, and its
$.6589$ is **Qwen3-8B fine-tuned**. This project has scaled the *reader* to 2.4 trillion parameters across
V-108/V-124 and has **never scaled the extractor**. A 2026 frontier model prompted for structured
extraction is a different instrument from a 2025 8B fine-tune, and nothing in the ledger measures it. It is
also the cheapest remaining shot at the bottleneck: no training, $\approx 600$ calls.

**Protocol.** Question-blind extraction, which is the deployment model the manuscript's reuse and
context-compression arguments depend on. **The question is NOT shown to the extractor** — V-98 established
that showing it is a different protocol ($.5511$ vs $.4678$) with no store reuse, and it is not adopted
here by the back door.
- Rows: the paper's `load_rows(300, "validation")` slice, unchanged.
- Input: the instance's ten paragraphs under the same $1200$-token budget the reader gets.
- Output: lines of `subject | relation | object`, parsed with a fixed strict parser; malformed lines counted
  and reported, never silently dropped.
- Extractors: `deepseek/deepseek-v4-flash` and `moonshotai/kimi-k3` via CommonStack, temperature $0$.
- Store: V-102's sharded code path **verbatim**, hashed identity, $d=2048$, exact algebra, built from the
  EXTRACTED facts only. Metric `em` after the registered normalisation. Two chained unbind-and-cleanup hops.
- Scored against the gold triples for recall and **chain recall** (the two triples the supporting facts
  identify), so V-98's diagnostic is recomputed rather than assumed.

**Predictions.**
- **P0 (validity gate):** the store built from GOLD triples on these same rows reproduces the banked
  $.9744$ within $\pm .01$. No model participates in the store, so a miss means the store code path moved
  and **nothing is read**.
- **P1 (PRIMARY):** frontier question-blind extraction recall $\ge 0.80$ — beating V-98's 8B fine-tune
  ($.6589$) and reaching the project's own $.798$ target.
- **P2:** end-to-end EM of the store built from extracted facts $\ge 0.6367$, V-98's fine-tune reference.
  This is the bar the route has never cleared; clearing it makes "LLM extraction + exact store" competitive
  with simply fine-tuning the model, for the first time on this record.
- **P3 (descriptive, no bar):** chain recall against overall recall. V-98 found them equal at 8B; whether
  that survives at frontier scale is unmeasured.
- **P4 (descriptive, no bar):** whether EM still tracks recall$^2$, and the malformed-line rate.

**My expected directions, stated separately from the bars.** I expect **P1 to HOLD** — structured
extraction is close to what these models are drilled on, and the $.0500$ frozen baseline reflects a 2025 8B
model, not the task's difficulty. I expect **P2 to be BORDERLINE**: at recall $.85$ the square is $.7225$,
which sits between the two fine-tune references, so P2 may hold against $.6367$ and still fail against
$.7333$. **I am registering that both references get reported, not whichever one the result clears.**

**Readings, fixed now.**
- **P1 and P2 both hold ⇒** the acquisition bottleneck is an artifact of extractor capacity, not a property
  of the task. That is a genuinely new positive result and it changes the paper: the honest caveat
  "everything here is conditional on gold facts" is replaced by a measured end-to-end number.
- **P1 holds, P2 fails ⇒** extraction is fixed but the recall$^2$ chain penalty still dominates, so the
  binding constraint is the *two-hop survival structure*, not extractor quality. Reported as such, and the
  paper's caveat stays with a sharper reason attached.
- **P1 fails ⇒ reported first.** A frontier model cannot do question-blind two-hop extraction either, so
  the bottleneck is the task and not the 8B model, and the manuscript's limitation is not merely real but
  robust to three orders of magnitude of extractor scale. I would rather publish that than leave it as an
  open door.
- **P0 fails ⇒** nothing is read; the store path is investigated first.

**Deviations in force on this gateway, from the V-127 record.** CommonStack exposes **no balance meter**
(so the V-108 rule is replaced by a hard cap of **900 calls**, and spend is not observable), **no provider
field** (so V-123's pinning rule cannot be honoured), and **no reasoning field**; per V-127's lesson
`max_tokens` is set to $2048$, well above the visible answer, and truncated output is treated as an
instrument ceiling rather than a transient error.

**Discipline.** Prereg + `v129.py` + the frozen `v129_score.py` committed before the first call.
`PYTHONHASHSEED=0` enforced by abort. Per-row dumps including the raw extraction text, so the defect that
made V-98's dumps unusable here — aggregates only, no per-row triples — is not repeated. Value-blind
completeness: an extractor arm with `n_scored != n_eval` is excluded on counts alone. Intervals are
bootstraps over rows, seeded. One tick held before scoring; the scorer is run verbatim.

**Scoping.** The store is re-measured **only** on gold triples as the P0 gate; its banked $.974$ is not
restated as a new result. Nothing here bears on the npj submission. If P1 and P2 hold, the natural
follow-up is V-106's evidence-scale sweep rebuilt on extracted facts — **named here, not run**, because
designing it after seeing this outcome is the hindsight this discipline exists to prevent.
