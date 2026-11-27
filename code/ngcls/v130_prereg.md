
## 2026-09-11 — V-129 VOID (instrument defects; P1/P2 NOT READ, no value entered the record)

V-129's P0 gate passed (store on gold triples $.9767$ against the then-banked $.9744$; the re-based
reference is $.9778$ and it passes that too). P1 and P2 are **not read**, because three defects in the
instrument — not in the extractors — make the numbers uninterpretable in either direction:

1. **Exact-string recall.** `recall_of` matches normalised `(subject, object)` strings, and the extractors
   obey the instruction to use surface forms as written in the passages, which are not the gold's Wikidata
   labels: `Polish-Russian War (film)` against gold `Polish-Russian War`; `actress Małgorzata Braunek`
   against `Małgorzata Braunek`. The recorded recall ($.4583$ deepseek-v4-flash, $.5450$ kimi-k3) is a lower
   bound of unknown looseness, so the P1 bar cannot be applied to it.
2. **Query-key mismatch in the store.** The store is keyed by the same exact strings, and the query uses
   `head_entity(row)` from the gold, so the end-to-end EM ($.0100$ / $.0200$) measures whether the
   extractor happened to spell the head entity exactly as Wikidata does — not whether it extracted the
   fact. The store readout on the extracted facts is therefore not the quantity P2 names.
3. **Truncation.** $17/300$ rows hit `max_tokens=2048` (paragraph sets with many relations); per the V-127
   rule these are an instrument ceiling, and $17$ is above the completeness tolerance of zero.

**What survives:** nothing quantitative. The per-row dumps (raw extraction text included) are kept, and
they are what exposed the defect — V-98's aggregates-only dumps could not have. The prereg's readings are
not applied to a void run. **Re-registration is named, not designed:** an extraction run needs an entity
normalisation / linking step that maps extracted surface forms to the store's query key, registered before
the call, and a token budget sized to the longest paragraph set. It takes a new ID when designed.

**Second-order finding, descriptive.** The defect is the paper's own $\varphi$-is-a-hash point turned on the
acquisition side: `cosine('Polish-Russian War', 'Polish-Russian War (film)') = 0.0107`, indistinguishable
from random ($0.0221$). Any extractor feeding this store must produce the store's exact keys, and that
linking step is a component the paper's caveat did not name.

## 2026-09-11 — V-130 PREREG: structured PLAIN TEXT at scale versus the store, with FRONTIER readers (user-directed: "给100个结构化信息，plain text形式 llm 依然错，vsa 形式则正确 — 去验证一下"; BLIND on P1–P4)

**The user's question.** Plain text can be structured too. Hand a model $100$ structured facts as text and
it still errs; hand the same $100$ facts to the store and it does not. Is that so?

**What the record already says, and what it cannot say.** V-106 measured exactly this axis: the same pooled
gold facts, rendered as `subject | relation | object` lines in the prompt versus loaded into the store,
$K \in \{8, 64, 128, 512, 1024\}$, $n = 120$. Prompt $.9500 \to .5417$, store $.9361 \to .9083$, LoRA
$.9167 \to .4250$. At $K = 128$ the prompt reads $.8500$ against the store's $.9194$. So the user's
hypothesis is **already confirmed for one reader** — Qwen3-8B. Nothing on the record says whether the
collapse of structured plain text at scale is a property of *reading structured text* or of an *8B model*.
The paper's ladder scaled the reader to $2.4$T on $K \approx 8$ in-instance facts only; the evidence-scale
axis has never seen a frontier reader. That is the unmeasured coordinate, and it is the one the user's
question actually turns on.

**Protocol.** V-106's harness with the reader swapped.
- Rows: V-106's $120$ evaluation rows from its $400$-row pool, same seed ($42$), same `pooled()` call, so
  the pooled fact sets are those V-106 used. Cached to `rows400.json` if the mirror allows; otherwise
  streamed as V-106 did.
- $K \in \{8, 64, 100, 128, 512, 1024\}$ — V-106's grid plus the user's $K = 100$.
- Renderer: V-106's `facts_prompt` **verbatim** (few-shot + `Facts:` block + "Answer with the entity
  only"). Every reader receives the byte-identical prompt; the arms differ only in who reads.
- Readers: **Qwen3-8B frozen, local GPU** (the P0b gate, re-run rather than copied), and
  `deepseek/deepseek-v4-flash`, `zai-org/glm-5.3`, `moonshotai/kimi-k3` via CommonStack at temperature
  $0$, `max_tokens = 1024` per the V-127 rule. Scored on the first output line after the registered
  normalisation, as V-106 scored the 8B; an echoed `Answer:` prefix is stripped.
- Store: V-102's `sharded_two_hop` **verbatim**, $64$ facts per shard, $d = 2048$, codebook seeds
  $\{0, 1, 2\}$ with V-106's exact seed offsets, per-row per-seed hits banked. Shard count is
  $\lceil K/64 \rceil$, identical to V-106's $\lfloor K/64 \rfloor$ at every V-106 grid point and $2$ at
  $K = 100$ so the per-shard load never exceeds the operating point.
- Completeness is value-blind: any reader-$K$ cell with `n_scored != n` is excluded on counts alone.
  Intervals are paired bootstraps over rows ($20{,}000$, seeded), store per-row accuracy being the mean
  over seeds.

**Predictions.**
- **P0a (validity):** the store reproduces V-106's store row within $\pm .01$ at the five shared $K$.
  Same code, seeds and pool — a miss means the pool or the store path moved and **nothing is read**.
- **P0b (validity):** the re-run Qwen3-8B prompt arm reproduces V-106's prompt row within $\pm .05$ at
  the five shared $K$. This is what certifies the renderer copy and the fact pool; a miss voids the run.
- **P1 (PRIMARY):** at $K = 1024$, for **every** complete frontier reader, store $-$ reader $\ge .05$ with
  the paired CI excluding zero. This is the user's hypothesis at frontier scale.
- **P2:** at the user's $K = 100$, for every complete frontier reader, store $-$ reader has a paired CI
  excluding zero (any positive gap).
- **P3 (descriptive, no bar):** per reader, the smallest $K$ at which the store leads by $\ge .02$ (V-106
  P4's rule), and the reader's drop from $K = 8$ to $K = 1024$.
- **P4 (descriptive, no bar):** prompt tokens and latency per query at $K = 1024$ against the store's.

**My expected directions, stated separately from the bars and against the user's hypothesis.** I expect
**P0a and P0b to hold**. I expect **P1 to FAIL** for at least one frontier reader: $1024$ clean
pipe-delimited lines are about $11{,}600$ tokens, a tenth of these models' context, and locating two
matching lines in clean structured text is close to what they are drilled on; I expect frontier accuracy at
$K = 1024$ in the $.85$–$.95$ band, i.e. the 8B collapse is a capacity effect that parameter count buys back.
I expect **P2 to FAIL** by a wider margin — a tie at $K = 100$. **I am registering this so that a P1 hold is
read as the user being right and me wrong, not as a result I steered toward.**

**Readings, fixed now.**
- **P1 holds ⇒** the user's hypothesis holds at frontier scale: structured plain text at $K = 1024$ still
  loses to the store for every reader we can reach. V-106's evidence-scale advantage is not an 8B artifact,
  and the paper's Sec 3.1 may cite it for the full ladder. This would be the first store-over-frontier
  accuracy result on the record.
- **P1 fails ⇒ reported first.** At least one frontier reader reads $1024$ structured facts as well as the
  store. V-106 becomes a small-model finding, in the same class as V-108's "algebra halves the residual",
  and the paper's evidence-scale claim is restricted to the 8B reader in the same commit that reports this.
  The cost claim (P4) survives regardless; the accuracy claim does not.
- **P2 holds ⇒** even $100$ structured facts are beyond exact reading for every frontier model; the user's
  intuition is right at the point they named.
- **P2 fails ⇒** $100$ structured facts are within what a frontier model reads exactly; the store's
  advantage, where it exists, begins higher, and P3 says where.
- **P0a or P0b fails ⇒** nothing is read.

**Deviations in force on this gateway (V-127/V-129 record).** No balance meter (hard cap $2{,}600$ calls;
spend not observable), no provider field (V-123's pinning rule cannot be honoured, so per-row claims carry
the $\pm .05$–$.19$ band; the bars read aggregates), no reasoning field (`max_tokens = 1024`).

**Discipline.** Prereg + `v130.py` + the frozen `v130_score.py` committed before the first call.
`PYTHONHASHSEED = 0` enforced by abort. Per-row dumps with raw output for every reader and per-seed hits for
the store. Hold one tick before scoring; run the scorer verbatim. This run takes a new ID; V-129 is void
above and is not reused.

**Scoping.** All arms receive gold triples: this measures the *use* of structured evidence at scale, not
its acquisition. Nothing here bears on the npj submission.
