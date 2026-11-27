
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
