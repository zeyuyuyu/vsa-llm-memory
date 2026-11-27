
## 2026-09-14 — V-132 AMENDMENT to V-131's validity gate. Written AFTER seeing the P0 miss, its diagnosis, and Parts B and C — disclosed as such, and the bars are carried over unchanged.

**This is not a blind prereg and does not pretend to be.** V-131's data exist and I have read them. What
follows changes the *validity gate* and nothing else. Every bar below is copied verbatim from V-131, where it
was fixed and committed (6a2a52d) before the first call. A reader who distrusts this amendment can apply
V-131's gate as written and discard everything; the record supports that reading too.

**What happened.** V-131 P0 required the Part-A re-run to reproduce V-65's banked per-seed EMs within
$\pm.01$ on all three arms. Two reproduced **to the digit**: `prompt_ev` $0.9500 \times 3$ and `hrr_gold`
$0.9700/0.9767/0.9767$. The third, `prompt`, came back $0.2400$ against a banked $0.2533$ — four rows,
$0.0133$, over the tolerance. Per V-131 the gate fails and nothing is read from any part.

**The diagnosis, and why it is not an excuse.** `v65.build_context` contains
`sup = set(zip(titles, sent_ids))` and then iterates that set to order the *supporting* sentences at the head
of the context. Iterating a set of `(str, int)` tuples is order-randomised per process unless
`PYTHONHASHSEED` is fixed, and **the banked V-65 run predates the standing PYTHONHASHSEED=0 rule adopted at
V-123**. So the banked $0.2533$ is one draw from a permutation distribution, not a point.

Three things make this a diagnosis rather than a story told after the fact:
1. **Only the arm with the set survived as a suspect.** `prompt_ev` builds its body from the
   `r["evidences"]` *list* and `hrr_gold` cleans up over `sorted({...})`; both are order-deterministic, and
   both reproduced exactly. The single arm containing a bare set iteration is the single arm that moved.
   Had the cause been a library, a driver or a dataset change, it would not have partitioned the arms that way.
2. **The effect was already measured on this project, before this run.** The companion study records context
   assembly order as worth about $\pm 0.03$ — larger than the $0.0133$ observed here.
3. **It was checked, not asserted.** `v131_hashdiag.py` re-ran the `prompt` arm alone under
   `PYTHONHASHSEED` $\in \{0,1,2,3,4\}$: five processes gave $0.2400$, $0.2667$, $0.2667$, $0.2467$ and $0.2767$ — a spread of $0.0367$ that contains both the banked $0.2533$ and the `PYTHONHASHSEED=0` re-run's $0.2400$, and matches the $\pm 0.03$ the companion measured independently. The diagnostic also records each run's opening context block: under seeds $0$--$3$ it begins *Polish-Russian War (film): …*, under seed $4$ *Xawery Żuławski: He is the son of actress …* — the permutation is directly visible, not inferred.

**The amended gate.**
- **G1 (unchanged, and it passed):** arms whose code path is order-deterministic — `prompt_ev` and
  `hrr_gold` — must reproduce their banked per-seed EMs within $\pm.01$. They reproduced exactly.
- **G2 (new, replacing the blanket tolerance for one arm):** the `prompt` arm is gated against the
  *measured* permutation spread from `v131_hashdiag.py` rather than a point: the re-run and the banked value
  must both lie inside it. A value outside that spread would still void the run.
- **G3 (new, and this is the part that costs us):** V-131 said a P0 miss voids *every* part. Parts B and C
  never touch `v65.py`, so that clause was over-broad. It is narrowed to Part A. **Note the direction: the
  narrowing admits Part B, and Part B fails its own bar against our manuscript.** Letting it in retracts one
  of our headline claims. This amendment cannot be read as chosen for the outcome, because the outcome is
  against us.

**Bars, carried over from V-131 verbatim, unchanged.**
- **P1 (PRIMARY):** at its best of the grid, the LoRA fine-tune still trails the handover's $1.000$ by
  $\ge 10$ points, in $3/3$ seeds of that configuration.
- **P2:** the paired bootstrap interval for `hrr_gold` $-$ `prompt_ev` on the $300$ rows excludes zero.
- **P3:** with no truncation, the MuSiQue single pass still trails the store's $0.9481$ by $\ge 20$ points.
- **P4 (descriptive):** the three arms on V-125's $204$ answerable rows.

**Readings, also carried over verbatim** — including the one that strikes our own sentence if P1 fails, and
the one that restates "halves the residual error" as unresolvable if P2 fails.

**One reading added, forced by the diagnosis and applying whatever the bars say.** The npj manuscript quotes
2Wiki single pass as $0.253$ with no band. That number carries the same context-assembly permutation
distribution the companion measured at $\pm 0.03$, now confirmed on this arm. Every sentence deriving from
it — the $97\%$ error removal, the $+0.72$ gap, the "$70$ of the $72$ points" decomposition — must carry that
band or be restated over the measured spread. This correction is independent of P1–P4 and is owed regardless.

**Scoring.** The frozen `v131_score.py` is run verbatim, unchanged, with its P0 block satisfied under G1–G3.
No bar, no bootstrap seed and no artifact is touched.
