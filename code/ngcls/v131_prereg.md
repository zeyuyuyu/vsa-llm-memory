
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
  counter-finding the paper already carries, and the ICLR companion's frontier-reader result stops being the
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
never the handover. Nothing here bears on the ICLR companion except by supplying it a paired interval it can
cite. No API spend, no money decision, no outward-facing act.
