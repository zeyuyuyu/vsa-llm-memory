# Ledger excerpt: Naming head against a LoRA fine-tune, bAbI tasks 1 and 2

Supplementary Table `tab:prereg`, row "bAbI (naming head against a LoRA fine-tune, tasks 1 and 2)" (marked "ledger" in the Supplementary Information).

**Source.** `ledger.md` of the experiment repository `ngcls` (private; see `README.md` in this folder), at commit `0732e13` (`0732e13ef2322f308a99dcac3bf0bd2e25a0da10`, tag `archive-2026-11-11`); the copy read has MD5 `6843647bad4782ac58fcd341448bf95c`, identical to that commit's working tree. Line numbers refer to that file. For each part, the commit is the one that added those lines, from `git blame` and checked with `git log -S` on the heading; dates are commit author dates in UTC+08:00 (the machine's clock). Parts not from `ledger.md` say so.

| # | Part | Ledger lines | Heading date | Added in commit | Commit date |
|---|---|---|---|---|---|
| 1 | PRE-REGISTRATION: V-50 (bar (d), task 2) | 13284-13326 | none in heading | `136ebe9` | 2026-08-24T03:49:17+08:00 |
| 2 | Verdict: V-50 (task 1 and task 2) | 13328-13379 | none in heading | `507689b` | 2026-08-24T04:29:18+08:00 |
| 3 | PRE-REGISTRATION: V-131 (P1, tuned fine-tune at two hops) | not in ledger.md: `v131_prereg.md` | 2026-09-14 | `6a2a52d` | 2026-09-14T16:50:46+08:00 |
| 4 | Deviation note: V-131 double launch (two identical copies in the ledger) | 23673-23710 | 2026-09-14 | `2784870` | 2026-09-14T17:33:19+08:00 |
| 5 | Amendment: V-132 (validity gate only; written after the data were read) | 23712-23769 | 2026-09-14 | `ef1be8c` | 2026-09-14T20:56:41+08:00 |
| 6 | Verdict: V-131 (P1 fails) | 23771-23826 | 2026-09-14 | `ef1be8c` | 2026-09-14T20:56:41+08:00 |
| 7 | Follow-up verdict: V-152 (configuration selected on validation) | 26010-26076 | none in heading | `bb9cb77` | 2026-10-09T05:18:50+08:00 |

**Depositor's notes** (written when this excerpt was deposited, 2026-11-21; not part of the ledger).

- **Row in the paper.** "lead >= 10 points at one hop, applied also at two hops"; outcome "failed at one hop (fine-tune 0.999) and at two hops (lead below three points)"; departures "the two-hop fine-tune was configured on the test set, which favours it". The ledger calls the naming head the "handover" (probe_argmax); its 1.000 was banked from earlier runs and not re-run here.
- **Matching registrations.** V-50 (part 1) bar (d): "THE HANDOVER SURVIVES A SUPERVISED OPPONENT iff 1.000 - lora_direct >= 0.10 on task 2 in 3/3". V-131 (part 3, a separate file `v131_prereg.md`, not ledger.md) P1: "at its best of 36 runs, the LoRA fine-tune still trails the supervised handover's 1.000 by >= 10 points, in 3/3 seeds of that configuration", on bAbI task 2.
- **Discrepancy: the direction of the extension is the reverse of the paper's.** The ten-point bar was pre-specified for two hops (task 2) only. No bar was registered for one hop (task 1); the V-50 verdict (part 2) applied the task-2 bar to task 1 after the run ("Bar (d) requires handover - lora_direct >= 0.10 in 3/3; it is .0007"). The one-hop failure is therefore not a pre-specified test. The paper should say that the bar was fixed in advance at two hops and applied afterwards at one hop.
- **Discrepancy: the first two-hop outcome passed.** Under V-50 the two-hop bar passed (leads 0.150, 0.305, 0.143 against LoRA rank 8, two epochs). V-131 then re-registered it against a LoRA tuned over 12 configurations x 3 seeds, selected on the test set (declared in the registration as a handicap on the handover); P1 failed (best rank 16, six epochs, lr 1e-4: 0.9722; lead 2.6 to 2.9 points). The row reports only the second outcome.
- **Amendment.** V-132 (part 5) changed only V-131's validity gate, after the data had been read (it says so); it admitted Part B, which then failed against the manuscript. Its text entered ledger.md in the same commit as the V-131 verdict (`ef1be8c`); the amendment commit itself (`e435591`, 2026-09-14T20:36:08+08:00) summarises it in the commit message and adds only files under `v131_logs/`.
- **Test-set selection.** V-152 (part 7; registration `v152_prereg.md`, commit `4a7dbdf`, 2026-10-08T19:17:48+08:00, not excerpted) selected the configuration on the validation split instead: it picked the same configuration, test 0.9722. The row's departure note could add that the selection made no difference.

---

## Part 1: PRE-REGISTRATION: V-50 (bar (d), task 2) (ledger.md lines 13284-13326)

SHA-256 of the lines below, each followed by a newline: `06f6b2513809b26e6c4d30052d0e935b3273658fc076786b8e0b4d5dda14911a`.

````text
## V-50 prereg: train the language model ITSELF through the VSA. Before any code.

**The objection this answers, and the half of it that is right.** The standing objection is that no improvement was
seen because the VSA was not in the training loop. Half of that is already tested and is the paper's central
negative: the `spans` and `frozen_cont` arms train a projection WITH gradients flowing through bundling,
unbinding and cleanup, and they reach chance in every synthetic arena and 37.6% of headroom at depth two on bAbI.
The other half is untested and the objection is correct about it: in every run in this paper the LANGUAGE MODEL IS
FROZEN. We have never backpropagated the binding objective into the model. So "the model learns to place symbols
where they bind well" has never been given a chance, and that is the strongest form of the hypothesis.

**Arms, on bAbI task 1 (all lengths) and task 2 (expressible subset), Phi-2, LoRA r=8 on the attention
projections, seeds 42/123/456, the same training instances every other arm used.**
  lora_direct        LoRA fine-tuned with cross-entropy on the answer location token. No VSA anywhere. This is
                     the supervised opponent the paper has never had on bAbI -- prompting is unsupervised and the
                     linear probe is weak, so without this the handover's margin is not attributable to structure.
  lora_through_vsa   LoRA fine-tuned with the loss taken AFTER bind, bundle, unbind and cleanup, gradients
                     flowing through the algebra and into the model. Symbol codes come from a projection of span
                     states learned jointly; NO identity labels are supplied, because the point is whether the
                     model can learn a bindable code on its own.
  Banked references, not recomputed: handover 1.000 / 1.000, prompting .792 / .386, last-token probe .771 / .245.

**Bars.**
(a) TRAINING THROUGH THE VSA WORKS iff lora_through_vsa >= 0.95 on task 1 AND >= 0.95 on task 2, in 3/3. Then the
    paper's central negative claim is wrong as stated for a trainable encoder, the abstract sentence "a projection
    trained through the loss never works" must be rewritten to say "never works with the encoder frozen", and
    this becomes the paper's main positive result rather than its main negative.
(b) IT STILL FAILS WITH A TRAINABLE ENCODER iff lora_through_vsa <= lora_direct - 0.05 in 3/3 on BOTH tasks. Then
    the negative generalises from frozen to trainable and the paper says so, which strengthens it.
(c) GRADED otherwise: report and license neither.
(d) SEPARATELY, the control the paper needs regardless: THE HANDOVER SURVIVES A SUPERVISED OPPONENT iff
    1.000 - lora_direct >= 0.10 on task 2 in 3/3. If lora_direct reaches the handover, then the fifty-eight point
    margin over prompting was about supervision and not about structure, and the capability claim comes out.

**Predictions on the record, and the honest uncertainty.** I expect lora_direct near .95 on task 1 -- one hop, and
prompting is already .792 -- and somewhere between .5 and .8 on task 2, so (d) holds but by less than the margin
over prompting suggests. I expect (b) rather than (a), because credit assignment through a cleanup arg-max is
brutal and 775-900 instances is little data for it. But I want to state plainly that I cannot rule out (a): with
the encoder trainable the model can move symbols into positions that bind, which is a mechanism the frozen
experiments could not test, and every previous negative in this family was measured with that door shut. Thirty
predictions tested today and twenty-one failed, so the prediction is worth the bar and no more.

**What I will not do.** If (a) comes out I will not keep the frozen-encoder negative as the headline; and if (d)
fails I will not keep the fifty-eight point comparison against prompting as a capability claim.
````

## Part 2: Verdict: V-50 (task 1 and task 2) (ledger.md lines 13328-13379)

SHA-256 of the lines below, each followed by a newline: `68f34b27cac11f03fd7990ad7c9e25cd0dbb18d478cda29cd876ec7a6a4ca538`.

````text
## V-50: the supervised opponent, and the trainable encoder. One capability claim earned, one killed, and the
## objection about training through the VSA is half right.

**Task 1 (one hop, all lengths).**
  handover (banked)   1.0000
  lora_direct         [.998 1.000 1.000]  0.9993
  prompting (V-49)    0.7920      last-token probe  0.7713
Bar (d) requires handover - lora_direct >= 0.10 in 3/3; it is .0007. **THE HANDOVER DOES NOT SURVIVE A SUPERVISED
OPPONENT ON TASK 1.** Per the prereg, the twenty-one point margin over prompting was about SUPERVISION and not
about structure, and the task-1 capability claim comes out. That also retires V-49's task-1 verdict, which held
by eight thousandths and is now moot: a LoRA fine-tune with no VSA anywhere reaches the same 1.000.

**Task 2 (two hops, expressible subset, 775 train / 420 eval).**
  handover (banked)      1.0000
  lora_direct            [.8500 .6952 .8571]  0.8008
  lora_through_vsa       [.5643 .6190 .6167]  0.6000     trainable encoder, NO identity labels
  frozen spans (V-37b)   [.4952 .5048 .5071]  0.5024     frozen encoder, NO identity labels
  prompting (V-49)       0.3860               last-token probe  0.2452
Bar (d): 1.000 - lora_direct = .150 / .3048 / .1429, all >= .10. **THE HANDOVER SURVIVES A SUPERVISED OPPONENT ON
TASK 2.** Structure is worth about twenty points over a fine-tune trained on the same labels and the same
instances, in three of three seeds. This is the paper's capability claim and it is a two-hop claim.

**The objection was half right, and the half that is right is measurable.** Unfreezing the encoder moves the
through-VSA route from .5024 to .6000 on the identical task, subset and label budget -- about ten points. So the
frozen encoder WAS handicapping it and every previous negative in this family was measured with that door shut.
The objection is wrong about the conclusion: at .6000 the route is still twenty points below a plain LoRA
fine-tune with no VSA at all (.8008) and forty below the discrete handover (1.0000). Bar (b) holds in 3/3.

**So the paper's central negative narrows rather than falls.** Not "a projection trained through the loss never
works" but: with the encoder frozen it stays at or below the floor; with the encoder trainable it improves by
about ten points and still loses to both a plain fine-tune and the discrete handover. The abstract sentence must
be rewritten to say that, and the improvement must be reported, not omitted.

**Coherent picture across the two tasks, and it matches V-48's mechanism.** One hop: the model already does it
(.792 zero-shot), fine-tuning finishes it (.9993), the algebra adds nothing. Two hops: the model cannot (.386),
fine-tuning gets partway (.8008), and the algebra with a discrete identity handover is exact (1.0000). V-48
showed the final-position state bridges one hop and not two, which is exactly where structure starts to pay.

**Limits to state in the manuscript, not in a footnote.** LoRA is r=8, two epochs, 775 instances; a longer or
larger fine-tune may close part of the twenty points, and the seed spread (.695 to .857) is wide. The handover
uses identity labels the fine-tune does not; the fine-tune uses only the answer label. So the comparison is
"structure plus identity supervision" against "answer supervision alone", which is the honest framing and is how
the sentence will read.

**Apparatus, disclosed above and repeated here.** The task-1 through-VSA arm was VOID on the first attempt
because I compared an offset location id against an unoffset label; it scored .08 identically across seeds with a
falling loss, which is a bug signature. Fixed, and a guard now ABORTS any run where more than half of the
training rows have the label outside the candidate list. Task 2 reported 0 of 775 rows skipped, so its numbers
are clean by that check.

**Tally.** Thirty-two predictions tested today, twenty-two failed. I predicted lora_direct near .95 on task 1
(.9993, right), .5 to .8 on task 2 (.8008, right at the top of the range), and bar (b) rather than (a) (right).
````

## Part 3: PRE-REGISTRATION: V-131 (P1, tuned fine-tune at two hops) (`v131_prereg.md` as committed in `6a2a52d`)

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

## Part 4: Deviation note: V-131 double launch (two identical copies in the ledger) (ledger.md lines 23673-23710)

SHA-256 of the lines below, each followed by a newline: `e976f602de0b6253db88e8146c7f35b1beea7e64ac68a98b33328762c9440913`.

````text
## 2026-09-14 in-flight deviation note (tick, 17:35+08): V-131 DOUBLE LAUNCH detected and de-duplicated; arms A and C complete, arm B continues on the ORIGINAL driver

V-131 (prereg 6a2a52d, as-executed 39f3f11, user-launched 16:51+08) suffered a duplicate launch: a second command
(PID 1628479, 17:01+08, origin not this loop -- the 17:0x tick took no action and recorded only the originals)
re-ran run_v131b.sh AND run_v131c.sh while the originals were live. Consequences found at 17:30: (1) TWO v131b.py
grid drivers racing on the same GPU, same output paths (original 1623578 started 16:53, duplicate 1628488 started
17:01); (2) arm C v131c.json OVERWRITTEN by an identical second run (16:57 -> 17:18; second run exited 0,
V131C-DONE, JSON parses, prereg field embedded; the 16:57 original was not preserved -- same harness, same seeds,
zero truncation on all arms per its log); (3) v131_logs/b.out truncated/interleaved by the duplicate redirect, so
its config-1/2 lines are not a reliable log.

Intervention (this tick, before any scoring): killed the DUPLICATE tree only (1628481/1628488 + its v50.py child
1635632, which was re-running config 2 whose completed JSON already existed). Verified after kill: only the
original driver 1623578 + its config-3 child 1637235 survive; b_runs/r8_e2_lr0.0001.json and r8_e2_lr0.0003.json
both parse. INTEGRITY CAVEAT for scoring: the on-disk b_runs JSONs for configs 1-2 may be the duplicate drivers

## 2026-09-14 in-flight deviation note (tick, 17:35+08): V-131 DOUBLE LAUNCH detected and de-duplicated; arms A and C complete, arm B continues on the ORIGINAL driver

V-131 (prereg 6a2a52d, as-executed 39f3f11, user-launched 16:51+08) suffered a duplicate launch: a second command
(PID 1628479, 17:01+08, origin not this loop -- the 17:0x tick took no action and recorded only the originals)
re-ran run_v131b.sh AND run_v131c.sh while the originals were live. Consequences found at 17:30: (1) TWO v131b.py
grid drivers racing on the same GPU, same output paths (original 1623578 started 16:53, duplicate 1628488 started
17:01); (2) arm C v131c.json OVERWRITTEN by an identical second run (16:57 -> 17:18; second run exited 0,
V131C-DONE, JSON parses, prereg field embedded; the 16:57 original was not preserved -- same harness, same seeds,
zero truncation on all arms per its log); (3) v131_logs/b.out truncated/interleaved by the duplicate redirect, so
its config-1/2 lines are not a reliable log.

Intervention (this tick, before any scoring): killed the DUPLICATE tree only (1628481/1628488 + its v50.py child
1635632, which was re-running config 2 whose completed JSON already existed). Verified after kill: only the
original driver 1623578 + its config-3 child 1637235 survive; b_runs/r8_e2_lr0.0001.json and r8_e2_lr0.0003.json
both parse. INTEGRITY CAVEAT for scoring: the on-disk b_runs JSONs for configs 1-2 may be the duplicate driver's
writes (mtimes 17:22/17:26 are consistent with either writer); the aggregate of record is the original driver's
final v131_logs/v131b.json; any per-config mismatch between b_runs files and the aggregate is explained by this
event. Configs 1-2 also ran under duplicated GPU load (fixed seeds; contention affects wall time, flagged for the
record). Arm A completed clean at 17:01 BEFORE the duplicate (V65-DONE, exit 0, banked verdicts reproduced
descriptively, v65_rows.json written) -- unaffected. No numbers scored this tick; frozen v131_score.py runs after
arm B exits + one-tick hold, per prereg. No rescue re-run of configs 1-2 registered -- whether the dup-window
configs need a clean re-measure is a scoring-time decision, to be preregistered if taken.
````

## Part 5: Amendment: V-132 (validity gate only; written after the data were read) (ledger.md lines 23712-23769)

SHA-256 of the lines below, each followed by a newline: `d292f61b525e0149a75b365090a146c5c8d1c1788ae6beddf73335a2a74574f4`.

````text
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
````

## Part 6: Verdict: V-131 (P1 fails) (ledger.md lines 23771-23826)

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

## Part 7: Follow-up verdict: V-152 (configuration selected on validation) (ledger.md lines 26010-26076)

SHA-256 of the lines below, each followed by a newline: `8c402a1abefbdb5b50ceb05037cf27212dc7bb720a0b53b9ea27b85166020d84`.

````text
## V-152 — the comparator selected honestly. T0 / T1 / T2 / T3 ALL HOLD: the disclosed test-set handicap cost nothing

Prereg + frozen scorer + runner 4a7dbdf, committed **before** the sweep. Runner `v152_v50.py` is `v50.py`
plus a default-off `--eval_val`; the diff removes three lines, each replaced by a superset. Artifact
`v152_logs/v152.json`. 12 configurations x 3 seeds, 4.0 h wall-clock.

**Why it was run.** V-131 Part B selected the LoRA comparator on the test set and disclosed it, giving as
its reason that "this harness has no validation split, inventing one would change the training set and
break comparability". **The premise was false**: `v5.load_babi` returns `{train, val, test}`, `v50.build_rows`
calls it with `n_val=200` and returns only two of the three. The validation split was loaded and discarded
on every run. Using it costs nothing — train stays at 775 and test at 420, and passage overlap between all
three pairs is 0, measured.

    config             val (80)   test (420)   V-131b banked
    r16_e6_lr1e-4       0.9750      0.9722        0.9722
    r8_e6_lr3e-4        0.9583      0.9635        0.9635
    r32_e6_lr1e-4       0.9583      0.9540        0.9540
    r8_e6_lr1e-4        0.9333      0.9484        0.9540
    r8_e2_lr3e-4        0.9417      0.9048        0.6548   <- the one cell that did not reproduce
    r32_e2_lr1e-4       0.8958      0.9000        0.9000
    r16_e2_lr1e-4       0.7833      0.8437        0.8437
    r8_e2_lr1e-4        0.7542      0.8008        0.7952
    r16_e2_lr3e-4       0.3500      0.4056        0.4056
    r16_e6_lr3e-4       0.1375      0.1690        0.1690
    r32_e2_lr3e-4       0.1250      0.1659        0.1659
    r32_e6_lr3e-4       0.1000      0.1611        0.1611

**T0 HOLDS.** 11 of 12 reproduce within 0.02 and ten of those reproduce to four decimals exactly; the banked
best is 0.9722 against 0.9722. The single cell that does not is `r8_e2_lr3e-4` at 0.9048 against a banked
0.6548 — V-131b's run of it had a collapsed seed (0.1619) and this one did not. **That cell is a knife edge,
not a number**, and it was flagged in flight before the sweep finished rather than after.

**T1 HOLDS.** The configuration chosen as the argmax of the validation mean has a test mean of **0.9722**,
against a bar of 0.9422.

**T2 HOLDS, and it is the headline.** The validation set picks **`r16_e6_lr1e-4`** — the same configuration
the test set picks. **A legitimate procedure finds exactly what the illegitimate one found.**

**T3 HOLDS trivially, because the two selections coincide.** Margin against the honestly selected comparator
**+0.0278**; against the test-selected comparator **+0.0278**. Identical. Neither clears the manuscript's
registered ten-point bar, which is the manuscript's own finding and is unchanged. Per-seed test accuracies
of the selected configuration: 0.9714 / 0.9738 / 0.9714.

**T4, and it corrects a worry of mine.** The prereg argued that 80 rows put the standard error near 0.97 at
about 0.019, so the top configurations would not be separable and the validation argmax could easily land
elsewhere. The standard error is 0.0175 as predicted and three configurations do sit within one of it — but
the **rank correlation between validation and test across all twelve is 0.993**. The argument was right about
the top of the table and wrong about the table: the configurations span 0.10 to 0.97, so ranking them is easy
and only the very top is close, and even there the argmax landed correctly.

**Prediction accuracy.** The prereg named T2 as the one I was least sure of and argued at length that it
would probably fail. It held. I was wrong, in the direction that costs nothing.

**What the manuscript may now say.** The disclosure stays — the configuration *was* selected on the test set,
and that sentence is load-bearing and stays. What can be added is that it **made no difference**: selecting
on a held-out validation split picks the same configuration and yields the same 0.9722, so the 2.9 / 2.6 /
2.9-point margins are what an honest procedure gives, not an artifact of a generous one. This converts a
disclosed weakness into a verified non-issue, and it is the second time in two runs that a suspected
instrument fault turned out to matter in only one direction (V-151 was the other, and there it mattered a
great deal).

**What it may not say.** That the fine-tune is reliable. One of twelve configurations is a coin flip between
0.16 and 0.95 on the same seeds, four of twelve sit at 0.10-0.41, and the comparator's 0.9722 is the best of
twelve, not what a practitioner gets by default.

SCOPE: bAbI task 2, Phi-2 frozen but for the adapter, V-50's instance set and answer-candidate restriction.
The handover's 1.000 is banked and was not re-run.
````
