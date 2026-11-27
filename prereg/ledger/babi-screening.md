# Ledger excerpt: Screening of the twenty bAbI tasks and the task-4 prediction

Supplementary Table `tab:prereg`, row "bAbI (screening of the twenty tasks)" (marked "ledger" in the Supplementary Information).

**Source.** `ledger.md` of the experiment repository `ngcls` (private; see `README.md` in this folder), at commit `0732e13` (`0732e13ef2322f308a99dcac3bf0bd2e25a0da10`, tag `archive-2026-11-11`); the copy read has MD5 `6843647bad4782ac58fcd341448bf95c`, identical to that commit's working tree. Line numbers refer to that file. For each part, the commit is the one that added those lines, from `git blame` and checked with `git log -S` on the heading; dates are commit author dates in UTC+08:00 (the machine's clock). Parts not from `ledger.md` say so.

| # | Part | Ledger lines | Heading date | Added in commit | Commit date |
|---|---|---|---|---|---|
| 1 | PRE-REGISTRATION: V-51 (taxonomy and screening bars) | 13381-13421 | none in heading | `0889a0a` | 2026-08-24T04:38:13+08:00 |
| 2 | Verdict: V-51 | 13423-13469 | none in heading | `175bacd` | 2026-08-24T04:49:46+08:00 |
| 3 | Earlier registration covering task 4: V-52 (task-4 arm never run) | 13471-13514 | none in heading | `a6a18fc` | 2026-08-24T04:51:35+08:00 |
| 4 | Verdict: V-52 (task 15 only) | 13516-13553 | none in heading | `4af9bc7` | 2026-08-24T05:34:58+08:00 |
| 5 | PRE-REGISTRATION: V-56, Part A is the task-4 prediction | 13719-13756 | none in heading | `86d5fec` | 2026-08-24T05:53:45+08:00 |
| 6 | Amendment: V-56 (Part A bars kept) | 13758-13794 | none in heading | `0c9080f` | 2026-08-24T05:54:59+08:00 |
| 7 | Verdict: V-58 (task 4; transcribed by the scheduled tick) | 13875-13917 | none in heading | `2340f2e` | 2026-08-24T09:33:26+08:00 |
| 8 | Verdict: V-58, second entry | 13919-13946 | none in heading | `354bce6` | 2026-08-24T10:29:11+08:00 |

**Depositor's notes** (written when this excerpt was deposited, 2026-11-21; not part of the ledger).

- **Row in the paper.** "expressibility taxonomy fixed before any number was seen; task 4: the algebra capped below 0.90 and a fine-tune above it"; outcome "prediction held (0.691 against 1.000)".
- **Matching registrations.** V-51 (part 1) fixes the taxonomy ("The expressibility taxonomy is fixed HERE, before any number is seen") and bars (a) candidate iff expressible and best prompting <= 0.60 in 3/3, (b) the search succeeds iff a task other than 2 is a candidate. Verdict (part 2): candidates 2, 4 and 15; SEARCH SUCCEEDS. The task-4 prediction is V-56 Part A (part 5): "CONDITION PREDICTS CORRECTLY iff parser <= 0.90 AND lora_direct - probe_argmax >= 0.05, in 3/3 seeds"; the amendment (part 6) re-derived it and kept the Part A bars. Verdicts (parts 7 and 8, the run is labelled V-58): parser 0.6907, fine-tune 1.000, CONDITION PREDICTS CORRECTLY. These match the row.
- **Discrepancy: nineteen tasks, not twenty.** The V-51 screen skipped task 19 ("skipped: no single-token answers"). The Supplementary Note says the model was prompted "on all twenty bAbI tasks".
- **"Before any number was seen".** True of the screen's own numbers; prompting results on tasks 1 and 2 (V-49) already existed when the taxonomy was fixed.
- **The task-4 bar, exactly.** The fine-tune had to lead the handover (probe_argmax) by at least 0.05 in three of three seeds, and the algebra (parser) had to stay at or below 0.90. The registration also declared a hybrid design: the four direction words get codebook rows, and the inverse relation is supplied from outside the algebra. A registered `spans` arm was not run (disclosed in part 7); no bar used it.
- **An earlier, opposite registration for task 4.** V-52 (part 3) first registered task 4 with bars "STRUCTURE PAYS iff handover - lora_direct >= 0.10" and predicted STRUCTURE PAYS there. Its task-4 arm was never run (the V-52 verdict, part 4, covers task 15 only). After the task-15 result, V-56 re-registered task 4 with the opposite prediction, before any task-4 arm ran. The paper's "prediction held" refers to V-56; V-52's superseded prediction should be disclosed beside it.

---

## Part 1: PRE-REGISTRATION: V-51 (taxonomy and screening bars) (ledger.md lines 13381-13421)

SHA-256 of the lines below, each followed by a newline: `8a2af1dcb143e5ac444cf46e8fef402ec2d596c5f6f5873a15f3bfb7947d504c`.

````text
## V-51 prereg: screen for where the model FAILS, before choosing where to apply structure. Before any code.

**The one regularity that survived today.** Structure pays where the model's own mechanism fails, and nowhere
else: at one hop the model is at .792 and a plain fine-tune reaches .999, so the algebra adds nothing; at two hops
the model is at .386 and the fine-tune stops at .801, so the handover's 1.000 is worth twenty points. V-48 gives
the mechanism -- the final-position state bridges one hop and not two. So the way to find further gains is not to
guess a task but to MEASURE where the model fails and then ask which of those failures the algebra can express.

**Screen.** Prompting only, no training: zero-shot and four-shot on bAbI tasks 1 through 20, next-token
distribution restricted to that task's own answer set, three prompt draws. Cheap, and it produces the candidate
list rather than letting me choose one.

**The expressibility taxonomy is fixed HERE, before any number is seen.** A task is EXPRESSIBLE iff its answer
can be obtained by unbinding a key from a bundle of role-filler pairs, with at most an ordering rule supplied
from outside the algebra (as used on tasks 1 and 2). By that definition, and committed now:
  EXPRESSIBLE: 1, 2, 11, 12, 13 (single and compound coreference are bundles of entities), 4, 5, 15, 16
  NOT EXPRESSIBLE: 3 (asks "before", a temporal order a superposition cannot hold), 6/9/10 (yes/no and
    indefinite knowledge -- the answer is not an entity retrieved by unbinding), 7 (counting), 8 (lists, the
    answer is a set with cardinality), 14 (temporal), 17/18/19 (positional and path reasoning over relations we
    do not encode), 20 (motivation, not retrieval)
Any task I later want to move between these lists must be moved in a new ledger entry with a reason, not
silently.

**Bars.**
(a) A task is a CANDIDATE iff it is EXPRESSIBLE by the list above AND the better prompting arm is <= 0.60 in 3/3
    draws. Both conditions, and the expressibility half was fixed before the numbers.
(b) THE SEARCH SUCCEEDS iff at least one task other than 2 is a candidate. Those tasks then get the full arm set
    -- handover, lora_direct, prompting -- in V-52, and any gain is reported against a supervised opponent as on
    task 2, never against prompting alone.
(c) THE SEARCH FAILS iff task 2 is the only candidate. Then the twenty-point gain is a single-task result, the
    paper says so, and I stop looking for more of them on bAbI rather than widening the taxonomy until something
    qualifies.

**Predictions on the record.** I expect tasks 2 and 16 to be candidates, and I expect 11 and 13 NOT to be --
coreference here is two sentences long and a 2.7B model resolves it. I also expect several NOT EXPRESSIBLE tasks
to score badly, which is exactly why the taxonomy had to be fixed first: a low number on task 3 or 7 is not an
opportunity for this algebra. Thirty-two predictions tested today and twenty-two failed.

**What I will not do.** I will not move a task into EXPRESSIBLE because it turned out to be hard, and I will not
report a prompting-only gap as a capability result. Task 2's twenty points exist because a supervised fine-tune
was measured on the same data; anything found here gets the same opponent or it does not get claimed.
````

## Part 2: Verdict: V-51 (ledger.md lines 13423-13469)

SHA-256 of the lines below, each followed by a newline: `60c5025914c259922a7ef461f754e278c37d750bb3c87892c457c230dc496f78`.

````text
## V-51 SEARCH SUCCEEDS: the failures were measured, and they form one family

**Screen (Phi-2, prompting only, answer set restricted per task, 300 eval instances, three draws).**
  task  class            best prompting   prior    verdict
  1     EXPRESSIBLE      .7978            .1767    --
  2     EXPRESSIBLE      .4356            .2267    CANDIDATE
  3     not-expressible  .2733            .2200    hard, not an opportunity (asks "before")
  4     EXPRESSIBLE      .4411            .1933    CANDIDATE
  5     EXPRESSIBLE      .8167            .1967    --
  6     not-expressible  .8467            .5100    --
  7     not-expressible  .6367            .5267    counting
  8     not-expressible  .8345            .4196    lists
  9     not-expressible  .8800            .6267    --
  10    not-expressible  .7267            .4367    --
  11    EXPRESSIBLE      .8356            .1900    --
  12    EXPRESSIBLE      .8067            .1900    --
  13    EXPRESSIBLE      .9422            .2000    --
  14    not-expressible  .4778            .2100    hard, not an opportunity (temporal)
  15    EXPRESSIBLE      .3867            .3433    CANDIDATE
  16    EXPRESSIBLE      .7467            .2900    --
  17    not-expressible  .5189            .5467    hard, not an opportunity (positional)
  18    not-expressible  .7911            .5267    --
  19    skipped: no single-token answers
  20    not-expressible  .8000            .1967    --
Candidates: 2, 4, 15. Hard but inexpressible, and therefore NOT opportunities for this algebra: 3, 14, 17.

**The candidates form one family and it is the family the mechanism predicts.** Task 2 is object to holder to
location. Task 4 is a two-argument relational lookup. Task 15 is entity to class to property. All three are
relational or multi-hop retrieval; none of the one-hop tasks qualified, and the two coreference tasks -- 11 and
13 -- came out at .8356 and .9422, which is the model resolving two-sentence pronouns without help. V-48 said the
final-position state bridges one hop and not two; the screen says the model's failures sit exactly where that
bridge ends.

**My prediction was right about the family and wrong about the members.** I predicted 2 and 16 would qualify and
11 and 13 would not. 2 qualified, 11 and 13 did not, 16 did NOT (.7467), and I missed 4 and 15 entirely. Which is
the argument for screening rather than choosing: had I picked by hand I would have run 16 and missed both new
candidates. Thirty-three predictions tested today, twenty-three failed.

**The taxonomy held, and it earned its keep.** Three tasks came in under .60 that a looser reading would have
called opportunities -- 3 at .2733, 14 at .4778, 17 at .5189 -- and all three are inexpressible for reasons fixed
in the prereg before any number existed: "before" is a temporal order a superposition cannot hold, task 14 is
temporal, task 17 is positional relations we do not encode. Without the taxonomy fixed first, task 3's .2733
would have looked like the best opportunity on the board.

**Next.** V-52 gives tasks 4 and 15 the full arm set -- handover, lora_direct, prompting -- because the prereg
requires any gain to be measured against a supervised opponent, never against prompting alone. Task 2's twenty
points exist only because lora_direct was measured on the same data, and these will be held to the same standard.
````

## Part 3: Earlier registration covering task 4: V-52 (task-4 arm never run) (ledger.md lines 13471-13514)

SHA-256 of the lines below, each followed by a newline: `661dc6fbfa0779cfd743049b79debe274df63c009d6bd8b745c4c7727c54ee42`.

````text
## V-52 prereg: the two new candidates, with the supervised opponent and a memorisability check. Before any code.

**What is being tested.** V-51's screen produced candidates 2, 4 and 15 from a taxonomy fixed before any number
was seen. Task 2 already has its full arm set. This entry gives tasks 4 and 15 the same treatment: handover,
lora_direct, prompting, and the through-loss projection, so any gain is measured against an opponent trained on
the same instances with the same answer labels -- never against prompting alone.

**Structures, with the outside rules named.**
  Task 15 (entity to class to property): membership statements bind class to entity, class statements bind
    property to class; the query unbinds the entity to get a class, cleans up against the classes present, then
    unbinds the class to get the property. Two genuine hops. Plural and singular mentions of one animal -- "Mice"
    and "mouse" -- must map to the SAME symbol, which is a mention partition the identity head has to learn.
    No outside rule is needed.
  Task 4 (two-argument relational lookup): each statement "A is DIR of B" is stored twice, as A keyed by
    (DIR, B) and as B keyed by (inverse(DIR), A), because the questions come in both directions. The inverse
    mapping north/south and east/west is SUPPLIED FROM OUTSIDE the algebra, exactly as the ordering rule is on
    tasks 1 and 2, and the manuscript must say so in the same sentence as any result.

**The memorisability check, which comes FIRST and can void a task outright.** The four task-15 instances I
inspected carry identical statements, so the knowledge base may be constant across the corpus. If it is, a
fine-tune can simply memorise "gertrude -> wolf" while prompting fails only because it was never told the
knowledge base, and any margin would be an artifact of who saw the corpus rather than of structure. So, before
any arm runs: count the distinct answer sets per query key across the training corpus.
  MEMORISABLE iff the query key determines the answer uniquely in more than 95% of training instances. A
  MEMORISABLE task is reported as such and NO capability claim is made from it, whatever the arm numbers say.
This applies to task 4 as well.

**Bars, per task, only for tasks that are not memorisable.**
(a) ADMISSION: parser - prompting >= 0.20, else void for lack of room.
(b) STRUCTURE PAYS iff handover - lora_direct >= 0.10 in 3/3 seeds.
(c) NO GAIN iff handover - lora_direct <= 0.02 in 3/3.
(d) GRADED otherwise.
Reported alongside without bars: the through-loss projection, and the label asymmetry -- the handover is given
identity labels the fine-tune is not, so every comparison reads "structure plus identity supervision against
answer supervision alone", as on task 2.

**Predictions on the record.** Task 15 is MEMORISABLE and will be voided -- the statements repeat verbatim, so
lora_direct should approach 1.0 by memorising the knowledge base. Task 4 is not memorisable, because the room
pairs differ per instance, and there I expect STRUCTURE PAYS, since the failure is a relation-inversion error
and the algebra is given the inverse rule while the fine-tune must learn it. Thirty-three predictions tested
today and twenty-three failed.

**What I will not do.** I will not report a margin over prompting as a gain on a memorisable task, and I will not
drop the memorisability check if it voids the task I would rather keep.
````

## Part 4: Verdict: V-52 (task 15 only) (ledger.md lines 13516-13553)

SHA-256 of the lines below, each followed by a newline: `2cbc96f34bbc1bd843823d4b440e7f40931c9f00df66b2fbf570375c0e5b9a58`.

````text
## V-52 NO GAIN on task 15 -- and it corrects the condition under which structure pays

**Numbers (bAbI task 15, 4 animals, 4 entity names, codebook 8, 900 train / 500 eval, 3 seeds).**
  lora_direct    [.992 .992 .990]  0.9913
  parser         [.824 .848 .836]  0.8360      exact codes, learns nothing
  probe_argmax   [.824 .848 .836]  0.8360      identical to parser, per seed
  spans          [.462 .470 .492]  0.4747      at the floor, as in every synthetic arena
  floor          [.442 .510 .468]  0.4733      last-token probe, the paper's convention
  prompting      0.3867 (banked, V-51)
Admission passes (parser - prompting = .4493). Verdict NO GAIN by bar (c), and the accurate statement is stronger
than the bar's wording: the fine-tune BEATS the handover by .168, .144 and .154, in three of three seeds.

**The ceiling is the algebra's, not the handover's.** probe_argmax equals parser to the seed, so identity is not
the limitation -- the supervised head is already reproducing exact codebook rows. The parser itself stops at
.8360, which means about one instance in six cannot be expressed by this bundle at all. Where the algebra is not
exact, a supervised fine-tune goes past it.

**Which corrects the condition I had been reading off task 2.** Two hops is NOT the condition.
  task 2   object -> holder -> location   parser and handover exact at 1.000   structure +20 over lora_direct
  task 15  entity -> class -> property    parser and handover cap at .836      structure -15 against lora_direct
The condition is that the ALGEBRA MUST BE EXACT on the task. Where it is, the handover inherits that exactness
and beats a fine-tune; where it is not, the fine-tune passes it. Task 2's twenty points are therefore a
conditional result and the manuscript must state the condition, not the hop count.

**My prediction was wrong twice over on this task.** I predicted task 15 would be MEMORISABLE (it is not: 225
distinct statement blocks, key->answer unique in 0.0000) and, having been cleared, that it would be a candidate
worth the arm set -- which it was, and the arms said no. Thirty-five predictions tested today, twenty-four
failed.

**A hypothesis for the .836, registered as a hypothesis and not asserted.** In this task a class can appear both
as a key and as a value inside one bundle -- "Wolves are afraid of cats" makes wolf a key while "Mice are afraid
of wolves" makes wolf a value -- and binding one symbol into a bundle in both roles creates interference that
HRR's cleanup cannot always resolve. Task 2 has no such reuse: objects, holders and locations are disjoint sets.
That is testable by measuring accuracy split on whether the queried chain touches a reused symbol, and it is
registered next as V-53 rather than written into the paper now.

**What goes in the manuscript.** The task-15 numbers, the statement that the fine-tune wins there, and the
corrected condition. Not a hop-count story.
````

## Part 5: PRE-REGISTRATION: V-56, Part A is the task-4 prediction (ledger.md lines 13719-13756)

SHA-256 of the lines below, each followed by a newline: `6428b594579fb9b681c98a0639428c49e6fdaf7b69f30cd41504908b08e34bdf`.

````text
## V-56 prereg: use the condition to PREDICT, and test whether the gain scales. Before any code.

**Why these two.** V-55 licensed a two-part condition: structure pays when the model's own single pass fails AND
the relation's key and value vocabularies are disjoint. A condition that only describes what already happened is
worth little. Part A turns it into a prediction that can fail. Part B asks whether, where the condition holds, the
gain grows with the number of hops -- which is the difference between a narrow result and a scaling one.

**Part A: task 4, predicted to FAIL by my own condition.** In bAbI task 4 every argument of the relation is a
room: "The office is north of the kitchen" makes office a value and kitchen part of the key, and elsewhere kitchen
is a value. Key and value vocabularies are the SAME set, so V-55 says the bundle should cap. Prompting is .4411,
so the model does fail there and the first half of the condition holds; the second half does not. Arms: floor,
parser, probe_argmax, spans, lora_direct, exactly as on task 15. The inverse relation (north/south, east/west) is
supplied from outside the algebra, and -- registered here as an amendment, because it is a real weakening -- the
four direction words are given CODEBOOK ROWS rather than codes derived from states, since after inversion a
direction may have no mention in the text. Rooms still come from span states via the handover. Any task-4 result
is therefore a hybrid and the manuscript must say so.
  CONDITION PREDICTS CORRECTLY iff parser <= 0.90 AND lora_direct - probe_argmax >= 0.05, in 3/3 seeds. The
  algebra caps and the fine-tune passes it, as reuse predicts.
  CONDITION FAILS iff probe_argmax - lora_direct >= 0.10 in 3/3. Structure would then win on a reused-role task,
  the condition would be wrong as stated, and I would withdraw it rather than add an exception.
  GRADED otherwise.

**Part B: does the gain scale with hops where the condition holds?** Arena C, roles and fillers drawn from
disjoint vocabularies by construction, at HOPS = 1, 2, 3, with the full arm set including lora_direct fine-tuned
on the same instances. bAbI has no three-hop disjoint-role task, so this half is synthetic and the manuscript
will say that.
  THE GAIN SCALES iff (probe_argmax - lora_direct) is >= 0.10 at HOPS = 2 AND strictly larger at HOPS = 3 than at
  HOPS = 2, in 3/3 seeds.
  THE GAIN DOES NOT SCALE iff the difference at HOPS = 3 is not larger than at HOPS = 2 in 3/3.
  Admission per depth: parser - floor >= 0.20, else that depth is void.

**Predictions on the record.** Part A: CONDITION PREDICTS CORRECTLY. Part B: THE GAIN SCALES, because the model's
single pass degrades with hops while the algebra with disjoint roles does not until capacity runs out, and V-43
put that capacity limit at four hops, not three. Thirty-seven predictions tested today and twenty-five failed.

**What I will not do.** If Part A comes out CONDITION FAILS I will withdraw the two-part condition rather than
patch it with an exception for task 4. If Part B does not scale, the twenty points stays a single-depth result and
the manuscript says the gain does not grow.
````

## Part 6: Amendment: V-56 (Part A bars kept) (ledger.md lines 13758-13794)

SHA-256 of the lines below, each followed by a newline: `909254b89af5a226116f31e9f9dcae0e510174a7841f7ecf6ff408ef7a3644fa`.

````text
## V-56 AMENDMENT, registered before Part B runs: my own condition is refuted by data already in this paper

**What I got wrong, and it is in the manuscript as of twenty minutes ago.** I wrote that the relation's key and
value vocabularies must be DISJOINT. Arena C, which has been in this paper all along, draws roles and fillers from
the SAME vocabulary -- v13's generator takes both from range(N_SYM) -- and its parser arm reaches .934 to 1.000.
By the coarse condition it should have collapsed to one half. It does not. So the coarse condition is false, and
it is false against evidence I already had, which I should have checked before writing it.

**The refinement, which explains four datasets rather than two.** V-55's collapse needs the spurious cross-term to
be a candidate at cleanup. Unbinding a bundle by key k returns the intended value plus, cleanly, whatever item k
was bound to as a value elsewhere -- but that item only wins if the cleanup is allowed to choose it. So:
  COLLAPSE HAPPENS iff the cleanup candidate set CONTAINS the spurious item.
  V-55 SHARED: candidates are the whole shared pool, the spurious item is in it -> .50, collapse.
  Arena C: candidates are that level's FILLERS while the spurious item is a ROLE -> not in the set -> .934-1.000.
  Task 15: candidates are the animals and the spurious item is also an animal -> in the set -> .836 ceiling.
  Task 2: candidates are locations while the spurious item is an object or a person -> not in the set -> 1.000.
The operative property is therefore the CANDIDATE SET, not the vocabulary. Disjoint vocabularies are one way to
get it; a candidate set restricted to one role's inventory is another, and is what arena C and task 2 have.

**Consequences I am committing to now.**
1. The manuscript sentence stating the disjointness condition is WRONG and comes out, replaced by the candidate-set
   version, with arena C named as the case that refutes the coarse reading. This is a correction of my own text,
   not a rewording.
2. V-56 Part B changes accordingly: arena C already has a restricted candidate set, so it cannot test the
   condition by varying vocabularies. Part B keeps only its scaling question -- does the gain over lora_direct
   grow with hops -- and the condition question moves to a new cell:
   CANDIDATE-SET TEST, in the algebra alone as V-55 was: two-hop retrieval with SHARED vocabularies, cleanup
   restricted to the true value inventory versus cleanup over the whole pool. Predicted: restricted recovers
   1.000, unrestricted stays at .50. If restricted does NOT recover, the refinement is wrong too and I withdraw
   the mechanism claim entirely rather than refine it a third time.
3. Part A's prediction for task 4 is re-derived under the refinement rather than kept: in task 4 the candidate set
   is the rooms and the spurious item is also a room, so the refined condition still predicts a cap. The Part A
   bars stand unchanged.

**Prediction on the record.** The candidate-set test recovers 1.000 when restricted. Thirty-eight predictions
tested today and twenty-five failed, and this is the second time tonight that a condition of mine was refuted by
data already in the paper -- the first was the hop-count reading that task 15 killed.
````

## Part 7: Verdict: V-58 (task 4; transcribed by the scheduled tick) (ledger.md lines 13875-13917)

SHA-256 of the lines below, each followed by a newline: `e86c80ae73d44325a3674f2344507b3bf5d0f772379908c8cedebf034d77ffab`.

````text
## V-58 CONDITION PREDICTS CORRECTLY — task 4 caps exactly as the refined condition said it would

[Entry written by the scheduled tick at 2026-08-24T01:35Z, per the standing rule registered at the 00:30Z tick:
the run finished 23:33Z and the live session has been silent for two consecutive ticks, so the tick transcribes
the scorer's registered verdict verbatim from v58_logs/v58.json rather than leave a finished experiment
unrecorded. Nothing here is new judgment; the live session may amend on return.]

**Numbers (bAbI task 4, 900 train / 500 eval, seeds 42/123/456; scorer bars from 86d5fec Part A, re-derived
unchanged at 0c9080f; run commit 608d73b).**
  floor          [.618 .610 .616]  .6147
  parser         [.674 .702 .696]  .6907
  probe_argmax   [.674 .702 .696]  .6907
  lora_direct    [1.0  1.0  1.0 ]  1.0000
  prompting (banked)               .4411

**Verdict by the registered bars: CONDITION PREDICTS CORRECTLY.** parser <= .90 in 3/3 (the algebra caps) AND
lora_direct - handover >= .05 in 3/3 (the fine-tune passes it). Both halves are what the refined condition
predicted BEFORE the run: in task 4 every argument is a room, so the spurious cross-term at cleanup is
same-type and in the candidate set, and the algebra should cap while a fine-tune does not. LoRA's training
loss collapses to ~0 by epoch 2 in all three seeds and it evaluates at 1.000.

**This is the first V-56 arm to CONFIRM a registered prediction.** Prediction tally: forty tested across this
arc's day, twenty-seven failed (V-57's scaling prediction failed; this one held).

**What this does and does not license.** It does NOT restore the mechanism claim withdrawn at V-56b — there is
still no passing isolating test, and the manuscript keeps the four-dataset pattern with the explanation
labelled unproven. What it adds is a fifth measured cell, and the first PREDICTIVE one: task 4 was called in
advance, from the same-type-spurious reading, and came out as called (.6907 cap vs 1.000 fine-tune). The
manuscript may report it as a successful out-of-sample prediction of the PATTERN, in "consistent with"
language, per the limit registered at 8f3daa3. The hybrid design must be stated wherever task 4 is reported:
direction words got codebook rows rather than span-derived codes, and the inverse relation is supplied from
outside the algebra (registered amendment at 86d5fec).

**Disclosures, transcribed as found.**
(1) The prereg arm list at 86d5fec named a `spans` arm ("exactly as on task 15"); v58.json contains no spans
    arm. No registered bar references spans, so the verdict is unaffected, but the arm list as run differs
    from the arm list as registered and that is recorded here rather than glossed.
(2) probe_argmax equals parser exactly, seed by seed — the handover sits on the algebra's ceiling here too,
    the same invariance every prior cell showed.
(3) parser - floor = .076, well under the .20 admission gate Part B uses; Part A registered no admission gate,
    so the bar stands as written, but the small parser-over-floor margin is on the record: most of task 4 is
    reachable without the algebra, and the registered claim is only about where the CAP sits, not about a gap
    over floor.
````

## Part 8: Verdict: V-58, second entry (ledger.md lines 13919-13946)

SHA-256 of the lines below, each followed by a newline: `083a67d861cefa2ba9c8c3a5b7b172e2625d790640a4689cfccdee7983027630`.

````text
## V-58 CONDITION PREDICTS CORRECTLY -- the first pre-registered prediction to hit on an unseen task

**Numbers (bAbI task 4, 6 rooms, hybrid design as declared: direction words on codebook rows, rooms from span
states, inverse rule from outside; 900 train / 500 eval, 3 seeds).**
  floor          [.618 .610 .616]  0.6147     last-token probe
  parser         [.674 .702 .696]  0.6907     exact codes -- THE ALGEBRA CAPS, as predicted
  probe_argmax   [.674 .702 .696]  0.6907     equal to parser per seed, as everywhere
  lora_direct    [1.000 1.000 1.000]          THE FINE-TUNE PASSES IT, as predicted
  prompting      0.4411 (banked, V-51)
Both registered halves hold in 3/3: parser <= .90 (it is .69), and lora - handover >= .05 (it is .31). The
prediction was written at 86d5fec before any task-4 arm data existed, from one property readable off the task
statement: every argument is a room, so the spurious cross-term at cleanup is same-type.

**What this makes the collection.** Seven mechanism-relevant cells, one of them predicted in advance:
  cross-type spurious item:  arena C .88-1.00 | task 2 1.000            -- algebra exact, structure +20 to +48
  same-type spurious item:   task 15 .836 (-15 vs FT) | task 4 .691 (-31 vs FT, PREDICTED)
The regularity "the algebra caps when the retrieval target's type also occupies the other role" has now made one
successful out-of-sample prediction. Still not a law -- the isolating experiment (V-56b) remains void and the
manuscript keeps "consistent with" -- but it is no longer merely a description of data already seen.

**The task-4 numbers also sharpen the practical guidance.** The fine-tune solves task 4 outright (1.000, loss
.0009 by epoch two) where prompting sits at .44: this is a task the MODEL can learn but the ALGEBRA cannot
express cleanly, the mirror image of arena C, where the algebra is exact and the fine-tune cannot learn at all.
Between them they bracket exactly when each tool applies.

**Prediction tally: forty tested today, twenty-seven failed** -- and the ones that held share a shape: they were
derived from a measured mechanism (V-48's one-hop shortcut, V-52's exactness condition, this one) rather than
from hope.
````
