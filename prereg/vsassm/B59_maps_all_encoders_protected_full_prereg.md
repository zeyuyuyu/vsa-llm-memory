# B59 pre-registration: map x entity intervals for every encoder; the protected-store search on the full budget grid with the uncoded store inside it

Date: 2026-11-21 (US Eastern 2026-11-20 evening; machine clock Asia/Shanghai, about 02:10). Written and committed, then
OpenTimestamps-stamped, before any B59 run. Nothing has been run for B59. The B59 code (b59_run.py for part 1,
b59_store.py for part 2, b59_verdict.py, b59_check.py) is written after this commit and committed before any
development run; it implements this document by calling B57's and B58's code unchanged wherever possible.

## Why (referee 15)

(1) B57 measured the variance over random feature maps (three new maps) only on Qwen3-8B, e5-base and MiniLM-L6, and
the paper states that "no verdict against the margin changed on the means over maps"; the other four encoders
(Granite-3.3-8B, Llama-3.1-8B, Mistral-7B, bge-small) have entity-only intervals. (2) B58's protected-store search was
cut by its compute notes (P-LDPC at octave resolution, test window +-2, one code family at e05-k32, no extension) and
the uncoded store was compared after the fact, not searched over. Both are repaired here.

## Part 1 (PRIMARY of B59): map x entity intervals on all seven encoders

**Fixed (B57 part 2, unchanged).** Panel, splits, stores (dev 6 + 3, test 20 + 5 per V at M = 256 / 1,024), queries,
budgets 1-64 bits per fact, ideal accounting primary, raw secondary, V in {2, 16}, region R1 = V 2 / 16 x 4, 8, 16 bits
per fact x both M (12 cells), R1-V2 and R1-V16 its halves. Map 0 = the registered map (B53 / B56 / B49 files). Maps
m = 1, 2, 3: seed tag `{enc}#b57m{m}` replaces the encoder name in every seed of the store's random components and of
LAM's rp projection blocks, exactly as b57_run.tag. Per map, each with its own development choice, exactly B57's
procedure (b57_run.mf_dev / b32_choose / engine / lamm / choose):
- the store **VSAS**: B53's 72-configuration matched-filter grid (B32's rule) -> on that bundle B56's LMMSE read
  (q x alpha, 15) and perceptron retraining (q x E x eta, 18), B56's STAR rule (higher dev mean, ties LMMSE);
- **LAMRF**: B56's LAM-RF on the map's chosen store features (q x alpha);
- **LAM**: B49's 48 configurations (the 24 rp ones re-run with the map's projection, the 24 pca ones B49's dev rows),
  B49's rule;
- **BEST**: B49's ten-family dev choice with LAM's dev accuracy replaced by the map's LAM;
- BEST36 (fixed, no map).

**New runs.** Encoders Granite-3.3-8B (b46), Llama-3.1-8B, Mistral-7B, bge-small: maps 1-3, all of the above (dev and
test). Qwen3-8B, e5-base, MiniLM-L6: B57's map runs and dev choices are used as they are (committed in B57).
**Secondary arm VSA** (the final design's matched-filter read, B53's rule, the numbers the paper quotes for the final
design): per map the MF-chosen configuration (results/b57_choicem_* for the three B57 encoders; the new B59 choice
files for the other four) read by the matched filter on the test stores (b32_run.vsa_grid with the chosen
configurations only, the path of b57_run.mf_dev); map 0 = B53's VSA predictions in the merged files. New for all
seven encoders (maps 1-3); dev choices for the three B57 encoders were committed in B57.

**Statistics.** For store in {VSAS (primary), VSA (secondary)} and competitor y in {LAM, BEST} (primary), {LAMRF,
BEST36} (secondary): per map the R1 difference (mean over the 12 cells of per-cell accuracy differences), the map mean
over the four maps, the between-map SD, and B57's **two-level bootstrap** of the map mean (b57_verdict.twolevel: maps
drawn with replacement from the four; entity clusters resampled once per draw, shared across maps; 10,000 draws; seed
59 + encoder index); 95% and 90% two-level intervals; the entity-only interval of the map mean; the registered map's
rank. Separately for R1, R1-V2 and R1-V16. Ideal primary; raw reported.

**Labels (on the map mean, two-level 95% interval).** *within the margin* = the interval lies inside (-2, +2) points
(the paper's definition: two one-sided tests at alpha = 0.025 per side); *not behind* = lower bound >= -2 (B56 / B57's
label); *behind* = upper bound < 0; *behind, not within the margin* = behind and not within the margin.

**PRIMARY BAR B1 (does every verdict against the +-2 margin that the paper states for the final design survive on the
map means?).** The paper states, from entity-only intervals on the registered map: (V1) the store is within the margin
of least squares on the four language models; (V2) the store is behind the best of ten on six of seven encoders, not
within the margin, and within it on Llama-3.1-8B. On the map means of VSAS with two-level intervals:
- V1 survives iff R1 VSAS - LAM is *within the margin* on each of Qwen3-8B, Granite, Llama, Mistral.
- V2 survives iff R1 VSAS - BEST is *behind, not within the margin* on exactly the six encoders Qwen3-8B, Granite,
  Mistral, e5, bge, MiniLM, and *within the margin* on Llama.
- B1 = SURVIVES iff V1 and V2 both survive; otherwise the changed verdicts are named (encoder, comparison, old and new
  label). The same two verdicts are evaluated for the VSA arm (secondary; the paper's quoted final-design numbers).
- Secondary verdicts (reported with the same labels): (V3) the sentence encoders against least squares (paper: behind;
  B56: behind beyond the margin on e5 and MiniLM, within on bge); (V4) per V on the four language models: least squares
  ahead with two values (R1-V2 upper < 0) and the store ahead with sixteen (R1-V16 lower > 0), 8 of 8.

## Part 2 (SECONDARY of B59): the protected store on the full budget grid, uncoded store inside the search

**Fixed (B58's protocol, registered prereg/B58_protected_store_prereg.md).** B38's 50 RULER test stores (4,776 probed
keys), B34's 12 dev stores (1,254), exact-key metric, cells, budget grid B = 2^(i/4) 8 ... 4,096, target 99.5%, the
seven i.i.d. 30-day conditions k1 ... k32 (e05 reported, not in the label), B38's table N* (best table T-rep / E-BCH /
E-LDPC / E-CAM; i.i.d. E-LDPC from B38, as B58), the store V-last, b58_store.evaluate unchanged (P-LDPC: B38's IRA
codes x 7 rates x information budget b in {76.1, 90.5, 107.6, 128.0}, OWN decoder batched, 60 iterations; P-BCH: the
tables' closed-form (n, t) x b), the censoring caps (dev 7, test 40 failed keys per store).

**Three families of store design.** U = the uncoded store (B38's V-last at information budget b = B, one design);
P-BCH (4 designs per budget); P-LDPC (28).

**Development (B58's registered procedure, no compute-note cut).** Pass 1 = every design at 90.5, 128, 256, ..., 4,096
(B58's dev1a / dev1b files: same code, same seeds, deterministic, reused). Pass 2 = every design at the quarter steps
within 3 grid steps of each (family, condition)'s pass-1 N* (top 4 budgets if not reached): P-BCH = B58's dev2 file (it
followed the registered rule); **P-LDPC pass 2 is run now** (B58 cut it, compute notes 2-3). U = B38's dev V-last
results at every grid budget (results/b38_dev.json.gz, 12 dev stores). Per (family, condition, budget) the design with
the highest dev share (B58's rule: ties the next harder condition, then design-list order); dev N* per family.
**Dev-selected union P-dev**: per (condition, budget) the dev-best design among U and the coded designs dev evaluated
at that budget (ties: next harder condition, then U, P-BCH, P-LDPC in list order).

**Test (run once) on the full grid.** Per (family, condition): every grid budget from the family's smallest feasible
budget (U 8; codes 90.5, the smallest budget at which a code fits over b = 76.1) to 4,096, each with that family's dev
design at the budget (where dev did not evaluate a budget, B58 compute note 2's rule: the nearest dev-evaluated budget's
design among those that fit, ties the larger budget). U's test evaluations are B38's wherever B38 evaluated V-last
(identical arm); missing U budgets are run with b38_eq.run_job (arm V-last). Coded test evaluations already in B58's
test files with the identical (store, family, design, budget, condition) are deterministic and reused (checked by
recomputation on a sample; see checks). Run order (for the time cut below): (T1) U's missing budgets; (T2) both code
families at their dev N* +- 4 grid steps; (T3) every remaining budget below those windows; (T4) every remaining
budget above them; (T5) damage, if time allows (below).

**Curves and N*.** Per family the test share at every evaluated budget (None where a store was censored).
**P-best (gi)** = the largest test share among U, P-BCH and P-LDPC at budget gi (the store's best design at every
budget; selected with knowledge of the test, which favours the store). **P-dev (gi)** = the test share of the
dev-selected union design. N* = the smallest grid budget with >= 99.5% there and at every larger evaluated budget
(B38's rule); an N* is reported as a bound when its region is not fully evaluated.

**PRIMARY BAR of part 2, H1-P (per k1 ... k32).** Delta(k) = grid steps(best table N*) - grid steps(P-best N*); "store
fewer cells" if Delta >= +2, "table fewer cells" if <= -2, "tie" if |Delta| <= 1; TABLE WIN / VSA WIN / MIXED / TIE as
B38's H1 over k1, k2, k4, k8, k16, k32. Store-cluster bootstrap (2,000 resamples, B58's method) of Delta and of P-best
N*. The same for P-dev (secondary). **G (does coding help the store?)**: per k, steps(U N*) - steps(min(P-BCH N*,
P-LDPC N*)): "helps" >= +2, "hurts" <= -2, else tie; and whether P-best's N* is set by a coded design that needs >= 2
steps fewer than U; the share of the seven k where coding helps. Per family N* at every k (both code families).

**Damage (lowest priority, only if T1-T4 finish before 03:40 machine clock).** Failed lines and stuck clusters,
natural layout, at k4 and k8 (4 conditions): each family with its matched i.i.d. dev designs at every grid budget from
the family's matched i.i.d. dev N* - 4 to + 4 steps, then B38's extension rule; P-best as above; compared with the
best non-HD table (B58's re-scored E-LDPC) and with U.

## Predictions (stated so that they can be wrong)

Part 1.
- P1. V1 survives: R1 VSAS - LAM within the margin on the map means on all four language models.
- P2. V2 survives: VSAS - BEST behind, not within the margin on the same six encoders; Llama within.
- P3. Between-map SD of R1 VSAS - LAM <= 0.6 points on each of the four new encoders.
- P4. Two-level 95% intervals of R1 VSAS - LAM are 5-60% wider than the entity-only interval of the map mean on each
  of the four new encoders.
- P5. V4 holds on the map means: 8 of 8 (LAM ahead at V 2, the store ahead at V 16, on the four language models).
- P6. bge-small: the B56 reading "within the margin of least squares" does NOT survive on the map mean (two-level lower
  bound below -2).
- P7. The VSA (MF) arm gives the same B1 outcome as VSAS.
Part 2.
- Q1. H1-P TABLE WIN; Delta <= -4 at every k (k1 ... k32).
- Q2. Coding helps the store at 0 of 7 k (share 0); P-best's N* equals U's at every k (within 1 step).
- Q3. P-LDPC pass 2 lowers P-LDPC's dev N* by >= 1 grid step at >= 3 of the 7 k.
- Q4. Test N*: P-LDPC <= P-BCH at k16 and k32; P-BCH <= P-LDPC at e05 and k8.
- Q5. P-dev's N* equals P-best's at >= 6 of 7 k.
- Q6 (damage, if run). In the natural layout P-best's N* is within 1 step of U's in all 4 conditions (coding does not
  rescue nor spoil the store's position independence because U is in the search).

## Checks (fixed now)

Part 1 (b59_check.py): plain-loop re-scoring of every cell accuracy and every per-map R1 point (|difference| < 1e-9);
bits audit of every new map configuration (store d by B32's rule, LAM p by B49's rule, 0 over budget); **cross-machine
agreement**: one bge development part (map 1, center:sgn) and one Granite engine dev job (map 1, M = 256, V = 2) run
both on the box and on the local machine, accuracies / predictions compared (0 differences expected; any difference
reported and its cell count given); an independent re-implementation of the two-level bootstrap (explicit loops,
different seed) on one encoder per group, intervals within 0.15 points; **sample reading by the AI-assisted process**
(results/b59_sample_reading.txt): on Granite and bge, the R1 cell with the largest between-map spread of VSAS - LAM, 10
queries where the maps disagree, with the alias, the gold value and each map's answer.
Part 2 (b59_check.py, B58's methods): second scorer by plain loops on a sample of test cells of every family; bit and
cell accounting of every new evaluation (0 over budget); N* recomputed by a separately written rule (0 mismatches);
reuse identity: a sample of >= 20 reused B58 test evaluations recomputed, failed-key lists identical; U evaluations run
anew vs B38 where both exist (identical); sample reading by the AI-assisted process of failed keys one step below N*
at k16 for each family.

## How the paper should read each outcome (fixed now)

- B1 SURVIVES: the paper may keep "no verdict against the margin changed on the means over maps" and extend it to all
  seven encoders. B1 changes: the changed verdict is stated with the two-level interval, and the sentence is restricted.
- H1-P TABLE WIN with U in the search and the full grid: "searched over the uncoded store and both channel codes at
  every budget, the store's best design still needs N cells against the table's M". Any TIE / store win: name the k.
- G helps at some k: the protected store is reported as the store's best form at that k.

## Compute, scope and time cut (registered now)

CPU only. Part 1 on the A800 box's CPUs (fresh directory ~/b59_run; at most 48 cores in total for B59, e.g. 12 jobs x
3 threads; nice 10; no other user's process touched), with the code and the encoders' data copied there; results pulled
back and merged locally. Part 2 dev pass 2 and test on the local CPU (<= 8 workers, free memory watched; other
sessions' processes run on this machine) and on the box's remaining cores within the 48. Time budget 2.5 hours from
01:57 machine clock. Cut rules, applied by a compute note committed before any test output is read: part 1 is not cut
(priority); part 2 tiers T5, T4, T3 are cut in that order if they cannot finish by 04:00 machine clock; an unfinished
tier is reported as not run, and N* in regions not evaluated are reported as bounds.

## Files

Code: b59_run.py, b59_store.py, b59_verdict.py, b59_check.py, b59_box.sh. Results: results/b59_choice_*.json,
results/b59_choicem_*.json (part 1 dev), results/b59_part_*.json, results/b59_verdict.{txt,json} (part 1);
results/b59_dev2_ldpc.json(.gz), results/b59_choice_store.json, results/b59_test*.json.gz, results/b59_store_verdict.{txt,json}
(part 2); results/b59_check.txt, results/b59_sample_reading.txt, results/b59_predictions.txt; notes
prereg/B59_compute_note*.md, prereg/B59_note_after_test.md. Dumps under b59_out/ (not committed).
