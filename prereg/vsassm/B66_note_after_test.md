# B66 note after the test (written after the verdicts; results/b66_verdict.txt and the part files)

Date: 2026-11-23, about 14:25 (machine clock). Tests run once; one cut, registered order, before any test output was
read: (C)'s KF6 (B36's six key-free families at 72 configurations) was not run (compute note 2), so (C) has no
best-of-six or best-of-seven draws. The paper is not edited by B66.

## Results (ideal accounting, R1 = V 2 / 16 x 4-16 bits per fact x M 256 / 1,024; points)

**(A1) Reconciliation, first design - SDMt, Qwen3-8B / Llama-3.1-8B.** Pipeline code: -1.89 / -1.20 on B64's grid (504
configurations), +1.86 / +2.03 on the clean room's grid (32). Clean-room code (cr.py verbatim + B64's grid): -0.95 /
-0.16 on B64's grid, +2.44 / +2.22 on its own grid (its original numbers, reproduced exactly: 108 of 108 arm-cell test
accuracies). Grid effect +3.57 / +2.81, code effect +0.76 / +0.62, interaction -0.36 / -0.84. **Registered attribution:
GRID on both encoders.** On B64's grid both codes find SDMt level with or ahead of the first design (clean room: SDMt
leads on Qwen3-8B, level on Llama); on the clean grid both find the first design ahead. The remaining code difference
(about +0.7 in the store's favour in the clean room) is the clean room's own stores, store and size search (bisection,
no default-value bits); the sample reading shows a cell (Qwen3-8B, M 256, V 16, 4 bits per fact) where the two codes'
dev choices differ and the clean-room SDMt reads 16.7 against 12.9.

**(A2) Random maps (maps 0-3; two-level bootstrap over maps).** First design - SDMt map means: SDMt leads on 6 of 7
(Granite -2.46, Llama -1.33, Mistral -2.58, e5 -1.19, bge -4.48, MiniLM -3.94); Qwen3-8B level, -0.80 [-1.84, +0.20]
(the registered map, -1.89, was the most favourable of the four to SDMt there). Between-map SD 0.24-1.17. Final design -
best of seven map means: level on Qwen3-8B (+0.81), Granite (-0.80) and Llama (+0.70); memory leads on Mistral (-1.12),
e5 (-0.90), bge (-3.39), MiniLM (-3.48). B36's bar: passes on none of 7 for either design. Map 0 of the first-design
code reproduces B49's dev choices and dev accuracies in 168 of 168 cells.

**(B) LSfc (least squares with LAMc's corrected options on the store's own features).** Store - LSFR (identical random
features; only the write differs): least squares leads on 7 of 7 over R1 (-1.49 to -2.26), with two values on 7 of 7,
with sixteen on 6 (Llama level). LSFR - LAMc (same least-squares write, the store's random map vs LAMc's dev-chosen
features): LAMc leads on 7 of 7 (-1.92 to -4.82). On the store's principal features the write-rule gap is larger:
VSAPc - LSFP -4.01 to -11.19, least squares leads on 7 of 7. LSFX (per cell the better of the two feature sets) equals
LAMc within 0.8 points on every encoder (LAMc's whiten principal features and the store's whiten principal map are the
same features). **Registered reading: "both"** (and the write-rule criterion alone is met on 7 of 7 for both feature
sets): least squares leads by its write rule at the store's own features, and its principal-component features add
1.9-4.8 points on top.

**(C) Matched grids (4 language models).** Choosing from 72 random configurations instead of the full grid lowers the
chosen configuration's R1 test accuracy by a median 0.7-1.3 points for LAMc and 1.5-2.0 for SDMt. Store - LAMc at 72:
the sign holds in 20 of 20 draws on all four (medians -3.0 / -4.9 / -4.2 / -4.4 against -4.1 / -5.6 / -5.6 / -5.2 at the
full grid; range -1.5 to -6.0). First design - SDMt at 72: medians +0.1 / -0.3 / +0.3 / -2.1 against -1.9 / -2.3 / -1.2 /
-3.6, the full-grid sign in only 10 / 12 / 7 / 16 of 20 draws. Final design - SDMt at 72: medians +3.5 / +1.7 / +2.4 /
+0.5 against +1.5 / -0.2 / +0.9 / -1.0. By the registered reading none of these is "not material".

**(D) Store-level intervals.** Under the registered two-level interval (stores, then entities within store) the claims
(1) least squares leads the store over R1 on all 7, (2) with sixteen values on all 7, (4) least squares leads the
per-condition random / principal store on all 7, (5) least squares leads every Hebbian store variant (final and first
design, VSA-internal read rules, principal-feature store, per-condition choice) on all 7, and (6) the one-pass delta rule
leads the store on all 7 **hold**. (3) "with two values on the language models, e5 and bge" fails on bge (-1.05 [-2.51,
+0.29], level). (7) SDMt leads the first design on all 7 fails (Llama -1.20 [-2.43, +0.05], e5 -0.92 [-2.06, +0.21]).
(8) the best of seven leads the first design on all 7 fails (Llama). (9) the final design - best of seven labels change
on Llama (+0.91, level) and Mistral (-1.02, level); the bar count (0 of 7) does not. 13 of 189 labels change, all on
effects of at most 1.51 points. Caveat that bounds this: the two-level interval is 1.32-1.39 times as wide as the
entity-only one, about what the scheme gives when stores do not differ at all (it counts within-store sampling twice;
1.34-1.40 expected), and the store-only (cluster) bootstrap, the consistent one for this design, is 0.87-1.02 times as
wide and keeps every claim. So the data show little extra between-store variance; the two-level labels are a
conservative bound, which is how the paper should use them.

## Predictions

Held: PA1, PA2, PA3, PA5, PB1-PB4, PD1, PX. Partly failed: PA4 (SDMt leads on 6 of 7 map means and map 0 reproduces
B49 - held; between-map SD <= 0.8 on every encoder - failed on Llama 1.17, Mistral 0.92, bge 0.95); PC1 (LAMc's 72-draw
medians within 1.5 - held; SDMt's within 1.5 only on Llama and Mistral; KF6 not run); PC2 (store - LAMc keeps its sign -
held; first design - SDMt keeps it in at least 18 of 20 draws on none of 4 - failed); PD2 (claims 1, 2, 4, 5 hold and
claim 7 fails - held; claim 6 was predicted to fail somewhere and held on all 7 - failed). I underestimated how much of
SDMt's lead comes from its grid and overestimated how fragile the delta-rule claim is.

## Checks

(A) plain-loop scorer 1,428 values, max |difference| 1.4e-16; bits: SDMTCR / SDMTm 1,151,472 dev and test rows 0 over
budget, VSARm 6,300 store sizes 0 over budget and 0 not maximal; float64 re-derivation (b64_check's independent code
with the map's order seed; own float64 store for VSARm): SDMTCR 56 stores min agreement 0.9989, SDMTm 72 stores min
0.9988 (N identical 128 of 128), VSARm 36 stores 1.0000. (B) scorer 1,050 values 1.4e-17; bits 113,064 rows 0 over, 0 not
maximal, 0 formula mismatches; float64 re-derivation (own features from the seeds, own dual ridge, own quantizer) 56
(store, budget) pairs min 0.9949. (C) plain-loop scorer from the raw job files (b66_check.py cscore, written after the
test, disclosed) 168 region values, 1.7e-16; bits 10,390 new rows 0 over budget. (D) 357 point statistics by plain
loops, 1.4e-17. All pass. Sample reading by the AI-assisted process: results/b66_sample_reading.txt (no leak or scoring
artefact; the least-squares-only queries are near-paraphrases of the label lost to cross-talk).

## What the paper may say, must change, and must drop

1. **The SDM sentence** (B64's "an SDM whose hard locations are the training split's codes beat the store's first design
   over the region on every encoder"): **must change.** The registered reading requires "grid" (met) and "SDMt leads" on
   all 7 map means under (D) (not met: Qwen3-8B level over four maps; Llama and e5 level under the two-level interval).
   May say: the earlier disagreement with the clean-room implementation was the tuning grid, not the code: on the same
   504-configuration grid both implementations find SDMt level with or ahead of the first design (clean room -0.95 /
   -0.16, pipeline -1.89 / -1.20 on Qwen3-8B / Llama), on the clean room's 32-configuration grid both find the first
   design ahead (+1.9 to +2.4). Averaged over four random maps SDMt led the first design on 6 of 7 encoders (1.2 to 4.5
   points) and was level on Qwen3-8B; its lead depends on its larger grid: with 72 configurations, as many as the store
   searches, the first design and SDMt were level on three of the four language models (median -0.3 to +0.3; Mistral
   -2.1). Must not say "on every encoder" for SDMt vs the first design.
2. **"Least squares beat every Hebbian store variant on every encoder": survives, and may be strengthened.** It holds
   under the store-level interval (claim 5) and with least squares tuned on 72 configurations (sign in 20 of 20 draws on
   all four language models). New: it holds when least squares uses exactly the store's own random features (1.5 to 2.3
   points, all 7 encoders) and the store's own principal features (4.0 to 11.2), so the gap is the write rule; LAMc's
   principal-component features add a further 1.9 to 4.8 points over the store's random map. The size at a matched grid
   should be given beside the full-grid one (median 3.0 to 4.9 instead of 4.1 to 5.6 on the language models).
3. **Per-ingredient superposition verdict:** B64's restatement stands and is made more cautious: at a few bits per fact
   the Hebbian store met the pre-specified +2-point margin over the best key-free memory on no encoder; over four maps
   the final design was level with the best of seven on three language models and behind on Mistral and the three
   sentence encoders; the best key-free memory's edge over the first design depends on its tuning grid. Add (B): least
   squares' lead over the Hebbian write is present at identical features, so it is the write rule (decorrelation) that
   the Hebbian sum lacks, with principal-component features as a second, separate gain. "No bits on keys" stands as the
   shared mechanism; "superposition per se loses" still must not be said (all winning memories superpose).
4. **The unequal-grid limitation must be stated with numbers:** the store searched 72 configurations, LAMc 288, SDMt 504,
   the key-free six 468. Re-chosen from 72 random configurations, least squares stays ahead of the store in every draw
   (by 1.5 to 6.0 points), but SDMt's lead over the first design disappears on three of four language models, and the
   final design moves ahead of SDMt on three (median +1.7 to +3.5). The key-free six were not re-run at 72 (cut), so the
   effect of grid size on the best of six is not measured.
5. **"On every encoder" claims under (D):** keep (1), (2), (4), (5), (6) as stated. Restate (3): with two values least
   squares led on the four language models and e5, and on bge only under the entity bootstrap (level when test-store
   variation is included). (7), (8): see item 1. (9): the final design's lead over the best of seven on Qwen3-8B and
   Llama (+1.25 / +0.91) must not be called a lead: both are level over four maps (+0.81 / +0.70), and Llama is also
   level under the store-level interval (Qwen3-8B's registered-map lead survives it, +1.25 [+0.07, +2.45]). Methods may say
   that store-level intervals were computed and that a cluster bootstrap over test stores reproduced every label, while
   the conservative two-stage bootstrap changed 13 of 189, all effects of at most 1.51 points.

## Literature (checked after the result; search snippets, papers not re-opened)

Two-stage bootstrap: Field & Welsh (2007, JRSS-B 69:369-390, "Bootstrapping clustered data") distinguish the cluster
bootstrap from the two-stage (cluster then within-cluster) bootstrap and show the cluster bootstrap's variance estimates
are consistent; the two-stage scheme's double counting of within-cluster variance is in Davison & Hinkley (1997, sec.
3.8). Consistent with (D): our store-only (cluster) interval is the consistent one and keeps every label; the registered
two-stage interval is the conservative bound. Write rule: the optimal linear (pseudo-inverse / Wiener-Hopf) associative
memory removes the cross-talk that the correlation-matrix (Hebbian) memory keeps for correlated keys (Kohonen 1972;
Poggio 1975); recent capacity analyses of linear memories make the same comparison (arXiv 2605.05189; and a 2026 arXiv
paper on correlation-matrix memories under correlated keys, arXiv 2611.12374, reporting 4.2-4.5x cross-talk energy for
semantic keys). Supporting (B)'s reading; no conflict found. Data-placed SDM locations: Ratitch & Precup (2004), as in
B64. Sources: https://ideas.repec.org/a/bla/jorssb/v69y2007i3p369-390.html, https://arxiv.org/abs/2605.05189,
https://arxiv.org/pdf/2611.12374.

## Limitations

R1 only; (A1) on two encoders; the clean room's stores differ from the pipeline's (its own seeds), so its cells are
compared by region statistics, not query by query; SDMt's maps re-draw only its location order (its one random
component) and the six key-free families keep map 0; (C) without KF6; (D) uses each experiment's registered map only.

## Disclosures and compute

Placement: local (clean-room code, D, verdicts, checks) and the A800 box (~/b66_run; SDMTm, VSARm, LSfc dev and test,
SDMTCR test, (C)'s SDMT / LAMC tests; at most about 48 workers at a time; no retry loops; only my own processes touched). Rule deviation: the long
transfers and starts ran in the background, but several short status checks and two job starts ran as foreground ssh
commands with a timeout (none state-changing beyond starting registered jobs). Incidents (compute notes 2-4): a local `pkill` pattern matched
its own shell (no other process affected); two uploads to the box were cut by `timeout` and a re-plan truncated the
uploaded SDMTm plans (0 jobs had run; plans re-uploaded gzip-compressed and md5-checked); three clean-room test-log lines
were seen at 14:01 after every choice was committed. b66_check.py's (C) scorer and b66_sample_reading.py were written
after the test. Time: prereg 13:38, tests done 14:04, verdicts 14:16, notes 14:25 (3-hour box from 12:50). No GPU, no
paid API, no OpenAI call.
