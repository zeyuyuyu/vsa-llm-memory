# B53 pre-registration: the store's value codes (antipodal at V = 2, orthogonal at V = 16), B49's panel otherwise unchanged

Date: 2026-11-20 (US Eastern; machine clock Asia/Shanghai). Written and committed before any memory with B53's value codes has answered a dev or test query (the only runs so
far are b53_run.py's selftest on random toy data and a dry run of b53_verdict.py on B49's own e5 files with the VSA
unchanged, to test the code path). A first placement on the A800 box CPU was abandoned before any run started (code-file
copy at about 30 kB/s, box load 108 of 128 from other sessions); nothing ran there.

## Why

Referee (9th read), M2a: in B49 the VSA store binds keys to independent random +-1 value codes (b32_run.Feat: rows of a
random sign matrix), while the least-squares associative memory (LAM) is trained on centred one-hot targets, which for
V = 2 are antipodal. With two random codes about half of the coordinates carry the same code for both values; those
coordinates add the same amount to both values' scores and cannot separate them, so at V = 2 the store uses about half
of what it stores. B49's registered loss to LAM over R1 (VSA - LAM -1.9 to -3.4 points on the seven encoders) is driven
by V = 2 (LAM ahead by 1-16 points per cell), while at V = 16 the store was ahead of LAM in 20 of the 24 LLM cells.
Fix: re-run with antipodal (V = 2) and orthogonal (V = 16) value codes, report V = 2 and V = 16 separately, and report
each arm's tuning-grid size. B53 does that; the result goes into the paper whatever it shows.

Algebra (stated now, tested in P2): for V = 2 the decision is the sign of score(0) - score(1) = sum_j phi_j(q) Q(S)_j
(c_j(0) - c_j(1)). With random codes only the coordinates where c(0) != c(1) (about d / 2) enter; with antipodal codes
all d do, and the store is then the d-dimensional Hebbian (outer-product) store. So the antipodal store at b bits per fact
should read like B49's random-code store at about 2b. For V = 16 no +-1 code can do much better than random codes on
pairwise separations: two distinct rows of a Hadamard matrix also differ on exactly half of the coordinates, and by
Plotkin's bound the average pairwise distance of 16 +-1 codes is at most d x 8 / 15 (a simplex, 7% above d / 2). What
orthogonality removes is the code cross-talk term (random codes' inner products, about +-sqrt(d)), which is small next to
the projection noise. So B53 expects a gain at V = 2 and little change at V = 16 (P1-P3).

## Design (the ONLY change from B49 is the store's value codes)

Panel, splits, stores, queries, budgets (1, 2, 4, 8, 16, 32, 64 bits per fact), accountings (ideal primary, real
secondary), competitors and their dev choices, BEST (the dev-chosen family among IDX, SF, CM, BF, FBF, SDM, LAM, LHI,
LHC, LHS), statistics and regions are B49's (prereg/B49_panel_mechanism_competitors_prereg.md). The competitors' test
predictions are B49's merged files (b49_out/fam/{enc}/test/all49_*), reused unchanged; only the VSA's predictions are
replaced.

**Value codes (b53_run.py, code_block):** r = row 0 of B49's random sign block for (encoder, prep, fmap, block) (same
seed, so c(0) is B49's code of value 0 and every projection, store, subset order and quantizer seed is B49's).
- V = 2, antipodal: c(0) = r, c(1) = -r.
- V = 16, orthogonal (pre-declared; not a simplex): c_j(v) = r_j H16[v, j mod 16], H16 the Sylvester-Hadamard matrix of
  order 16: the 16 codes are 16 rows of a Sylvester-Hadamard matrix of order 16 x 2^k (times a fixed +-1 diagonal)
  restricted to the store's d coordinates; exactly orthogonal on every aligned window of 16 coordinates (d is cut at
  arbitrary length, so at most 15 trailing coordinates break exact orthogonality).
- V in {128, 1024}: B49's random codes, kept and not re-run (B49's predictions reused; reported as random codes).
Codes are fixed, drawn from a seed or constructed, the same for every store: free under B32's accounting, as before.

**Store configuration:** re-selected on dev per cell (M, V, bpf, acct) with B32's rule (highest mean dev accuracy over
the 18 dev stores of V in {2, 16}; within 0.1 points tied, first in grid order) over B32's 72 configurations (prep
{center, whiten} x fmap {lin, sgn, sgn3} x q {1, 2, 4} x stored share {1, 1/2, 1/4, 1/8}); 1-2 and 4-64 bits per fact
in one pass (each configuration's predictions equal those of running it alone: b53_run.selftest). Test once (25 test
stores per V: 20 at M = 256, 5 at M = 1,024), after the choices are committed.

**Arms in the verdict:** VSA (B53 codes); VSAR (B49's random-code store, B49's test predictions, for the paired effect of
the codes); LAM; BEST; BEST36, BESTNL, ORACLE as in B49 (secondary).

## Tuning-grid sizes (configurations searched per cell on dev; reported in the paper's table)

| arm | configurations per cell (V 2 / 16) |
| --- | --- |
| VSA | 72 (2 preps x 3 feature maps x 3 quantizers x 4 stored shares) |
| IDX | 48 (2 x 6 code families x 4 shares) |
| SF | 176 (2 x 88) |
| CM | 132 (2 x 66) |
| BF | 60 (2 x 30) |
| FBF | 12 (2 x 6) |
| SDM | 40 (2 x 20) |
| LAM | 48 (2 preps x 2 feature maps x 4 ridge alphas x 3 quantizers; p set by the budget) |
| LHI / LHC / LHS | 20 / 66 / 88, each on a learned map chosen once per encoder among 16 (s, r) by B32's criterion |
| BEST | the best of the ten families' chosen configurations: 690 configurations per cell (+16 learned maps) |

## Bars (ideal accounting; 95% cluster bootstrap, entity within store, 10,000 resamples; B49's code)

- **Primary, per encoder: R1 (V 2 / 16, 4-16 bits per fact, both M; 12 cells) VSA - LAM.** Label: "VSA ahead" if the
  interval lies above 0; "LAM ahead" if it lies below 0; otherwise "level". **Panel count: the number of encoders with
  VSA >= LAM, i.e. not "LAM ahead"** (and, separately, the number with "VSA ahead"), of 7, of the 4 LLM families, of the
  3 sentence encoders. B49's +-2-point rule is reported beside it.
- Per V (required): the same statistic over R1-V2 and R1-V16 (6 cells each), and over R0 / RH by V.
- Secondary: R1 VSA - BEST with B49's H1-B49 bar (>= +2 and interval above 0; loss <= -2 and interval below 0); R0, RH,
  R2 against LAM and BEST; real accounting; VSA - VSAR (the effect of the codes) by region and V; per-cell table of
  V in {2, 16} with Bonferroni over the 28 re-run cells per encoder and accounting.
- R2 (V = 1,024) and every V in {128, 1024} cell are unchanged by construction (identical predictions); reported as such.

Readings (fixed now):
- VSA >= LAM on all 7 encoders: the paper may say that with antipodal / orthogonal value codes the store is level with
  or ahead of the least-squares associative memory over R1 on every encoder, with the per-V numbers beside it.
- LAM ahead on k >= 1 encoders: those encoders are named; for them "a least-squares associative memory beats the store at
  equal bits over R1" stands. If LAM is still ahead at V = 2 on most encoders, the paper must say that the store's V = 2
  deficit is not only the value codes.
- H1-B49 against BEST: as in B49 (passes named; no "LLM codes" statement unless all four LLM families pass).
- The code effect (VSA - VSAR) is reported for every encoder and V whatever its sign; B49's VSA numbers are then
  superseded for V in {2, 16} in the paper, with B49's kept as the random-code store.

## Predictions (stated so that they can be wrong)

Numbers come from B49's curves under the algebra above (B53 at b ~ B49's random-code store at 2b).

P1. V = 2: VSA - VSAR over R1-V2 is above 0 with the interval above 0 on all 7 encoders; the mean over encoders is between
    +1.5 and +5 points.
P2. Half the coordinates: over R1-V2, |VSA(b) - VSAR(2b)| <= 2 points (mean over the 6 cells) on at least 5 of 7 encoders.
P3. V = 16: |VSA - VSAR| over R1-V16 <= 1.5 points on every encoder.
P4. Primary: R1 VSA - LAM rises on every encoder relative to B49; LAM stays ahead over R1-V2 (interval below 0) on at
    least 5 of 7 encoders; over all of R1, VSA >= LAM on at most 3 of 7 encoders (LAM ahead on at least 4).
P5. R1 VSA - BEST: H1-B49 passes on 0 of 7; BEST ahead (interval below 0) on at least 6 of 7.
P6. R1-V16: VSA ahead of LAM on all 4 LLM families; not ahead on any of the 3 sentence encoders.
P7. R0-V2: VSA - VSAR > 0 on all 7 encoders.
P8. Checks: independent scorer |difference| < 1e-9; 0 chosen configurations over budget (and d maximal); float64
    re-derivation of the B53 store agrees on >= 99% of the re-derived queries.

## Checks (second angle; b53_check.py)

- Independent scorer: plain loops over the merged files, every re-run cell's VSA - LAM, VSA - BEST and VSA - VSAR
  against the verdict (|difference| < 1e-9).
- Bits audit: for every test record and cell, the logged dimension d of the chosen configuration from an own
  re-implementation of the size rule: bits(d) <= budget and bits(d + 1) > budget.
- Re-derivation: numpy float64 re-implementation of the B53 store (own Hadamard matrix from scipy.linalg.hadamard, own
  quantizer and reading; the generators' seeds are the registered ones) on 10% of the test stores of every encoder and
  cell (one store per (encoder, M, V, bpf), ideal accounting).
- Hand reading: Qwen3-8B and e5-base-v2, the R1-V2 cell where LAM leads most after the fix: 10 queries only the VSA
  answers right and 10 only LAM answers right (alias, label, gold), notes by me.

## Compute

Local CPU (16 cores; nice 10; no GPU), b53_local.sh: dev in two queues that bound the memory (the four LLMs' sgn3 parts
2 at a time, every other part 4 at a time), B32's choice rule, choices committed, then the test (LLMs one at a time).
Codes are the repository's files (md5: Qwen3-8B 1e4f5a91..., Granite c474cf93..., Llama a55892a0..., Mistral 52207e3e...,
e5 836cff40..., bge 5cda7b45..., MiniLM 9d2d9bcc..., as B49). No paid API.

## Files

b53_run.py (codes, dev, choice, test, merge, selftest), b53_local.sh, b53_verdict.py, b53_check.py.
Results: results/b53_*. Dumps under b53_out/ (not committed).
