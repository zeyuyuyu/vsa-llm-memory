# V-151 pre-registration — the capacity exponent at a resolution that can actually resolve it

Written and committed BEFORE the V-151 sweep and before any V-151 number exists.

## Why this run

V-149 (fc874c8, corrected b06af87) swapped only the algebra across HRR / MAP / BSC / FHRR and measured a
capacity ladder per model. Its P3 bar — that every model's exponent in `k* ~ d^alpha` sits in [0.85, 1.15] —
FAILED, at MAP 0.700 and FHRR 0.600. The entry's own reading is that **the failure is the instrument, not
the models**:

> The k grid is octave-spaced (32/64/128/256/512), so k* can only take octave values and a single cell
> crossing the 0.70 line halves it. MAP's alpha is 0.700 for one reason: its d=4096 k=256 cell is 0.680,
> 1.2 points under the line, where HRR's same cell is 0.732. [...] All four ladders are consistent with
> linear-in-d; the registered exponent bar cannot tell "sub-linear" apart from "one cell landed two points
> short." I registered a bar my grid cannot support.

It also named the fix: "a finer k grid, or k* by interpolation across the 0.70 crossing". This registration
does both, and decides whether the sub-linearity is real.

## Two censoring faults in V-149's grid, fixed here

Reading V-149's banked curve rather than its summary:

- **BSC is left-censored at d=512.** Its k=32 cell is 0.655, already under the 0.70 line at the smallest k
  the grid contains, so its k* is below the grid and V-149 recorded it as 0. The grid must extend down.
- **The top is close.** FHRR and HRR both reach k*=256 at d=4096 on the octave grid, so their crossings lie
  between 256 and 512; a grid stopping at 512 leaves almost no headroom. V-146 was invalidated by exactly
  this (right-censoring made 8x growth look like 4x), and it is not repeated.

So the grid runs `k in {8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512, 724, 1024}` — fifteen values
at `sqrt(2)` spacing, bracketing every crossing V-149 saw with room on both sides.

## Design

Everything except the k grid and the k* definition is V-149's, unchanged, so that any difference is the
resolution and not the protocol: four algebras (HRR, MAP, BSC, FHRR) from Survey Part I sec 2.3; the V-101
capacity protocol on the same 400-instance validation slice with the same pooling; `d in {512, 1024, 2048,
4096}`; `n = 200` instances per cell; capacity seeds {0, 1, 2}; 200,003-row codebook; `PYTHONHASHSEED=0`.
No language model, no API, no training.

**`k*` is defined by interpolation, not by the last clearing grid point.** The accuracy curve falls
monotonically in k; `k*` is the k at which it crosses 0.70, by linear interpolation **in log k** between the
two bracketing grid points. A cell whose accuracy at the smallest k is already below 0.70 is **left-censored**,
one whose accuracy at the largest k is still at or above 0.70 is **right-censored**; either is recorded as
such and excluded from every fit, with the exclusion printed.

## Bars

**S0 --- validity gate.** HRR's interpolated `k*` at each d lies in `[v, 2v)` for V-149's banked octave value
`v = 32 / 64 / 128 / 256`. That is what "the same measurement at finer resolution" means, and if it fails the
protocol drifted and nothing is read.

**S1 --- is the sub-linearity real?** For each of the four models, fitting `log k* = alpha * log d + c` by
least squares, the 95% bootstrap CI for `alpha` **contains 1.0**. Holds iff all four are consistent with
linear-in-d once the grid can resolve them.

**S2 --- one law, different constants.** `max(alpha) - min(alpha)` across the four models is **< 0.25**. This
is V-149's qualitative claim — "separated by a constant factor, not by a different law" — stated as something
that can fail.

**S3 --- the constants themselves.** At every d where both models are interior: `k*(BSC) / k*(HRR)` lies in
**[0.35, 0.65]**, and at `d = 512`, `k*(FHRR) / k*(HRR) >= 1.4`. These are V-149's "BSC at 1/2 and FHRR at 2x
HRR's capacity per dimension", restated as numbers.

**S4 --- descriptive, no bar.** The full interpolated grid; each model's `k*` against V-149's octave value;
any censored cell; and the fitted constant `c` per model, which is the quantity S3 is about.

## What I expect (recorded before the sweep, separately from the bars)

- S0 passes.
- **S1 holds.** This is V-149's own self-correction, and the whole reason for this run is that its grid could
  not test it. If S1 fails for MAP or FHRR at this resolution, the sub-linearity is real and V-149's
  correction was itself wrong — which would be the more interesting outcome, and I would rather find that
  than confirm my own earlier reading.
- S2 holds.
- **S3 is the one I am least sure of, and I want that on the record.** V-149 read "half" and "double" off an
  octave grid, where one step IS a factor of two by construction; the measurement could not have produced any
  other number. At finer resolution the true ratio can be anything, and [0.35, 0.65] is a band I am taking
  from an instrument that could not resolve it. S3 failing while S1 and S2 hold would mean the qualitative
  claim survives and the specific factors do not, and the manuscript must then stop saying "half" and "twice".

## Scope

2WikiMultihopQA validation slice, V-101's pooling, two-hop unbind-and-cleanup. Nothing here is about
analogical mapping (V-146 is Kanerva-style holistic transformation and V-148/V-150 are the Gayler & Levy
replicator; their exponents are different mechanisms and must never be pooled with this one). Nothing here
touches the language-model comparison. The open problem is Kleyko et al., Survey Part I sec 4.1.

## Protocol

`v151_score.py` frozen and committed with this file before the sweep runs, and run verbatim. Censored cells
are excluded from fits by the scorer, not by hand.
