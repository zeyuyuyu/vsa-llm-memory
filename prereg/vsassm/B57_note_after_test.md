# B57 note after the test (test run once; results/b57_verdict.txt, b57_predictions.txt, b57_check.txt)

Primary (registered bar): R1 VSAP - LAM. The store on principal-component features is not behind LAM on 4 of 7
encoders: all four LLMs, TOST-equivalent at +-2. Its 95% interval excludes 0 on Qwen3-8B (-0.75 [-1.35, -0.15]), but that
is within the margin. It is far behind on the three sentence encoders (-8.2 to -10.7) and ahead on none. The cause is
V = 16: R1-V16 VSAP - VSAS is -1.9 to -4.9 on the LLMs and -13 to -14 on the sentence encoders. This is the mechanism
registered for Q2: with Hadamard codes each residue class reads a fixed 1/16 of the principal coordinates. The hand
reading shows near-copies of labels misread at V = 16. At V = 2 the principal-component features help the store:
VSAP - VSAS is +1.4 to +4.0 on 5 of 7 (interval above 0). Even so, LAM on the same kind of features stays ahead at V = 2
on 6 of 7 (R1-V2 VSAP - LAM -0.9 to -4.8; Mistral +0.2, level).

Secondary, registered: VSAPX (per cell the dev-best of VSAS and VSAP) - LAM over R1 is ahead on all four LLMs (+0.7 to
+1.6, intervals above 0), level on bge (-0.85, not behind) and behind beyond the margin on e5 (-1.52 [-2.17, -0.87]) and
MiniLM (-1.67 [-2.40, -0.95]). Chance-normalised, VSAPX - LAM is level on Qwen3-8B, Granite and Llama, ahead on Mistral
(+1.23 [+0.15, +2.37]), and behind on e5 and MiniLM (bge -1.07 [-2.09, +0.01]). VSAPX - BEST over R1: BEST ahead on 6 of 7,
Llama level.

Maps (qwen8, e5, MiniLM; registered + 3 new):
- The between-map SD over R1 is 0.30-0.54 points for VSAS - LAM, 0.25-0.42 for VSAS - BEST and 0.24-0.44 for VSAS - BEST36.
- The two-level intervals are 10-46% wider than the entity-only ones.
- The registered map was the most favourable of the four on qwen8, on all three comparisons. VSAS - LAM was +0.37 against a map mean of -0.39 [-1.07, +0.32].
- On e5 the registered map was the least favourable of the four, on all three comparisons.
- On MiniLM it was the most favourable against LAM and BEST, and third of four against BEST36.
- On the map mean, VSAS stays not behind LAM on qwen8 and behind on e5 (-2.34) and MiniLM (-2.27). Against BEST36 it is ahead on qwen8 (+9.96) and e5 (+1.48) and behind on MiniLM (-1.34).

Online writes:
- The one-pass normalised delta rule is behind LAM over R1 on 6 of 7 (-0.9 to -2.75; Mistral level). It is ahead of the store at V = 2 on 5 of 7 and behind it at V = 16 on 6 of 7.
- RLS on principal components equals ridge LAM restricted to principal components. The recursion matches the batch solve to 3e-15. Over R1 it is +0.05 to +0.93 against LAM.

Write cost (median over the R1 cells), in multiply-adds and working numbers:
- Hebbian bundle: about 2e6, working memory d.
- LAM batch: about 3-5e8, with about 1e6 numbers of working memory.
- NLMS: about 2-5e6, with about p c numbers.
- RLS: 1e8-1e9, with p^2 numbers.
- Store retraining, where it is chosen: 4e6-3e9, with ns d cached features.

Accounting: raw (bits stored) moves R1 VSAS - BEST 0.6-2.5 points toward the store and VSAS - LAM by -0.2 to +0.75. VSAS's not-behind count against LAM is the same under raw accounting (5 of 7). VSAPX stays ahead of LAM under raw accounting on Qwen3-8B, Llama and Mistral; Granite becomes level (+0.49 [-0.18, +1.14]).

Failed predictions:
- Q3: VSAP was not behind on 4 encoders, not at most 3.
- Q5: on qwen8 the registered map lies above the two-level interval of the map mean.
- Q6: on e5 the registered map is the lowest on VSAS - BEST36, by more than 1 between-map SD.
- Q9: the re-derivation agreement on MiniLM is 0.970 (LMMSE read) and 0.955 (retrained), below 0.99. All of that disagreement sits in one configuration, center:psgn with d = n = 384. Its last two principal coordinates are numerically zero (median |value| 5e-11 and 3e-17), so their signs are rounding noise that differs between the float32 run and the float64 re-derivation. With the run's own basis the agreement is the same, 0.973 / 0.960.
  - This is a defect of the 'psgn' map at d = n on rank-deficient preps: it should drop null directions. It affects the MiniLM cells that chose center:psgn (3 of 12 R1 cells) and any other cell where psgn reaches the null directions.
- Held: Q1, Q2, Q4, Q7, Q8.

Other checks: plain-loop scorer 0 difference; bits 0 over budget and 0 not maximal (38,500 configurations).

Disclosed after the test, analysis only:
- b57_check.py gained a diagnostic mode (B57_LAMFEAT_BASIS) and per-cell logging of re-derivation cells below 0.99.
- No arm, choice or bar changed.
