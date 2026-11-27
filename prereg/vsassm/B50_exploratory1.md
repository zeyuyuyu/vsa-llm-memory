# B50 exploratory 1 (registered before it runs; reported beside the registered result, never instead of it)

Date: 2026-11-20, about 01:45 box time (commit b47f767; the first version of this line said 02:00 by mistake). Written after the registered B50 verdict was read (B50-1 = SURVIVES: every
standard OOD gate on DISC, matched on dev to the VSA gate's trained-type open rate, stays open on 94.5-99.6% of the
addition items, which DISC reads as bitwise OR; all close on integer division; HO(VSA) - HO(best DISC gate, MSP) =
+0.431 [+0.389, +0.475]).

Why: the VSA gate's score is a linear function of the same state. op_s = max_c (W_enc h) . a_c, and the encoder W_enc
was fitted by least squares (RMSE) to the problem vector, in which the type atom a_c has weight 1 and the bound operand
codes are nearly orthogonal to every type atom. So op_s is, up to cross-talk, the output of a least-squares linear
regression of the one-hot type indicator on h, thresholded at 0.8. DISC's type head is the same kind of linear map
trained by cross-entropy. If the held-out behaviour comes from the regression objective (outputs fitted to 1 on the
trained types, no softmax), a plain least-squares readout with no vector-symbolic structure should reproduce it, and
the result would not be something the vector form adds.

Run (b50_expl1.py; no LLM forward; B40's cached layer-20 states): ordinary least squares with intercept (float64,
torch.linalg.lstsq) from the 16,000 training states to the 8-way one-hot type (MSE-ONEHOT); score = the largest of the
8 outputs. Gates: (a) score > 0.8 (DE25's number, as the VSA gate); (b) the registered matching rule (open iff score
>= the 797th largest of the 800 dev scores, dev from B40's dev cache). Evaluated on B50's test-state cache (2,500
items; a separate batched forward, so a few items near a threshold may differ from a live run): trained-type and
held-out open rates, AUROC; accuracy composed from DISC's forced-open runs (seeds 0-2) and BASE (open -> DISC's output,
closed -> BASE's); HO(VSA) - HO(DISC with this gate) with the paired bootstrap CI of the registered bar. Nothing is
tuned (no ridge, no threshold search beyond the two fixed rules).

Reading (exploratory): if MSE-ONEHOT at either gate keeps TR within 0.02 of DISC-0.8 and HO within 0.02 of VSA's (or
better), the held-out advantage is a property of the least-squares readout, not of the VSA format, and the paper must
say so; if it opens on held-out items like the CE-trained gates, the advantage is tied to the VSA encoder's target
(the HRR problem vector) as far as this test can tell.

Prediction: MSE-ONEHOT closes on >= 90% of addition and division items at both gates, opens on >= 99% of trained-type
items at gate (b), and its AUROC (trained vs held-out) is >= 0.99; HO(VSA) - HO(DISC with MSE-ONEHOT) in [-0.03, +0.02].

## Live check (added after the cached/composed result above was read, before the live runs)

Cached/composed result: MSE-ONEHOT AUROC 0.9999; gate > 0.8 opens on 99.95% of trained-type items, 1.6% of addition,
0% of division; composed TR 0.9293, HO 0.9720, HO(VSA) - HO -0.007 [-0.017, +0.003]; dev-matched gate (tau 0.8902)
TR 0.9258, HO 0.9800, HO(VSA) - HO -0.015 [-0.026, -0.006]. To rule out cache or composition artefacts, the same gate
(the same least-squares weights, saved by b50_expl1.py fit as out/lsq_W.pt) is run live in DISC's forward pass
(b50_run.py --gate lsq), DISC seeds 0-2, thresholds 0.8 and 0.8901834484174533, full test split, once each. Reported
beside the composed numbers; nothing else changes. (b50_run.py gains the 'lsq' gate; every registered gate is
unchanged.)
