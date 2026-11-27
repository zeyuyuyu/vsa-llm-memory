# B40 exploratory 1 (registered before it runs; reported beside the registered result, never instead of it)

Date: 2026-11-19, about 07:50 box time. Written after the registered task-N verdict was read (R2 = DISC WIN,
VSA - DISC = -0.032 [-0.046, -0.019]).

Why: the hand-read of the VSA arm's task-N errors (results/b40_handread.txt) shows that 14 of 16 sampled operand
misreads are a digit read as 0. DE25's clean-up keeps a digit only if its best similarity exceeds 0.5 and otherwise
writes 0 (llama/vsa_engine.query_digit). On the DEV cache (seed-0 reader, layer 20), plain argmax clean-up (no
threshold) reads type and both operands exactly on 0.971 of dev rows, against 0.896 with DE25's threshold and 0.946
for the discrete heads. Under read noise the VSA reads stay fragile with or without the threshold (dev, rho 0.25:
0.636 argmax / 0.549 threshold / 0.920 DISC), so the noise result is not a threshold artefact.

Run: the three trained task-N VSA seeds, unchanged (same encoder and decoder weights), evaluated once on the test
split with digit_thr = 0 (argmax clean-up); nothing retrained. Name: vsaargmax_s{0,1,2}.
Reading: R2's bars against DISC (the registered DISC runs), reported as EXPLORATORY. Prediction: VSA-argmax read exact
>= 0.96 on test and VSA-argmax - DISC in [0, +0.04] (the write-back, trained on thresholded reads, loses about one
point).
Code: VSAN takes digit_thr (default 0.5 = DE25, unchanged for every registered run); b40_run --digit_thr.
