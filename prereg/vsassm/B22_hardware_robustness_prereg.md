# B22 pre-registration: does the RULER vector store stay exact under hardware-like precision, noise and failures?

Date: 2026-11-14. Written before any code for this experiment. Motivation: an internal referee read of the VSA-LLM
manuscript (r13) found that the paper argues hardware relevance in one paragraph and tests none of it. In-memory
hardware stores vector components at a few bits, reads them with noise, and has failed cells. B21 showed that a
sharded Fourier HRR store (d = 8192, at most 16 pairs per vector on average) returns exactly what a table returns on
5,400 RULER samples, in float32. B22 asks how much of that survives low precision, read noise and failed components,
and whether the loss can be bought back with shorter loads (more vectors) or longer vectors.

## Stores (fixed here)
The real B21 stores (b21_test outputs, the reader's grounded pairs, prereg/B21 + amendments 1-2), rebuilt on CPU from
the saved pairs: the first 10 samples (in file order) of each of niah_multikey_2 at 131072 and 32768, niah_multikey_3
at 131072, niah_multivalue at 131072 and vt at 131072 (50 stores; the largest hold about 6,000 pairs in 375 vectors).
In each store, up to 150 keys are read (all keys if fewer; otherwise an evenly spaced subset in table order).

## Metric
A key is read exactly if the store returns the same values in the same order as the table (the B21 read: unbind
key and position j, clean up against the store's value codebook, accept if the similarity is at least 0.5, stop at
the first j below). Reported: the fraction of keys read exactly over all 50 stores, and the number of stores in which
every probed key is read exactly. The threshold stays at 0.5 in every condition (no retuning).

## Conditions (each applied alone to the float32 B21 store; d = 8192, 16 pairs per vector on average)
- Q (precision, b = 1, 2, 3, 4, 6, 8 bits): every code (key, value, position) has its phase rounded to one of 2^b
  levels (b = 1 gives bipolar +-1 codes), and the stored vectors keep the real and imaginary part of each component
  at b bits: a symmetric mid-rise uniform quantizer clipped at 3 times the vector's RMS (b = 1: the sign), after which
  each vector is rescaled to its original RMS (one stored scale per vector).
- N (read noise): complex Gaussian noise added to each stored vector at read time, standard deviation r times the
  vector's RMS per component, r = 0.25, 0.5, 1, 2.
- F (failed components): in each stored vector, an independent random fraction p of components reads as zero,
  p = 0.05, 0.1, 0.2, 0.3.
- Trade (recovery): at b = 1, b = 2, r = 2 and p = 0.3, the same stores re-sharded at 8 and 4 pairs per vector, and at
  d = 16384 with 16 pairs per vector.
Random draws (noise, failure masks) use seed 0 per store.

## Bars
- H1: at b = 4 the store reads at least 99.5% of keys exactly.
- H2: with read noise r = 0.5 the store reads at least 99.5% of keys exactly.
- H3: in every condition where the default store falls below 99.5%, at least one of the trade settings (8 or 4 pairs
  per vector, or d = 16384) restores at least 99.5%.

## Predictions
Per read, the cross-talk has a standard deviation of about sqrt(load / d), 0.044 at the average load and 0.068 at the
largest load (38) seen in B21; a 1-bit quantizer keeps about sqrt(2/pi) = 0.80 of the signal.
- b >= 3 and r <= 0.5 and p <= 0.1: indistinguishable from float (>= 99.9% of keys exact).
- b = 1: between 90% and 99.5% (losses on the most loaded vectors); b = 2: >= 99.5%.
- r = 2 and p = 0.3: below 99.5%. All three bars hold.

## Compute
CPU only, local. No model is run.

## Amendment 1 (2026-11-14, after the registered noise and failure levels were scored, before any further level was run)
Every registered noise level (r up to 2) and failure level (p up to 0.3) read all 4,776 probed keys exactly, so the
registered ranges do not show where the store breaks; the prediction that r = 2 and p = 0.3 fall below 99.5% was wrong.
To locate the break, we add r = 3, 4, 6, 8 and p = 0.4, 0.5, 0.6, 0.7 on the same stores and keys, each alone, plus
the three trade settings at the first added level that falls below 99.5% (if any). These levels are descriptive: no
bar is attached to them, and they are reported as added after the registered levels were seen.

## Amendment 2 (2026-11-14, after the registered conditions were scored; post hoc, labelled as such)
The registered 1-bit quantizer is a poor model of bipolar hardware, and the scored results show why: with b = 1 every
code is real (+-1), so the imaginary part of every stored vector is exactly zero, and the registered sign quantizer
maps zero to +1, adding a constant to every component; and a sum of an even number of +-1 terms is often exactly zero,
so ties are frequent (37.5% of components at 4 pairs per vector) and all break to +1. This is why 4 pairs per vector
read worse than 16 at b = 1 (0.954 against 0.973). The registered results stand as registered (a lower bound for 1-bit
storage). As a post hoc variant we add the usual bipolar implementation: at b = 1 the store keeps only the real part,
as the sign, with ties broken at random (seed 0 per store), at 16, 8 and 4 pairs per vector and at d = 16384.
Reported beside the registered 1-bit numbers, not in place of them; no bar.
