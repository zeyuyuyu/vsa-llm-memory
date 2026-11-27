# B35 priors, written before any B35 GPU run (2026-11-18, about 06:10 UTC)

Context: B35 asks whether VSA bundles handed to a language model as a few soft input positions let it answer more
facts at an equal context budget than learned memory tokens without VSA structure (prereg/B35_soft_token_vsa_prereg.md,
to be written after a train/dev pilot). The only B35 numbers known when this was written are CPU ones: the exact
external decoding of the bundles on the dev split and a capacity sweep (b35_ext.py; MAP bipolar codes, D = B x 2048):
single hop 1.000 wherever N / B <= 64, 0.967 (random codes) / 0.933 (embedding codes) at N = 128, B = 1, within a few
points of the Gaussian theory (0.958); two hop 0.78 / 0.77 at N = 128, B = 1 (theory 0.77), >= 0.98 elsewhere.

My priors (to be checked against the dev pilot and the test run, whatever they show):
1. The bundles hold the facts: exact external decoding matches the theory within a few points everywhere.
2. The question-agnostic VSA arm (the LM must unbind the bundle itself; the projector cannot see the question) stays
   near the floor (<= 0.15 single hop) at N >= 32 for every B; at N = 8 it may rise above the floor but stays <= 0.5.
3. The learned memory tokens (Perceiver over the parsed facts) beat the VSA arm by >= 0.2 at N = 8, and both are near
   the floor at N = 128 with B <= 2.
4. The full text with LoRA scores >= 0.95 single hop at every N and >= 0.8 two hop.
5. The text truncated to B tokens is at chance (B <= 8 tokens hold less than one ~10-token fact).
6. The question-conditioned VSA arm (unbinding done outside the LM with the parsed question key) scores >= 0.9 single
   hop wherever the external decoding is >= 0.95.
7. Two hop: both question-agnostic soft arms <= 0.1 everywhere except possibly N = 8.
Reason for 2 and 7: reading one key out of a superposition needs a product of a question-dependent key with the
memory (a bilinear map of rank ~ D); a frozen LM with a low-rank LoRA has no obvious way to compute it, and there is
no precedent of a pretrained LM learning HRR/MAP unbinding from a single input vector.
