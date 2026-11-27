# B58 compute note 1 (during dev; no test run yet, no test output read)

- **Time in the pre-registration.** Its first paragraph says "machine clock Asia/Shanghai, 20:40"; that was a typo.
  The commit (616acbb) is at 20:09:29 and the OpenTimestamps proof commit (467eced) at 20:09:38 machine clock; these
  are the registration times.
- **Part (c) ran first** (synthetic codewords and B38's own decoder-check cells only; no B58 dev or test store):
  - c1 (B38's 10 check cells, 1,007 codewords): agreement 792 / 1,007 (B38's 79%), PKG decodes 705, OWN 874; PKG's
    posterior log-probability ratios are non-finite in 173 of the 175 codewords that OWN decodes and PKG does not.
    The registered mechanism test (channel probabilities floored at 1e-6, |LLR| <= 13.8) restores agreement only to
    818 / 1,007: flooring the input does not stop messages from overflowing inside the iterations.
  - c2 (exhaustive ML on K = 16 IRA codes, 72 settings x 400 frames): in channel A (i.i.d.) PKG and OWN agree with
    each other in every setting (both below ML where BP is weak, e.g. R = 1/2, eps 0.276, rho 4: ML 235, both 88);
    in channel B (stuck copies) both fall short of ML by more than 3 points in many settings; with saturated inputs
    (rho 16, 30) OWN is far closer to ML at R <= 1/2 (e.g. R = 1/4, eps 0.1226, rho 16: ML 392, OWN 166, PKG 54) and
    PKG closer at R = 2/3 (N = 24; rho 30, eps 0.1226: ML 122, PKG 62, OWN 43). **Registered rule outcome: neither
    decoder is within 3 points of ML in every setting, so OWN is used for P-LDPC and for the E-LDPC re-score (stated
    as the rule's default), with the ML-closer decoder reported per setting.**
  - **Disclosed additions to (c1)** (after the registered floor test failed to restore agreement; no B58 store
    involved): (1) an emulation of the package's product-sum arithmetic in numba (b58_dec.py emu: received word
    decoded as an error pattern against its syndrome, forward / backward tanh products, message log((1 + p) / (1 - p))
    without clipping, forward / backward bit sums, 'error' when the posterior <= 0, IEEE inf / NaN) reproduces PKG
    exactly: 1,007 / 1,007 c1 codewords and 3,600 / 3,600 channel-B frames, with non-finite posteriors in the same
    348 codewords. (2) The same emulation with OWN's two guards added (tanh input within +-40, |product| <= 1 -
    1e-12; b58_dec.py emuclip) decodes exactly what OWN decodes (874 of 1,007 codewords; identical output on 995 / 1,007, the rest
    non-converged words ending on different hard decisions; no non-finite value; results/b58_dec_emuclip.json). So the cause is the package's
    unguarded product-sum update: a product of tanh terms that rounds to +-1 gives an infinite message, and inf - inf
    or inf x 0 then give NaN, which the decision 'posterior <= 0' reads as 'no error'.
- **Decoder implementation for P-LDPC and the re-score.** The numba port (b58_bp.NbDecoder) is not bit-identical to
  b38_check._sp_decode on saturated inputs (port check: 1,581 / 1,988 synthetic words identical; the differences sit
  in channel-B settings with stuck copies, where non-converged words end on different hard decisions after
  last-digit differences in tanh / log / exp). It is therefore not used. Instead b58_bp.sp_decode_batch runs
  _sp_decode's numpy code over many codewords at once and is bit-identical to it (port check 1,988 / 1,988 words;
  two B38 E-LDPC test cells' decoded streams identical to b38_eq's B38_LDPC_OWN path). P-LDPC and the E-LDPC
  re-score use it, so OWN in B58 is exactly B38's second decoder.
- **Dev pass 1 was restarted once** (20:22 -> 20:29) for speed, after 9 of 456 jobs, with two exact changes: shards
  are decoded and read in batches of 8 (one matrix product per batch instead of per shard; censoring is decided on
  the same total of failed keys); identical results on the profiled cells (exact counts, residual errors, failed
  codewords). The 9 finished jobs were discarded and rerun. A `pkill -f` pattern in a check command also matched its
  own shell (no effect on results).
- **Dev pass 1 was stopped a second time** at 20:34 (17 of 456 jobs finished, free memory down to 2 GB with 12
  workers at 0.8-1.7 GB each) and resumed at 20:35 with the 16 jobs saved in the checkpoint kept
  (results/b58_dev1a.json, 532 evaluations) and only the missing evaluations run (results/b58_dev1b.json); the
  workers now drop the float32 copies of the codebooks that b34_inmem.Store keeps (B58 reads only Store38's int8
  copies) and are recycled every 4 jobs: 0.5-0.6 GB per worker, 13-14 GB free. No result changes (same code path).
- Workers: 12 (16 cores); another session's process (stale_ablation) also runs on this machine.
- The test extension rule skips budgets where the chosen design does not fit (b / R > B) or that were already tried
  (an infeasible evaluation is not a result, so without this it would be re-added each pass).
- Nothing about designs, grids, seeds, bars or predictions changed.
