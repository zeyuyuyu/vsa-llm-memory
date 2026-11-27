# B25 pre-registration: the RULER vector stores on a measured phase-change-memory device model

Date: 2026-11-17. Written before any condition is run on the real stores. The code (b25_pcm.py) was exercised only on
synthetic random pairs (`--selftest` and a 6,000-pair synthetic store); where those runs informed a prediction below,
the prediction says so.

Motivation. A referee of the VSA-LLM manuscript wrote that "realistic device models (PCM drift and conductance
variation), higher loads, and an energy/area comparison" are needed, and that "complex FHRR codes have no stated mapping
to real-valued in-memory arrays". B22 tested idealized precision, i.i.d. read noise and zeroed components, and (post hoc,
amendment 2) a bipolar 1-bit store that read all 4,776 probed keys exactly. B25 puts the bipolar store on a PCM device
model whose every parameter is a published measurement, maps it to crossbars in the way in-memory hyperdimensional
computing hardware does (Karunaratne et al. 2020), and adds a cited energy estimate.

## Stores, keys, loads (fixed here)
The B22 stores and keys: the 50 real B21 stores rebuilt from the saved pairs (b21_dl/b21_test, first 10 samples of
niah_multikey_2 at 131072 and 32768, niah_multikey_3, niah_multivalue and vt at 131072), up to 150 probed keys per store,
4,776 keys in all, d = 8192. Bipolar variant (B22 amendment 2): every key, value and position code has its phase rounded
to 0 or pi, i.e. it is a real +-1 vector; a stored vector is the integer sum m of key * position * value over its pairs.
Two stored forms:
- B1 (primary): sign(m), ties broken at random (seed 0 per store, exactly B22 amendment 2), with the per-vector scale
  rms(m) kept digitally.
- ML (secondary): the integer sum m itself, as a multi-level analog value.
Loads: 16 pairs per vector (the RULER/B21 setting; mean 15.5, at most 38 per vector), and the same stores re-sharded at
32 and 64 pairs per vector (as B22's trades). B22's sanity check found the ideal 1-bit store holds 64 but not 128 pairs
per vector on the three largest stores.

## Hardware mapping (fixed here)
- Store crossbar: one row per stored vector, each component a differential pair of PCM devices (G+, G-), value
  proportional to G+ - G-. B1: +1 -> (G_ON, RESET), -1 -> (RESET, G_ON). ML: |m_c| mapped linearly onto [0, G_ON] per
  vector (scale max_c |m_c| kept digitally), m_c > 0 on G+, m_c < 0 on G-, the other device RESET. A read takes the row
  device by device through the ADC (as on the prototype chip of Joshi et al.: devices "serially read individually with the
  8-bit on-chip ADC"): B1 keeps the sign of the digitized G+ - G- (an ADC zero is a tie, broken at random); ML rescales
  the digitized difference to m units.
- Unbinding: digital. With +-1 codes, unbinding a key and a position is a component-wise sign flip.
- Clean-up crossbar: the store's value codebook (N codes of length d, +-1) as d x N differential pairs, +1 -> (G_ON,
  RESET), -1 -> (RESET, G_ON), as in Karunaratne et al. 2020, where the associative memory holds the prototype
  hypervectors in a PCM crossbar and "the query hypervector ... is input as voltages on the wordline driver, to perform
  the AM search using an in-memory dot product primitive". Inputs: B1 query components are +-1 (1-bit inputs, exact); ML
  query components are quantized to 8 bits (127 levels per sign, per-query scale kept digitally). The crossbar is tiled
  in 256-row tiles (the 256 x 256 cores of Le Gallo et al. 2023); each tile's column outputs are digitized by an 8-bit
  ADC whose full scale is the largest possible tile output (256 x max input x G_ON, so it never clips), and the tiles are
  summed digitally. Similarities are the summed output divided by alpha * G_ON * d and multiplied by the digital scales;
  the read then follows B21 exactly (unbind position j, accept the best value if its similarity is >= 0.5, stop at the
  first j below). Threshold 0.5 in every condition, no retuning.

## Device model (Joshi et al. 2020, Nat. Commun. 11, 2473; Supplementary Note 2), fixed here
Conductances in microsiemens (uS), times in seconds after programming.
| quantity | value | source |
|---|---|---|
| programmed target G_ON | 25 uS | main text: iterative programming "within the range of 0-25 uS"; Gmax = 25 uS |
| RESET device target | 0 uS | main text: the other device "was RESET close to 0 uS" |
| drift law | G(t) = G(t0) (t/t0)^(-nu) | main text and SI Eq. 1 |
| drift exponent nu | Gaussian per device, mean mu_nu(G_T) = max(-0.0045 G_T + 0.08, min(0.06, max(-0.00169 G_T + 0.0825, 0.055))), sd sigma_nu(G_T) = 1/(8 G_T + 30.0320); so mu = 0.055, sd = 0.0043 at 25 uS and mu = 0.08, sd = 0.033 at 0 uS | SI Eqs. 8-9 (targets defined at 20 s); main text "nu ~ 0.06 on average" |
| programming noise | two events, at T0 = 27.36 s and T1 = 55.39 s (SI Eqs. 13-14); sd 1.2 uS per event for a programmed device and 0.2 uS for a RESET device | main text "For all levels, we achieve a standard deviation less than 1.2 uS"; SI Fig. 16 shows about 1.07 uS (T0) and 1.12 uS (T1) at 25 uS and about 0.15-0.2 uS at 0 uS. We round up (conservative). |
| read (1/f) noise | G(t) = G_drift(t) + N(0, sigma_nG), sigma_nG = Q_s(G_T, t) G_drift(t) sqrt((f_max^(1-g) - f_min^(1-g))/(1-g)), g = 1.21, f_max = 1/(2 T_read), T_read = 250 ns, f_min = 1/(t + T_read); Q_s = 0.0388 / G_T^0.5729 (t / 1.089e5 s)^(-nu_q), nu_q = min(0.081, 0.04 G_T + 0.02) - [G_T > 13.21] min(0.035, -0.2048 + 0.0155 G_T) | SI Eqs. 10, 11, 16, 17 and the text around Eq. 5 |
| ADC | 8 bits | Methods: "the on-chip 8-bit cyclic analog-to-digital converter"; 8-bit I/O also in Le Gallo et al. 2023 |
| drift compensation (GDC) | alpha(t) = (summed conductance of all devices of the array, one calibration read at time t) / (summed target conductances); outputs divided by alpha | Methods, "Global drift compensation (GDC) method" (summed column currents under a calibration voltage on all rows, divided by the stored sum of the known programmed conductances; in their experiments over all columns) |

Rules we add (not in the paper, needed to run the model, stated here): conductances are clamped at >= 0 after each step;
nu is truncated at 0 (drift is relaxation; at the registered parameters this touches 0.8% of RESET devices and no
programmed device); Eq. 10 diverges at G_T = 0, so Q_s is capped at 0.2 (the cap used by IBM's aihwkit PCM model).
Each device's nu and programming noise are drawn once (at programming, seeded per store) and kept for all times; the read
noise is drawn once per device per time point (the array state read at time t; 1/f noise is dominated by its low
frequencies, so reads at the same time see nearly the same deviation). A descriptive variant redraws it on every MVM.
Le Gallo & Sebastian 2020 (J. Phys. D 53, 213002, Sec. 4.2) give nu "typically ... 0.1 for the RESET state" and "lower
for the SET state (< 0.05)"; the Joshi model (0.055 at 25 uS) is for iteratively programmed multi-level states, so for
B1's fully SET devices it is if anything pessimistic.

## Times (fixed here)
60 s, 1 h, 1 day, 30 days and 1 year after programming (the model is defined for t >= T1 = 55.39 s, so the first time is
60 s, not 1 s). The model was fitted to measurements up to 1e5 s (about 1.2 days); 30 days and 1 year are
extrapolations, as in Joshi et al. (who use the model "for extrapolating results over a longer period of time").

## Conditions
Registered grid (180 + 6): stored form {B1, ML} x stage {S: store crossbar on PCM, clean-up digital; C: store digital,
clean-up crossbar on PCM; SC: both on PCM} x compensation {GDC, none} x 5 times x 3 loads; plus the ideal (no device)
store at each form and load. Stage S and C isolate which stage limits accuracy. "none" (outputs divided by G_ON only)
shows what GDC is buying. For B1-S the compensation is irrelevant (the store is read as signs).
Descriptive extras (no bars), all at 30 days, 16 pairs per vector, GDC: (a) device-noise multiplier k = 2, 4, 8, 16, 32
applied to every device noise source at once (both programming-noise sds, sigma_nu, read-noise sd), for S, C and SC, both
forms, to locate the break; (b) ADC (and ML input DAC) at 6 and 4 bits, SC; (c) read noise redrawn on every MVM, SC.
Seeds: per store from its identity; device draws are shared across times, loads use their own store-array draws, and the
k multipliers scale the same underlying draws.

## Metric
B22's: a key is exact if the store returns the table's values in order; reported as the fraction of the 4,776 probed keys
read exactly and the number of the 50 stores in which every probed key is exact.

## Registered check
The ideal B1 path must reproduce B22 amendment 2 on every store at 16 pairs per vector (key counts asserted per store
against b22_robust.run_store(bits=1, bipolar=True)); B22 read 4,776 / 4,776.

## Bars
- H1 (primary): B1-SC with GDC at 16 pairs per vector reads >= 99.5% of keys exactly at 1 day.
- H2: the same at 30 days.
- H3: ML-SC with GDC at 16 pairs per vector reads >= 99.5% of keys exactly at 1 day.
- H4: B1-SC with GDC at 1 day at 32 and at 64 pairs per vector is within 0.5 points of the ideal B1 store at the same load.

## Predictions
Per component, the programmed devices carry about 7% conductance error (1.2 uS x sqrt(2) on 25 uS), a drift-exponent
spread of 0.0043 (about 5% conductance spread after a month) and a read noise of about 4.5% at 1 day. A clean-up
similarity sums d = 8,192 such terms, so device noise adds about 0.004 to a similarity whose cross-talk sd is 0.044
(sqrt(16/8192)); the common drift, however, shrinks every similarity by (t/20 s)^(-0.055): 0.94 at 60 s, 0.75 at 1 h,
0.63 at 1 day, 0.52 at 30 days, 0.46 at 1 year, against a B1 match similarity of about 0.8 and a threshold of 0.5.
- P1: with GDC, every arm (B1 and ML; S, C, SC) at 16 and 32 pairs per vector reads >= 99.9% of keys at every time up to
  1 year. All four bars pass.
- P2: with GDC at 64 pairs per vector, every arm is within 0.2 points of the ideal store at that load; the ideal B1 store
  at 64 reads >= 99%.
- P3 (no compensation, at 16 pairs per vector): B1-C and B1-SC read >= 99.5% at 60 s and at 1 h, fall below 99.5% by 1 day (the match
  similarity 0.8 x 0.63 straddles 0.5; a synthetic 6,000-pair store read 57%-85% at 1 day) and below 20% at 30 days;
  B1-S is unchanged by compensation. ML-S and ML-C read >= 99% at 1 day, below 80% at 30 days and below 20% at 1 year.
  ML-SC (both scales multiply) falls below 99.5% by 1 hour and below 5% at 1 day.
- P4 (descriptive extras; informed by the synthetic stores): B1-SC with GDC stays >= 99.5% up to k = 4 and falls below
  99.5% at k = 8 (at large k the RESET devices' clamped noise inflates the GDC sum, so the clean-up gain is
  under-corrected); B1-S stays >= 99.5% up to k = 4, is above 90% at k = 8 (store bit errors about 12%) and falls below
  50% by k = 16; ML-SC stays >= 99.5% at k = 2 and falls below 99.5% at k = 4. 6-bit ADC: >= 99.5% for both forms; 4-bit:
  no prediction (with the registered full-scale tile ADC, 4 bits leave ML tile outputs mostly below one step). Read noise
  redrawn per MVM: within 0.1 points of the per-time draw.
- Which stage limits: with GDC none does at the registered parameters; without compensation, the clean-up stage (its
  scale multiplies a threshold test) limits B1, and both stages limit ML.

## Energy estimate (formula and constants fixed here; arithmetic only, labelled as an estimate)
One clean-up step (unbind position j and clean up against N value codes of length d) is N x d multiply-accumulates
(MACs); a key with n values takes n + 1 steps (the last one stops the read). Largest store: N = 5,993, d = 8,192,
N x d = 49.1 M MACs.
- In-memory, per MAC: (a) Karunaratne et al. 2020 Table 1: PCM associative-memory search 9.44 nJ per query, for c = 22
  classes x d = 10,000 (Extended Data Table 1, language task, used for the energy study) -> 9.44 nJ / 220,000 = 42.9 fJ per
  binary MAC (their estimate: 90-nm PCM chip parameters, 65-nm CMOS periphery). (b) Le Gallo et al. 2023 (Nat. Electron.
  6, 680-693), measured: 9.76 TOPS/W (1-phase read) and 2.48 TOPS/W (4-phase, high precision) for 8-bit input/output MVMs,
  "1 multiply-and-accumulate = 2 OPs" -> 0.205 and 0.806 pJ per MAC. Store-row read (B1, d sense-amplifier reads at 9.8
  fJ each, Karunaratne Extended Data Table 2) is about 80 pJ, negligible; digital unbinding (d sign flips) is neglected.
- GPU, NVIDIA A100 SXM datasheet: 400 W TDP; INT8 tensor 624 TOPS, FP16 tensor 312 TFLOPS, FP32 19.5 TFLOPS dense;
  HBM2e 2,039 GB/s. Lower bound (compute at peak efficiency, e.g. many reads batched against one codebook): 2 ops per
  MAC x 400 W / peak -> 1.28 pJ (INT8), 2.56 pJ (FP16), 41 pJ (FP32) per MAC. Upper bound for one unbatched read
  (memory-bound: stream an N x d INT8 codebook at full bandwidth while drawing TDP): 400 W / 2,039 GB/s = 196 pJ per
  byte -> 196 pJ per MAC.
- Area/footprint: 2 x N x d = 98 M devices for the largest codebook (Karunaratne: 0.2 um^2 per device -> about 20 mm^2
  of devices), about 12 times the > 4 M weights (64 x 256 x 256) of one Le Gallo et al. 2023 chip.

## Compute
CPU only, local; 4 worker processes with OMP_NUM_THREADS=3 (another long CPU job may be running). No model, no GPU, no
paid API.
