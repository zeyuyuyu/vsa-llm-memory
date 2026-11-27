# B55 pre-registration: the argmax-read vector module as a registered arm, clean and under read noise, on task N (trained and held-out types) and task R

Date: 2026-11-20 (box clock Asia/Shanghai; US Eastern -13 h). Extension of B40 (prereg/B40_inforward_vsa_prereg.md, its
amendments 1-2, exploratory 1, dev record, verdict) and B50 (prereg/B50_ood_gates_cot_budget_prereg.md, its dev record,
verdict, exploratory 1). Written, with the runner (b55_run.py), the verdict generator (b55_verdict.py), the box worker
(b55_worker.sh) and the PACE job (b55_pace.sbatch), before any B55 run of any kind. b40_core.py, b40_data.py and
b40_run.py are imported unchanged (md5 835e9f4f..., 67b29c80..., fef8081d..., identical on the box and in git).

## Why (referee, 9th read, M2e)

Table 1 reports DE25's published thresholded read for the in-forward-pass VSA module on task N (a digit is kept only if
its best clean-up similarity exceeds 0.5, otherwise written as 0): VSA 0.898 vs the same-shape discrete module (DISC)
0.930. With plain argmax clean-up the same trained seeds reached 0.960 and led DISC by +0.030 [+0.021, +0.040] (B40
exploratory 1, registered after B40's verdict was read). The noise comparison (VSA 0.60 / 0.20 / 0.12 vs DISC 0.91 /
0.81 / 0.47 at read noise 0.25 / 0.5 / 1.0) used the thresholded read and the text does not say so. Needed: the argmax
read as a registered arm, under the same read-noise levels, on task N and task R, three seeds, against DISC (and, on the
held-out types, against DISC with its best OOD gate from B50).

## What I have already seen (so the reader can judge what is blind)

- Task N, noise 0, the argmax read (digit_thr = 0, B40 exploratory 1, test, all three seeds): TR 0.9602, read exact
  0.9702; VSA-argmax - DISC +0.030 [+0.021, +0.040]; held-out addition 0.9707, division 0.9600; vs DISC-MSP (B50)
  +0.431 on held-out. The B55 noise-0 task-N cell is therefore NOT a blind test: it is a registered re-run of a result I
  know, with pure argmax (see below) instead of digit_thr = 0.
- Every B40 / B50 number in their committed verdicts: VSA (thresholded) and DISC at every noise level on trained types
  (task N) and on all items (task R), task-R VSA-soft, B50's gates at noise 0. I have NOT computed held-out-type
  accuracies or gate-open rates of any arm under noise, and no arm with argmax digit clean-up or with cleaned task-R
  mention reads has ever been run under noise or on task R.
- Dev, seed-0 readers, read accuracy only (B40 exploratory 1 / amendment 1): task N read exact at rho 0 / 0.25: argmax
  0.971 / 0.636, threshold 0.896 / 0.549, DISC 0.946 / 0.920; task R per-mention entity read (VSA encoder + argmax over
  the 50 codes) 1.000 at rho <= 1, 0.851 at 2, 0.309 at 4, 0.099 at 8 (DISC 1.000, 1.000, 0.975, 0.558).

## Arms (evaluation only; nothing is trained; B40's trained seeds 0, 1, 2)

- **N-ARG** (the arm under test, task N): B40's VSA module (layer 20, reader out/rd/N_vsa_e1000_s{s}, write-back
  out/dev/N_vsa_recon_lr1e-3_s0 and out/fin/N_vsa_s{1,2}) with ONE change: each operand digit is read by argmax over the
  ten digit atoms with no similarity threshold (b40_core.VSAN with digit_thr = -inf; B40 exploratory 1 used digit_thr = 0,
  which still writes 0 when every similarity is <= 0). The type read (argmax over the type atoms), the type gate
  (max similarity > 0.8, DE25), the Python function, the HRR answer code, the decoder and the 50/50 mix are unchanged.
- **R-ARG** (the arm under test, task R). B40's task-R VSA read was ALREADY argmax: every hop result is cleaned up by
  argmax over the 50 entity codes, and there is no threshold anywhere in it. What it does not clean up is the encoder
  output of each mention before it is bound into the memory M = sum a (*) P(b). The analogue of the task-N change is to
  read every mention (both mentions of every fact and the question mention) by argmax over the 50 codes, i.e. replace
  the encoder output by its nearest code, before M is built; the hops, the hop clean-up, the room gate, the decoder and
  the mix are unchanged (b55_run.VSARArg; B40 reader out/rd/R_vsa_e100_s{s}, write-back out/dev/R_vsa_lr1e-3_s0,
  out/fin/R_vsa_s{1,2}). DISC's closed-world query rule is NOT given to R-ARG (that would change more than the read).
- **Comparison arms (unchanged B40 / B50 modules):** N-THR = B40's VSA (digit_thr 0.5, the published read); N-DISC =
  B40's DISC (softmax gate > 0.8); N-DISC-MSP = B40's DISC seeds with B50's MSP gate (log-odds of the max type softmax
  >= the B50 dev-matched threshold of seed s, out/thresholds.json; B50's DISC-best, fixed now, not re-chosen); R-VSA =
  B40's task-R VSA; R-DISC = B40's task-R DISC; BASE = B40's frozen-LLM run (read noise does not touch the LLM's stream,
  so BASE does not depend on rho).
- **Noise:** rho in {0, 0.25, 0.5, 1, 2, 4, 8}: B40's injector, isotropic Gaussian noise of norm rho x the state's norm
  added per position to the READ only, torch.manual_seed(12345) before the first batch, B40's batching (longest prompt
  first, batch 32), so on the same GPU type every arm reads exactly the same noisy states as B40's runs.
- **Evaluation:** full test splits (task N 2,500 items: 2,000 trained-type TR, 500 held-out HO = 250 addition + 250
  integer division; task R all 2,000 items), greedy, 6 new tokens, B40's parser; one run per (arm, seed, rho).

## Where the runs execute, and what the comparison arms are (decided now)

- **A800 plan** (preferred; same GPU type as B40): runs N-ARG and R-ARG at all 7 levels (42 runs) and N-DISC-MSP at the
  6 noisy levels (18 runs). Comparison arms = B40's existing files (N-THR, N-DISC, R-VSA, R-DISC at every level) and
  B50's discmsp_s* files at rho 0. Same noise is CHECKED, not assumed: N-ARG and N-THR must log the identical type
  read and similarity on every item (the type read does not depend on the digit rule); N-DISC-MSP and N-DISC must log the
  identical reads (op, op_p, x, y); R-ARG and R-VSA must log the identical per-item read flag. If any of these differ
  on more than 0.5% of items, the A800 comparison is not same-noise and the comparison arms are re-run in B55 (the PACE
  plan's list) before the verdict is read.
- **PACE plan** (fallback, paid account, only if no A800 GPU is idle; one GPU of type H200 / H100 / A100 / L40S): the
  CUDA noise draws depend on the GPU, so the same job also re-runs every comparison arm at every level on the same GPU
  (N-THR, N-DISC, R-VSA, R-DISC at 7 levels, N-DISC-MSP at all 7; 147 runs in all); B40's / B50's files are then only a
  cross-GPU reproduction check.
- Both are submitted now; the first to start its test jobs is kept, and the other is stopped (STOP file / scancel)
  before it starts any test job. If both write a test file, the A800 files are used and the other platform's files are
  deleted unread (reported).
- A smoke run on 16 TRAINING rows per task (seed 0, rho 0 and 1) precedes the test jobs on whichever platform runs; it
  never touches the test split.

## Bars (b55_verdict.py)

Accuracy = mean over items of the seed-mean correctness; Delta = difference of two arms on the same items; 95% CI =
paired bootstrap over items, 10,000 resamples (B40's convention). Label of a Delta: **WIN** if CI lower > 0; **LOSE** if
CI upper < 0; otherwise **TIE** if the CI lies inside [-0.02, +0.02], else **NOT RESOLVED**. Beside each label: whether
the inequality holds in all three seeds, and B40's R2 label (which also needs |Delta| >= 0.03 and all seeds).

- **B55-N1 (primary):** TR(N-ARG) - TR(N-DISC) at each of the 7 noise levels.
- **B55-N2:** HO(N-ARG) - HO(N-DISC-MSP) at each level (the held-out comparison against DISC's best B50 gate); HO(N-ARG)
  - HO(N-DISC) and HO(N-ARG) - BASE beside it.
- **B55-R1:** R(R-ARG) - R(R-DISC) on all 2,000 task-R items at each level.
- Reported beside them (no bar): N-ARG - N-THR (TR, HO) and R-ARG - R-VSA per level; read-exact and gate-open rates;
  task-R per-cell accuracies (size x depth) per level; TR(N-ARG) - TR(N-DISC-MSP).

## Predictions (stated so they can fail)

- Q1 noise 0: N-ARG reproduces B40 exploratory 1 (pred identical on >= 7,490 of 7,500 seed-items); B55-N1 = WIN.
- Q2 B55-N1 = LOSE at rho 0.25, 0.5 and 1.0; N-ARG TR in [0.62, 0.76] at 0.25, [0.15, 0.45] at 0.5, <= 0.25 at 1.0.
- Q3 at rho 2, 4, 8: |TR(N-ARG) - TR(N-DISC)| <= 0.05 (both near or below BASE's 0.193).
- Q4 N-ARG - N-THR on TR: >= +0.03 with CI lower > 0 at rho 0 and 0.25; |.| <= 0.03 at rho >= 1.
- Q5 B55-N2 = WIN at rho 0, 0.25 and 0.5; HO(N-ARG) <= 0.5 at rho >= 2 (the VSA type gate opens on noisy held-out
  reads); HO(N-DISC-MSP) <= 0.3 at rho >= 2 (input noise inflates the type logits, so the MSP gate opens).
- Q6 R-ARG >= 0.995 at rho 0, 0.25, 0.5 and >= 0.98 at rho 1 (R-ARG - R-VSA >= +0.20 at rho 1: B40's task-R fragility at
  rho 1 comes from binding uncleaned reads); R-ARG in [0.20, 0.80] at rho 2 and <= 0.45 at rho 4 and 8.
- Q7 B55-R1 = WIN (CI rule; R2 bar TIE) at rho 0, TIE or WIN at 0.25-1, LOSE at rho 2 and 4; LOSE or NOT RESOLVED at 8.
- Q8 Same-noise checks: 0 differences in the type read (N-ARG vs N-THR), the DISC reads (N-DISC-MSP vs N-DISC) and the
  read flag (R-ARG vs R-VSA) on the A800 plan.
- Q9 Second scorer (b40_check.py): 0 gold mismatches and 0 correctness mismatches on every B55 file.
- Q10 Compute <= 3 GPU-hours (A800 plan) / <= 5 (PACE plan).

## Checks from other angles

- Second scorer: b40_check.py (B40's independent scorer: gold rebuilt from the prompt text, outputs re-normalised by its
  own rules) on every B55 test file.
- The same-noise checks above; MSP gate re-derived from the logged score and tau; noise-0 N-ARG against B40's
  exploratory-1 files item by item; on PACE, the clean comparison re-runs against B40's A800 files.
- Hand-read (fixed seed): 12 trained-type items where N-ARG and N-DISC disagree at rho 0 and 12 at rho 0.25 (seed 0;
  read error vs write-back not followed vs gate), 12 task-R items where R-ARG and R-DISC disagree at rho 2.
- Full test sets only; held-out and per-cell numbers beside the pooled ones.

## What the paper may / must say (decided now)

- Table 1 must name the read of every VSA row. If B55-N1 at rho 0 = WIN: the paper may report the argmax-read VSA module
  beside DE25's published read (Delta vs DISC [CI]) and must say that the argmax read is a deviation from DE25's
  published module, introduced by us after B40's registered test (exploratory 1) and registered here as a re-run, so the
  clean comparison is not a blind test.
- The noise sentence must state which read it uses and give both. If B55-N1 = LOSE at a noise level for the argmax read,
  the paper says the discrete module is more robust to read noise at that level with either read; if TIE / WIN, it must
  say that the thresholded read, not the vector format, caused the earlier fragility there.
- Held-out: B55-N2 per level, with the B50 caveat unchanged (the rejection comes from the least-squares readout, B50
  exploratory 1), so held-out gating is still not listed as something the vector format adds.
- Task R: the paper must say that B40's task-R module already read by argmax at every hop; R-ARG reports what cleaning
  the mention reads does; B55-R1 per level as measured.
- In every case: no generalisation beyond Llama-3.1-8B-Instruct, these two task families and these noise levels; the
  noise is an isotropic perturbation of the read only (a probe of read margins, not a model of a real failure).

## Compute

A800 box, one idle GPU claimed by the shared lock rule (never GPU 3; nobody else's process touched); PACE paid account
(gts-nimam6-paid) as the registered fallback while the box has no idle GPU (it has none at writing: all seven usable
GPUs are running another user's jobs). Wall time per run to logs/gpu_hours.txt. No paid API.
