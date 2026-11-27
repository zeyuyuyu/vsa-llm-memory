# B32 compute note 1 (2026-11-18, US Eastern): OPQ trained with 4 iterations; where the dev grid runs

Written after the pre-registration (e301410) and before any index dev result exists (the two index dev processes
were stopped before they finished their first store; their output files are empty). No VSA dev number has been read
for selection. Nothing on test data has run.

1. **OPQ iterations.** The index family "OPQ + 8-bit PQ" was registered without an iteration count, so FAISS's
   default applied (OPQMatrix.niter = 50). Measured on this CPU (shared with other jobs): 2 iterations take 25 s on
   the 768-number e5 codes and more than 180 s on the 4,096-number Qwen3-8B codes, so 50 iterations for each of the
   up to 6 sub-quantizer counts and 2 preprocessings would take about 2 hours (e5) and more than 12 hours (Qwen3-8B).
   OPQ is now trained with **4 iterations** (b32_common.OPQ_NITER; b32_check.py the same). This can only weaken this
   one family; the other five index families (LSH, ITQ, PCA-sign, PQ with 4- and 8-bit sub-quantizers) are
   unchanged, and the dev grid picks the best family per cell. In pilot 1 (bge codes, 50 iterations) OPQ was the best
   index family in no cell.
2. **Where things run.** All A800 GPUs except GPU 3 were taken by another job (B31) when the B32 VSA dev grid was due,
   so the GPU job (b32_run_gpu.sh SELFTEST, waiting in its claim loop, no GPU ever claimed, ledger unchanged) was
   stopped, and the Qwen3-8B VSA dev grid runs on the local CPU (b32_run.py dev qwen8 ... --part vsa, torch on CPU:
   the same code). GPU-hours so far: 0.025 (the Qwen3-8B encoding).
