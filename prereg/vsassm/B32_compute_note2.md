# B32 compute note 2 (2026-11-18, US Eastern): OPQ fits for the Qwen3-8B codes computed in parallel and cached

Written before any Qwen3-8B index dev result exists (the Qwen3-8B index dev process was stopped after 61 minutes, still
inside its first store; its output file is empty). The e5 dev grid, the e5 test run and the Qwen3-8B VSA test run are
complete; no test accuracy has been read.

- The Qwen3-8B index dev grid spent its first hour fitting OPQ on 4,096-number codes, mostly single-threaded: the grid
  needs OPQ with 1, 2, 4, 8, 16, 32 and 64 sub-quantizers (stored shares below 1 leave more bits per fact) for both
  preprocessings, 14 fits in one process.
- The fits depend only on the training split, the preprocessing and the number of sub-quantizers. They are now
  computed by 14 parallel processes (b32_run.py fit qwen8 PREP opq8 M) and written to b32_out/fitcache/ (FAISS
  write_index); b32_common.IndexFamily reads a fitted PQ / OPQ index from there when B32_FIT_CACHE is set, and writes
  any PQ / OPQ fit it makes itself. The Qwen3-8B index dev grid and test run use the same cache, so they use the same
  fitted transforms. Nothing about a configuration, the grid or the choice rule changes.
- FAISS's thread count for fitting can be set with B32_FAISS_THREADS (default 8, as before; 1 for the parallel fits).
