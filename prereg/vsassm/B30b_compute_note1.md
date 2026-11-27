# B30b compute note 1 (2026-11-18, ~22:45 UTC; no design change; no test output read)

Written after the first G-E launch and before any B30b output has been read or scored.

1. **G-E impact shard a crashed at engine start-up** on GPUs 0-1 (`CUDA error: an illegal memory access` in the NCCL
   all-reduce during CUDA-graph capture, 22:37 UTC, logs/GE_impact_a.log on the box). Shard b on GPUs 2-6 started
   normally and kept running. Cause: NCCL peer-to-peer transfers over this PCIe pair. Fix, for every B30b job with
   tensor parallelism from now on: `NCCL_P2P_DISABLE=1` (all-reduce through host shared memory) in b30b_run.sh. With
   two ranks an all-reduce adds two numbers, so the transport cannot change the result.
2. To restart cleanly, I stopped my own b30b_run_a.sh shells (PIDs 934944, 934951, 936868 and the waiting wrapper
   936869) and the window shard-a process they had just started (937060 and its vLLM engine and workers, still
   loading the model; nothing generated). G-E impact shard b (PID 936054) was left running. `b30b_run_a2.sh`
   continues part (a) in the registered order: impact shard a again, window shards a and b, impact and window forcing
   when their shards end, then S32 R1 (two jobs), then G-E location.
3. The 32B dev smoke (24 dev prompts, real budget) was still running on GPUs 4-5 at this point; its numbers go into
   the verdict's compute section.
