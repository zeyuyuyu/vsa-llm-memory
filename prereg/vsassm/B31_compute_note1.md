# B31 compute note 1 (2026-11-18 ~06:00 UTC; no design change, no bar or prediction changed)

1. **Budget.** The main session relayed at about 05:55 UTC that the user allows PACE (paid account included) and that
   the GPU-hour cap may rise to 1.5 x the registered one (60 -> 90 GPU-hours) if that speeds the work up. The
   registered projection (about 48 GPU-hours) is unchanged; the higher cap is used only if needed, and the verdict's
   ledger reports the total against both numbers. P17 stays as registered (<= 60).
2. **PACE is unreachable from this machine:** `ssh login-phoenix.pace.gatech.edu` timed out at 05:15, 05:44 and 05:56
   UTC; the GlobalProtect interface gpd0 is DOWN, and bringing it up needs the user's password and Duo push. All B31
   compute therefore stays on the A800 box: six GPUs (0, 1, 4, 5, 6, 7) are in use by B31 workers; GPU 2 belongs to
   another user and GPU 3 is never used.
3. **Crash and re-runs.** The first ten light jobs (the nine G0 runs and the encoder-free exact-symbol final of arena
   A) died after their first seed because the output directory did not exist (`json.dump` to `<out>.part` before
   `os.makedirs`). The directories were created at 05:44 UTC, the ten jobs were re-queued unchanged, and both scripts
   now create the directory first (b31_arms.py, b31_lora.py). Jobs started after 05:44 were unaffected; the Qwen3-8B
   LoRA tuning runs started at 05:39 write their output only at the end, after the directory existed.
4. **Worker log fix.** `b31_worker.sh` logged `rc=$?` after a `$(date)` substitution, so every exit code read 0; it now
   captures the status first. Completed jobs are checked by the presence of their output file instead.
