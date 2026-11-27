# V5 packet addenda and procedural deviations (logged before the affected prompt is sent)

- 2026-11-23 17:10 (machine clock). Gateway: the api_zheng gateway sits behind a Cloudflare quick tunnel that ends any
  request after about 125 s (HTTP 524), streamed or not; the first attempts at reasoning effort "high" (non-streamed,
  then streamed) all ended in 524 or in HTTP 429 (scheduler busy) before any reply. No code had been returned. Change:
  reasoning effort "low" and max_completion_tokens 20,000 so that a reply fits within the tunnel's limit; if a part
  still does not fit, the script is requested in consecutive pieces within one conversation (same packet, same rules),
  which counts as one iteration. The prompts' content is unchanged.
- 17:12. Part A, iteration 1 (one reply at effort "low", 107.7 s) segfaulted in --smoke inside scipy.linalg.eigh. The
  orchestrator reproduced the crash in a six-line test independent of GPT's code (OPENBLAS/OMP threads set to 1 before
  import, then threadpoolctl raising the limit) and sent the traceback plus that environment fact. Iteration 2 was
  returned in two consecutive pieces, concatenated verbatim (v5_out/iter_a2/).
- 17:13. Part B, iteration 1 was requested in three consecutive pieces within one conversation (a single reply does not
  fit the tunnel's limit), concatenated verbatim (v5_out/iter_b1/). The same thread-limit environment fact was stated in
  the first piece's prompt (before any Part B code existed).
- 17:20. Part B iteration 1 failed in --smoke with a joblib un-pickling traceback (sent verbatim). For fixes, GPT
  returns only the complete top-level definitions it changes; v5_patch.py substitutes each by name, mechanically, with
  no other edit (v5_out/iter_b2/). Iteration 2 passed --smoke.
- 17:30. Runs on the A800 (ssh cavin, conda env dwm: Python 3.10.20, numpy 1.26.4, scipy 1.15.3, numba 0.65.0): the
  only change to GPT's scripts is the data-path constant(s), substituted by sed (v5_gpt_a.py line 62 DATA ->
  /path/to/HOME/b60_run/b49_out/b32_data, whose files have the same md5 as the local ones; v5_gpt_b.py lines 48-49 ->
  ~/v5_run/b34_out/, copies of the same files). Part A Qwen3-8B runs locally (8 workers), Part A Llama-3.1-8B and Part B
  on the box (20 and 24 workers, nice 5).
- 17:38. Part B iteration 2 passed --smoke, but its full run was too slow (dense dev sweep about 1.1 min per grid
  budget; extrapolated many hours). Stopped by PID after 8 minutes (partial output kept in
  v5_out/iter_b2/full_run_stopped/). Iteration 3 prompt = a resource fact only (timings and the 45-minute limit), no
  result. GPT's patch (Numba scoring for the dense and sparse stores, early stop of a sweep once 99.5% is impossible)
  passed --smoke with exactly the same smoke counts as iteration 2. Its full run is started as four processes, one
  per memory (--memory dense / sbc / bch / cuckoo, each --k 1,8,16), the script's own command-line options; the
  memories are independent in the script, so this changes nothing but wall time.
- 17:47. Iteration 4 prompt = a resource fact only (bch made 2 evaluations in 4.5 minutes; the limit). GPT's patch
  replaced only BCH, cuckoo-layout and worker-cache functions (the dense and sparse-block code untouched; the replaced
  worker function only adds a layout cache used by the tables); --smoke identical to iteration 3 in every dev and test
  count. The iteration-3 bch and cuckoo processes were stopped by PID and restarted with iteration 4 at 17:49; the
  iteration-3 dense and sbc processes continue (their code is identical in iteration 4). v5_gpt_b.py = iteration 4.
- 18:00. Iteration 5 prompt (cuckoo only) = a resource fact (the cuckoo sweep's rate and the limit). GPT's patch
  replaced only cuckoo functions and added cuckoo helpers; --smoke identical to iteration 4 in every cuckoo dev and test
  count. The iteration-4 cuckoo process was stopped by PID (partial output kept in v5_out/iter_b4/cuckoo_run_stopped/)
  and restarted with iteration 5 at 18:00 (6 workers). v5_gpt_b.py = iteration 5; dense and sbc run iteration-3 code,
  bch iteration-4 code, all identical to iteration 5 on their own code paths (the patches replaced no function on
  those paths except the worker's layout cache, which the dense and sparse stores do not use).
- 18:15. Iteration 6 (the last allowed; prompt = a resource fact for the cuckoo memory) returned a patch of two cuckoo
  transport functions that GPT itself said would not meet the limit. It was NOT run: restarting the cuckoo process
  would have discarded 15 minutes of its iteration-5 progress with 40 minutes left before the cut. v5_gpt_b.py stays
  iteration 5. The machine was also shared from about 18:05 with another agent's job (four processes of 16 threads),
  which slowed every V5 run.
