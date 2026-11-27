# B47 compute note 2 (2026-11-19 ~17:45 UTC; no design change, no bar or prediction changed)

1. **Session restart; PACE unreachable.** The parent session restarted (about 12 h lost). At 17:4x UTC
   login-phoenix.pace.gatech.edu did not answer (ssh connection timed out); the state of the PACE jobs queued under
   gts-nimam6-paid (800 tuning runs, 8,000 grids, naming head at 8,000; compute note 1) is unknown. **Twins rule, fixed
   now:** every fine-tune is (re)run on the A800 box; the box outputs are the registered ones. Any PACE output of the
   same configuration is a duplicate: it is not read for the selection or the test, and it is reported (with its
   GPU-hours) when PACE is reachable again; any PACE job still pending then is cancelled.
2. **Box setup.** /path/to/work/b47 with the committed code; tokenisation rebuilt there by b47_tok.py: the content
   checksums (b44_tokmd5) of all six files equal the PACE ones (k32 train f9e46b4d..., dev 2036b1f9..., test
   9f0c5775...; std train 08fee1e5...), self-test passes.
3. **Worker.** b47_worker.sh = b44_worker.sh with B47 paths and the shared claim directory (never GPU 3, idle = < 500 MiB
   and no compute process, re-checked after 20 s, never another process); a lane whose queue is empty waits for
   <queue>.close so commands appended after a selection find the GPU. Two workers, two lanes each; every run uses
   --mbs 4 (batch 8 as two micro-batches with gradient accumulation, allowed by the B31 recipe) so that a full fine-tune
   (~50 GB) and a LoRA run fit one 80 GB GPU together.
4. **GPUs.** At 17:39 UTC only GPU 5 was idle; at 17:43 UTC every usable GPU held a process (two claimed by B44 / B48
   workers, five by processes outside the claim directory). The workers wait for a GPU to become idle.
5. **Order (the registered cut order applied to the remaining time):** primary 800 grid (gen x 6, fullgen x 3), the
   800 selections and finals; then 8,000 narrowed to the rate selected at 800 (cut 4) and its finals; then the naming
   head at 8,000; then scratch; then fullcls. What has not finished by about 23:45 UTC is reported as not run.
6. **Second scorer for the sweeps and bootstrap** rerun locally (the first run was killed by the restart before writing
   anything).
