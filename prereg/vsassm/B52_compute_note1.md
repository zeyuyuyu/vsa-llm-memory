# B52 compute note 1 (2026-11-20 about 03:35 UTC; no design change, no bar or prediction changed; no test output read)

1. **Orchestrator crash and restart (part a).** `b52_queues.py orch` crashed in its first loop (03:27 UTC) because
   `write_select` read a cell's binding-loss choice before one existed (KeyError). Fixed (skip cells without a choice yet)
   and restarted at 03:29 UTC. It had appended nothing; the workers had only started the queued tuning runs.
2. **cw tuning is queued on the priority queue (part a).** The orchestrator now appends each cell's cw grid (made after the
   cell's binding-loss lr choice) to `q/final`, which the workers serve before `q/tune`, so arena-C cells are not held up
   behind arena-A tuning. Ordering only; the grid, the runs and the selection rule are unchanged.
3. **Part (c) relaunch.** The first launch (03:27 UTC) never ran a job: in the launching shell the interpreter variable was
   set inside a backgrounded command list, so the pipeline command saw it empty ("-u: not found"). Relaunched at 03:29 UTC
   with `b52_c1d.sh` (same steps, arguments and workers: dev, choose, plan, test, extend, closed, verdict).
4. Part (a) worker start: 10 CPU lanes x 6 threads (nice 10) at 03:28 UTC; the idle-GPU worker is waiting (all usable
   GPUs busy at launch). Part (b): the PQ banks of maps 1-3 were fitted from 03:26 UTC (training split only), then the VSA
   dev grid of maps 0-3 and the map-0 gates.
