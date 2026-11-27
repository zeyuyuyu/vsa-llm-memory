# B52 compute note 2 (2026-11-20 about 03:52 UTC; coordinator deadline; scope cut, no bar, rule or prediction changed)

The coordinator set a hard deadline (final report within about two hours) and the priority (c), (b), then (a) only for
arena C on Qwen3-8B's dev-chosen middle layer. Part (a) is therefore restricted to the primary cell **qwen8L27 x C**
(`b52_refocus.sh`): the queues keep only that cell's lines; this experiment's own runs on the other cells (four arena-A
binding-loss tuning runs and one qwen8L0 x C cw tuning run) were stopped; the orchestrator was restarted restricted to the
cell (`B52_CELLS=qwen8L27_C`). As registered for cells that do not complete, **qwen8L0 x C, qwen8L9 x A and qwen8L0 x A are
reported as "not run"**: their dev tuning outputs (binding-loss lr runs; qwen8L0 x C: binding-loss lr 3e-4 chosen) are
dev only, and no test output of them is read. The embedding naming-head finals (qwen8L0, A and C) had already run as
references for the dropped cells; their test outputs are not read either. Parts (b) and (c) are unchanged.
