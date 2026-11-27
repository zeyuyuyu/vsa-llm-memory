# B31 compute note 3 (2026-11-18 ~07:40 UTC; no design change)

The Qwen3-8B queue after the stage-1 selections of arenas A and B holds about 27 jobs of 0.5-1 GPU-hour each; at two
Qwen workers B31 would finish around 17:30 UTC. The scheduler cap is set to 5 B31 workers on the box, at most 3 of them
on Qwen3-8B queues: one GPU fewer than the six B31 held at 07:00, as the main session asked ("a GPU or two"), and about
three hours sooner than the cap of 4. The three PACE jobs (compute note 2) are still pending (Priority).
