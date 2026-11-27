# B65 compute note 1 (2026-11-23; procedure only; written during dev, no test run exists)

- Pre-registration committed e119b9e at 13:08:51 (machine clock, Asia/Shanghai; 05:08:51Z), OpenTimestamps proof
  committed 4b9006a at 13:09:02. The header says "about 13:05" (writing time); the commit time is the registration
  time. The self-test (training rows only) ran before the commit, 05:01:57-05:07:40Z, and is reported in the
  pre-registration.
- Box code = committed code: md5 b65_run.py fc01b2e7..., b65_plan.py 2b7ca5fd..., b65_orch.sh 30e84f13...,
  b65_worker.sh 2d1d9617..., b40_core.py 835e9f4f..., b40_data.py 67b29c80..., b40_run.py fef8081d... (identical in
  /path/to/work/b65, in /path/to/work/b40 for the three B40 modules, and in git).
- GPUs at the dev launch (05:10Z): 1, 2, 4, 5, 6, 7 idle (1 MiB, no compute process); GPU 0 another user's job
  (43 GB), GPU 3 rm_server.py (4.5 GB); neither is touched (the worker's GPU list is 1 2 4 5 6 7, idle = < 1,024 MiB and
  no compute process, re-checked after 20 s, shared claim directory). /data1 170 GB free; B65 writes about 2.2 GB of
  operand-state caches plus checkpoints (VSA decoder 33 MB, DISC embedding 0.8 MB per run).
- The dev phase runs as `b65_orch.sh dev1` (dev operand caches, ops layer scan, readers at both positions, then the
  first write-back grids: 12 runs per position, three workers x four lanes per position) and `b65_orch.sh dev2 <pos>`
  (edge extensions until interior, then seeds 1 and 2). Every choice is b65_plan.py's registered rule; nothing is
  chosen by hand.
