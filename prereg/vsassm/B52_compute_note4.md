# B52 compute note 4 (2026-11-20 about 04:20 UTC; deadline; procedure only)

Part (a), qwen8L27 x C: under the shared load (box load about 123 of 128) one run takes 15-25 minutes; 44 tuning runs and
18 per-seed finals remained at 04:16 UTC, more than the time left. (1) The orchestrator now queues each family's three
per-seed finals as soon as that family's grid is complete (the binding-loss finals at once, since its lr was chosen at
03:41 UTC); the selection rule per family is unchanged, and the finals share the priority queue with the remaining cw
tuning. (2) Moving the remaining runs to PACE GPUs was tried and abandoned: copying the layer-27 features off the box ran at
about 0.8 MB/s (3.9 GB needed); the partial copy was deleted, nothing ran on PACE (0 PACE GPU-hours). (3) Whatever has not
finished by the report deadline is reported as "not run" (dev tuning, if complete, as dev only); no test output of a family
is read before all three of its seeds have finished.
