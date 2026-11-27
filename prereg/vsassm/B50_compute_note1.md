# B50 compute note 1: four gated test runs crashed out of memory before writing anything; re-run unchanged

Written 2026-11-19, about 13:45 box time (05:45Z).

The worker claimed GPU 6 when it was idle (05:23Z). After the claim another user's job (pid 3283408, not ours, not
touched) started on the same GPU and grew to 10-22 GB. With our COT2048 task-N run (about 30-35 GB as its cache grows)
and two DISC evaluation lanes (17.6 GB each), the card filled, and four seed-0 DISC test runs (gates ENERGY, MSP, MAHA,
MAXLOGIT) died with `torch.OutOfMemoryError` while loading the model, at 05:37:47-05:38:13Z. A prediction file is
written only at the end of a run; none of the four wrote one (preds/N held only discopen_s0, discp08_s0). So no
output of these runs was ever produced or seen.

Action: the remaining gated runs were moved out of the lane queues (q/hold_*), and after the two runs in flight
finished (mahalw_s0, open_s1, both rc 0), the four crashed commands, unchanged, and the held ones were queued on ONE
DISC lane beside the COT lane. Nothing in any command, threshold or rule changed. Each registered test evaluation still
produces exactly one output file.
