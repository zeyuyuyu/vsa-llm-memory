# B72 compute note 5 (2026-11-23, about 20:25 machine clock; no test output read)

- (B) development restarted as four per-encoder runners of 12 workers each (still 48 in total), so that each worker
  derives an encoder's whitening / principal bases once instead of at every change of encoder in a mixed queue. The
  first attempt (20:18) did nothing: the ssh connection closed before the remote command ran (box state checked
  afterwards: old runner intact, no partial file). The restart was then run as a detached script on the box
  (~/b72_run/restart_b.sh, log b72_out/logs/restart_b.log): old runner (PID 4093830, its chain parent 4088908 and 48
  workers) stopped by exact PIDs at about 20:22, 0 partial files, 1,224 of the 2,376 (B) dev jobs finished before the
  stop (0 failures), 4 x 288 left. Results depend on registered seeds only.
- Language-model (A) test (local): the 85 M = 1,024 jobs finished, 380 M = 256 jobs running.
