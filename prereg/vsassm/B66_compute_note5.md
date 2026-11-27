# B66 compute note 5 ((C)'s test jobs finished; before (C)'s verdict is computed or read)

Date: 2026-11-23, 14:06 (machine clock; compute note 4 says "about 14:06" but was committed at 14:02:44, its commit time is the authoritative one).

- (C)'s test jobs ran once on the box, 13:57-14:04 (b66_grids.py runplan --kinds SDMT,LAMC, 20 workers; plan md5
  a718a567): SDMT 200 jobs, LAMC 345 jobs; job files pulled back. KF6 / KF6IDX not run (cut, compute note 2), so KF6's
  draws, BEST7_72 and the VSA / VSAR - BEST7_72 contrasts come out as "no complete draw" (the verdict leaves out draws
  with a missing prediction and prints n); they are not reported as numbers.
- Nothing of (C)'s test output has been read. Next: b66_grids.py verdict, then the plain-loop check of its region values.
