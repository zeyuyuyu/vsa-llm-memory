# B67 compute note 1 (before any theory number on real data)

Machine clock 2026-11-23 15:12 (Asia/Shanghai). Two corrections to the pre-registration's header, disclosed:

- The header's "about 16:35" and "time box from about 15:20" are wrong; the pre-registration was committed at 14:56
  (git f642851) and stamped at 14:56 (95e5063); work began at about 13:50, so the 2.5-hour box ends at about 16:20.
- Selftest C3 (b67_theory.py selftest; synthetic isotropic keys, no real data), run before any real-data theory number:
  theory H against direct simulation of the store: 0.767 / 0.772, 0.714 / 0.716, 0.642 / 0.647, 0.586 / 0.587 (within
  0.6 points; passes). Theory L against direct ridge simulation: V = 2: 0.608 / 0.623, 0.695 / 0.718, 0.646 / 0.629
  (within 2.3 points; the registered 2-point bar fails narrowly in one case); V = 16: 0.372 / 0.425 (5.3 points under
  the simulation; FAILS the bar). The registered theory is run unchanged on the real data; theory L's V = 16 numbers
  therefore carry a known under-prediction of about 5 points at small p in the isotropic test, to be read with that
  caveat. Not tuned on real data.
- Placement: (a) one process per encoder on this machine (language models 4 BLAS threads each), (b) one process.
