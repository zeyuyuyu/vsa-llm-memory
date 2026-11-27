# B61 compute note 4 (11:02 machine clock; test finished; the verdict has not been computed and no test share or N* has been read)

- The test ran once on the box, 10:54:12-11:00:05 (b61_box.sh test, 44 workers, code MD5-identical to 504680d's
  b61_hd.py, plan results/b61_plan_test.json MD5 819cc65c...), 4,600 jobs, all finished (rc = 0). The output
  results/b61_test1.json was pulled back MD5-identical (667b2cbc...; committed gzipped).
- B38's extension rule (b61_hd.extend, mechanical) added **0** budgets: for every (arm, condition) the top evaluated
  budget passes and no H-HAD N* sits at its lowest evaluated budget (only the count of additions was read).
- **No cut**: the whole registered test (seven i.i.d. conditions for the three arms, four natural-layout damage
  conditions for H-SBC / H-SIC) finished before the 12:05 cut time.
- Next: the verdict (b61_verdict.py, committed 47b08b7), then the checks (b61_check.py, committed 504680d) and the
  predictions (b61_predictions.py, written during the test run from the pre-registration's R1-R8, committed with this
  note). The exploratory dev-only lam = 0.25 run (compute note 3) is still running locally.
