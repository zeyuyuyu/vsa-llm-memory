# B61 compute note 3 (10:59 machine clock; test running on the box since 10:54:12; no test output has been read)

- Compute note 2's header says 10:55; it was committed at 10:53 (47b08b7). The test started at 10:54:12 (4,600 jobs).
- **Exploratory dev-only check (not registered; cannot change any test arm or bar).** Because lam = 0.35, the lower edge
  of the registered grid, was chosen at k32 (compute note 2), H-SBC / H-SIC are evaluated on the 12 dev stores at
  lam = 0.25 at the budgets within 3 grid steps of the dev N* at k16 (362) and k32 (1,448), locally (8 workers; the
  local load fell to about 1 at 10:58). Output results/b61_dev_x025.json; the dev shares at lam 0.25 are compared with the
  registered choice at the same budgets and reported in the note after the test as a statement on the grid's lower
  edge only. No test evaluation at lam 0.25 is run.
