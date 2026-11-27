# B59 compute note 6 (03:24 machine clock; after the part 2 test output was read; analysis only)

- **Damage: P-best's per-budget maximum is not well defined there; P-any is reported.** The registered P-best takes,
  at every budget, the largest test share among U, P-BCH and P-LDPC, and N* is the smallest budget passing there and at
  every larger evaluated budget. On the i.i.d. conditions U is evaluated on the whole grid (B38 + T1), so the maximum
  is always over U and the codes, and P-best equals the smallest of the three families' own N* (P-any) at all seven
  k. In the damage conditions U has only B38's evaluated budgets (no T1 for damage was registered), so at budgets
  where only a failing code family was evaluated the maximum reads as a failure: at lines-nat-k8 and clust-nat-k8
  P-best by the maximum is 724 and 1,722, although U itself passes from 152 and fails nowhere above. That is an
  artefact of unequal coverage, not a property of the store. The damage table therefore reports P-any (each family's
  N* on its own budgets, the smaller of the three; B58's P-any) for the label, and shows the per-budget maximum beside
  it, marked. Q6 is evaluated on P-any and its literal reading is stated.
- **T6 budgets.** The re-instated exactly-filling P-LDPC budgets (compute note 5: 152 at k1-k8, 304 and 362 at k8 and
  in damage) turned out infeasible on 23 of the 50 test stores (32 of 50 at 362): with the store's per-store rounding
  of floor(b n) and floor(B n), a code that fills the budget exactly at one copy does not fit every store. By B38's
  rule a budget counts only when every store is evaluated, so these budgets stay out of N* (they are listed in the
  verdict's coverage table). Nothing changes in any N*.
- Damage extension: B38's rule needed three passes (P-BCH up to 2,896 at clust-nat-k8, P-LDPC up to 609); a fourth
  pass added nothing. All of part 2 (T1-T6 and the extensions) finished at 03:19.
