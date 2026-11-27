# B58 compute note 3 (21:18 machine clock, during dev pass 1; before dev pass 2 and before any test run)

Dev pass 1 is slower than measured at 20:48: the four 6,000-pair dev stores take about 1 hour (single jobs up to 20
minutes, where belief propagation fails on every word at large b and high noise while the store still reads), so
pass 1 ends about 21:45. With the reduced P-LDPC pass 2 of compute note 2 (about 20 minutes), the test could not start
before about 22:10, and the damage conditions would hit the registered 23:20 cut. Further cut, fixed now:

- **No P-LDPC dev pass 2.** P-LDPC designs come from pass 1 (budgets 90.5, 128, 256, ..., 4,096, every one of the 28
  designs at each); dev N* for P-LDPC is at octave resolution. Test budgets between octaves use compute note 2's rule
  (nearest dev budget's design that fits). P-BCH keeps its registered pass 2 (+-3 quarter steps; 4 designs, cheap).
  Effect: P-LDPC designs at quarter-step test budgets are not tuned at those budgets, which can only raise P-LDPC's
  test N* (by at most the gap to a tuned design); this works against the protected store and is reported as such.
- Everything else as registered and in compute note 2 (i.i.d. first; damage after, cut at 23:20 if unfinished).
