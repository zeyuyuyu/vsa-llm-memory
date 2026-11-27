# B27 exploratory run 1 (not registered): the arena-A InfoNCE arm trained for 30 epochs

Date: 2026-11-17, about 15:40 UTC. Written after the label-free tuning and selection (dev only) and before any
label-free test number exists.

## Why

- A's dev-selected label-free configuration is nce, lambda 100, lr 1e-3, qc index. Its dev accuracy was still rising
  at the last registered epoch:
  - epochs 1-10: 0.572, 0.602, 0.654, 0.694, 0.742, 0.744, 0.778, 0.768, 0.788, 0.814;
  - training accuracy over the same epochs: 0.677 to 0.948.
- The registered final runs (10 epochs) therefore measure a projection that has not finished training. Whether more
  training brings a label-free projection to the naming head (0.914) is the referee's question.

## Run

- The same configuration (b27_lf.py --arena A --mode final --reg nce --lam 100 --lr 1e-3), 30 epochs, seeds 42,
  123 and 456.
- Test accuracy is read at each seed's dev-best epoch, as in the registered runs.

## Status

- **Exploratory.** It is reported beside the registered 10-epoch result, with its own bars computed the same way
  (recovery against floor 0.509 and naming head 0.914; MATCH, PARTIAL and FAIL as registered). It does not replace
  the registered result.
- It runs alongside the registered label-free finals on the same GPU, so its GPU time is mostly shared. It is
  counted in the ledger either way.
