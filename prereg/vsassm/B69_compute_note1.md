# B69 compute note 1 (15:12 machine clock; no dev, test or part (A) output has been read)

- **Registration time.** The pre-registration's header says "about 15:20"; it was committed at 15:07 (4449bc4) and
  stamped, the proof committed at 15:07 (41bf357). The check code b69_check.py was committed at 15:10 (e587f61), before
  any B69 run output existed (one fix before that commit: a one-copy BCH design is now also checked to fit its cells).
- **Sweep stores** generated locally after the registration (b69_vocab.py stores, 15:07): b69_out/sweep_dev.json.gz
  (12 stores, MD5 bc0908b3...) and b69_out/sweep_test.json.gz (32 stores, MD5 57a6f8f8...); gzip headers carry a time
  stamp, so a regeneration has the same content but another MD5. Only the store counts were printed.
- **Dev plan** results/b69_plan_dev1.json (b69_vocab.py devplan, from results/b69_theory.json): 2,592 evaluations per
  store set of the four N, 7,776 on the 12 dev stores (H-SBC, E-LDPC, O-BCH, O-LDPC; U, T-rep and E-BCH have no
  development).
- **Box**: ~/b69_run, the code of 4449bc4 and the stores uploaded (MD5 compared before the dev run; the link is slow).
- **Part (A)** is being computed locally (b69_vocab.py partA results/b69_partA, started 15:10, stdout to a log); its
  output is not read before compute note 2.
- The timing run of the pre-registration's self-test record (a 'timing' split store, N = 24,000) printed shares for
  that store only; they do not enter any choice.
