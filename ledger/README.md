# Ledger

The timing record of the pre-specifications, without the development history of the source repositories:

- `prereg_timeline.tsv`: for each experiment, the first and last commit (hash and date) of each included
  preregistration, amendment, verdict and output file. A preregistration predates its result when its first commit
  is older than the first commit of the verdict.
- `prereg_checks.txt`: that comparison for every preregistration, paired with the verdict of the same experiment tag
  (amendments made after a first verdict say so in their own text).

Commit dates are internal records (the workstation clock is Asia/Shanghai, +08:00); the public evidence of timing is
the OpenTimestamps proofs next to the pre-specifications. Absolute paths and user names are redacted as in the rest
of the package.
