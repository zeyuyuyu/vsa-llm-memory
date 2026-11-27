# Third-party timestamps (OpenTimestamps)

`manifest_2026-11-19.txt` lists the SHA-256 of every pre-registration (`prereg/*.md`) and verdict file
(`results/*verdict*.txt`), and the HEAD commits of this repository and of the paper repository, as of
2026-11-19 05:11 UTC. Its digest was submitted to three OpenTimestamps calendars
(`manifest_2026-11-19.txt.ots`). Only the digest left the machine. After Bitcoin confirmation, run `ots upgrade` to embed the
attestation, then `ots verify manifest_2026-11-19.txt.ots` to check it.

What it proves: these exact files existed no later than the attested time. It does NOT prove that earlier
pre-registrations preceded their own results. Pre-registrations written from 2026-11-19 on (B47-B50) are stamped
individually before any test run (`prereg/*.md.ots`), which does prove that order.
