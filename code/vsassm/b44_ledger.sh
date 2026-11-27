#!/bin/bash
# B44 GPU-hour ledger from PACE accounting: every b44_* job since 2026-11-18 22:00 UTC (17:00 local PACE time), elapsed
# x GPUs, by state and GPU type. Free-tier account gts-nimam6 unless stated.
timeout 120 ssh -i $HOME/.ssh/id_ed25519_pace -o BatchMode=yes <user>@login-phoenix.pace.gatech.edu \
  'sacct -u <user> -S 2026-11-18T17:00 -n -X -P -o JobID,JobName,Account,Partition,State,ElapsedRaw,AllocTRES' \
  | awk -F'|' '$2 ~ /^b44_/ {g=0; if (match($7, /gres\/gpu=[0-9]+/)) g=substr($7, RSTART+9, RLENGTH-9); h=$6*g/3600; tot+=h; byp[$4]+=h; bys[$5]+=h; bya[$3]+=h; n++}
     END {printf "jobs %d  GPU-hours %.2f\n", n, tot; for (p in byp) printf "  partition %-14s %.2f\n", p, byp[p]; for (s in bys) printf "  state %-14s %.2f\n", s, bys[s]; for (c in bya) printf "  account %-16s %.2f\n", c, bya[c]}'
