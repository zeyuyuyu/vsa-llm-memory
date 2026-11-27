# B30b compute note 3 (2026-11-18, ~00:05 UTC; no design change, no S32 output scored)

S32 R2 (seed 1, the registered secondary replicate) is started now rather than after R1, on the GPU pairs that free up
while R1 runs (b30b_run_r2.sh). The registered condition is that the ledger after R1 and part (b) at 128k stays at or
below 55 GPU-hours; at this point the ledger stands at about 8 GPU-hours and R1 projects to about 12, so the condition
holds whatever R1's tail does. The comparison rules for R2 are unchanged (reported, not a bar).
