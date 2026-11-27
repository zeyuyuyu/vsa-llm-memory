# B30b compute note 2 (2026-11-17, 23:40 UTC; no design change)

1. G-E was evaluated for impact and window before any S32 output existed (both S32 R1 jobs started at 23:36 and
   23:38 UTC, after the window forcing ended): impact re-run 0.296 vs published 0.278, window 0.183 vs 0.202, both
   within 0.03 (gate passes); identical parsed answers 113/230 and 251/460 (below the predicted 80%).
2. G-E location (registered after S32 R1) is started now on the GPU pair that S32 R1 left free (b30b_run_ge_where.sh,
   the same commands as b30b_run_a2.sh's last step); when b30b_run_a2.sh reaches that step, the original runners find
   every problem answered and do nothing. This changes only when the job runs.
3. Part (b): the reader pass (125,436 distinct prompts, 5.7 min on one GPU) and G-B passed at every length before
   any notepad arm ran (500/500, 125/125, 125/125 memory answers equal to banked B18).
