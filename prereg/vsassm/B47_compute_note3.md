# B47 compute note 3 (2026-11-19 ~18:10 UTC; no design change)

1. **PACE reachable again (~17:47 UTC); compute note 2 item 1 superseded.** sacct shows that every PACE job queued in
   compute note 1 ran (gts-nimam6-paid, H200 nodes, 10:06-10:57 UTC) and completed: the 800 grids of gen (6), fullgen
   (3), fullcls (3) and scratch (2), the 8,000 grids of fullgen (3) and gen at ranks 8 and 64 (6), and the Phi-2 naming
   head at 8,000 (b47_name8k). No B47 run had started on the A800 box (its workers never found an idle GPU; their queue
   was emptied at 17:47 UTC before any claim), so there are no twins; the PACE outputs are the registered ones.
2. **Selections** (b47_select.py on the tuning runs' train/dev numbers; b47_out/select.json): gen 800 rank 64 lr 3e-4
   (no configuration fits; best dev 0.524); fullgen 800 lr 3e-6 (no configuration fits; dev 0.514); fullcls 800 lr 3e-5
   (no fit; dev 0.548); scratch 800 lr 3e-4 (fits; dev 1.000, early stop); gen 8,000 at the 800 rank (64), lr 3e-5 (dev
   0.534); fullgen 8,000 lr 3e-6 (dev 0.530). The three rank-8 gen runs at 8,000 are run and not used (compute note 1
   item 7).
3. **Finals and seed-42 test scoring** submitted to PACE (gts-nimam6-paid): one job scores the six selected seed-42
   states once (b47_score42.sh); seeds 123 / 456 of the six selected configurations as --final runs (0:40 segments for
   800, 1:15 for 8,000). If a box GPU becomes idle while some of these are still pending, the pending PACE job is
   cancelled (squeue-checked) before the same command is queued on the box, never both.
4. **Exploratory 1** (prereg/B47_exploratory1.md, committed before it runs): answer-token positive control, two runs on
   PACE (train / dev only).
