# B59 compute note 2 (02:42 machine clock; part 1 dev choices written, before any part 1 test run)

- **Part 1 dev ran on the box**: dev1 (72 MF grid parts + 12 LAM rp dev jobs) 02:13-02:28, all 84 rc = 0; MF choices
  02:28; dev2 (48 engine dev jobs, 8 jobs x 3 threads) 02:28-02:38, all rc = 0. Choices written locally by
  b59_run.choose (results/b59_choice_{b46,llama,mistral,bge}.json, 1,176 per encoder; results/b59_choicem_*.json).
- **Gate (dev data only)**: b59_run.choose run on B57's own dev files reproduces B57's committed map choices for
  Qwen3-8B and e5-base in 1,176 / 1,176 entries each, so the per-map rules are B57's.
- **Cross-machine check (registered; dev data only)**: bge MF part (map 1, center:sgn) box vs local: 105 of 3,024
  configuration accuracies (18 dev stores) differ, largest 1.8 points on one store; Granite engine dev job (map 1,
  M = 256, V = 2) box vs local: 44 of 1,139,040 predictions differ, in 12 of 630 configurations. The two environments
  (numpy 1.26 / torch 2.4 on the box, numpy 2.2 / torch 2.11 locally) round float32 sums differently, which flips
  near-tied argmax reads. The new maps' dev and test both run on the box, so each map is internally consistent; the
  registered map (map 0) was computed locally by B53 / B56. **Disclosed addition (before the test):** to measure what
  this does on test data, B53's registered-map VSA test for bge (its committed choice results/b53_choice_bge.json,
  b53_run.test unchanged, output to a scratch directory) is re-run on the box and compared prediction by prediction
  with B53's local test file; the count of differing predictions and the change of the R1 bge VSA - LAM point are
  reported in the check file. This is a check, not an arm; nothing in the bars uses it.
- Part 1 test: b59_box.sh test with 8 jobs x 3 threads (the P-LDPC dev shards hold 24 box cores until they end).
- Nothing about designs, grids, seeds, bars or predictions changed.
