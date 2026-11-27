# B73 (equal bits) compute note 5 (about 21:42 machine clock; written before any language-model test output is read)

- **Encoding** (box, b73_box.sh embed, the paper's recipes and model files): Qwen3-8B on GPU 1 13:30:11Z-13:32:08Z,
  Llama-3.1-8B on GPU 4 13:30:37Z-13:32:23Z, Mistral-7B on GPU 5 from 13:31:02Z, Granite-3.3-8B on GPU 6
  13:31:27Z-13:33:42Z (all rc 0; ledger ~/b73_run/logs/gpu_ledger.txt). Each GPU idle at its claim (4 MiB used, no
  compute app, re-checked 20 s later); GPUs 0, 2, 3 not used. Codes stored as float16 (MD5s in logs/codes_md5.txt).
  GPU time about 8 minutes (about 0.14 GPU-hours with the claim checks).
- **Test** ran once on the box, 21:33:44-21:40:16 (b73_run.py runplan, 48 workers, nice 5): 184 units (46 per
  language model), 0 failures, **0 cut** (the clock rule did not bind; every registered arm and encoder ran). Only the
  completion lines have been read.
- **Post-test steps.** b73_box_all.sh had assembled the records and started the verdict's per-encoder parts one after
  the other; to save wall time its shell (PID 4124105) and the running part process (PID 4128264) were stopped by PID
  at 21:40:3x, before any part file was written, and b73_box_post.sh (committed fe4e6b5) runs the same commands with
  the four parts, the bits audit, a determinism re-run (8 units into run_det) and the Qwen3-8B sample reading in
  parallel, then the panel and the plain-loop scorer. No statistic or rule changes.
- **Sentence encoders**: the best of six is not run (registered); every other registered arm ran on all 7 encoders.
- CPU: language-model test 184 units in 6.5 minutes on 48 workers (unit times in logs/runplan_llm.log); sentence
  encoders' test 0.22 CPU-hours locally.
