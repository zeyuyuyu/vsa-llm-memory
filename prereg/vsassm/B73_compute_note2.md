# B73 (equal bits) compute note 2 (about 20:38 machine clock; written before any test output is read)

- **Fetch** (curl, compute note 1) 20:24-20:27, one query per class, none failed: rows (item x alias) human 4,089, film
  4,049, literary work 1,054, album 989, TV series 1,382, single 467, city 6,810, football club 5,948, business 3,493
  (b73_out/fresh/fetch_log.json). **Build** (registered rule): 8,472 entities after the quotas, 50 dropped for a shared
  label, aliases 13,016 -> 12,744 after the ambiguity filter; 8,422 entities (train 2,500 / dev 1,736 / test 4,186; with
  an alias 2,489 / 1,731 / 4,176), 21,166 strings. Classes below quota: human 2,466 of 2,600, literary work 620 of 800,
  album 646 of 700, single 240 of 400 (all their survivors kept, as registered). **Leakage check** (b73_data.py leak):
  0 URIs, 0 exact strings and 0 normalized strings of the fresh set in PopQA's entity file or raw subjects (12,244
  PopQA URIs, 20,471 normalized PopQA strings). Committed at 20:28:18 (e5bc808), before any code was encoded.
- **Sentence encoders** encoded on this machine's CPU (b32_embed.py st, the paper's model ids, HF cache, offline),
  20:30-20:34: emb_e5_base (21,166 x 768, md5 fb2534d1...), emb_bge_small (x 384, 699742b3...), emb_minilm (x 384,
  5679413d...).
- **Deviation (machine only).** The link to the A800 box stalled on every transfer above a few hundred kB between 20:22
  and 20:36 (two uploads of the code and the entity file timed out; a fresh login at 20:33 worked). So that the time box
  holds, the sentence encoders' test ran **on this machine** instead of the box (same code, b73_run.py runplan, 8
  workers, nice 5): b73_out/plan_local_sent.txt (138 units: LAMC, NLMSC, STORE, VSAS, VSAPC, VSAR, SDMT for e5, bge,
  minilm), 20:34:20-20:35:59, 138 / 138 units rc = 0. Only the completion lines (unit, rc, seconds) have been read.
  Each unit ran once. The language models' codes are to be encoded on the box (GPUs 1, 2, 4, 5, 6, 7 if idle) and their
  test run there; the files needed on the box are copied from earlier run directories when MD5-identical to this
  repository's, the rest uploaded in small parts.
- No test output (prediction, accuracy, row) has been read; the verdict has not been run.
