# B69 compute note 2 (15:19 machine clock; dev running on the box; part (A) finished, its output not yet read)

- **Part (A)** ran once locally, 15:10-15:11 (b69_vocab.py partA results/b69_partA, the code of 4449bc4); its output
  files exist and are read after this note is committed. No change to the part (A) code.
- **Store upload.** The box link is slow (about 4 KB/s); the first upload of the store files was cut by its 300 s timeout
  after 1 MB of sweep_dev.json.gz (the partial file was deleted on the box). Instead the stores were generated on the
  box with the same generator: wonderwords is not installed there, so b69_vocab.words() now falls back to the same two
  sorted word lists exported locally (b69_out/words.json.gz, 910 adjectives, 6,782 nouns). The decompressed content of
  both store files is identical on the box and locally (MD5 56b2bfa9... dev, b75bc0e4... test); every code file on
  the box is MD5-identical to the local one.
- **Verdict code fix (before any output exists).** verdict_b labelled each N over every condition run; the
  pre-registration labels over k1, k8, k16 with k4 reported beside it. The call now passes the primary conditions only.
- **Dev** started on the box at about 15:18 (b69_box.sh run, 60 workers, nice 10, results/b69_plan_dev1.json).
- The waiting loops of this session used pgrep -f patterns that matched their own shells and were stopped; no box or
  local process of B69 was affected and no other process was touched.
