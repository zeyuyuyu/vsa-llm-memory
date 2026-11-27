"""NSR1 reader (prereg/NSR1_prereg.md, arm 1's state): the shipped Readable Memory reader exactly as committed at 707351b
(nsr/vendor_707351b/run_real.py), with only its data and output directories redirected:

  --split dev    reads nsr/data/<scenario>_dev.jsonl, copies of apps/real/data's dev logs (checks that this copy
                 reproduces apps/real/out/reader_dev_verify_v4.jsonl)
  --split nsr1   reads nsr/data/<scenario>_nsr1.jsonl (the held-out logs)
Output always goes to nsr/out/reader_<split>_verify<tag>.jsonl.

python3 nsr/nsr1_reader.py reader --model <Qwen3-1.7B> --adapter b16_models/student_s0 --split nsr1 --verify --tag _v4
"""
import os
import sys

NSR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(NSR, "vendor_707351b"))
import run_real  # noqa: E402

# Every split is read from nsr/data (the dev check uses copies of apps/real/data/<scenario>_dev.jsonl placed there;
# the vendored module's own default, nsr/vendor_707351b/data, does not exist).
run_real.DATA = os.path.join(NSR, "data")
run_real.OUT = os.path.join(NSR, "out")

if __name__ == "__main__":
    run_real.main()
