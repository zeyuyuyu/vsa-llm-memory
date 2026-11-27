"""B28 gate G0 (environment): run the ORIGINAL scripts (verbatim copies in b28_ngcls/, md5-checked) on their
published configuration, with outputs redirected to the B28 directory. No new arm, no encoding change.

  python b28_orig.py v58 --arms parser probe_argmax --out <dir>/g0_v58.json
  python b28_orig.py v52 --arms parser probe_argmax --out <dir>/g0_v52.json
  python b28_orig.py v37b --arms floor parser probe_argmax --out <dir>/g0_v37b.json
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b28_common as C                                                # noqa: E402

if __name__ == "__main__":
    name = sys.argv[1]
    assert name in ("v58", "v52", "v37b"), name
    M = C.load_ngcls()
    sys.argv = [name + ".py"] + sys.argv[2:]
    M[name].main()
