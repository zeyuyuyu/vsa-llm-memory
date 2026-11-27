"""B68 (prereg/B68_early_layer_labelfree_prereg.md): runs the UNCHANGED B42 / B31 arm scripts on the input-embedding and
early-layer features of Phi-2 and Qwen3-8B (b68_feats.py). One thing is patched at import time and nothing else: the
--enc_model choices of b42_arms.py / b31_arms.py are widened to the B68 feature sets (<enc>L<L>, L in 0, 2, 4).

  python b68_run.py b42_arms.py --enc_model qwen8L0 --arena A --fam vq --mode tune ... --feats <b68 feats>
  python b68_run.py b31_arms.py --enc_model phi2L2 --arena A --arm name --mode final ... --feats <b68 feats>
"""
import argparse
import os
import runpy
import sys

B68_ENCS = ["%sL%d" % (e, L) for e in ("phi2", "qwen8") for L in (0, 2, 4)]
_add = argparse.ArgumentParser.add_argument


def _add_patched(self, *args, **kw):
    if args and args[0] == "--enc_model":
        kw["choices"] = B68_ENCS
    return _add(self, *args, **kw)


argparse.ArgumentParser.add_argument = _add_patched

if __name__ == "__main__":
    script = os.path.abspath(sys.argv[1])
    assert os.path.basename(script) in ("b42_arms.py", "b31_arms.py"), script
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(script))
    runpy.run_path(script, run_name="__main__")
