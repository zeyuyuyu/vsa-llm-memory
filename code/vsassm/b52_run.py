"""B52 (prereg/B52_robustness_layers_maps_1day_prereg.md), part (a): runs the UNCHANGED B42 / B31 arm scripts on other
Qwen3-8B feature layers. Two things are patched at import time and nothing else:
  1. argparse: the --enc_model choices of b42_arms.py / b31_arms.py are widened to the B52 feature sets
     (qwen8L0 = input embeddings, hidden_states[0]; qwen8L9 / qwen8L27 = B31 exploratory-1 layers);
  2. torch.cuda.get_device_name: returns "cpu (N threads)" for a CPU device (b31_arms.py records the GPU name and
     would fail on the box CPU otherwise; b42_arms.py already handles CPU).

  python b52_run.py b42_arms.py --enc_model qwen8L27 --arena C --fam vq --mode tune ... --device cpu
  python b52_run.py b31_arms.py --enc_model qwen8L0 --arena C --arm name --mode final ... --device cpu
"""
import argparse
import os
import runpy
import sys

import torch

B52_ENCS = ["qwen8L0", "qwen8L9", "qwen8L27"]
_add = argparse.ArgumentParser.add_argument


def _add_patched(self, *args, **kw):
    if args and args[0] == "--enc_model":
        kw["choices"] = B52_ENCS
    return _add(self, *args, **kw)


argparse.ArgumentParser.add_argument = _add_patched
_gdn = torch.cuda.get_device_name


def _gdn_patched(device=None):
    if device is not None and not str(device).startswith("cuda"):
        return "cpu (%d threads)" % torch.get_num_threads()
    return _gdn(device)


torch.cuda.get_device_name = _gdn_patched

if __name__ == "__main__":
    script = os.path.abspath(sys.argv[1])
    assert os.path.basename(script) in ("b42_arms.py", "b31_arms.py"), script
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(script))
    runpy.run_path(script, run_name="__main__")
