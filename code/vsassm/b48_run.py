"""B48 part A wrapper (prereg/B48_one_form_and_alias_naming_prereg.md): runs an unchanged B31 / B42 script
(b31_arms.py, b42_arms.py, b31_probe.py) with arena C replaced by arena C1 (one surface form per symbol;
b48_struct.patch). The features passed with --feats must be the C1 features (b48_prep.py featsC1): every arm checks the
md5 of the instance texts against the feature file, so C features cannot be read by mistake.

  python b48_run.py b42_arms.py --enc_model phi2 --arena C --fam vq ... --feats feats --out out/tune/....json
"""
import os, runpy, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b48_struct                                                           # noqa: E402

b48_struct.patch()
if "--device" in sys.argv and sys.argv[sys.argv.index("--device") + 1] == "cpu":
    # CPU runs (compute note 3): b31_arms records torch.cuda.get_device_name(a.device), which fails for "cpu"; record
    # the CPU instead (as b42_arms does). No numerical change.
    import torch
    torch.cuda.get_device_name = lambda *a_, **k_: "cpu (%d threads)" % torch.get_num_threads()
script = sys.argv[1]
assert script in ("b31_arms.py", "b42_arms.py", "b31_probe.py"), script
sys.argv = sys.argv[1:]
runpy.run_path(os.path.join(HERE, script), run_name="__main__")
