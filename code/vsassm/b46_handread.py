"""B46 hand reading (prereg/B46_second_llm_family_prereg.md, 'Checks'): b36_handread.py, unchanged except for the
code-file table (substituted in its source text, asserted), run in b46_out/ on the B46 dumps.

python3 b46_handread.py            # -> results/b46_handread.txt
"""
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
WD = os.environ.get('B46_WD', os.path.join(ROOT, 'b46_out'))  # B46_WD: smoke tests only
RES = os.path.join(ROOT, 'results') if 'B46_WD' not in os.environ else os.path.join(WD, 'results')  # smoke: stay in WD

if __name__ == '__main__':
    os.chdir(WD)
    sys.path.insert(0, ROOT)
    tag = json.load(open('b46_encoder.json'))['file_tag']
    src = open(os.path.join(ROOT, 'b36_handread.py')).read()
    old = "pool = C.Pool({'qwen8': 'qwen8_mean_L9_f16', 'e5': 'e5_base'}[enc])"
    assert src.count(old) == 1
    src = src.replace(old, f"pool = C.Pool({tag!r})")
    sys.argv = ['b36_handread.py', 'b36_out/test', 'b46']
    exec(compile(src, 'b36_handread.py (B46 substitution)', 'exec'), {'__name__': '__main__'})
    shutil.copyfile('results/b36_handread_b46.txt', os.path.join(RES, 'b46_handread.txt'))
