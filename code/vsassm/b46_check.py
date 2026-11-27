"""B46 checks (prereg/B46_second_llm_family_prereg.md, 'Checks'): B36's and B32's checkers, unchanged except for the
encoder list and the code-file table, which are substituted in their source text (every substitution is asserted to
occur the expected number of times), run in b46_out/.

python3 b46_check.py score                 # b36_check.score on the B46 dumps (vs b46_out/results/b36_verdict.json)
python3 b46_check.py bits                  # b36_check.bits on the B46 dumps
python3 b46_check.py rederive [--frac 0.1] # b36_check.rederive (new families) -> results/b46_rederive.json
python3 b46_check.py rederive32 [--frac 0.1]  # b32_check.rederive (VSA / index) -> results/b46_rederive32.json
"""
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
WD = os.environ.get('B46_WD', os.path.join(ROOT, 'b46_out'))  # B46_WD: smoke tests only
RES = os.path.join(ROOT, 'results') if 'B46_WD' not in os.environ else os.path.join(WD, 'results')  # smoke: stay in WD


def load(fn, subs, glob_extra):
    src = open(os.path.join(ROOT, fn)).read()
    for old, new, n in subs:
        c = src.count(old)
        assert c == n, (fn, old, c, n)
        src = src.replace(old, new)
    g = {'__name__': f'{fn[:-3]}_b46', '__file__': os.path.join(ROOT, fn)}
    g.update(glob_extra)
    exec(compile(src, fn + ' (B46 substitution)', 'exec'), g)
    return g


def main():
    a = sys.argv[1:]
    os.chdir(WD)
    sys.path.insert(0, ROOT)
    tag = json.load(open('b46_encoder.json'))['file_tag']
    frac = float(a[a.index('--frac') + 1]) if '--frac' in a else 0.1
    if a[0] in ('score', 'bits', 'rederive'):
        g = load('b36_check.py', [("for enc in ('qwen8', 'e5'):", 'for enc in ENCS46:', 3),
                                  ("tag = {'qwen8': 'qwen8_mean_L9_f16', 'e5': 'e5_base'}[enc]", 'tag = TAGS46[enc]', 1)],
                 {'ENCS46': ('b46',), 'TAGS46': {'b46': tag}})
        if a[0] == 'score':
            g['score']('b36_out/test')
        elif a[0] == 'bits':
            g['bits']('b36_out/test')
        else:
            g['rederive']('b36_out/test', 'b46', frac)
            shutil.copyfile('results/b36_rederive_b46.json', os.path.join(RES, 'b46_rederive.json'))
    elif a[0] == 'rederive32':
        g = load('b32_check.py', [("files = {'qwen8': 'b32_data/emb_qwen8_mean_L9_f16.npy', 'e5': 'b32_data/emb_e5_base.npy'}",
                                   "files = {'b46': f'b32_data/emb_{TAG46}.npy'}", 1),
                                  ("for e, p in (('qwen8', chq), ('e5', che)) if e in encs",
                                   "for e, p in (('b46', chq),) if e in encs", 1)],
                 {'TAG46': tag})
        g['rederive']('b32_out/test', 'results/b32_choice_b46.json', None, frac, ('b46',))
        shutil.copyfile('results/b32_rederive_b46.json', os.path.join(RES, 'b46_rederive32.json'))


if __name__ == '__main__':
    main()
