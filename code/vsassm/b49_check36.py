"""B49 checks of B36's / B32's families on the panel's new encoders (prereg/B49 section 9): b36_check.py's bits audit
and independent re-derivation, and b32_check.py's re-derivation of the VSA / index, unchanged except for the encoder
list and the code-file table (substituted in their source text; every substitution asserted), run in the encoder's
working directory b49_out/enc_{ENC}/ (as b46_check.py did for B46).

python3 b49_check36.py ENC bits
python3 b49_check36.py ENC rederive [--frac 0.1]      # -> results/b36_rederive_{ENC}.json in the working directory
python3 b49_check36.py ENC rederive32 [--frac 0.1]    # -> results/b32_rederive_{ENC}.json in the working directory
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def load(fn, subs, glob_extra):
    src = open(os.path.join(ROOT, fn)).read()
    for old, new, n in subs:
        c = src.count(old)
        assert c == n, (fn, old, c, n)
        src = src.replace(old, new)
    g = {'__name__': f'{fn[:-3]}_b49', '__file__': os.path.join(ROOT, fn)}
    g.update(glob_extra)
    exec(compile(src, fn + ' (B49 substitution)', 'exec'), g)
    return g


def main():
    enc, a = sys.argv[1], sys.argv[2:]
    os.chdir(os.path.join(ROOT, 'b49_out', f'enc_{enc}'))
    sys.path.insert(0, ROOT)
    tag = json.load(open('b49_encoder.json'))['file_tag']
    frac = float(a[a.index('--frac') + 1]) if '--frac' in a else 0.1
    if a[0] in ('bits', 'rederive'):
        g = load('b36_check.py', [("for enc in ('qwen8', 'e5'):", 'for enc in ENCS49:', 3),
                                  ("tag = {'qwen8': 'qwen8_mean_L9_f16', 'e5': 'e5_base'}[enc]", 'tag = TAGS49[enc]', 1)],
                 {'ENCS49': (enc,), 'TAGS49': {enc: tag}})
        if a[0] == 'bits':
            g['bits']('b36_out/test')
        else:
            g['rederive']('b36_out/test', enc, frac)
    elif a[0] == 'rederive32':
        g = load('b32_check.py', [("files = {'qwen8': 'b32_data/emb_qwen8_mean_L9_f16.npy', 'e5': 'b32_data/emb_e5_base.npy'}",
                                   "files = {ENC49: f'b32_data/emb_{TAG49}.npy'}", 1),
                                  ("for e, p in (('qwen8', chq), ('e5', che)) if e in encs",
                                   "for e, p in ((ENC49, chq),) if e in encs", 1)],
                 {'TAG49': tag, 'ENC49': enc})
        g['rederive']('b32_out/test', f'results/b32_choice_{enc}.json', None, frac, (enc,))


if __name__ == '__main__':
    main()
