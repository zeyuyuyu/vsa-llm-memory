"""B49 panel (prereg/B49_panel_mechanism_competitors_prereg.md): B32's and B36's code, unchanged, on the codes of the
panel's new encoders (Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, bge-small-en-v1.5, all-MiniLM-L6-v2), exactly as
b46_run.py ran it on Granite. One working directory per encoder, b49_out/enc_{ENC}/: its b32_data holds B32's
entities.json and emb_strings.json (copies, md5-checked) and the encoder's codes; b32_out/, b36_out/ and results/ there
receive what B32 / B36 wrote at the repository root. This file only registers the encoder tag in B32's and B36's
encoder tables and calls their functions; no grid, rule, seed or accounting is changed.

python3 b49_run.py ENC setup CODES.npy
python3 b49_run.py ENC selftest
python3 b49_run.py ENC fit PREP FAMILY P[,P..]
python3 b49_run.py ENC b32dev vsa|idx
python3 b49_run.py ENC b32choose
python3 b49_run.py ENC dev32
python3 b49_run.py ENC pqfit
python3 b49_run.py ENC dev [--workers N] [--fam ...]
python3 b49_run.py ENC choose
python3 b49_run.py ENC b32test vsa|idx
python3 b49_run.py ENC b32merge
python3 b49_run.py ENC test [--workers N]
python3 b49_run.py ENC test32
python3 b49_run.py ENC merge
"""
import hashlib
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
NEW_ENCS = {'llama': 'llama31i_mean_L24_f16', 'mistral': 'mistral7i_eol_L24_f16', 'bge': 'bge_small',
            'minilm': 'minilm'}


def wd(enc):
    return os.path.join(ROOT, 'b49_out', f'enc_{enc}')


def md5(fn):
    return hashlib.md5(open(fn, 'rb').read()).hexdigest()


def setup(enc, codes):
    W = wd(enc)
    for d in ('b32_data', 'b32_out/fitcache', 'b32_out/test', 'b36_out/test', 'b36_out/pqcache', 'results', 'logs'):
        os.makedirs(os.path.join(W, d), exist_ok=True)
    for f in ('entities.json', 'emb_strings.json'):
        src, dst = os.path.join(ROOT, 'b32_data', f), os.path.join(W, 'b32_data', f)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        assert md5(src) == md5(dst), f
    base = os.path.basename(codes)
    assert base == f'emb_{NEW_ENCS[enc]}.npy', (base, NEW_ENCS[enc])
    dst = os.path.join(W, 'b32_data', base)
    if os.path.abspath(codes) != dst:
        shutil.copyfile(codes, dst)
    json.dump({'enc': enc, 'file_tag': NEW_ENCS[enc], 'md5': md5(dst)}, open(os.path.join(W, 'b49_encoder.json'), 'w'),
              indent=1)
    print('setup', enc, NEW_ENCS[enc], md5(dst))


def register(enc):
    os.chdir(wd(enc))
    sys.path.insert(0, ROOT)
    os.environ.setdefault('B32_FIT_CACHE', 'b32_out/fitcache')
    tag = json.load(open('b49_encoder.json'))['file_tag']
    import b32_run as R32
    import b36_run as R36
    R32.ENCS[enc] = tag
    R36.ENCS[enc] = tag
    return R32, R36


def main():
    enc, a = sys.argv[1], sys.argv[2:]
    assert enc in NEW_ENCS, enc
    wk = int(a[a.index('--workers') + 1]) if '--workers' in a else 1
    if a[0] == 'setup':
        setup(enc, a[1])
        return
    R32, R36 = register(enc)
    if a[0] == 'selftest':
        R32.selftest('cpu')
        R36.selftest()
    elif a[0] == 'fitlist':  # every m a key of <= 600 bits can use (64 bits per fact at a stored share >= 1/8)
        import numpy as np
        n = np.load(f'b32_data/emb_{R32.ENCS[enc]}.npy', mmap_mode='r').shape[1]
        nb = 4 if a[1] == 'pq4' else 8
        print(','.join(str(m) for m in range(1, n + 1) if n % m == 0 and m * nb <= 600))
    elif a[0] == 'fit':
        import b32_common as C
        pool = C.Pool(R32.ENCS[enc])
        Tr = pool.train_matrix()
        fam = C.IndexFamily(a[2], C.Prep(a[1], Tr)(Tr), cache_tag=f'{enc}_{a[1]}')
        for p in a[3].split(','):
            fam._fit(int(p))
            print('fitted', enc, a[1], a[2], p, flush=True)
    elif a[0] == 'b32dev':
        R32.dev(enc, f'b32_out/dev_{enc}_{a[1]}.jsonl', a[1], 'cpu')
    elif a[0] == 'b32choose':
        R32.choose(enc, f'b32_out/dev_{enc}_vsa.jsonl,b32_out/dev_{enc}_idx.jsonl', f'results/b32_choice_{enc}.json')
    elif a[0] == 'dev32':
        R36.dev32(enc, f'b36_out/dev32_{enc}')
    elif a[0] == 'pqfit':
        import time
        S = R36.Setup(enc)
        for p in R36.PREPS:
            t0 = time.time()
            if S.banks[p].cent is None:
                S.banks[p].fit()
            print('pq fitted', enc, p, f'{time.time() - t0:.0f}s', flush=True)
    elif a[0] == 'dev':
        fams = a[a.index('--fam') + 1].split(',') if '--fam' in a else R36.FAMS
        R36.dev(enc, f'b36_out/dev_{enc}.jsonl', fams, wk)
    elif a[0] == 'choose':
        R36.choose(enc)
    elif a[0] == 'b32test':
        R32.test(enc, f'results/b32_choice_{enc}.json', 'b32_out/test', a[1], 'cpu')
    elif a[0] == 'b32merge':
        R32.merge('b32_out/test', [enc])
    elif a[0] == 'test':
        R36.test(enc, f'results/b36_choice_{enc}.json', 'b36_out/test', wk)
    elif a[0] == 'test32':
        R36.test32(enc, 'b36_out/test')
    elif a[0] == 'merge':
        R36.merge(enc, 'b36_out/test')
    else:
        raise SystemExit(f'unknown mode {a[0]}')


if __name__ == '__main__':
    main()
