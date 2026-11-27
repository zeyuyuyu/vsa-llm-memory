"""B46 (prereg/B46_second_llm_family_prereg.md): B32's and B36's code, unchanged, on the B46 encoder's codes.

Runs in the working directory b46_out/ (created by 'setup'): b46_out/b32_data holds B32's entities.json and
emb_strings.json (copies, md5-checked against the repository's) and the chosen codes; b32_out/, b36_out/ and results/
there receive what B32 / B36 wrote at the repository root. This file only registers the encoder tag 'b46' in B32's and
B36's encoder tables and calls their functions; no grid, rule, seed or accounting is changed.

python3 b46_run.py setup CODES.npy                # copy data, register the chosen codes (b46_out/b46_encoder.json)
python3 b46_run.py selftest                       # b32_run.selftest + b36_run.selftest from the working directory
python3 b46_run.py fit PREP FAMILY P[,P..]         # PQ / OPQ fits into b32_out/fitcache (B32 compute note 2)
python3 b46_run.py b32dev vsa|idx                 # B32 dev grid at 4-64 bits per fact
python3 b46_run.py b32choose                      # B32 choice -> results/b32_choice_b46.json
python3 b46_run.py dev32                          # B32 grids at 1-2 bits per fact (b36_run.dev32)
python3 b46_run.py pqfit                          # B36 PQ codebooks of the projection pool
python3 b46_run.py dev [--workers N] [--fam ...]  # B36 new families, dev grid
python3 b46_run.py choose                         # B36 choice (VSA / IDX / families / BEST) -> results/b36_choice_b46.json
python3 b46_run.py b32test vsa|idx                # B32 test (4-64) -> b32_out/test
python3 b46_run.py b32merge
python3 b46_run.py test [--workers N]             # B36 new families on the test stores
python3 b46_run.py test32                         # B32 arms at 1-2 bits per fact on the test stores
python3 b46_run.py merge                          # -> b36_out/test/all_b46_M{M}_V{V}.jsonl.gz
"""
import hashlib
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
WD = os.environ.get('B46_WD', os.path.join(ROOT, 'b46_out'))  # B46_WD: smoke tests only
ENC = 'b46'


def md5(fn):
    return hashlib.md5(open(fn, 'rb').read()).hexdigest()


def setup(codes):
    for d in ('b32_data', 'b32_out/fitcache', 'b32_out/test', 'b36_out/test', 'b36_out/pqcache', 'results'):
        os.makedirs(os.path.join(WD, d), exist_ok=True)
    for f in ('entities.json', 'emb_strings.json'):
        src, dst = os.path.join(ROOT, 'b32_data', f), os.path.join(WD, 'b32_data', f)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        assert md5(src) == md5(dst), f
    base = os.path.basename(codes)
    assert base.startswith('emb_') and base.endswith('.npy'), base
    dst = os.path.join(WD, 'b32_data', base)
    if os.path.abspath(codes) != dst:
        shutil.copyfile(codes, dst)
    tag = base[4:-4]
    json.dump({'enc': ENC, 'file_tag': tag, 'md5': md5(dst)}, open(os.path.join(WD, 'b46_encoder.json'), 'w'),
              indent=1)
    print('setup', tag, md5(dst))


def register():
    os.chdir(WD)
    sys.path.insert(0, ROOT)
    os.environ.setdefault('B32_FIT_CACHE', 'b32_out/fitcache')
    tag = json.load(open('b46_encoder.json'))['file_tag']
    import b32_run as R32
    import b36_run as R36
    R32.ENCS[ENC] = tag
    R36.ENCS[ENC] = tag
    return R32, R36


def main():
    a = sys.argv[1:]
    wk = int(a[a.index('--workers') + 1]) if '--workers' in a else 1
    if a[0] == 'setup':
        setup(a[1])
        return
    R32, R36 = register()
    if a[0] == 'selftest':
        R32.selftest('cpu')
        R36.selftest()
    elif a[0] == 'fit':  # fit PREP FAMILY P[,P...]: one or more fits, one process
        import b32_common as C
        pool = C.Pool(R32.ENCS[ENC])
        Tr = pool.train_matrix()
        fam = C.IndexFamily(a[2], C.Prep(a[1], Tr)(Tr), cache_tag=f'{ENC}_{a[1]}')
        for p in a[3].split(','):
            fam._fit(int(p))
            print('fitted', ENC, a[1], a[2], p, flush=True)
    elif a[0] == 'b32dev':
        R32.dev(ENC, f'b32_out/dev_{ENC}_{a[1]}.jsonl', a[1], 'cpu')
    elif a[0] == 'b32choose':
        R32.choose(ENC, f'b32_out/dev_{ENC}_vsa.jsonl,b32_out/dev_{ENC}_idx.jsonl', f'results/b32_choice_{ENC}.json')
    elif a[0] == 'dev32':
        R36.dev32(ENC, f'b36_out/dev32_{ENC}')
    elif a[0] == 'pqfit':
        import time
        S = R36.Setup(ENC)
        for p in R36.PREPS:
            t0 = time.time()
            if S.banks[p].cent is None:
                S.banks[p].fit()
            print('pq fitted', ENC, p, f'{time.time() - t0:.0f}s', flush=True)
    elif a[0] == 'dev':
        fams = a[a.index('--fam') + 1].split(',') if '--fam' in a else R36.FAMS
        R36.dev(ENC, f'b36_out/dev_{ENC}.jsonl', fams, wk)
    elif a[0] == 'choose':
        R36.choose(ENC)
    elif a[0] == 'b32test':
        R32.test(ENC, f'results/b32_choice_{ENC}.json', 'b32_out/test', a[1], 'cpu')
    elif a[0] == 'b32merge':
        R32.merge('b32_out/test', [ENC])
    elif a[0] == 'test':
        R36.test(ENC, f'results/b36_choice_{ENC}.json', 'b36_out/test', wk)
    elif a[0] == 'test32':
        R36.test32(ENC, 'b36_out/test')
    elif a[0] == 'merge':
        R36.merge(ENC, 'b36_out/test')
    else:
        raise SystemExit(f'unknown mode {a[0]}')


if __name__ == '__main__':
    main()
