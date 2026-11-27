"""B60 VSAPc (prereg/B60_corrected_lam_prereg.md, store arms): B57's VSAP procedure re-run with the corrected whiten
principal basis. B57's FeatP read b49_common.LamFeat's basis, which for prep = whiten is the principal basis of the
whitened codes (not principal; b60_run overlap). Here the whiten LamFeat object handed to b57_run is replaced by U = U0
(eigenvectors of the raw codes' centred training covariance, descending; the decomposition Prep('whiten') whitens with)
and mu = 0, so phi(y) = U0[:, :d]^T P_w(y): the whitened code's principal coordinates. Center parts are B57's dev rows
(unchanged code, seeds and basis), copied. Everything else is b57_run's code, called unchanged.

python3 b60_vsapc.py ENC devp            # whiten:plin and whiten:psgn MF grid parts on the dev stores
python3 b60_vsapc.py ENC choosep         # B32's rule over the 48 -> results/b60_choicepc_ENC.json
python3 b60_vsapc.py ENC decp SPLIT M V  # B56's LM / RT reads on the chosen bundle (test: chosen only)
python3 b60_vsapc.py ENC choose          # PLM / PRT / PSTAR (B57's rules) appended to results/b60_choicepc_ENC.json
"""
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ['B57_OUT'] = os.path.join(ROOT, 'b60_out', 'vsapc')
sys.path.insert(0, ROOT)
import b57_run as R57  # noqa: E402

_orig_setup = R57.setup


class CorrectedBasis:
    def __init__(self, Tr):
        Tr = np.asarray(Tr)
        mu = Tr.mean(0)
        w, U = np.linalg.eigh(np.cov((Tr - mu).T))
        self.U = U[:, np.argsort(-w)].astype(np.float32)
        self.mu = np.zeros(Tr.shape[1], np.float32)
        self.n = Tr.shape[1]


_CACHE = {}


def setup(enc):
    if enc not in _CACHE:
        pool, preps, Tr, lfs = _orig_setup(enc)
        lfs = dict(lfs)
        lfs['whiten'] = CorrectedBasis(Tr)
        _CACHE.clear()
        _CACHE[enc] = (pool, preps, Tr, lfs)
    return _CACHE[enc]


R57.setup = setup


def choicefile(enc):
    return f'{ROOT}/results/b60_choicepc_{enc}.json'


def main():
    enc, mode = sys.argv[1], sys.argv[2]
    od = R57.odir(enc)
    if mode == 'devp':
        for p in ('center_plin', 'center_psgn'):
            src = os.path.join(ROOT, 'b57_out', enc, f'devp_{enc}_{p}.jsonl')
            dst = os.path.join(od, f'devp_{enc}_{p}.jsonl')
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)
        for part in sys.argv[3:] or ['whiten:plin', 'whiten:psgn']:
            R57.mf_dev(enc, 0, part, os.path.join(od, f'devp_{enc}_{part.replace(":", "_")}.jsonl'))
    elif mode == 'choosep':
        R57.b32_choose([os.path.join(od, f'devp_{enc}_{p}_{f}.jsonl') for p, f in R57.PARTS_P], R57.PFMAPS,
                       choicefile(enc))
    elif mode == 'decp':
        split, M, V = sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
        ch = json.load(open(choicefile(enc)))['choice']
        chp = {k: v for k, v in ch.items() if k.endswith('|VSA')}
        sel = None
        if split == 'test':
            ch2 = {k.replace('|PLM', '|LM').replace('|PRT', '|RT'): v for k, v in ch.items()
                   if k.split('|')[-1] in ('PLM', 'PRT')}
            sel = R57.select_from(ch2, M, V, ['LM', 'RT'])
        R57.engine(enc, 0, split, M, V, chp, ['LM', 'RT'], sel, os.path.join(od, f'decp_{split}_{enc}_M{M}_V{V}.jsonl.gz'))
    elif mode == 'choose':
        import gzip
        J = json.load(open(choicefile(enc)))
        ch = J['choice']
        for M in R57.M_GRID:
            for V in R57.VN:
                R_ = [json.loads(l) for l in gzip.open(os.path.join(od, f'decp_dev_{enc}_M{M}_V{V}.jsonl.gz'), 'rt')]
                tmp = {}
                R57.dec_choose(R_, ['LM', 'RT'], tmp, M, V)
                for k, v in tmp.items():
                    ch[k.replace('|LM', '|PLM').replace('|RT', '|PRT')] = v
                for bpf in R57.BPF:
                    for acct in R57.ACCT32:
                        lm, rt = tmp[f'{M}|{V}|{bpf}|{acct}|LM'], tmp[f'{M}|{V}|{bpf}|{acct}|RT']
                        ch[f'{M}|{V}|{bpf}|{acct}|PSTAR'] = {'fam': 'PRT' if rt['dev_acc'] > lm['dev_acc'] + 0.001
                                                             else 'PLM', 'dev': {'PLM': lm['dev_acc'], 'PRT': rt['dev_acc']}}
        json.dump(J, open(choicefile(enc), 'w'), indent=1, sort_keys=True)
        print('choices ->', choicefile(enc))
    else:
        raise SystemExit(mode)


if __name__ == '__main__':
    main()
