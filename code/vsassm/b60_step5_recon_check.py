# Reconstruction check: the step6 selection (L0 / L3 levels) on b60_step5.py's output vs the diagnosis's printed values.
import pickle, numpy as np
D = '/path/to/vsassm/b60_out/step5/'
REF = {(256, 2): ([73.1, 76.7, 79.8], [73.6, 76.4, 79.0]), (256, 16): ([14.9, 19.4, 32.3], [22.6, 31.9, 44.3]),
       (1024, 2): ([72.4, 74.1, 74.6], [71.6, 73.4, 74.5]), (1024, 16): ([15.2, 20.9, 31.7], [17.6, 25.0, 40.6])}
for (M, V), (r0, r3) in REF.items():
    R = pickle.load(open(f'{D}step5_M{M}_V{V}.pkl', 'rb'))
    keys = list(R['dev'][0]['pred'].keys())
    out = {}
    for lev, filt in (('L0', lambda k: k[0] == 'PL'), ('L3', lambda k: k[0] == 'FX')):
        accs = []
        for b in (4, 8, 16):
            cand = [k for k in keys if filt(k)]
            dv = {k: np.mean([(R['dev'][i]['pred'][k][b] == R['dev'][i]['qgold']).mean() for i in R['dev']]) for k in cand}
            top = max(dv.values()); k = [k for k in cand if dv[k] >= top - 0.001][0]
            t = np.concatenate([R['test'][i]['pred'][k][b] == R['test'][i]['qgold'] for i in sorted(R['test'])]).mean()
            accs.append(round(100 * t, 1))
        out[lev] = accs
    print(M, V, 'L0', out['L0'], 'ref', r0, '| L3', out['L3'], 'ref', r3)
