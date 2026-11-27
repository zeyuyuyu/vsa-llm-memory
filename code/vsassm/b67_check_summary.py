"""B67: summary of the C1 / C2 check files -> results/b67_check.txt."""
import glob
import json
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
out = []
cfg = {}
for e in ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']:
    ch = json.load(open(f'{R}/b53_choice_{e}.json'))['choice']
    fa, fl = f'{R}/b67_check_a_{e}.json', f'{R}/b67_check_L_{e}.json'
    if os.path.exists(fa):
        d = json.load(open(fa))['cells']
        lin = [abs(v['H_diff']) for k, v in d.items() if ch[f'{k}|ideal|VSA']['cfg'][1] == 'lin']
        sg = [abs(v['H_diff']) for k, v in d.items() if ch[f'{k}|ideal|VSA']['cfg'][1] != 'lin']
        out.append(f'{e:8s} H: lin cells {len(lin)} max |diff| {100 * max(lin):.2f} pts; sign cells {len(sg)} max '
                   f'{(100 * max(sg)) if sg else float("nan"):.2f} pts')
    if os.path.exists(fl):
        d = json.load(open(fl))
        c = [v for v in d.values() if v.get('checked')]
        if c:
            out.append(f'{e:8s} L: checked {len(c)} of {len(d)} arm-cells, max |diff| L {100 * max(abs(v["diff"]) for v in c):.3f}'
                       f' pts, L\' {100 * max(abs(v.get("diff2", float("nan"))) for v in c):.3f} pts')
for f in sorted(glob.glob(f'{R}/b67_check_c2_*.json')):
    for e, v in json.load(open(f)).items():
        out.append(f'{e:8s} C2: rows by split {v["rows_by_split"]}, guard raises on test: {v["guard_raises_on_test"]}')
fb = f'{R}/b67_check_b.json'
if os.path.exists(fb):
    for k, v in json.load(open(fb)).items():
        out.append(f'(b) {k}: agree {v["agree"]}')
open(f'{R}/b67_check.txt', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
