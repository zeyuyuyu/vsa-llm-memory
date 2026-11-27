"""B46: mechanical check of the registered predictions P1-P10 (prereg/B46_second_llm_family_prereg.md); P11 (checks)
is filled in from the checker outputs by hand in results/b46_predictions.txt.

python3 b46_predcheck.py      # reads results/b46_encoder_choice.json, results/b46_verdict.json, results/b46_choice_b36.json
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
R = lambda f: json.load(open(os.path.join(ROOT, 'results', f)))


def main():
    enc = R('b46_encoder_choice.json')['chosen']
    J = R('b46_verdict.json')
    J36 = R('b36_verdict.json')
    ch = R('b46_choice_b36.json')['choice']
    reg = lambda name, vs='BEST', acct='ideal': J['regions'][f'b46|{acct}|{name}|{vs}']
    out = []

    def P(k, ok, txt):
        out.append(f'{k}: {"HELD" if ok else "FAILED"} - {txt}')

    pool, layer = enc['variant'].split('_L')
    depth = {'llama31i': 32, 'mistral7i': 32, 'granite33i': 40}[enc['tag']]
    P('P1', enc['tag'] == 'llama31i' and pool == 'mean' and int(layer) == depth // 4 and 0.62 <= enc['score_float32'] <= 0.74,
      f'chosen {enc["tag"]} {enc["variant"]}, criterion {enc["score_float32"]:.4f} (predicted Llama-3.1-8B-Instruct, mean, '
      f'1/4 depth, 0.62-0.74)')
    r1 = reg('R1')
    P('P2', 0.02 <= r1['diff'] <= 0.09 and r1['ci95'][0] > 0,
      f'R1 VSA - BEST {100 * r1["diff"]:+.2f} [{100 * r1["ci95"][0]:+.2f}, {100 * r1["ci95"][1]:+.2f}] (predicted +2 to +9, CI > 0)')
    r = reg('R1', 'IDX')
    P('P3', 0.04 <= r['diff'] <= 0.11, f'R1 VSA - IDX {100 * r["diff"]:+.2f} (predicted +4 to +11)')
    kf = [ch[f'{M}|{V}|{b}|ideal|BEST']['fam'] for M in (256, 1024) for V in (2, 16) for b in (4, 8, 16)]
    n_kf = sum(f != 'IDX' for f in kf)
    P('P4', n_kf >= 4, f'key-free family is the dev BEST in {n_kf} of 12 R1 cells (predicted >= 4); {kf}')
    r = reg('R0')
    P('P5', -0.03 <= r['diff'] <= 0.04, f'R0 VSA - BEST {100 * r["diff"]:+.2f} [{100 * r["ci95"][0]:+.2f}, '
      f'{100 * r["ci95"][1]:+.2f}] (predicted -3 to +4)')
    r = reg('R2')
    n_idx = sum(ch[f'{M}|1024|{b}|ideal|BEST']['fam'] == 'IDX' for M in (256, 1024) for b in (1, 2, 4, 8, 16, 32, 64))
    P('P6', r['diff'] <= -0.02 and r['ci95'][1] < 0 and n_idx >= 10,
      f'R2 VSA - BEST {100 * r["diff"]:+.2f} [{100 * r["ci95"][0]:+.2f}, {100 * r["ci95"][1]:+.2f}]; BEST = index in '
      f'{n_idx} of 14 (predicted <= -2 with CI < 0, index >= 10)')
    r = reg('RH')
    P('P7', -0.03 <= r['diff'] <= 0.07, f'RH VSA - BEST {100 * r["diff"]:+.2f} (predicted -3 to +7)')
    r = reg('R1', 'ORACLE')
    P('P8', r['diff'] >= 0.01 and r['ci95'][0] > 0, f'R1 VSA - ORACLE {100 * r["diff"]:+.2f} [{100 * r["ci95"][0]:+.2f}, '
      f'{100 * r["ci95"][1]:+.2f}] (predicted >= +1, CI > 0)')
    ok9 = []
    for M in (256, 1024):
        for V in (2, 16):
            x = J['rates'][f'b46|ideal|{M}|{V}|0.25']
            ok9.append((M, V, x['ratio_flag'] + f'{x["ratio_BEST_over_VSA"]:.2f}',
                        x['ratio_BEST_over_VSA'] >= 1.5 and x['ratio_flag'] != '<=' and x['ratio_flag'] != '?'))
    P('P9', sum(o[3] for o in ok9) >= 3, f'rate ratio BEST / VSA at lift 0.25: {[o[:3] for o in ok9]} (predicted >= 1.5 in >= 3 of 4)')
    gaps = []
    for M in (256, 1024):
        for V in (2, 16, 128, 1024):
            a = J['cells'][f'b46|ideal|{M}|{V}|4']['EXACT']
            b = J36['cells'][f'qwen8|ideal|{M}|{V}|4']['EXACT']
            gaps.append(round(100 * (a - b), 2))
    P('P10', all(abs(g) <= 6 for g in gaps), f'EXACT(b46) - EXACT(Qwen3-8B) per (M, V): {gaps} (predicted within 6)')
    txt = '\n'.join(out)
    print(txt)
    return txt


if __name__ == '__main__':
    main()
