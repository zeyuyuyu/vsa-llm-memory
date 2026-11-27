"""B32 dev record: what the dev grid chose (prereg/B32_vector_form_equal_bits_prereg.md, 'Grid, development and
choice'). Dev numbers only.

python3 b32_devsum.py results/b32_choice_qwen8.json results/b32_choice_e5.json
"""
import collections
import json
import sys


def main():
    for fn in sys.argv[1:]:
        d = json.load(open(fn))
        ch = d['choice']
        print(f'## {d["enc"]} ({fn})')
        cnt = collections.Counter()
        for acct in ('ideal', 'raw'):
            print(f'  {acct}:  M     V  bpf   dev VSA  dev IDX   diff  | VSA cfg                    | IDX cfg')
            for M in (256, 1024):
                for V in (2, 16, 128, 1024):
                    ex = ch[f'{M}|{V}|None|None|EXACT']
                    for b in (4, 8, 16, 32, 64):
                        v, x = ch[f'{M}|{V}|{b}|{acct}|VSA'], ch[f'{M}|{V}|{b}|{acct}|IDX']
                        print(f'        {M:5d} {V:5d} {b:4d}   {100 * v["dev_acc"]:6.2f}   {100 * x["dev_acc"]:6.2f} '
                              f'{100 * (v["dev_acc"] - x["dev_acc"]):+6.2f}  | {str(v["cfg"]):26s} | {x["cfg"]}')
                        if acct == 'ideal':
                            cnt['vsa_fmap', v['cfg'][1]] += 1
                            cnt['vsa_q', v['cfg'][2]] += 1
                            cnt['vsa_prep', v['cfg'][0]] += 1
                            cnt['vsa_f', v['cfg'][3]] += 1
                            cnt['idx_family', x['cfg'][1]] += 1
                            cnt['idx_f', x['cfg'][2]] += 1
                            if b <= 16:
                                cnt['vsa_fmap<=16', v['cfg'][1]] += 1
                                cnt['vsa_q<=16', v['cfg'][2]] += 1
                            if b <= 8:
                                cnt['idx_f<1 at <=8', x['cfg'][2] < 1] += 1
                    print(f'        {M:5d} {V:5d}  EXACT dev {100 * ex["dev_acc"]:.2f} ({ex["cfg"][0]})')
        print('  ideal-accounting choice counts:', dict(sorted(cnt.items(), key=str)))


if __name__ == '__main__':
    main()
