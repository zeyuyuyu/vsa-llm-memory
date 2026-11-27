"""Recompute the equal-bit point estimates of the manuscript from the archived per-condition test accuracies.

Region R1: 256 and 1,024 facts x 2 and 16 values x 4, 8 and 16 bits per fact (12 conditions), ideal accounting.
Each number is the mean over the region's conditions of (memory A accuracy - memory B accuracy), in points.
Inputs: inputs/b60_part_<enc>.json (B60), b62_part_<enc>.json (B62), b64_part_<enc>.json (B64), field "cells".
Intervals (cluster bootstrap over entities) need the per-query outputs and come from the verdict files.
Usage: python3 equal_bit.py
"""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
ENC = [('Qwen3-8B', 'qwen8'), ('Granite-3.3-8B', 'b46'), ('Llama-3.1-8B', 'llama'), ('Mistral-7B', 'mistral'),
       ('e5-base-v2', 'e5'), ('bge-small', 'bge'), ('MiniLM-L6', 'minilm')]
CONDS = [(M, V, b) for M in ('256', '1024') for V in ('2', '16') for b in ('4', '8', '16')]

def region(cells, key, a, b, Vs=('2', '16')):
    d = [cells[key(M, V, bb)][a] - cells[key(M, V, bb)][b] for (M, V, bb) in CONDS if V in Vs]
    return 100.0 * sum(d) / len(d)

def table(title, part, keyfmt, rows):
    print('\n' + title)
    print('%-15s' % 'encoder' + ''.join('%22s' % r[0] for r in rows))
    for name, e in ENC:
        c = json.load(open(os.path.join(H, 'inputs', '%s_%s.json' % (part, e))))['cells']
        key = lambda M, V, b, e=e: keyfmt % dict(e=e, M=M, V=V, b=b)
        print('%-15s' % name + ''.join('%22.2f' % region(c, key, a, b, Vs) for _, a, b, Vs in rows))

ALL = ('2', '16')
table('B60: store (final design) and variants minus corrected least squares (LAMc)', 'b60_part', '%(e)s|ideal|%(M)s|%(V)s|%(b)s',
      [('STORE-LAMC R1-V2', 'STORE', 'LAMC', ('2',)), ('STORE-LAMC R1-V16', 'STORE', 'LAMC', ('16',)),
       ('STORE-LAMC R1', 'STORE', 'LAMC', ALL), ('VSAPCX-LAMC R1', 'VSAPCX', 'LAMC', ALL), ('STORE-BESTC R1', 'STORE', 'BESTC', ALL)])
table('B62: store minus online least squares (NLMSc, RLSc)', 'b62_part', '%(e)s|%(M)s|%(V)s|%(b)s',
      [('STORE-NLMSC R1-V2', 'STORE', 'NLMSC', ('2',)), ('STORE-NLMSC R1-V16', 'STORE', 'NLMSC', ('16',)),
       ('STORE-NLMSC R1', 'STORE', 'NLMSC', ALL), ('STORE-RLSC R1', 'STORE', 'RLSC', ALL), ('NLMSC-LAMC R1', 'NLMSC', 'LAMC', ALL)])
table('B64: first design (VSAR) and final design (VSA) against the SDM at training codes and the best of seven', 'b64_part',
      '%(e)s|ideal|%(M)s|%(V)s|%(b)s',
      [('VSAR-SDMT R1', 'VSAR', 'SDMT', ALL), ('VSAR-BEST7 R1', 'VSAR', 'BEST7', ALL), ('VSA-BEST7 R1', 'VSA', 'BEST7', ALL),
       ('VSAR-BEST36 R1', 'VSAR', 'BEST36', ALL)])
