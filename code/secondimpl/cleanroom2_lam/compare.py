"""Compare my_cells.json (clean room) with the pipeline's b60_part_*.json, per condition. Run only after my_cells.json exists."""
import json, sys, os
mine = json.load(open('my_cells.json'))
PIPE = {'qwen8': '/path/to/vsassm/results/b60_part_qwen8.json', 'e5': '/path/to/vsassm/results/b60_part_e5.json'}
rep = {}
for enc, path in PIPE.items():
    if enc not in mine or not os.path.exists(path):
        continue
    pipe = json.load(open(path))
    pc = pipe['cells']
    print(enc, 'pipeline cell keys sample:', list(pc)[:3])
    print(' sample cell:', json.dumps(pc[list(pc)[0]])[:400])
    rows = []
    for ck, x in mine[enc]['cells'].items():
        _, _, M, V, b = ck.split('|')
        cand = [k for k in pc if k.split('|')[1] == 'ideal' and k.split('|')[2:] == [M, V, b]]
        if not cand:
            print(' missing', ck); continue
        p = pc[cand[0]]
        ps, pl = p['STORE'], p['LAMC']
        if ps <= 1.0 and pl <= 1.0:
            ps, pl = 100 * ps, 100 * pl
        rows.append(dict(M=int(M), V=int(V), b=int(b), my_store=x['STORE'], my_ls=x['LAMC'], pipe_store=ps,
                         pipe_ls=pl, my_cfg_store=x['store_cfg'], my_cfg_ls=x['lamc_cfg']))
    rows.sort(key=lambda r: (r['V'], r['M'], r['b']))
    print(f"{'V':>3} {'M':>5} {'b':>3} | {'store me':>8} {'pipe':>6} {'d':>6} | {'LS me':>6} {'pipe':>6} {'d':>6} | "
          f"{'S-LS me':>7} {'pipe':>6} {'d':>6}")
    for r in rows:
        dm, dp = r['my_store'] - r['my_ls'], r['pipe_store'] - r['pipe_ls']
        print(f"{r['V']:>3} {r['M']:>5} {r['b']:>3} | {r['my_store']:8.2f} {r['pipe_store']:6.2f} "
              f"{r['my_store']-r['pipe_store']:+6.2f} | {r['my_ls']:6.2f} {r['pipe_ls']:6.2f} "
              f"{r['my_ls']-r['pipe_ls']:+6.2f} | {dm:+7.2f} {dp:+6.2f} {dm-dp:+6.2f}")
    reg = {}
    for name, f in [('V2', lambda r: r['V'] == 2), ('V16', lambda r: r['V'] == 16), ('R1', lambda r: True)]:
        rr = [r for r in rows if f(r)]
        m = sum(r['my_store'] - r['my_ls'] for r in rr) / len(rr)
        p = sum(r['pipe_store'] - r['pipe_ls'] for r in rr) / len(rr)
        ms = sum(r['my_store'] for r in rr) / len(rr); psx = sum(r['pipe_store'] for r in rr) / len(rr)
        ml = sum(r['my_ls'] for r in rr) / len(rr); pl = sum(r['pipe_ls'] for r in rr) / len(rr)
        reg[name] = dict(mine=m, pipe=p, diff=m - p, my_store=ms, pipe_store=psx, my_ls=ml, pipe_ls=pl)
        print(f"  region {name:4s}: store-LS mine {m:+6.2f} pipe {p:+6.2f} diff {m-p:+5.2f} | store mine {ms:6.2f} "
              f"pipe {psx:6.2f} | LS mine {ml:6.2f} pipe {pl:6.2f}")
    rep[enc] = dict(rows=rows, regions=reg)
json.dump(rep, open('comparison.json', 'w'), indent=1)
