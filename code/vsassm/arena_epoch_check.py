"""Arena epoch check: what does each arm score WITHOUT choosing its epoch on the test set?

A referee objected that the controlled arena experiments (paper2_vsallm_rewrite, Fig. 5a / tab:threearena) report
"the best epoch of each arm on the test set". Every arena script evaluates the test set after each epoch and banks
max-over-epochs; none has a validation split. The per-epoch test accuracies survive in the runs' stdout logs, which
this script parses. Analysis only: no training, no GPU.

Run artifacts (read-only), on the A800 box (ssh alias cavin):
  root B = /path/to/work/sync_h20/rewardbench_experiment/zeyu/hybrid_neurosymbolic  (V-1, V-4, V-8, V-9, V-11/V-18)
  root D = /path/to/work/ngcls                                                      (V-9 armA, V-13, V-14, V-15, V-15b, V-57, V-165)
The banked JSONs the figure reads (paper repo figdata/) are byte-identical copies of the box files (md5 checked 2026-11-17).

Usage:
  python arena_epoch_check.py --logs-b DIR_B --logs-d DIR_D      # parse local mirrors of the two roots
  python arena_epoch_check.py                                    # use the per-epoch snapshot embedded below
Either way the script asserts, seed by seed, that max over the parsed epochs equals the banked value in figdata/
(so the logs are the runs behind the printed numbers), then writes results/arena_epoch_check.txt.
"""
import argparse, json, os, re, sys
from statistics import mean

FIGDATA = "/path/to/paper/figdata"
OUT = "/path/to/vsassm/results/arena_epoch_check.txt"
SEEDS = [42, 123, 456]

# ----------------------------------------------------------------------------------------------- parsers
TS = re.compile(r"\] ep (\d+): loss=([0-9.]+) eval_acc=(\d\.\d{4})")        # run_systematic_chain.train_one


def _read(p):
    return open(p, encoding="utf-8", errors="replace").read().replace("\r", "\n")


def parse_train_one(path, summary_re):
    """V-1 / V-4 logs: '[ts] ep k: loss=L eval_acc=A' per epoch (eval = test split), then a per-seed summary line.
    lora.log carries duplicate, truncated lines from an orphan second process (driver.log, 13:03); complete
    duplicates of the same epoch are identical and are dropped, truncated ones do not match the regex."""
    out, cur, seen = [], [], set()
    for line in _read(path).split("\n"):
        m = TS.search(line)
        if m:
            k = (int(m.group(1)), m.group(2), m.group(3))
            if k in seen:
                continue
            seen.add(k)
            cur.append({"test": float(m.group(3)), "train_loss": float(m.group(2))})
        s = re.search(summary_re, line)
        if s:
            out.append({"seed": int(s.group(1)), "acc": float(s.group(2)), "epochs": cur})
        if re.search(r"sd\d+\s+acc=\d", line):              # any per-seed summary closes the current block (v4.log sweeps)
            cur, seen = [], set()
    return out


def parse_simple(path, ep_re, summary_re):
    """V-8 / V-9 / V-11 / V-13 / V-15 logs: one 'epN test|eval=A' line per epoch, then a per-seed summary."""
    out, cur = [], []
    for line in _read(path).split("\n"):
        m = re.search(ep_re, line)
        if m:
            cur.append({"test": float(m.group(1))})
            continue
        s = re.search(summary_re, line)
        if s:
            out.append({"seed": int(s.group(1)), "acc": float(s.group(2)), "epochs": cur})
            cur = []
    return out


def parse_grouped(path, arm, block_re, epochs=2):
    """V-14 / V-57 logs: v13.run() epoch lines without per-seed summaries; seeds run in order 42, 123, 456, and a
    block line ('k=2   floor [...]' or 'depth 1 floor [...]') closes each arm x setting."""
    out, cur, key = {}, [], None
    for line in _read(path).split("\n"):
        m = re.search(r"^\s+%s\s+ep(\d+) test=(\d\.\d{4})" % arm, line)
        if m:
            cur.append(float(m.group(2)))
            continue
        b = re.search(block_re % arm, line)
        if b:
            assert len(cur) == epochs * 3, (path, arm, b.group(1), cur)
            out[b.group(1)] = [{"seed": s, "epochs": [{"test": x} for x in cur[i * epochs:(i + 1) * epochs]],
                                "acc": max(cur[i * epochs:(i + 1) * epochs])} for i, s in enumerate(SEEDS)]
            cur = []
    return out


def parse_all(B, D):
    j = os.path.join
    v13 = lambda arm: parse_simple(j(D, "v13_logs/v13_fixed.log"), r"^\s+%s\s+ep\d+ test=(\d\.\d{4})" % arm,
                                   r"^\s{2}%s\s+sd(\d+)\s+acc=(\d\.\d{4})" % arm)
    v18 = lambda f, arm: parse_simple(j(B, "v11_logs", f), r"^\s+%s\s+ep\d+ eval=(\d\.\d{4})" % arm,
                                      r"^\s{2}%s\s+sd(\d+)\s+acc=(\d\.\d{4})" % arm)
    v4 = lambda mode, eps: parse_train_one(j(B, "v4_logs/v4.log"), r"^\s+%s\s+eps=%s sd(\d+)\s+acc=(\d\.\d{4})" % (mode, eps))
    lora15 = lambda f, ar: parse_simple(j(D, f), r"lora/%s ep\d+ test=(\d\.\d{4})" % ar,
                                        r"lora arena=%s sd(\d+)\s+acc=(\d\.\d{4})" % ar)
    R = {
        ("parser", "A"): v4("perocc", r"0\.00"),
        ("parser", "B"): v18("v18_spanfix.log", "parser"),
        ("parser", "C"): v13("parser"),
        ("probe", "A"): parse_train_one(j(B, "v1_logs/frozen_llm.log"), r"frozen_llm\s+sd(\d+) acc=(\d\.\d{4})"),
        ("probe", "B"): v18("v18_spanfix.log", "floor"),
        ("probe", "C"): v13("floor"),
        ("loss", "A"): parse_simple(j(B, "v8_logs/v8.log"), r"^\s+ep\d+ eval=(\d\.\d{4})", r"vsa_spans sd(\d+)\s+acc=(\d\.\d{4})"),
        ("loss", "B"): v18("v18b_spans.log", "spans"),
        ("loss", "C"): v13("spans"),
        ("lora", "A"): parse_train_one(j(B, "v1_logs/lora.log"), r"^\s+lora\s+sd(\d+) acc=(\d\.\d{4})"),
        ("lora", "B"): lora15("v15_logs/v15.log", "B"),
        ("lora", "C"): lora15("v15_logs/v15.log", "C"),
        ("lora10", "C"): lora15("v15b_logs/v15b.log", "C"),
        ("naming", "A"): parse_simple(j(B, "v9_logs/v9_gpu1_rerun_partial.log"), r"extract_then_bind ep\d+ eval=(\d\.\d{4})",
                                      r"extract_then_bind sd(\d+)\s+acc=(\d\.\d{4})"),
        ("naming", "B"): v18("v18_spanfix.log", "probe_argmax"),
        ("naming", "C"): v13("probe_argmax"),
        ("relabel", "A"): v4("consistent", r"1\.00"),
        ("relabel", "B"): v18("v18b_spans.log", "consistent"),
        ("relabel", "C"): v13("consistent"),
        ("nosnap", "B"): v18("v18_spanfix.log", "frozen_cont"),
        ("nosnap", "C"): v13("frozen_cont"),
    }
    v9 = json.load(open(j(D, "runs/v9/v9.json")))                     # V-9 armA keeps its per-epoch list itself
    R[("nosnap", "A")] = [{"seed": r["seed"], "acc": r["armA"]["best"], "epochs": [{"test": x} for x in r["armA"]["epochs"]]}
                          for r in v9]
    v165 = json.load(open(j(D, "v165_logs/v165.json")))["seeds"]      # rerun of loss/C; keeps last-batch training loss
    R[("loss165", "C")] = [{"seed": s["seed"], "acc": s["acc_best"],
                            "epochs": [{"test": e["acc"], "train_loss": e["loss_last"]} for e in s["epochs"]]} for s in v165]
    sweeps = {}
    for arm in ("floor", "parser", "probe_argmax"):
        sweeps[("v14", arm)] = parse_grouped(j(D, "v14_logs/v14_fixed.log"), arm, r"^\s{2}k=(\d+)\s+%s\s+\[")
        sweeps[("v57", arm)] = parse_grouped(j(D, "v57_logs/v57.log"), arm, r"^\s{2}depth (\d) %s\s+\[")
    return R, sweeps


# ----------------------------------------------------------------------------------------------- banked values
def banked():
    J = lambda p: json.load(open(os.path.join(FIGDATA, p)))
    a = lambda L, **eq: {r["seed"]: r["acc"] for r in L if all(r.get(k) == v for k, v in eq.items())}
    b18, c13 = J("v11_logs/v18_spanfix.json"), J("v13_logs/v13_fixed.json")
    v15 = J("v15_logs/v15.json")
    return {
        ("parser", "A"): a(J("v4_logs/v4_perocc.json"), epsilon=0.0), ("parser", "B"): a(b18, arm="parser"), ("parser", "C"): a(c13, arm="parser"),
        ("probe", "A"): a(J("v1_logs/frozen_llm.json")), ("probe", "B"): a(b18, arm="floor"), ("probe", "C"): a(c13, arm="floor"),
        ("loss", "A"): a(J("v8_logs/v8.json")), ("loss", "B"): a(b18, arm="spans"), ("loss", "C"): a(c13, arm="spans"),
        ("lora", "A"): a(J("v1_logs/lora.json")), ("lora", "B"): a(v15, arena="B"), ("lora", "C"): a(v15, arena="C"),
        ("lora10", "C"): a(J("v15b_logs/v15b.json"), arena="C"),
        ("naming", "A"): a(J("v9_logs/v9_gpu1_rerun_partial.json"), arm="extract_then_bind"),
        ("naming", "B"): a(b18, arm="probe_argmax"), ("naming", "C"): a(c13, arm="probe_argmax"),
        ("relabel", "A"): a(J("v4_logs/v4_consistent.json"), epsilon=1.0), ("relabel", "B"): a(b18, arm="consistent"),
        ("relabel", "C"): a(c13, arm="consistent"),
        ("nosnap", "A"): {r["seed"]: r["armA"]["best"] for r in J("runs/v9/v9.json")},
        ("nosnap", "B"): a(b18, arm="frozen_cont"), ("nosnap", "C"): a(c13, arm="frozen_cont"),
        ("loss165", "C"): a(c13, arm="spans"),
    }


SNAPSHOT = (  # per-epoch test accuracies parsed from the box logs on 2026-11-17 (--dump output, compact)
    '{"runs":{"lora10|C":[{"acc":0.5375,"epochs":[{"test":0.4725},{"test":0.515},{"test":0.465},{"test":0.525},{"te'
    'st":0.485},{"test":0.4925},{"test":0.4975},{"test":0.5375},{"test":0.515},{"test":0.4925}],"seed":42},{"acc":0'
    '.52,"epochs":[{"test":0.515},{"test":0.485},{"test":0.485},{"test":0.485},{"test":0.515},{"test":0.485},{"test'
    '":0.52},{"test":0.4875},{"test":0.4975},{"test":0.5025}],"seed":123},{"acc":0.515,"epochs":[{"test":0.485},{"t'
    'est":0.515},{"test":0.485},{"test":0.515},{"test":0.485},{"test":0.485},{"test":0.4875},{"test":0.5025},{"test'
    '":0.5125},{"test":0.48}],"seed":456}],"lora|A":[{"acc":0.493,"epochs":[{"test":0.492,"train_loss":0.706},{"tes'
    't":0.493,"train_loss":0.6993}],"seed":42},{"acc":0.508,"epochs":[{"test":0.508,"train_loss":0.7092},{"test":0.'
    '508,"train_loss":0.7022}],"seed":123},{"acc":0.515,"epochs":[{"test":0.5,"train_loss":0.7051},{"test":0.515,"t'
    'rain_loss":0.7037}],"seed":456}],"lora|B":[{"acc":0.98,"epochs":[{"test":0.9125},{"test":0.98}],"seed":42},{"a'
    'cc":0.895,"epochs":[{"test":0.51},{"test":0.895}],"seed":123},{"acc":0.51,"epochs":[{"test":0.51},{"test":0.49'
    '}],"seed":456}],"lora|C":[{"acc":0.515,"epochs":[{"test":0.485},{"test":0.515}],"seed":42},{"acc":0.5175,"epoc'
    'hs":[{"test":0.5175},{"test":0.485}],"seed":123},{"acc":0.515,"epochs":[{"test":0.4725},{"test":0.515}],"seed"'
    ':456}],"loss165|C":[{"acc":0.472,"epochs":[{"test":0.464,"train_loss":0.8566094040870667},{"test":0.472,"train'
    '_loss":0.7392123937606812}],"seed":42},{"acc":0.502,"epochs":[{"test":0.502,"train_loss":0.8793203830718994},{'
    '"test":0.486,"train_loss":0.6423845887184143}],"seed":123},{"acc":0.476,"epochs":[{"test":0.46,"train_loss":0.'
    '783550500869751},{"test":0.476,"train_loss":0.5744794607162476}],"seed":456}],"loss|A":[{"acc":0.511,"epochs":'
    '[{"test":0.504},{"test":0.511}],"seed":42},{"acc":0.499,"epochs":[{"test":0.498},{"test":0.499}],"seed":123},{'
    '"acc":0.493,"epochs":[{"test":0.493},{"test":0.486}],"seed":456}],"loss|B":[{"acc":0.502,"epochs":[{"test":0.5'
    '02},{"test":0.492}],"seed":42},{"acc":0.498,"epochs":[{"test":0.488},{"test":0.498}],"seed":123},{"acc":0.522,'
    '"epochs":[{"test":0.518},{"test":0.522}],"seed":456}],"loss|C":[{"acc":0.472,"epochs":[{"test":0.464},{"test":'
    '0.472}],"seed":42},{"acc":0.502,"epochs":[{"test":0.502},{"test":0.486}],"seed":123},{"acc":0.476,"epochs":[{"'
    'test":0.46},{"test":0.476}],"seed":456}],"naming|A":[{"acc":0.915,"epochs":[{"test":0.871},{"test":0.915}],"se'
    'ed":42},{"acc":0.913,"epochs":[{"test":0.885},{"test":0.913}],"seed":123},{"acc":0.913,"epochs":[{"test":0.89}'
    ',{"test":0.913}],"seed":456}],"naming|B":[{"acc":0.99,"epochs":[{"test":0.99},{"test":0.99}],"seed":42},{"acc"'
    ':1.0,"epochs":[{"test":1.0},{"test":1.0}],"seed":123},{"acc":0.998,"epochs":[{"test":0.998},{"test":0.998}],"s'
    'eed":456}],"naming|C":[{"acc":0.932,"epochs":[{"test":0.932},{"test":0.932}],"seed":42},{"acc":0.932,"epochs":'
    '[{"test":0.932},{"test":0.932}],"seed":123},{"acc":0.938,"epochs":[{"test":0.938},{"test":0.938}],"seed":456}]'
    ',"nosnap|A":[{"acc":0.671,"epochs":[{"test":0.651},{"test":0.671}],"seed":42},{"acc":0.672,"epochs":[{"test":0'
    '.654},{"test":0.672}],"seed":123},{"acc":0.677,"epochs":[{"test":0.657},{"test":0.677}],"seed":456}],"nosnap|B'
    '":[{"acc":0.978,"epochs":[{"test":0.978},{"test":0.978}],"seed":42},{"acc":0.99,"epochs":[{"test":0.99},{"test'
    '":0.99}],"seed":123},{"acc":0.988,"epochs":[{"test":0.988},{"test":0.988}],"seed":456}],"nosnap|C":[{"acc":0.9'
    '22,"epochs":[{"test":0.922},{"test":0.922}],"seed":42},{"acc":0.944,"epochs":[{"test":0.944},{"test":0.944}],"'
    'seed":123},{"acc":0.906,"epochs":[{"test":0.906},{"test":0.906}],"seed":456}],"parser|A":[{"acc":0.918,"epochs'
    '":[{"test":0.917,"train_loss":0.6907},{"test":0.918,"train_loss":0.6757}],"seed":42},{"acc":0.919,"epochs":[{"'
    'test":0.919,"train_loss":0.7039},{"test":0.919,"train_loss":0.6899}],"seed":123},{"acc":0.918,"epochs":[{"test'
    '":0.917,"train_loss":0.6843},{"test":0.918,"train_loss":0.6694}],"seed":456}],"parser|B":[{"acc":0.99,"epochs"'
    ':[{"test":0.99},{"test":0.99}],"seed":42},{"acc":1.0,"epochs":[{"test":1.0},{"test":1.0}],"seed":123},{"acc":0'
    '.998,"epochs":[{"test":0.998},{"test":0.998}],"seed":456}],"parser|C":[{"acc":0.932,"epochs":[{"test":0.932},{'
    '"test":0.932}],"seed":42},{"acc":0.932,"epochs":[{"test":0.932},{"test":0.932}],"seed":123},{"acc":0.938,"epoc'
    'hs":[{"test":0.938},{"test":0.938}],"seed":456}],"probe|A":[{"acc":0.508,"epochs":[{"test":0.493,"train_loss":'
    '0.7086},{"test":0.508,"train_loss":0.705}],"seed":42},{"acc":0.511,"epochs":[{"test":0.511,"train_loss":0.7018'
    '},{"test":0.501,"train_loss":0.7033}],"seed":123},{"acc":0.508,"epochs":[{"test":0.502,"train_loss":0.7041},{"'
    'test":0.508,"train_loss":0.7012}],"seed":456}],"probe|B":[{"acc":0.794,"epochs":[{"test":0.74},{"test":0.794}]'
    ',"seed":42},{"acc":0.818,"epochs":[{"test":0.718},{"test":0.818}],"seed":123},{"acc":0.832,"epochs":[{"test":0'
    '.692},{"test":0.832}],"seed":456}],"probe|C":[{"acc":0.512,"epochs":[{"test":0.512},{"test":0.512}],"seed":42}'
    ',{"acc":0.53,"epochs":[{"test":0.518},{"test":0.53}],"seed":123},{"acc":0.504,"epochs":[{"test":0.486},{"test"'
    ':0.504}],"seed":456}],"relabel|A":[{"acc":0.867,"epochs":[{"test":0.867,"train_loss":1.1605},{"test":0.862,"tr'
    'ain_loss":1.1059}],"seed":42},{"acc":0.871,"epochs":[{"test":0.871,"train_loss":1.1544},{"test":0.866,"train_l'
    'oss":1.1183}],"seed":123},{"acc":0.869,"epochs":[{"test":0.869,"train_loss":1.057},{"test":0.849,"train_loss":'
    '1.0904}],"seed":456}],"relabel|B":[{"acc":0.996,"epochs":[{"test":0.996},{"test":0.996}],"seed":42},{"acc":0.9'
    '96,"epochs":[{"test":0.996},{"test":0.996}],"seed":123},{"acc":0.99,"epochs":[{"test":0.99},{"test":0.99}],"se'
    'ed":456}],"relabel|C":[{"acc":0.94,"epochs":[{"test":0.94},{"test":0.94}],"seed":42},{"acc":0.914,"epochs":[{"'
    'test":0.914},{"test":0.914}],"seed":123},{"acc":0.934,"epochs":[{"test":0.934},{"test":0.934}],"seed":456}]},"'
    'sweeps":{"v14|floor":{"16":[{"acc":0.51,"epochs":[{"test":0.504},{"test":0.51}],"seed":42},{"acc":0.494,"epoch'
    's":[{"test":0.494},{"test":0.48}],"seed":123},{"acc":0.53,"epochs":[{"test":0.53},{"test":0.494}],"seed":456}]'
    ',"2":[{"acc":0.806,"epochs":[{"test":0.754},{"test":0.806}],"seed":42},{"acc":0.808,"epochs":[{"test":0.808},{'
    '"test":0.782}],"seed":123},{"acc":0.772,"epochs":[{"test":0.77},{"test":0.772}],"seed":456}],"24":[{"acc":0.53'
    '4,"epochs":[{"test":0.508},{"test":0.534}],"seed":42},{"acc":0.546,"epochs":[{"test":0.518},{"test":0.546}],"s'
    'eed":123},{"acc":0.522,"epochs":[{"test":0.502},{"test":0.522}],"seed":456}],"32":[{"acc":0.512,"epochs":[{"te'
    'st":0.512},{"test":0.512}],"seed":42},{"acc":0.53,"epochs":[{"test":0.518},{"test":0.53}],"seed":123},{"acc":0'
    '.504,"epochs":[{"test":0.486},{"test":0.504}],"seed":456}],"4":[{"acc":0.576,"epochs":[{"test":0.552},{"test":'
    '0.576}],"seed":42},{"acc":0.576,"epochs":[{"test":0.53},{"test":0.576}],"seed":123},{"acc":0.56,"epochs":[{"te'
    'st":0.56},{"test":0.508}],"seed":456}],"8":[{"acc":0.524,"epochs":[{"test":0.514},{"test":0.524}],"seed":42},{'
    '"acc":0.542,"epochs":[{"test":0.542},{"test":0.516}],"seed":123},{"acc":0.534,"epochs":[{"test":0.516},{"test"'
    ':0.534}],"seed":456}]},"v14|parser":{"16":[{"acc":0.97,"epochs":[{"test":0.97},{"test":0.97}],"seed":42},{"acc'
    '":0.972,"epochs":[{"test":0.972},{"test":0.972}],"seed":123},{"acc":0.974,"epochs":[{"test":0.974},{"test":0.9'
    '74}],"seed":456}],"2":[{"acc":1.0,"epochs":[{"test":1.0},{"test":1.0}],"seed":42},{"acc":1.0,"epochs":[{"test"'
    ':1.0},{"test":1.0}],"seed":123},{"acc":1.0,"epochs":[{"test":1.0},{"test":1.0}],"seed":456}],"24":[{"acc":0.95'
    ',"epochs":[{"test":0.95},{"test":0.95}],"seed":42},{"acc":0.954,"epochs":[{"test":0.954},{"test":0.954}],"seed'
    '":123},{"acc":0.948,"epochs":[{"test":0.948},{"test":0.948}],"seed":456}],"32":[{"acc":0.932,"epochs":[{"test"'
    ':0.932},{"test":0.932}],"seed":42},{"acc":0.932,"epochs":[{"test":0.932},{"test":0.932}],"seed":123},{"acc":0.'
    '938,"epochs":[{"test":0.938},{"test":0.938}],"seed":456}],"4":[{"acc":0.998,"epochs":[{"test":0.998},{"test":0'
    '.998}],"seed":42},{"acc":0.998,"epochs":[{"test":0.998},{"test":0.998}],"seed":123},{"acc":0.998,"epochs":[{"t'
    'est":0.998},{"test":0.998}],"seed":456}],"8":[{"acc":0.994,"epochs":[{"test":0.994},{"test":0.994}],"seed":42}'
    ',{"acc":0.996,"epochs":[{"test":0.996},{"test":0.996}],"seed":123},{"acc":1.0,"epochs":[{"test":1.0},{"test":1'
    '.0}],"seed":456}]},"v14|probe_argmax":{"16":[{"acc":0.97,"epochs":[{"test":0.97},{"test":0.97}],"seed":42},{"a'
    'cc":0.972,"epochs":[{"test":0.972},{"test":0.972}],"seed":123},{"acc":0.974,"epochs":[{"test":0.974},{"test":0'
    '.974}],"seed":456}],"2":[{"acc":0.98,"epochs":[{"test":0.896},{"test":0.98}],"seed":42},{"acc":0.988,"epochs":'
    '[{"test":0.902},{"test":0.988}],"seed":123},{"acc":0.998,"epochs":[{"test":0.92},{"test":0.998}],"seed":456}],'
    '"24":[{"acc":0.95,"epochs":[{"test":0.95},{"test":0.95}],"seed":42},{"acc":0.954,"epochs":[{"test":0.954},{"te'
    'st":0.954}],"seed":123},{"acc":0.948,"epochs":[{"test":0.948},{"test":0.948}],"seed":456}],"32":[{"acc":0.932,'
    '"epochs":[{"test":0.932},{"test":0.932}],"seed":42},{"acc":0.932,"epochs":[{"test":0.932},{"test":0.932}],"see'
    'd":123},{"acc":0.938,"epochs":[{"test":0.938},{"test":0.938}],"seed":456}],"4":[{"acc":0.996,"epochs":[{"test"'
    ':0.974},{"test":0.996}],"seed":42},{"acc":0.994,"epochs":[{"test":0.95},{"test":0.994}],"seed":123},{"acc":0.9'
    '94,"epochs":[{"test":0.97},{"test":0.994}],"seed":456}],"8":[{"acc":0.994,"epochs":[{"test":0.994},{"test":0.9'
    '94}],"seed":42},{"acc":0.996,"epochs":[{"test":0.992},{"test":0.996}],"seed":123},{"acc":1.0,"epochs":[{"test"'
    ':1.0},{"test":1.0}],"seed":456}]},"v57|floor":{"1":[{"acc":0.54,"epochs":[{"test":0.496},{"test":0.54}],"seed"'
    ':42},{"acc":0.528,"epochs":[{"test":0.528},{"test":0.524}],"seed":123},{"acc":0.554,"epochs":[{"test":0.542},{'
    '"test":0.554}],"seed":456}],"2":[{"acc":0.512,"epochs":[{"test":0.512},{"test":0.512}],"seed":42},{"acc":0.53,'
    '"epochs":[{"test":0.518},{"test":0.53}],"seed":123},{"acc":0.504,"epochs":[{"test":0.486},{"test":0.504}],"see'
    'd":456}],"3":[{"acc":0.528,"epochs":[{"test":0.518},{"test":0.528}],"seed":42},{"acc":0.518,"epochs":[{"test":'
    '0.518},{"test":0.508}],"seed":123},{"acc":0.544,"epochs":[{"test":0.544},{"test":0.484}],"seed":456}]},"v57|pa'
    'rser":{"1":[{"acc":0.954,"epochs":[{"test":0.954},{"test":0.954}],"seed":42},{"acc":0.96,"epochs":[{"test":0.9'
    '6},{"test":0.96}],"seed":123},{"acc":0.96,"epochs":[{"test":0.96},{"test":0.96}],"seed":456}],"2":[{"acc":0.93'
    '2,"epochs":[{"test":0.932},{"test":0.932}],"seed":42},{"acc":0.932,"epochs":[{"test":0.932},{"test":0.932}],"s'
    'eed":123},{"acc":0.938,"epochs":[{"test":0.938},{"test":0.938}],"seed":456}],"3":[{"acc":0.884,"epochs":[{"tes'
    't":0.884},{"test":0.884}],"seed":42},{"acc":0.88,"epochs":[{"test":0.88},{"test":0.88}],"seed":123},{"acc":0.8'
    '76,"epochs":[{"test":0.876},{"test":0.876}],"seed":456}]},"v57|probe_argmax":{"1":[{"acc":0.954,"epochs":[{"te'
    'st":0.954},{"test":0.954}],"seed":42},{"acc":0.96,"epochs":[{"test":0.96},{"test":0.96}],"seed":123},{"acc":0.'
    '96,"epochs":[{"test":0.96},{"test":0.96}],"seed":456}],"2":[{"acc":0.932,"epochs":[{"test":0.932},{"test":0.93'
    '2}],"seed":42},{"acc":0.932,"epochs":[{"test":0.932},{"test":0.932}],"seed":123},{"acc":0.938,"epochs":[{"test'
    '":0.938},{"test":0.938}],"seed":456}],"3":[{"acc":0.884,"epochs":[{"test":0.884},{"test":0.884}],"seed":42},{"'
    'acc":0.88,"epochs":[{"test":0.88},{"test":0.88}],"seed":123},{"acc":0.876,"epochs":[{"test":0.876},{"test":0.8'
    '76}],"seed":456}]}}}'
)


# ----------------------------------------------------------------------------------------------- analysis
NAMES = {"parser": "exact codes (parser)", "probe": "probe (floor)", "loss": "map trained on the answer (binding loss)",
         "lora": "LoRA fine-tune (rank 16)", "naming": "naming head", "relabel": "exact codes, relabelled",
         "nosnap": "naming head read without snapping", "lora10": "LoRA on C, 10-epoch extension (V-15b)",
         "loss165": "binding-loss map on C, V-165 rerun"}
TRAINED = {"parser": "no (2 readout scalars; arena-A readout learned, B/C threshold fitted on train)", "probe": "yes (linear probe)",
           "loss": "yes (5.2M projection)", "lora": "yes (10.5M LoRA)", "naming": "yes (identity head)",
           "relabel": "no (as parser)", "nosnap": "projection fitted to codebook, then frozen; 2 readout scalars"}
PRINTED = {("parser", "A"): .918, ("parser", "B"): .996, ("parser", "C"): .934, ("probe", "A"): .509, ("probe", "B"): .815,
           ("probe", "C"): .515, ("loss", "A"): .501, ("loss", "B"): .507, ("loss", "C"): .483, ("lora", "A"): .505,
           ("lora", "B"): .795, ("lora", "C"): .516, ("naming", "A"): .914, ("naming", "B"): .996, ("naming", "C"): .934,
           ("relabel", "A"): .869, ("relabel", "B"): .994, ("relabel", "C"): .929, ("nosnap", "A"): .673,
           ("nosnap", "B"): .985, ("nosnap", "C"): .924, ("lora10", "C"): .524, ("loss165", "C"): .483}
SOURCE = {("parser", "A"): "B/v4_logs/v4.log (perocc eps=0.00)", ("parser", "B"): "B/v11_logs/v18_spanfix.log",
          ("parser", "C"): "D/v13_logs/v13_fixed.log", ("probe", "A"): "B/v1_logs/frozen_llm.log",
          ("probe", "B"): "B/v11_logs/v18_spanfix.log (floor)", ("probe", "C"): "D/v13_logs/v13_fixed.log (floor)",
          ("loss", "A"): "B/v8_logs/v8.log", ("loss", "B"): "B/v11_logs/v18b_spans.log (spans)",
          ("loss", "C"): "D/v13_logs/v13_fixed.log (spans)", ("lora", "A"): "B/v1_logs/lora.log",
          ("lora", "B"): "D/v15_logs/v15.log", ("lora", "C"): "D/v15_logs/v15.log", ("lora10", "C"): "D/v15b_logs/v15b.log",
          ("naming", "A"): "B/v9_logs/v9_gpu1_rerun_partial.log (extract_then_bind)",
          ("naming", "B"): "B/v11_logs/v18_spanfix.log (probe_argmax)", ("naming", "C"): "D/v13_logs/v13_fixed.log (probe_argmax)",
          ("relabel", "A"): "B/v4_logs/v4.log (consistent eps=1.00)", ("relabel", "B"): "B/v11_logs/v18b_spans.log (consistent)",
          ("relabel", "C"): "D/v13_logs/v13_fixed.log (consistent)", ("nosnap", "A"): "D/runs/v9/v9.json armA.epochs",
          ("nosnap", "B"): "B/v11_logs/v18_spanfix.log (frozen_cont)", ("nosnap", "C"): "D/v13_logs/v13_fixed.log (frozen_cont)",
          ("loss165", "C"): "D/v165_logs/v165.json seeds[].epochs (acc, loss_last)"}


def summarize(runs):
    by = {r["seed"]: r for r in runs}
    assert sorted(by) == SEEDS, sorted(by)
    ep = [[e["test"] for e in by[s]["epochs"]] for s in SEEDS]
    best = [max(x) for x in ep]
    final = [x[-1] for x in ep]
    worst = [min(x) for x in ep]
    tl = None
    if all("train_loss" in e for s in SEEDS for e in by[s]["epochs"]):          # epoch with the lowest training loss
        tl = []
        for s in SEEDS:
            es = by[s]["epochs"]
            tl.append(min(es, key=lambda e: e["train_loss"])["test"])
    return {"epochs": ep, "best": best, "final": final, "worst": worst, "tl": tl,
            "n_ep": len(ep[0]), "invariant": all(len(set(x)) == 1 for x in ep)}


def check_banked(R):
    bk = banked()
    for key, runs in R.items():
        for r in runs:
            b = bk[key][r["seed"]]
            mx = max(e["test"] for e in r["epochs"])
            assert abs(mx - b) < 6e-5, ("max over logged epochs != banked", key, r["seed"], mx, b)
            assert abs(r["acc"] - b) < 6e-5, ("log summary != banked", key, r["seed"], r["acc"], b)
    return True


def fmt(xs):
    return "/".join("%.3f" % x for x in xs)


def write_report(R, sweeps, path):
    S = {k: summarize(v) for k, v in R.items()}
    L = []
    P = L.append
    P("ARENA EPOCH CHECK  (referee: 'best epoch of each arm on the test set')   generated by /path/to/vsassm/arena_epoch_check.py")
    P("=" * 118)
    P("What the scripts do: every arena script (V-1/V-4 train_one, V-8, V-9, V-11/V-18, V-13, V-14, V-15, V-15b) evaluates the")
    P("TEST split after each epoch and banks max over epochs. None has a validation split. 2 epochs everywhere (V-15b: 10).")
    P("No training accuracy is logged anywhere. Training LOSS is logged per epoch only by run_systematic_chain.train_one")
    P("(V-1 probe/LoRA on arena A, V-4 exact codes on A) and as last-minibatch loss by V-165 (rerun of the binding-loss map on C,")
    P("which reproduces V-13's per-epoch test accuracies exactly). The decision threshold of the B/C arms is fitted in closed")
    P("form on the TRAINING instances, so the threshold itself never sees the test set.")
    P("Check: for every arm x arena x seed, max over the logged epochs == the banked value in figdata/ (asserted; all pass),")
    P("so these logs are the runs behind the printed numbers. figdata JSONs are md5-identical to the box originals.")
    P("")
    P("Columns: printed = paper value (mean of 3 seeds, best test epoch); final = last epoch (no selection); worst = min over")
    P("epochs (lower bound for ANY epoch rule); train-loss-sel = epoch with the lower training loss (only where logged).")
    P("'val-selected' is not computable: no validation split, no training accuracy in any log. (Methods prints a final-epoch")
    P("TRAINING accuracy for LoRA on C, 0.532/0.520/0.502; it is not in v15.log and a ledger grep did not find it. Being")
    P("final-epoch only, it could not select an epoch anyway.)")
    P("")
    hdr = "%-44s %-5s %-7s %-7s %-7s %-7s %-6s  %s" % ("arm", "arena", "printed", "best", "final", "worst", "tl-sel", "per-seed final (42/123/456)")
    P(hdr); P("-" * len(hdr))
    order = ["parser", "naming", "probe", "loss", "lora", "relabel", "nosnap"]
    for arm in order:
        for ar in "ABC":
            k = (arm, ar)
            if k not in S:
                continue
            s = S[k]
            P("%-44s %-5s %-7.3f %-7.3f %-7.3f %-7.3f %-6s  %s%s" % (
                NAMES[arm], ar, PRINTED[k], mean(s["best"]), mean(s["final"]), mean(s["worst"]),
                ("%.3f" % mean(s["tl"])) if s["tl"] else "n/a", fmt(s["final"]),
                "   [identical every epoch]" if s["invariant"] else ""))
    for k in [("lora10", "C"), ("loss165", "C")]:
        s = S[k]
        P("%-44s %-5s %-7.3f %-7.3f %-7.3f %-7.3f %-6s  %s" % (NAMES[k[0]], k[1], PRINTED[k], mean(s["best"]), mean(s["final"]),
                                                               mean(s["worst"]), ("%.3f" % mean(s["tl"])) if s["tl"] else "n/a",
                                                               fmt(s["final"])))
    P("")
    P("Per-epoch test accuracy, seeds 42 | 123 | 456  (source log; B = hybrid_neurosymbolic root, D = ngcls root)")
    for k in sorted(R, key=lambda k: (["parser", "naming", "probe", "loss", "loss165", "lora", "lora10", "relabel", "nosnap"].index(k[0]), k[1])):
        s = S[k]
        P("  %-9s %s  %-62s %s" % (k[0], k[1], " | ".join(fmt(x) for x in s["epochs"]), SOURCE[k]))
    P("")
    P("Was the arm trained?")
    for arm in order:
        P("  %-44s %s" % (NAMES[arm], TRAINED[arm]))
    P("  Exact codes on B/C (and relabelled, and the frozen no-snap projection) give the same test accuracy at every epoch:")
    P("  their score depends only on fixed codebook rows and a threshold refitted on training data, so best == final by construction.")
    P("")
    # ------------------------------------------------------------------ conclusions
    f = lambda arm, ar: mean(S[(arm, ar)]["final"])
    w = lambda arm, ar: mean(S[(arm, ar)]["worst"])
    P("CONCLUSIONS UNDER THE FINAL EPOCH (and under the worst epoch, i.e. any epoch rule at all)")
    c1 = all(f(a, ar) > 0.85 for a in ("naming", "parser") for ar in "AC") and \
         all(f(a, ar) < 0.56 for a in ("probe", "loss", "lora") for ar in "AC")
    P("(1) only naming head and exact codes clearly beat chance on A and C: %s" % ("HOLDS" if c1 else "CHANGES"))
    P("    final: naming %.3f / %.3f, exact %.3f / %.3f vs probe %.3f / %.3f, binding loss %.3f / %.3f, LoRA %.3f / %.3f (A / C)" % (
        f("naming", "A"), f("naming", "C"), f("parser", "A"), f("parser", "C"), f("probe", "A"), f("probe", "C"),
        f("loss", "A"), f("loss", "C"), f("lora", "A"), f("lora", "C")))
    P("    worst epoch: naming %.3f / %.3f (A / C); the naming head's epoch-1 score on A is %.1f points over the probe's final." % (
        w("naming", "A"), w("naming", "C"), 100 * (w("naming", "A") - f("probe", "A"))))
    lo = min(w(a, ar) for a in ("naming", "parser") for ar in "AC")
    hi = max(mean(S[(a, ar)]["best"]) for a in ("probe", "loss", "lora") for ar in "AC")
    P("    ANY epoch rule: the worst epoch of naming head / exact codes on A,C (%.3f) exceeds the BEST epoch of probe /" % lo)
    P("    binding loss / LoRA on A,C (%.3f) by %.1f points, so no choice of epoch can change conclusion (1)." % (hi, 100 * (lo - hi)))
    c2 = all(f("loss", ar) < 0.53 for ar in "ABC")
    P("(2) binding-loss map stays at chance: %s  (final %s; best %s)" % ("HOLDS" if c2 else "CHANGES",
      " / ".join("%.3f" % f("loss", ar) for ar in "ABC"), " / ".join("%.3f" % mean(S[("loss", ar)]["best"]) for ar in "ABC")))
    c3 = all(f("lora", ar) < 0.53 for ar in "AC")
    P("(3) LoRA stays at chance on A and C: %s  (final %.3f / %.3f, best %.3f / %.3f; C final per seed %s, test majority 0.515;" % (
        "HOLDS" if c3 else "CHANGES", f("lora", "A"), f("lora", "C"), mean(S[("lora", "A")]["best"]), mean(S[("lora", "C")]["best"]),
        fmt(S[("lora", "C")]["final"])))
    P("    10-epoch extension final %.3f, per seed %s). LoRA on B: final %.3f (%s) vs printed 0.795 -- still unreliable." % (
        f("lora10", "C"), fmt(S[("lora10", "C")]["final"]), f("lora", "B"), fmt(S[("lora", "B")]["final"])))
    P("")
    P("Derived numbers in the text under the final epoch (printed -> final):")
    gap = lambda ar: (f("parser", ar) - f("probe", ar)) * 100
    P("  exact codes minus probe: 40.9/18.1/41.9 -> %.1f/%.1f/%.1f points" % (gap("A"), gap("B"), gap("C")))
    P("  binding loss minus probe: -0.8/-30.7/-3.2 -> %.1f/%.1f/%.1f points" % tuple((f("loss", ar) - f("probe", ar)) * 100 for ar in "ABC"))
    P("  relabelling cost: 4.9/0.2/0.5 -> %.1f/%.1f/%.1f points  (A relabelled final %.3f vs printed 0.869: the only above-chance" % (
        *((f("parser", ar) - f("relabel", ar)) * 100 for ar in "ABC"), f("relabel", "A")))
    P("    row that moves by a point. On arena A the two trainable readout scalars set the decision (train_one, argmax of logits),")
    P("    so arena-A code arms can move between epochs; on B/C the threshold is refitted on training data and they cannot.)")
    rec = lambda arm, ar: 100 * (f(arm, ar) - f("probe", ar)) / (f("parser", ar) - f("probe", ar))
    P("  naming head recovers: 99/100/100%% -> %.0f/%.0f/%.0f%%;  no-snap recovers 40/94/98%% -> %.0f/%.0f/%.0f%%" % (
        rec("naming", "A"), rec("naming", "B"), rec("naming", "C"), rec("nosnap", "A"), rec("nosnap", "B"), rec("nosnap", "C")))
    P("  Fig. 5b filled diamond (binding-loss map after training, V-165 acc_best 0.483) -> final %.3f (seeds %s);" % (
        mean(S[("loss165", "C")]["final"]), fmt(S[("loss165", "C")]["final"])))
    v165 = {r["seed"]: [e["train_loss"] for e in r["epochs"]] for r in R[("loss165", "C")]}
    P("    appendix 'best test epoch 0.472, 0.502, 0.476' -> final %s. Training-loss selection picks the final epoch" %
      ", ".join("%.3f" % x for x in S[("loss165", "C")]["final"]))
    P("    in all three seeds (last-batch loss %s)." % ", ".join("%.2f->%.2f" % tuple(v165[s]) for s in SEEDS))
    P("")
    P("Fig. 5c (V-14, bindings per level, arena C) and Supplementary Fig. a (V-57, depth): best -> final, mean of 3 seeds")
    for tag, label in (("v14", "k"), ("v57", "depth")):
        for arm in ("parser", "probe_argmax", "floor"):
            d = sweeps[(tag, arm)]
            row = []
            for key in sorted(d, key=int):
                s = summarize(d[key])
                row.append("%s=%s %.3f->%.3f" % (label, key, mean(s["best"]), mean(s["final"])))
            P("  %-4s %-13s %s" % (tag, arm, "  ".join(row)))
    P("  V-57 lora_direct is evaluated ONCE after its last epoch (v57.py), so its printed values are final-epoch already.")
    P("  Naming head and exact codes are unchanged at every k and depth; only the probe moves, and it moves DOWN under the final")
    P("  epoch (e.g. k=16 0.511->0.495, depth 3 0.530->0.507), which widens rather than narrows the paper's contrast.")
    P("")
    head5 = [(a, ar) for a in ("parser", "naming", "probe", "loss", "lora") for ar in "ABC"]
    moved = [(k, mean(S[k]["best"]), mean(S[k]["final"])) for k in head5 if mean(S[k]["best"]) - mean(S[k]["final"]) > 5e-4]
    P("BOTTOM LINE: of the five headline arms, test-epoch selection moved only arms at or near chance (and LoRA on B), by at")
    P("most %.1f points in the mean: %s." % (max(100 * (b - x) for _, b, x in moved),
                                            ", ".join("%s/%s %.3f->%.3f" % (k[0], k[1], b, x) for k, b, x in moved)))
    P("Exact codes and the naming head give the same mean at the final epoch as at the best one in all three arenas. Outside")
    P("the headline arms, the relabelled-codes control on A drops 0.869->%.3f and the 10-epoch LoRA extension on C drops" % f("relabel", "A"))
    P("0.524->%.3f (further below chance). All three qualitative conclusions hold under the final epoch, and (1) holds under" % f("lora10", "C"))
    P("any epoch rule whatever.")
    open(path, "w").write("\n".join(L) + "\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-b", help="local mirror of the hybrid_neurosymbolic root (v1_logs, v4_logs, v8_logs, v9_logs, v11_logs)")
    ap.add_argument("--logs-d", help="local mirror of the ngcls root (v13_logs, v14_logs, v15_logs, v15b_logs, v57_logs, v165_logs, runs/v9)")
    ap.add_argument("--dump", help="write the parsed per-epoch snapshot to this JSON file")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.logs_b and a.logs_d:
        R, sweeps = parse_all(a.logs_b, a.logs_d)
        if SNAPSHOT is not None:
            snap = load_snapshot()
            assert json.dumps(_ser(R, sweeps), sort_keys=True) == json.dumps(snap, sort_keys=True), "logs differ from embedded snapshot"
    else:
        R, sweeps = _deser(load_snapshot())
    check_banked(R)
    if a.dump:
        json.dump(_ser(R, sweeps), open(a.dump, "w"), indent=0, sort_keys=True)
    print(write_report(R, sweeps, a.out))


def _ser(R, sweeps):
    return {"runs": {"%s|%s" % k: v for k, v in R.items()},
            "sweeps": {"%s|%s" % k: v for k, v in sweeps.items()}}


def _deser(d):
    R = {tuple(k.split("|")): v for k, v in d["runs"].items()}
    sw = {tuple(k.split("|")): v for k, v in d["sweeps"].items()}
    return R, sw


def load_snapshot():
    return json.loads(SNAPSHOT)


if __name__ == "__main__":
    main()
