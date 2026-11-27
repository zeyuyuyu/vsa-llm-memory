"""B33 generator check: does b33_gen reproduce the official bAbI task-2 distribution?

Reads the official training split (Muennighoff/babi task 2: 900 train + 100 validation questions, 200 stories; the
held-out test split is never read here) and compares it with generated stories on a battery of statistics.

  Split-half protocol. The structural rules (templates, steps of two statements, the question rule and its window,
  five questions per story) were read off all 200 official stories. The numeric parameters (INIT in {random, objlazy}: the
  places at which never-mentioned objects are first picked up by people whose place is stated are uniform over the six
  locations, so the released world file's fixed placement is excluded; P_GET, P_GET_HOLDER, P_DROP, R_GET, R_DROP,
  W_UNSEEN; P_THERE_DROP by its rate) are fitted on the first 100 stories only (random search, then a
  local search from the best 8, minimising the mean chi-square distance per degree of freedom over every
  statistic).
  The gate compares the generator with the OTHER 100 stories on every statistic: story-level parametric bootstrap
  (B = 2000 samples of 100 generated stories; resampling stories keeps the dependence of a story's questions),
  every p >= 0.05 / K (Bonferroni over K statistics), the tracker answers every official question, and the
  generated stories obey the question rule at every step end. Statistics: action mix (overall, given the actor
  holds nothing / something), same actor after a get / drop / move, "there" after a drop and a get, verbs within
  each action, story length, first-question position, question gaps, question about a held vs a dropped object,
  objects whose place is known at question time, gets by a person whose place was never stated, gets of a
  never-mentioned object, objects already held by the getter, dropper moved after the drop, objects with an event
  and persons with a move at question time (the bundle sizes), statements since the queried object's last event,
  statement-pair kinds, answers, queried objects.

  python b33_gencheck.py --official <official_task2_trainval.jsonl> --n_stories 20000 --seed 1 [--fit]
Dev seeds only (1, 2, 3); the registered held-out seed is never used here.
"""
import argparse, collections, itertools, json, os, random, sys

import numpy as np
from scipy.stats import chi2_contingency

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b33_gen as G                                                   # noqa: E402
from b33_track import Tracker, stories, parse_stmt                    # noqa: E402


def bucket(x, edges):
    for e in edges:
        if x <= e:
            return "<=%d" % e
    return ">%d" % edges[-1]


def story_stats(rows):
    """Per-story statistics: a list of {stat: Counter} (one per story) plus tracker agreement counts."""
    S = stories(rows)
    per = []
    n_ok = n_q = 0
    for s in S:
        C = collections.defaultdict(collections.Counter)
        T = Tracker()
        infos = []
        last_ev = {}                       # object -> (index, kind, actor)
        moved_after = collections.defaultdict(list)
        qpos = [q[0] for q in s["qs"]]
        qi = 0
        for i, x in enumerate(s["stmts"]):
            nh = sum(1 for h in T.held.values() if h == x.split()[0])
            e = T.step(x)
            e["holds_before"] = nh
            infos.append(e)
            if e["kind"] in ("get", "drop"):
                last_ev[e["arg"]] = (i, e["kind"], e["actor"])
            else:
                moved_after[e["actor"]].append(i)
            while qi < len(s["qs"]) and s["qs"][qi][0] == i + 1:
                _, o, ans = s["qs"][qi]
                got, mode = T.answer(o)
                n_q += 1; n_ok += int(got == ans)
                C["C_qtype"][mode] += 1
                C["C_known_objects"][str(sum(T.answer(ob)[0] is not None for ob in G.OBJS))] += 1
                li, lk, la = last_ev[o]
                if lk == "drop":
                    C["C_drop_then_moved"]["moved" if any(j > li for j in moved_after[la]) else "stayed"] += 1
                C["C_since_last_event"][bucket(i - li, [0, 1, 2, 3, 5, 8, 12])] += 1
                C["C_bundle_objects"][str(len(last_ev))] += 1
                C["C_bundle_persons"][str(len(moved_after))] += 1
                C["C_answer"][ans] += 1
                C["C_qobject"][o] += 1
                qi += 1
        C["C_story_len"][bucket(len(s["stmts"]), [12, 14, 16, 18, 20, 22, 24, 26, 28, 32])] += 1
        C["C_first_q"][bucket(qpos[0], [2, 4, 6, 8, 10, 12, 16])] += 1
        for a, b in zip(qpos, qpos[1:]):
            C["C_gap"][bucket(b - a, [2, 4, 6, 8])] += 1
        for e in infos:
            C["C_kind"][e["kind"]] += 1
            C["F1_kind_holds0" if e["holds_before"] == 0 else "F1_kind_holds1+"][e["kind"]] += 1
            C["C_verb_" + e["kind"]][e["verb"]] += 1
            if e["kind"] == "drop":
                C["F3_drop_there"][str(e["there"])] += 1
            if e["kind"] == "get":
                C["C_get_there"][str(e["there"])] += 1
                C["C_get_actor_place_unstated"][str(not e["actor_loc_known"])] += 1
                C["C_get_unseen_object"][str(e["obj_unseen"])] += 1
                C["C_get_already_held"][str(min(e["n_held_by_actor"], 2))] += 1
        for j in range(0, len(infos) - 1, 2):
            a, b = infos[j], infos[j + 1]
            C["C_pair_kinds"][a["kind"] + "-" + b["kind"]] += 1
        for a, b in zip(infos, infos[1:]):
            key = {"get": "F4_same_actor_after_get", "drop": "F4_same_actor_after_drop",
                   "move": "C_same_actor_after_move"}[a["kind"]]
            C[key][str(a["actor"] == b["actor"])] += 1
        per.append(C)
    return per, n_ok, n_q


def total(per, idx=None):
    out = collections.defaultdict(collections.Counter)
    for i in (range(len(per)) if idx is None else idx):
        for k, c in per[i].items():
            out[k].update(c)
    return {k: dict(v) for k, v in out.items() if sum(v.values())}


def stats(rows):
    per, n_ok, n_q = story_stats(rows)
    return total(per), n_ok, n_q, len(per)


def rule_check(rows, window):
    """At every step end (after an even number of statements): is a question asked exactly when some object is
    answerable within the window, and is the asked object answerable? Also: questions after an odd count."""
    out = collections.Counter()
    for s in stories(rows):
        T = Tracker(); last_ev = {}; moves = collections.defaultdict(list)
        qpos = {q[0]: q[1] for q in s["qs"]}
        for i, x in enumerate(s["stmts"]):
            e = T.step(x)
            if e["kind"] == "move":
                moves[e["actor"]].append(i)
            else:
                last_ev[e["arg"]] = (i, e["kind"], e["actor"])
            if (i + 1) % 2:
                out["question after an odd count"] += int((i + 1) in qpos)
                continue
            ans = [o for o, (j, k, a) in last_ev.items() if i - j <= window and T.answer(o)[0] is not None
                   and not (k == "get" and not any(m > j for m in moves[a]))]
            asked = (i + 1) in qpos
            out["slots"] += 1
            out["explained"] += int(asked == bool(ans) and (not asked or qpos[i + 1] in ans))
    return dict(out)


def distance(obs, ref):
    """Chi-square-type distance of observed counts from reference proportions (categories with expected count < 5
    pooled into one)."""
    cats = sorted(set(obs) | set(ref))
    n = sum(obs.values()); r = sum(ref.values())
    o = np.array([obs.get(c, 0) for c in cats], float)
    e = np.array([ref.get(c, 0) / r * n for c in cats], float)
    small = e < 5
    if small.any():
        o = np.append(o[~small], o[small].sum()); e = np.append(e[~small], e[small].sum())
    m = e > 0
    return float((((o - e) ** 2)[m] / e[m]).sum() + (o[~m] > 0).sum() * 1e6)


def compare(off_tot, gen_per, gen_tot, n_off, B, seed):
    """Story-level parametric bootstrap: is the official sample (n_off stories) a typical n_off-story sample of the
    generator? p = share of B generated samples whose distance from the generator's own proportions is >= the
    official sample's. Resampling whole stories keeps the within-story dependence of questions and statements."""
    rng = np.random.default_rng(seed)
    d_off = {k: distance(off_tot[k], gen_tot.get(k, {})) for k in off_tot}
    exceed = collections.Counter()
    for _ in range(B):
        samp = total(gen_per, rng.integers(0, len(gen_per), n_off))
        for k in off_tot:
            exceed[k] += int(distance(samp.get(k, {}), gen_tot.get(k, {})) >= d_off[k])
    return {k: (exceed[k] + 1) / (B + 1) for k in off_tot}


def gen_rows(n, seed, P):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        rows += G.story(rng, P)
    return rows


def objective(off_tot, gen_tot):
    """Fit criterion: mean over statistics of the chi-square-type distance per degree of freedom."""
    ds = []
    for k, o in off_tot.items():
        ncat = max(2, len(set(o) | set(gen_tot.get(k, {}))))
        ds.append(distance(o, gen_tot.get(k, {})) / (ncat - 1))
    return float(np.mean(ds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True)
    ap.add_argument("--n_stories", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--n_search", type=int, default=240)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    assert a.seed in (1, 2, 3), "dev seeds only"
    off_rows = [json.loads(l) for l in open(a.official)]
    oper, ok, nq = story_stats(off_rows)
    ns = len(oper)
    half = ns // 2
    fit_tot, chk_tot, all_tot = total(oper, range(half)), total(oper, range(half, ns)), total(oper)
    print("official: %d stories, %d questions; text tracker answers %d/%d; fit half = stories 1-%d, check half = "
          "stories %d-%d" % (ns, nq, ok, nq, half, half + 1, ns))
    rules = {w: rule_check(off_rows, w) for w in (2, 3, 4, 5, 6)}
    for w, r in rules.items():
        print("  question rule, window %d: %d of %d step ends explained; questions after an odd count: %d"
              % (w, r["explained"], r["slots"], r["question after an odd count"]))
    P = json.loads(json.dumps(G.PARAMS))
    fit = None
    if a.fit:
        rng = random.Random(a.seed)
        cands = []
        for _ in range(a.n_search):
            cands.append(dict(P, INIT=rng.choice(["random", "objlazy"]), P_GET=round(rng.uniform(0.2, 1.0), 2),
                              P_GET_HOLDER=round(rng.uniform(0.3, 1.0), 2),
                              P_DROP=round(rng.uniform(0.40, 0.65), 2), R_GET=round(rng.uniform(0.0, 0.4), 2),
                              R_DROP=round(rng.uniform(0.0, 0.3), 2), W_UNSEEN=round(rng.uniform(0.3, 1.5), 2)))
        scored = sorted(((objective(fit_tot, stats(gen_rows(1000, 11, c))[0]), i) for i, c in enumerate(cands)))
        top = [cands[i] for _, i in scored[:8]]
        # local search: from the best 8, perturb one numeric parameter at a time (3,000 stories per evaluation)
        cur = sorted(((objective(fit_tot, stats(gen_rows(3000, 12, c))[0]), j) for j, c in enumerate(top)))
        best_o, best = cur[0][0], top[cur[0][1]]
        steps = {"P_GET": 0.05, "P_GET_HOLDER": 0.05, "P_DROP": 0.03, "R_GET": 0.04, "R_DROP": 0.03, "W_UNSEEN": 0.15}
        for _ in range(3):
            for k, st in steps.items():
                for sgn in (-1, 1):
                    c = dict(best); c[k] = round(min(max(c[k] + sgn * st, 0.0), 1.0 if k != "W_UNSEEN" else 3.0), 3)
                    o = objective(fit_tot, stats(gen_rows(3000, 12, c))[0])
                    if o < best_o:
                        best_o, best = o, c
        refined = [(best_o, 0)]
        top = [best]
        d = fit_tot["F3_drop_there"]
        P.update({k: best[k] for k in ("INIT", "P_GET", "P_GET_HOLDER", "P_DROP", "R_GET", "R_DROP", "W_UNSEEN")})
        P["P_THERE_DROP"] = round(d["True"] / (d["True"] + d["False"]), 3)
        fit = {"search": [[o, cands[i]] for o, i in scored[:20]], "refined": [[o, top[j]] for o, j in refined]}
        print("fit (random search %d + local search from the best 8, on the fit half): %s  objective %.3f" % (
            a.n_search, json.dumps(P), refined[0][0]))
    else:
        print("using b33_gen.PARAMS: %s" % json.dumps(P))
    gper, okg, nqg = story_stats(gen_rows(a.n_stories, a.seed, P))
    sg = total(gper)
    print("generated: %d stories, %d questions (seed %d); tracker answers %d/%d" % (len(gper), nqg, a.seed, okg, nqg))
    pv_chk = compare(chk_tot, gper, sg, ns - half, a.boot, a.seed)
    pv_all = compare(all_tot, gper, sg, ns, a.boot, a.seed + 10)
    res = {"params": P, "fit": fit, "official_tracker": [ok, nq], "generated_tracker": [okg, nqg], "stats": {},
           "rule_check_official": rules,
           "rule_check_generated": rule_check(gen_rows(500, a.seed + 100, P), P["WINDOW"]), "boot": a.boot}
    keys = sorted(chk_tot)
    worst = min(pv_chk[k] for k in keys)
    print("  statistic                      p(check half)  p(all 200)   official(all)/generated proportions")
    for k in keys:
        cats = sorted(set(all_tot[k]) | set(sg.get(k, {})))
        to, tg = sum(all_tot[k].values()), sum(sg.get(k, {}).values())
        cells = "  ".join("%s %.3f/%.3f" % (c, all_tot[k].get(c, 0) / to, sg.get(k, {}).get(c, 0) / tg) for c in cats)
        print("  %-30s %.4f        %.4f       %s" % (k, pv_chk[k], pv_all[k], cells))
        res["stats"][k] = {"p_check_half": pv_chk[k], "p_all": pv_all[k], "official_all": all_tot[k],
                           "official_check_half": chk_tot[k], "generated": sg.get(k, {})}
    alpha = 0.05 / len(keys)
    rg = res["rule_check_generated"]
    gate = worst >= alpha and ok == nq and okg == nqg and rg["explained"] == rg["slots"]
    print("GATE G-DIST: %d statistics on the check half (story bootstrap, B = %d, p floor %.4f), min p = %.4f "
          "(threshold 0.05/%d = %.4f); tracker %d/%d official, %d/%d generated; question rule %d/%d generated -> %s"
          % (len(keys), a.boot, 1 / (a.boot + 1), worst, len(keys), alpha, ok, nq, okg, nqg, rg["explained"],
             rg["slots"], "PASS" if gate else "FAIL"))
    res.update({"min_p_check_half": worst, "alpha": alpha, "gate": gate,
                "n_below_alpha_all200": sum(pv_all[k] < alpha for k in keys)})
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
