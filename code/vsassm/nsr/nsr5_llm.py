"""NSR5 LLM arm on multi-day sites: prompts (merged asset list + catalog + the K day logs + question) and parsing.

python3 nsr/nsr5_llm.py prep --K 8
python3 nsr/nsr5_llm.py parse --K 8 --runs 'nsr/out/api/nsr5_K8_*.jsonl' [--normalize]
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nsr1_catalog as C  # noqa: E402
import nsr1_llm as L  # noqa: E402
import nsr5_sites as S  # noqa: E402

SC = ["warehouse", "biobank", "home", "ops"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "parse", "engine"])
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--runs")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--repair", action="store_true")  # frontier amendment 2
    ap.add_argument("--arm", default="frontier")  # output name for parse (frontier = gpt-6.1-sol; claude = 2nd arm)
    a = ap.parse_args()
    L.NORMALIZE = a.normalize
    L.REPAIR = a.repair
    ss = S.sites(a.K)
    if a.cmd == "prep":
        path = os.path.join(HERE, "out", f"nsr5_problems_K{a.K}.jsonl")
        with open(path, "w") as f:
            for s_i, (site, _read) in enumerate(ss):
                intro = L.INTRO[site["scenario"]].replace("today's record", f"the record of {a.K} consecutive days")
                for idx, q in enumerate(C.questions(site)):
                    p = (f"{intro} Answer the question about the state at the end of the record.\n\n{L.assets_block(site)}"
                         f"\n\n{L.catalog_block(site)}\n\n" + "\n".join(site["text_lines"]) + f"\n\nQuestion: {q['q']}\n{L.ASK}")
                    f.write(json.dumps({"task": f"NSR5K{a.K}", "n": s_i, "idx": idx,
                                        "id": f"{site['id']}|{q['qtype']}|{q['target']}", "prompt": p}) + "\n")
        print(path)
        return
    out = []
    if a.cmd == "engine":
        for site, read in ss:
            cat = C.catalog(site)
            for q in C.questions(site):
                out.append({"id": f"{site['id']}|{q['qtype']}|{q['target']}", "scenario": site["scenario"],
                            "pred": sorted(C.answer(q["qtype"], q["target"], read, cat))})
        name = "pipeline"
    else:
        by = {site["id"]: site for site, _ in ss}
        runs = {r["id"]: r for fp in glob.glob(a.runs) for r in map(json.loads, open(fp)) if "error" not in r}
        for site, _ in ss:
            for q in C.questions(site):
                key = f"{site['id']}|{q['qtype']}|{q['target']}"
                r = runs.get(key)
                txt = None if r is None or r.get("hit_limit") else r.get("answer_text")
                out.append({"id": key, "scenario": site["scenario"], "pred": L.resolve(txt, by[site["id"]])})
        name = a.arm + ("n" if a.normalize else "")
    path = os.path.join(HERE, "out", f"nsr5_K{a.K}_{name}.jsonl")
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in out)
    print(path)


if __name__ == "__main__":
    main()
