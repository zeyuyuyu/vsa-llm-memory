#!/usr/bin/env python3
"""Check the figure data layer against the package (run from anywhere; nothing is written).

1. Every pointer in figures/figdata/**/*.json and figures/FIGURE_VALUES.json (keys "file", "count_file", "check_file")
   is a relative path that resolves to a file inside this package (a ".gz" twin counts; the package keeps large
   outputs gzipped).
2. Every transcribed value that carries its verbatim source line ("line"/"lines") is re-checked as in
   make_figures.verify_sources: the line is in the file, the number's text is in the line (in the stated table cell
   where one is given), the number equals the stored value, and the paired-outcome and per-site counts are recounted.

    python3 figures/check_sources.py        # exit status 0 when everything resolves and matches
"""
import gzip
import json
import os
import re
import sys

FIG = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(FIG)
D = os.path.join(FIG, "figdata")
POINTER_KEYS = ("file", "count_file", "check_file")


def resolve(p):
    """Package path of a pointer, or None. Pointers are relative to figures/ (or to figures/figdata/ in INDEX.json)."""
    if not isinstance(p, str) or not p or "://" in p:
        return None
    if os.path.isabs(p):
        return None
    for base in (FIG, D):
        for cand in (p, p + ".gz"):
            q = os.path.realpath(os.path.join(base, cand))
            if q.startswith(os.path.realpath(PKG) + os.sep) and os.path.isfile(q):
                return q
    return None


_texts = {}


def text_of(p):
    q = resolve(p)
    if q is None:
        return None
    if q not in _texts:
        b = open(q, "rb").read()
        _texts[q] = (gzip.decompress(b) if q.endswith(".gz") else b).decode("utf-8")
    return _texts[q]


def pointers(node, where, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in POINTER_KEYS and isinstance(v, str) and ("/" in v or v.endswith((".json", ".txt", ".jsonl", ".md", ".csv"))):
                out.append((where + "/" + k, v))
            pointers(v, where + "/" + str(k), out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            pointers(v, "%s[%d]" % (where, i), out)


def verify(node, where, bad, seen):
    if isinstance(node, dict):
        if "file" in node and ("line" in node or "lines" in node):
            src = text_of(node["file"])
            if src is None:
                bad.append("%s: source %s not in the package" % (where, node["file"]))
            else:
                seen[0] += 1
                lines = node.get("lines", [node.get("line")])
                if "\n".join(lines) not in src:
                    bad.append("%s: line not found verbatim in %s" % (where, node["file"]))
                if "text" in node:
                    hay = lines[0]
                    if "cell" in node:
                        hay = [c.strip() for c in hay.strip().strip("|").split("|")][node["cell"]]
                    if node["text"] not in hay:
                        bad.append("%s: %r not in its source line" % (where, node["text"]))
                    if "value" in node:
                        num = node["text"].split("±")[0]
                        if "/" in num:
                            c, n = num.split()[-1].split("/")
                            got = [100.0 * int(c) / int(n)]
                        else:
                            got = [float(num), 100 * float(num)]
                        if min(abs(g - node["value"]) for g in got) > 1e-9:
                            bad.append("%s: value %s != text %s" % (where, node["value"], node["text"]))
                    if "sd" in node and abs(float(node["text"].split("±")[1]) - node["sd"]) > 1e-9:
                        bad.append("%s: s.d. %s != text %s" % (where, node["sd"], node["text"]))
        for k, v in node.items():
            verify(v, where + "/" + str(k), bad, seen)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            verify(v, "%s[%d]" % (where, i), bad, seen)


def main():
    bad, seen = [], [0]
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(D) for f in fs if f.endswith(".json")]
    files.append(os.path.join(FIG, "FIGURE_VALUES.json"))
    ptrs = []
    for f in files:
        try:
            d = json.load(open(f))
        except (json.JSONDecodeError, FileNotFoundError) as e:
            bad.append("%s: %s" % (os.path.relpath(f, PKG), e))
            continue
        pointers(d, os.path.relpath(f, PKG), ptrs)
    unresolved = [(w, p) for w, p in ptrs if resolve(p) is None]
    for w, p in unresolved:
        bad.append("%s: pointer %r does not resolve inside the package" % (w, p))
    J = lambda rel: json.load(open(os.path.join(D, rel)))
    for name in ("oplogs.json", "benchmarks/babilong_per_task.json", "gains.json", "benchmarks/ruler.json",
                 "benchmarks/b23.json", "benchmarks/b26.json"):
        verify(J(name), name, bad, seen)
    op = J("oplogs.json")
    for r in op["paired"]["rows"]:
        for block in (r, r.get("repaired_parse")):
            if block is None:
                continue
            if text_of(r["file"]) is None:
                bad.append("paired/%s: source not in the package" % r["id"])
                continue
            seen[0] += 1
            nums = [int(x) for x in re.findall(r"(?:right|RM|history|frontier|wrong) (\d+)[,;]", " ".join(block.get("lines", r["lines"])))]
            if nums[:4] != [block[k] for k in ("both_right", "only_ours", "only_gpt", "both_wrong")]:
                bad.append("paired/%s: counts differ from the source sentence" % r["id"])
    for s in op["per_site"]["sites"]:
        f = s["n"]["count_file"]
        t = text_of(f)
        if t is None:
            bad.append("per_site/%s: %s not in the package" % (s["id"], f))
            continue
        seen[0] += 1
        got = sum(1 for l in t.splitlines() if l.strip() and json.loads(l)["id"].startswith(s["n"]["count_prefix"]))
        if got != s["n"]["value"]:
            bad.append("per_site/%s: n %d, counted %d" % (s["id"], s["n"]["value"], got))
    print("check_sources: %d pointers (%d resolve inside the package), %d transcribed values re-checked: %s"
          % (len(ptrs), len(ptrs) - len(unresolved), seen[0], "all match" if not bad else "%d PROBLEM(S)" % len(bad)))
    for b in bad:
        print("   - " + b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
