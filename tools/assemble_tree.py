#!/usr/bin/env python3
"""Assemble runnable source trees from the package: work/<source>/ = code/<source> + data/<source> + results/<source>,
with the outputs that the packager gzipped expanded back to their original names (docs/PROVENANCE.tsv records which).
The scripts then find their inputs and outputs at the relative paths they use (run them from work/<source>/).

    python3 tools/assemble_tree.py [OUT_DIR]          # default: ./work
"""
import csv
import gzip
import os
import shutil
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(PKG, "work"))
n = 0
with open(os.path.join(PKG, "docs", "PROVENANCE.tsv")) as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["package_path"].split("/")[0] not in ("code", "data", "results"):
            continue
        src = os.path.join(PKG, row["package_path"])
        dst = os.path.join(OUT, row["source"], row["source_path"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if "gzip" in row["transforms"] and "as in source" not in row["transforms"]:
            with gzip.open(src, "rb") as a, open(dst, "wb") as b:
                shutil.copyfileobj(a, b)
        else:
            shutil.copy2(src, dst)
        n += 1
print("assembled %d files under %s" % (n, OUT))
