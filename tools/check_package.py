#!/usr/bin/env python3
"""Integrity check of the package: SHA256SUMS, the figure pointers (figures/check_sources.py), and that every path
named in README.md and docs/DETAILS.md exists.   python3 tools/check_package.py"""
import hashlib
import os
import re
import subprocess
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bad = 0
n = 0
for line in open(os.path.join(PKG, "SHA256SUMS")):
    h, p = line.rstrip("\n").split("  ", 1)
    n += 1
    if not os.path.isfile(os.path.join(PKG, p)) or hashlib.sha256(open(os.path.join(PKG, p), "rb").read()).hexdigest() != h:
        print("checksum mismatch or missing: " + p)
        bad += 1
print("SHA256SUMS: %d files, %d problems" % (n, bad))
readme = open(os.path.join(PKG, "README.md")).read() + open(os.path.join(PKG, "docs", "DETAILS.md")).read()
miss = []
for p in sorted(set(re.findall(r"`((?:code|prereg|results|data|figures|ledger|tools)/[^`]+)`", readme))):
    if "<" in p or ("`%s` (pending)" % p) in readme:
        continue
    q = os.path.join(PKG, p.rstrip("/"))
    if not (os.path.exists(q) or os.path.exists(q + ".gz")):
        miss.append(p)
print("README and docs/DETAILS.md paths: %d missing%s" % (len(miss), "" if not miss else ": " + ", ".join(miss)))
r = subprocess.run([sys.executable, os.path.join(PKG, "figures", "check_sources.py")])
sys.exit(1 if bad or miss or r.returncode else 0)
