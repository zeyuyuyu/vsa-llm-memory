"""Registry of the held-out problem families."""
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FAMILY_NAMES = ["meetings", "roster", "project", "islanders", "vans"]
LEVELS = [1, 2, 3, 4, 5]


def load(name):
    return importlib.import_module(name)


def all_families():
    return {n: load(n) for n in FAMILY_NAMES}
