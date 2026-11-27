"""B67: run b67_check's C1 (a: H and L / L') and C2 for one encoder in one process, sharing one DevData."""
import sys

import b67_theory as T
import b67_check as K

_orig, _cache = T.DevData, {}


def _dd(enc):
    if enc not in _cache:
        _cache[enc] = _orig(enc)
    return _cache[enc]


T.DevData = _dd
enc = sys.argv[1]
K.check_c2([enc])
K.check_a(enc)
K.check_L(enc)
