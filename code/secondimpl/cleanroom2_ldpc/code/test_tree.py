import sys, numpy as np, itertools
sys.path.insert(0,'.')
import ldpc_own as lo
rng=np.random.default_rng(6)
# tree-structured IRA code: every info bit has degree 1 -> Tanner graph is a tree, BP = exact bitwise MAP
lo.PROFILE[(1,2)] = (0.0, 1)
k=12
code=lo.build_ira.__wrapped__(k,(1,2)) if hasattr(lo.build_ira,'__wrapped__') else None
