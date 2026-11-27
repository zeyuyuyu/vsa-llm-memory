"""General modelling harness (design v1, milestone G0): run a CPMpy model written by a language model (or by hand) and
apply the solver primitives the certification layer needs, whatever the problem is.

Contract for a model program (Python, `from cpmpy import *` is already done):
  - add constraints to the given `model` with `add(constraint, "S3")` (the sentence ids it encodes; plain
    `model += c` is allowed and counts as untagged);
  - set `answer = {"key": expr or list/array of exprs, ...}` with the keys of the question's answer schema
    (if `answer` is not set, the keys are looked up as global variable names, the CP-Bench convention);
  - optionally `model.minimize(...)` / `model.maximize(...)`.
Primitives (one fresh process each; `op` is a dict):
  solve                      -> status, answer values, objective value
  fix    {assign, objective} -> is the model satisfiable with the answer variables fixed to `assign` (and, when the
                                model has an objective, does that assignment reach `objective`)?
  other  {assign}            -> is there a solution whose answer differs from `assign`? (uniqueness)
  enum   {cap, at_optimum}   -> up to `cap` distinct answers (at_optimum: only those reaching the optimum)
  subset {tags}              -> is the model satisfiable keeping only the tagged constraints that cite one of `tags`
                                (plus every untagged one)? Another formalisation's core, checked here: the general
                                form of the NSR7b reason check for "impossible"
  mus                        -> for an unsatisfiable model, a minimal unsatisfiable subset of the TAGGED constraints
                                (untagged ones are kept hard), as sentence tags
Trusted code (our gold models, CP-Bench references) may run locally; model-written code runs in the no-network Docker
image nsengine-sandbox:cpm (sandbox/Dockerfile.cpm).
"""
import json
import os
import subprocess
import sys
import tempfile

PRELUDE = r'''
import json as _nse_json, os as _nse_os, sys as _nse_sys
from cpmpy import *
import numpy as np
model = Model()
_NSE_TAGGED = []
def add(c, tag="?"):
    _NSE_TAGGED.append((c, str(tag)))
    globals()["model"] += c
_NSE_USES, _NSE_IGNORED = [], []
def uses(tag):
    _NSE_USES.extend(t.strip() for t in str(tag).split(",") if t.strip())
def ignore(tag, why=""):
    _NSE_IGNORED.extend(t.strip() for t in str(tag).split(",") if t.strip())
answer = None
'''

EPILOGUE = r'''
def _nse_main():
    import numpy as _np
    from cpmpy.expressions.utils import flatlist as _flat
    op = _nse_json.loads(_nse_os.environ.get("NSE_OP", '{"op": "solve"}'))
    T = float(op.get("time_limit", 60))
    g = globals()
    M = g["model"]
    ans = g.get("answer")
    if ans is None:
        ans = {k: g[k] for k in op.get("keys", []) if k in g}
    def val(e):
        try:
            v = e.value()
        except AttributeError:
            v = e
        v = _np.asarray(v, dtype=object).tolist() if not isinstance(v, (int, bool, type(None))) else v
        return _nse_json.loads(_nse_json.dumps(v, default=lambda x: int(x) if x is not None else None))
    def pairs(assign):
        cons = []
        for k, v in assign.items():
            if k not in ans:
                raise KeyError("answer key %r not in model" % k)
            es = _flat([ans[k]])
            vs = _np.asarray(v, dtype=object).flatten().tolist() if isinstance(v, list) else [v]
            if len(es) != len(vs):
                raise ValueError("shape mismatch for %r: %d vs %d" % (k, len(es), len(vs)))
            cons += [e == int(x) for e, x in zip(es, vs)]
        return cons
    from cpmpy.transformations.get_variables import get_variables as _getvars
    for _v in _getvars(_flat([ans[k] for k in ans])):   # an answer variable no constraint mentions still gets a value
        M += _v >= _v.lb
    has_obj = M.has_objective()
    out = {"op": op["op"]}
    if op["op"] == "solve":
        ok = M.solve(time_limit=T)
        out["status"] = "sat" if ok else ("unsat" if str(M.status().exitstatus).endswith("UNSATISFIABLE") else "unknown")
        out["cited"] = sorted({t.strip() for _, ts in _NSE_TAGGED for t in ts.split(",") if t.strip()} | set(_NSE_USES))
        out["ignored"] = sorted(set(_NSE_IGNORED))
        if ok:
            out["answer"] = {k: val(e) for k, e in ans.items()}
            out["objective"] = M.objective_value() if has_obj else None
            out["optimal"] = str(M.status().exitstatus).endswith("OPTIMAL") if has_obj else None
        out["cited"] = sorted({t.strip() for _, ts in _NSE_TAGGED for t in ts.split(",") if t.strip()} | set(_NSE_USES))
        out["ignored"] = sorted(set(_NSE_IGNORED))
    elif op["op"] == "fix":
        M2 = Model(list(M.constraints) + pairs(op["assign"]))
        ok = M2.solve(time_limit=T)
        out["status"] = "sat" if ok else ("unsat" if str(M2.status().exitstatus).endswith("UNSATISFIABLE") else "unknown")
        if ok and has_obj:                                # the model's own optimum, then the fixed answer's value
            M.solve(time_limit=T)
            M3 = Model(list(M.constraints) + pairs(op["assign"]))
            obj = M.objective_
            M3.minimize(obj) if M.objective_is_min else M3.maximize(obj)
            M3.solve(time_limit=T)
            out["objective_model"] = M.objective_value()
            out["objective_fixed"] = M3.objective_value()
    elif op["op"] == "other":
        cs = pairs(op["assign"])
        M2 = Model(list(M.constraints) + [~all(cs) if len(cs) else False])
        ok = M2.solve(time_limit=T)
        out["status"] = "sat" if ok else ("unsat" if str(M2.status().exitstatus).endswith("UNSATISFIABLE") else "unknown")
        if ok:
            out["answer"] = {k: val(e) for k, e in ans.items()}
    elif op["op"] == "enum":
        seen = []
        def cb():                                        # distinct answers (auxiliary variables may repeat one)
            a_ = {k: val(e) for k, e in ans.items()}
            if a_ not in seen:
                seen.append(a_)
        cons = list(M.constraints)
        if op.get("at_optimum") and has_obj:             # every answer that reaches the optimum
            M.solve(time_limit=T)
            out["objective"] = M.objective_value()
            cons.append(M.objective_ == M.objective_value())
        Model(cons).solveAll(display=cb, solution_limit=int(op.get("cap", 100)), time_limit=T)
        out["answers"] = seen
    elif op["op"] == "subset":                            # only the constraints tagged with one of these sentences
        want = {x.strip() for t in op["tags"] for x in str(t).split(",") if x.strip()}
        tagged = [(c, set(t.replace(" ", "").split(","))) for c, t in _NSE_TAGGED]
        hard = [c for c in M.constraints if not any(c is t for t, _ in tagged)]
        keep = [c for c, ts in tagged if ts & want]
        M2 = Model(hard + keep)
        ok = M2.solve(time_limit=T)
        out["status"] = "sat" if ok else ("unsat" if str(M2.status().exitstatus).endswith("UNSATISFIABLE") else "unknown")
    elif op["op"] == "mus":
        from cpmpy.tools.explain import mus
        tagged = [c for c, _ in _NSE_TAGGED]
        hard = [c for c in M.constraints if not any(c is t for t in tagged)]
        core = mus(soft=tagged, hard=hard)
        out["tags"] = sorted({x.strip() for c, t in _NSE_TAGGED if any(c is y for y in core) for x in t.split(",") if x.strip()},
                             key=lambda x: (len(x), x))
    print("NSE_RESULT " + _nse_json.dumps(out))
_nse_main()
'''


def build(code):
    return PRELUDE + "\n" + code + "\n" + EPILOGUE


def run(code, op, trusted=False, timeout=120):
    """Run one primitive on a model program. Returns the result dict, or {"error": ...}."""
    prog = build(code)
    env = dict(os.environ, NSE_OP=json.dumps(op))
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "prog.py"), "w").write(prog)
        if trusted:
            cmd = [sys.executable, os.path.join(d, "prog.py")]
        else:
            cmd = ["docker", "run", "--rm", "--network", "none", "--cpus", "2", "--memory", "4g", "--read-only",
                   "--tmpfs", "/tmp", "-e", "NSE_OP", "-v", d + ":/w:ro", "nsengine-sandbox:cpm", "timeout",
                   str(timeout), "python3", "-I", "/w/prog.py"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 30, env=env, cwd=d)
        except subprocess.TimeoutExpired:
            return {"error": "timeout"}
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("NSE_RESULT "):
            return json.loads(line[len("NSE_RESULT "):])
    return {"error": "no result", "rc": p.returncode, "stderr": p.stderr[-1500:]}
