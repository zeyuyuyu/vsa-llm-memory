"""Certification layer (design v1, G2): certify-or-abstain over K independent formalisations of one problem, with no
knowledge of the problem type. Every check goes through the solver primitives of engine/gm.py.

    verdict, answer, cert = certify([primary_code, code_2, ...], kind)

kind (from the question):
  "find"      any answer meeting the constraints is right (e.g. a maintenance plan);
  "unique"    the question has one answer (e.g. which services go down);
  "optimize"  the answer must reach the optimum (one optimal answer is enough);
  "all_opt"   every optimal answer must be listed (e.g. all minimum failure sets).
Rules, for the primary's result A:
  - A is an answer:  every other formalisation accepts A (fix -> sat);
      unique:   no formalisation has another answer (other -> unsat);
      optimize: A reaches each formalisation's own optimum;
      all_opt:  every formalisation enumerates the same set of optimal answers.
  - A is "impossible" (primary unsat): every formalisation is unsat, AND each stays unsat when only the constraints
    citing the primary's core sentences are kept (agreement on the reason, NSR7b/NSR7c), AND the primary's core is
    itself checked the same way in reverse (each other formalisation's core sentences make the primary unsat).
Coverage (when the number of sentences is given): every sentence is cited, used or ignored by every formalisation, and
(strict) no sentence is ignored by one and cited by another; a fact one formalisation leaves out shows up here.
Anything else, including any error, timeout or "unknown" from the solver, is an abstention.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gm  # noqa: E402


def _canon(a):
    return json.dumps(a, sort_keys=True)


def coverage_check(results, n_sentences, strict=True):
    """Every sentence S1..Sn is cited (a constraint or uses()) or ignored by every formalisation; with strict, no
    sentence is ignored by one formalisation and cited by another. Returns a reason to abstain, or None."""
    allids = {"S%d" % i for i in range(1, n_sentences + 1)}
    for i, r in enumerate(results):
        miss = allids - set(r.get("cited") or []) - set(r.get("ignored") or [])
        if miss:
            return "formalisation %d leaves %s unaccounted for" % (i, ",".join(sorted(miss, key=lambda x: int(x[1:]))[:5]))
    if strict:
        for i, r in enumerate(results):
            for j, r2 in enumerate(results):
                d = (set(r.get("ignored") or []) & set(r2.get("cited") or [])) & allids
                if i != j and d:
                    return "formalisations %d and %d disagree on whether %s matters" % (i, j, ",".join(sorted(d)))
    return None


def certify(codes, kind, trusted=False, timeout=180, cap=200, n_sentences=None, strict_coverage=True):
    run = lambda c, op: gm.run(c, op, trusted=trusted, timeout=timeout)  # noqa: E731
    prim, others = codes[0], codes[1:]
    cert = {"kind": kind, "checks": []}
    r0 = run(prim, {"op": "solve"})
    if r0.get("status") not in ("sat", "unsat"):
        return "abstain", None, dict(cert, why="primary: %s" % (r0.get("error") or r0.get("status")))
    if n_sentences:
        rs = [r0] + [run(c, {"op": "solve"}) for c in others]
        why = coverage_check(rs, n_sentences, strict_coverage)
        if why:
            return "abstain", None, dict(cert, why=why)
    if r0["status"] == "unsat":
        m0 = run(prim, {"op": "mus"})
        if "tags" not in m0:
            return "abstain", None, dict(cert, why="primary core not computed")
        cert["core_sentences"] = m0["tags"]
        for i, c in enumerate(others, 1):
            rb = run(c, {"op": "solve"})
            if rb.get("status") != "unsat":
                return "abstain", None, dict(cert, why="formalisation %d: %s" % (i, rb.get("status") or rb.get("error")))
            sb = run(c, {"op": "subset", "tags": m0["tags"]})
            if sb.get("status") != "unsat":
                return "abstain", None, dict(cert, why="formalisation %d does not fail for the primary's reason" % i)
            mb = run(c, {"op": "mus"})
            back = run(prim, {"op": "subset", "tags": mb.get("tags", [])}) if "tags" in mb else {}
            if back.get("status") != "unsat":
                return "abstain", None, dict(cert, why="formalisation %d's reason does not hold in the primary" % i)
            cert["checks"].append({"formalisation": i, "unsat": True, "same_reason": True, "core": mb["tags"]})
        return "certified", {"impossible": True}, cert
    A = r0["answer"]
    if kind == "optimize" and r0.get("optimal") is False:
        return "abstain", None, dict(cert, why="primary optimum not proven")
    if kind == "unique":
        o0 = run(prim, {"op": "other", "assign": A})
        if o0.get("status") != "unsat":
            return "abstain", None, dict(cert, why="primary has another answer")
    ref_set = None
    if kind == "all_opt":
        e0 = run(prim, {"op": "enum", "at_optimum": True, "cap": cap})
        if "answers" not in e0 or len(e0["answers"]) >= cap:
            return "abstain", None, dict(cert, why="primary enumeration failed or hit the cap")
        ref_set = sorted(_canon(a) for a in e0["answers"])
        A = {"optimum": e0.get("objective"), "answers": e0["answers"]}
    for i, c in enumerate(others, 1):
        if kind == "all_opt":
            eb = run(c, {"op": "enum", "at_optimum": True, "cap": cap})
            if "answers" not in eb or sorted(_canon(a) for a in eb["answers"]) != ref_set or eb.get("objective") != A["optimum"]:
                return "abstain", None, dict(cert, why="formalisation %d lists other optimal answers" % i)
            cert["checks"].append({"formalisation": i, "same_optimal_set": True})
            continue
        f = run(c, {"op": "fix", "assign": A})
        if f.get("status") != "sat":
            return "abstain", None, dict(cert, why="formalisation %d rejects the answer (%s)" % (i, f.get("status") or f.get("error")))
        if kind == "optimize" and f.get("objective_model") != f.get("objective_fixed"):
            return "abstain", None, dict(cert, why="answer is not optimal under formalisation %d" % i)
        if kind == "unique":
            ob = run(c, {"op": "other", "assign": A})
            if ob.get("status") != "unsat":
                return "abstain", None, dict(cert, why="formalisation %d admits another answer" % i)
        cert["checks"].append({"formalisation": i, "accepts": True})
    return "certified", A, cert
