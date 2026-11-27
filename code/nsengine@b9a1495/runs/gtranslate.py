"""Generic translator prompt (design v1, G1): English problem -> CPMpy model under the engine/gm.py contract.

One prompt for every problem family. The worked examples come from a separate pool written for this file (timetabling,
coin change, an impossible schedule); none is from a development or test family (SRE, ZebraLogic, CP-Bench, ...).
A family adapter supplies three things: the problem text, the answer keys (named slots, possibly a pattern such as
"down:<service>"), and the question kind for the certification layer (engine/gcert.py).
"""
import re

GUIDE = """You turn a problem written in English into a constraint model in Python with CPMpy, for a solver to solve. You do not solve the problem yourself.

The problem's sentences are numbered [S1], [S2], ...

Write Python code. `from cpmpy import *` and `import numpy as np` are already done, and a CPMpy `model = Model()` exists.
- Declare decision variables with intvar / boolvar, and give each one a name.
- Add every constraint with add(<constraint>, "S<i>"), naming the sentence(s) it comes from: add(x + y <= 5, "S3"), add(c, "S2,S7"). Every fact or rule in the text that matters for the question must become constraints.
- Use model.minimize(...) or model.maximize(...) only when the question asks for a smallest or largest.
- Account for every sentence: each one is cited by add(), or marked with uses("S<i>") when it shapes the objective or the answer keys without a constraint, or marked with ignore("S<i>") when it does not matter for the question.
- At the end, set answer = {...} with exactly the keys described under "Answer keys", each mapped to the variable or expression whose value answers that key.
- Do not call solve, do not print, and do not hard-code an answer: the solver finds it.
- When the question asks whether something is possible, write the model so that it has no solution exactly when it is impossible.
Reply with the code only, in one ```python block."""

EXAMPLES = [
    ("""[S1] Three exams, Algebra, Biology and Chemistry, must each be placed in one of the time slots 1, 2 and 3.
[S2] Ana takes Algebra and Biology, so those two cannot share a slot.
[S3] Chemistry cannot be in slot 1 because the lab is booked.
[S4] Biology must come before Chemistry.
[S5] Question: give a slot for each exam.""",
     "slot:<exam> for each exam: its slot (1, 2 or 3).",
     '''exams = ["Algebra", "Biology", "Chemistry"]
slot = {e: intvar(1, 3, name="slot_" + e) for e in exams}
add(slot["Algebra"] != slot["Biology"], "S2")
add(slot["Chemistry"] != 1, "S3")
add(slot["Biology"] < slot["Chemistry"], "S4")
uses("S5")
answer = {"slot:" + e: slot[e] for e in exams}'''),
    ("""[S1] A machine pays out coins worth 1, 5 and 12 cents.
[S2] It must pay exactly 16 cents.
[S3] The machine was serviced in May.
[S4] Question: with the fewest coins, how many of each coin does it pay?""",
     "count:<value> for each coin value (1, 5, 12): how many of that coin. Minimise the total number of coins.",
     '''values = [1, 5, 12]
count = {v: intvar(0, 16, name="count_%d" % v) for v in values}
add(sum(v * count[v] for v in values) == 16, "S1,S2")
ignore("S3")
model.minimize(sum(count.values()))
uses("S4")
answer = {"count:%d" % v: count[v] for v in values}'''),
    ("""[S1] Jobs P, Q and R run on one machine, one at a time, each taking 2 hours.
[S2] The machine is free from hour 0 to hour 5.
[S3] Q needs the output of P, so it starts after P ends.
[S4] Question: can all three jobs be finished by hour 5? If yes, give each start hour; if not, say it is impossible.""",
     "start:<job> for each job: its start hour.",
     '''jobs = ["P", "Q", "R"]
start = {j: intvar(0, 5, name="start_" + j) for j in jobs}
for j in jobs:
    add(start[j] + 2 <= 5, "S1,S2")
for i, a in enumerate(jobs):
    for b in jobs[i + 1:]:
        add((start[a] + 2 <= start[b]) | (start[b] + 2 <= start[a]), "S1")
add(start["Q"] >= start["P"] + 2, "S3")
uses("S4")
answer = {"start:" + j: start[j] for j in jobs}'''),
]


def number_sentences(text):
    """Every line is a unit (heading lines dropped, list markers such as "- " or "3. " removed); a line with several
    sentences is split at sentence ends. Returns (numbered text "[S1] ..", list of sentences)."""
    parts = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^(?:[-*]\s+|\d+[.)]\s+)", "", line)
        parts += [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Za-z\[(\"'`])", line) if x.strip()]
    return "\n".join("[S%d] %s" % (i + 1, x) for i, x in enumerate(parts)), parts


def prompt(text, answer_keys):
    numbered, _ = number_sentences(text)
    shots = "\n\n".join("Example %d\nProblem:\n%s\nAnswer keys: %s\n```python\n%s\n```" % (i + 1, p, k, c)
                        for i, (p, k, c) in enumerate(EXAMPLES))
    return (GUIDE + "\n\n" + shots + "\n\nNow the real problem.\nProblem:\n" + numbered + "\nAnswer keys: " + answer_keys)


def extract_code(reply):
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", reply or "", re.S)
    return m[-1] if m else None


# ------------------------------------------------------------------------------------------------ family adapters
def zebra_task(it):
    """(text, answer keys, kind) for a ZebraLogic grid puzzle."""
    attrs = eval(it["attributes"]) if isinstance(it["attributes"], str) else it["attributes"]
    n = int(str(it["size"]).split("*")[0])
    return (it["text"], ("house:<Attribute>:<value> for every value of every attribute: the number (1 to %d) of the house "
                         "it belongs to. Use these attribute names: %s; write each value exactly as the puzzle lists it."
                         % (n, ", ".join(attrs))), "unique")


def zebra_grid(it, ans):
    """Engine answer {"house:Attr:value": h} -> the scorer's grid."""
    n = int(str(it["size"]).split("*")[0])
    grid = {"House %d" % h: {} for h in range(1, n + 1)}
    for k, h in ans.items():
        parts = k.split(":", 2)
        if len(parts) == 3 and isinstance(h, int) and 1 <= h <= n:
            grid["House %d" % h][parts[1]] = parts[2]
    return {"grid": grid}


def sre_task(it):
    """(text, answer keys, kind) for an SRE item: runbook + rules + question; nothing SRE-specific reaches the solver."""
    q = it["query"]["type"]
    text = it["text"] + "\n\n" + it["rules"] + "\n\nQuestion: " + it["question"]
    if q == "blast_radius":
        return text, ("down:<service> for every service named in the runbook: 1 if that service is unavailable, "
                      "0 if it is available."), "unique"
    if q == "min_failure_sets":
        return text, ("fail:<host> for every host named in the runbook: 1 if that host fails. Minimise the number of "
                      "failed hosts; every answer that reaches the minimum will be listed."), "all_opt"
    return text, ("wave:<host> for every host to be patched: the wave it is patched in, numbered from 0. If no plan "
                  "exists, the model must have no solution."), "find"


def generic_task(it):
    """(text, answer keys, kind) for an item that states them itself (held-out families, bench/heldout/)."""
    kind = {"check": "find"}.get(it["kind"], it["kind"])
    return it["text"], it["answer_keys"], kind
