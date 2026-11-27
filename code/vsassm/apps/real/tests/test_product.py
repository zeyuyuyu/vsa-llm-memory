"""Engineering tests for the Readable Memory product (apps/real/product.py) and its SDK (readable_memory.SiteMemory).
No GPU and no model: the reader is a deterministic fake that answers from the logs' true events and flips some of
them, so wrong reads are exercised too.

    python3 -m pytest -q apps/real/tests
"""
import hashlib
import json
import os
import re
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "apps", "real"), ROOT]
os.chdir(ROOT)  # run_real.load reads apps/real/data relative to the repo
import run_real as rr  # noqa: E402
import product  # noqa: E402
from readable_memory import SiteMemory  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "apps", "real", "tools"))
import capacity  # noqa: E402

h = lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16)  # noqa: E731
LOGS = rr.load("test2")
TRUTH = {rr.message(l, x): x["events"] for l in LOGS for x in l["lines"] if x["kind"] == "human"}


def fake_reader(seen):
    def read(prompts):
        out = []
        for p in prompts:
            seen.append(p)
            if p.startswith("Message:") and "Choose one:" in p:
                out.append(("AB"[h(p) % 2], 1))
            elif p.startswith("Message:"):
                out.append(("yes" if h(p) % 5 else "no", 1))
            else:
                sent = re.findall(r"\nSentence: ([^\n]*)", p)[-1]
                evs = TRUTH.get(sent, [])
                if h(p) % 7 == 0 and evs:
                    evs = [[e[0], e[2], e[1]] for e in evs]
                out.append((json.dumps({"events": evs}), 3))
        return out
    return read


def lines(log):
    return [(rr.message(log, x) if x["kind"] == "human" else t, x["src"]) for t, x in zip(log["text_lines"], log["lines"])]


def new_memory(log, seen, memory):
    m = SiteMemory(registry=log["registry"], domain=log["domain"], connector=log["scenario"], reader=fake_reader(seen),
                   memory="single" if memory == "single" else "sharded")
    if memory == "tiny":  # 3 items a shard, a rebuild at 5 value codes: shards and rebuilds all day
        m.site.mem = product.ShardedMemory(log["domain"], m.b14, keys=3, values=5)
    return m


def outcome(m, log):
    return [m.site.ask(q["q"], q["kind"]) for q in log["questions"]], m.site.board(), m.alarms()


def tensors(m):
    mems = m.site.mem.shards if isinstance(m.site.mem, product.ShardedMemory) else [m.site.mem]
    return [(x.S, x.K.codes, x.V.codes) for x in mems]


# ------------------------------------------------------------------------------------------------ save and load
@pytest.mark.parametrize("memory", ["single", "sharded", "tiny"])
def test_resume_after_save_is_identical(memory, tmp_path):
    """Save half-way through every test2 log, load into a new process's object, read the rest: the reader's prompts,
    every answer, the board and the state matrices are those of the uninterrupted run, bit for bit."""
    for log in LOGS:
        ls = lines(log)
        seen_a = []
        a = new_memory(log, seen_a, memory)
        for t, ch in ls:
            a.observe(t, channel=ch)
        seen_b = []
        b = new_memory(log, seen_b, memory)
        for t, ch in ls[:len(ls) // 2]:
            b.observe(t, channel=ch)
        path = tmp_path / f"{log['id']}.pt"
        b.save(path)
        del b
        c = SiteMemory.load(path, reader=fake_reader(seen_b))
        for t, ch in ls[len(ls) // 2:]:
            c.observe(t, channel=ch)
        assert seen_b == seen_a, log["id"]
        assert outcome(c, log) == outcome(a, log), log["id"]
        ta, tc = tensors(a), tensors(c)
        assert len(ta) == len(tc)
        assert all(torch.equal(x, y) for p, q in zip(ta, tc) for x, y in zip(p, q)), log["id"]


def test_load_takes_a_newer_asset_list(tmp_path):
    """A site adds a pallet after the save: the loaded memory keeps every belief and resolves the new pallet."""
    log = next(l for l in LOGS if l["scenario"] == "warehouse")
    m = new_memory(log, [], "single")
    for t, ch in lines(log):
        m.observe(t, channel=ch)
    before = m.beliefs()
    m.save(tmp_path / "w.pt")
    reg = json.loads(json.dumps(log["registry"]))
    reg["items"].append({"id": "LPN 0777001", "aliases": ["the new pallet"]})
    n = SiteMemory.load(tmp_path / "w.pt", reader=fake_reader([]), registry=reg)
    assert n.beliefs() == before
    n.observe("06:30  RF-01  RECEIVE   LPN 0777001  @ DOOR 1  user=test")
    assert n.ask("Where is the new pallet?")["answer"] == "DOOR 1"


def test_load_refuses_other_settings(tmp_path):
    log = LOGS[0]
    m = new_memory(log, [], "single")
    m.save(tmp_path / "s.pt")
    st = torch.load(tmp_path / "s.pt", weights_only=True)
    st["site"]["b14"]["MERGE"] = "suffix"
    torch.save(st, tmp_path / "s2.pt")
    with pytest.raises(ValueError, match="b14 settings"):
        SiteMemory.load(tmp_path / "s2.pt", reader=fake_reader([]))
    st["site"]["version"] = 99
    torch.save(st, tmp_path / "s3.pt")
    with pytest.raises(ValueError, match="version"):
        SiteMemory.load(tmp_path / "s3.pt", reader=fake_reader([]))


def test_saved_size_does_not_grow_with_lines(tmp_path):
    """Re-reading lines that name nothing new leaves the saved memory the same size."""
    log = next(l for l in LOGS if l["scenario"] == "warehouse")
    m = new_memory(log, [], "single")
    struct = [t for t, x in zip(log["text_lines"], log["lines"]) if x["kind"] == "struct"]
    for t in struct:
        m.observe(t)
    m.save(tmp_path / "a.pt")
    for _ in range(5):
        for t in struct:
            m.observe(t)
    m.save(tmp_path / "b.pt")
    assert os.path.getsize(tmp_path / "a.pt") == os.path.getsize(tmp_path / "b.pt")


# ------------------------------------------------------------------------------------------------ connectors
@pytest.mark.parametrize("split", ["dev", "test", "test2"])
def test_connectors_give_every_system_record_its_events(split):
    """Every system line of every log: the connector's events change the true state exactly as the line's true events
    do (a cycle count re-states where a pallet already is: no true event, a no-op write), and no human line is taken
    for a system record."""
    def step(state, evs):
        s = dict(state)
        for op, a, b in evs:
            if op == "move":
                s[a] = b
            else:  # swap: everything in a goes to b and back
                s = {o: b if p == a else a if p == b else p for o, p in s.items()}
        return s

    bad, n = [], 0
    for log in rr.load(split):
        state = {}
        for t, x in zip(log["text_lines"], log["lines"]):
            got = product.connect(log["scenario"], t)
            if x["kind"] == "struct":
                n += 1
                if got is None or step(state, got) != step(state, x["events"]):
                    bad.append((log["id"], t, got, x["events"]))
            elif got is not None:
                bad.append((log["id"], t, got, "human line"))
            state = step(state, x["events"])
    assert n and not bad, bad[:5]


# ------------------------------------------------------------------------------------------------ the registry
def registry(digit_tails):
    items = [{"id": i, "aliases": []} for i in ("db1021", "es1021", "LPN 0435871")]
    return {"registry": {"items": items, "places": [{"id": "s4 eqiad primary", "aliases": ["s4 eqiad"]}],
                         "digit_tails": digit_tails}, "domain": "locations"}


def test_digit_tails_off_keeps_prefixed_ids_apart():
    r = product.Registry(registry(False))
    assert r.resolve("es1021", r.items) == "es1021"
    assert r.resolve("db1021", r.items) == "db1021"
    assert r.mentioned("Promote es1021 to es4 primary") == {"es1021"}
    assert r.resolve("5871", r.items) is None


def test_digit_tails_on_matches_a_number_that_ends_one_id():
    r = product.Registry(registry(True))
    assert r.resolve("5871", r.items) == "LPN 0435871"
    assert r.resolve("1021", r.items) is None  # ends two IDs: not a name
    assert r.resolve("s4 eqiad", r.places) == "s4 eqiad primary"


# ------------------------------------------------------------------------------------------------ capacity
@pytest.mark.parametrize("n_items,n_places", [(1024, 40), (16, 1024)])
def test_sharded_memory_is_exact_past_one_state(n_items, n_places):
    """Past 256 items or 256 places one b14.Memory goes wrong; ShardedMemory stays exact (true events, no reader)."""
    r = capacity.run(n_items, n_places, "sharded")
    assert r["where"] == 1.0 and r["what_in"] == 1.0, r


# ------------------------------------------------------------------------------------------------ provenance and alarm
def small_site():
    reg = {"items": [{"id": "LPN 0435871", "aliases": ["the Dyson pallet"]}, {"id": "LPN 0400002", "aliases": []}],
           "places": [{"id": p, "aliases": []} for p in ("DOOR 2", "A-10-02", "B-11-03", "QA HOLD")]}
    return SiteMemory(registry=reg, connector="warehouse", reader=lambda p: [])


def test_alarm_when_a_person_overrides_the_system_and_clears_on_the_next_record():
    m = small_site()
    m.observe("06:05  RF-03  RECEIVE   LPN 0435871  @ DOOR 2  user=kowalski")
    m.site.apply([["move", "the Dyson pallet", "B-11-03"]], from_reader=True, channel="chat",
                 text="[08:25] Nguyen: Dyson pallet is in B-11-03 now")
    assert m.ask("Where is LPN 0435871?")["answer"] == "B-11-03"  # the alarm changes no answer
    al = m.alarms()
    assert set(al) == {"LPN 0435871"}
    assert al["LPN 0435871"]["person_said"] == "B-11-03" and al["LPN 0435871"]["system_said"] == "DOOR 2"
    assert m.why("the Dyson pallet")["written"]["line"].endswith("in B-11-03 now")
    m.observe("09:10  RF-03  PUTAWAY   LPN 0435871  B-11-03 -> A-10-02  user=kowalski")
    assert m.alarms() == {}
    assert m.why("LPN 0435871")["written"]["by"] == "system"


def test_no_alarm_when_the_person_agrees_or_no_system_record_exists():
    m = small_site()
    m.observe("06:05  RF-03  RECEIVE   LPN 0435871  @ DOOR 2  user=kowalski")
    m.site.apply([["move", "LPN 0435871", "DOOR 2"]], from_reader=True, channel="chat", text="5871 still at door 2")
    m.site.apply([["move", "LPN 0400002", "QA HOLD"]], from_reader=True, channel="chat", text="0400002 to QA")
    assert m.alarms() == {}
    assert m.why("LPN 0400002")["written"]["by"] == "reader"


def test_why_reports_a_swap_after_the_write():
    m = small_site()
    m.observe("06:05  RF-03  RECEIVE   LPN 0435871  @ DOOR 2  user=kowalski")
    m.site.apply([["swap", "DOOR 2", "QA HOLD"]], from_reader=True, channel="chat",
                 text="swapped everything between door 2 and QA")
    w = m.why("LPN 0435871")
    assert w["belief"] == "QA HOLD" and w["then_swapped"]["other"] == "DOOR 2"


def test_correction_clears_an_alarm():
    m = small_site()
    m.observe("06:05  RF-03  RECEIVE   LPN 0435871  @ DOOR 2  user=kowalski")
    m.site.apply([["move", "LPN 0435871", "B-11-03"]], from_reader=True, channel="chat", text="5871 to B-11-03")
    assert m.alarms()
    m.correct("LPN 0435871", "B-11-03")
    assert m.alarms() == {}
