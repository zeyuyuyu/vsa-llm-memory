"""B40 second scorer (independent of b40_run.parse / b40_data.op_answer / the chain follower in b40_data).

For every prediction file: rebuild the gold answer from the PROMPT TEXT alone (task N: regex over the question and
own arithmetic; task R: regex over the story sentences and own chain following), re-normalise the saved raw output
with its own rules, and compare (a) gold vs the main scorer's gold, (b) correctness vs the main scorer's 'correct'.
usage: python b40_check.py <pred.jsonl> [...]   (prints one line per file and the first mismatches)
"""
import json, re, sys
import b40_data as D  # only to regenerate the prompt text of each id (the text is then parsed here)

ROOM_WORDS = None


def own_n_answer(q):
    m = re.search(r"What is (.+)\?", q)
    s = m.group(1)
    pats = [(r"^(\d+) times (\d+) mod (\d+)$", lambda a, b, c: (a * b) % c),
            (r"^(\d+) plus (\d+)$", lambda a, b: a + b),
            (r"^(\d+) // (\d+)$", lambda a, b: a // b),
            (r"^(\d+) mod (\d+)$", lambda a, b: a - b * (a // b)),
            (r"^the GCD of (\d+) and (\d+)$", None),
            (r"^the LCM of (\d+) and (\d+) mod (\d+)$", None),
            (r"^(\d+)\^2 mod (\d+)$", lambda a, b: pow(a, 2, b)),
            (r"^(\d+) AND (\d+)$", lambda a, b: a & b),
            (r"^(\d+) XOR (\d+)$", lambda a, b: a ^ b),
            (r"^(\d+) OR (\d+)$", lambda a, b: a | b)]
    for p, f in pats:
        mm = re.match(p, s)
        if not mm:
            continue
        v = [int(t) for t in mm.groups()]
        if "GCD" in p:
            a, b = v
            while b:
                a, b = b, a % b
            return a
        if "LCM" in p:
            a, b, c = v
            g, x = a, b
            while x:
                g, x = x, g % x
            return (a // g * b) % c
        return f(*v)
    raise ValueError(s)


def own_r_answer(user):
    story = user.split("Story: ", 1)[1].split("\nQuestion:")[0]
    q = re.search(r"Where is (?:the )?(\w+)\?", user).group(1)
    where = {}
    for sent in re.split(r"(?<=\.)\s+", story.strip()):
        m = re.fullmatch(r"The (\w+) is in the (\w+)\.", sent)
        if m:
            where[m.group(1)] = m.group(2); continue
        m = re.fullmatch(r"(\w+) has the (\w+)\.", sent)
        if m:
            where[m.group(2)] = m.group(1); continue
        m = re.fullmatch(r"(\w+) is in the (\w+)\.", sent)
        if m:
            where[m.group(1)] = m.group(2); continue
        raise ValueError(sent)
    x, n = q, 0
    while x in where:
        x = where[x]; n += 1
        assert n < 20
    return x, n


def own_norm(task, out, cot):
    t = out.strip()
    if task == "N":
        t = t.replace(",", "")
        if cot:
            seg = t.split("Final Answer")[-1] if "Final Answer" in t else None
            if seg is not None:
                nums = re.findall(r"\d+", seg)
                if nums:
                    return nums[0]
            nums = re.findall(r"\d+", t)
            return nums[-1] if nums else ""
        m = re.match(r"\D*?(\d+)", t)
        return m.group(1) if m else ""
    words = re.findall(r"[a-z]+", t.lower())
    if cot:
        low = t.lower()
        if "final answer" in low:
            seg = re.findall(r"[a-z]+", low.split("final answer")[-1])
            for w in seg:
                if w in ROOM_WORDS:
                    return w
        rw = [w for w in words if w in ROOM_WORDS]
        return rw[-1] if rw else ""
    return words[0] if words else ""


def check(path):
    global ROOM_WORDS
    rows = [json.loads(l) for l in open(path)]
    task = rows[0]["id"].split("-")[0]
    split = rows[0]["id"].split("-")[1]
    cot = path.split("/")[-1].split(".")[0].endswith("cot")  # cot.jsonl / N_cot.jsonl
    src = {r["id"]: r for r in D.gen(task, split if split != "dev_e2e" else "dev")}
    ROOM_WORDS = set(D.ROOMS)  # the answer vocabulary is part of the task definition
    gm = cm = 0
    ex = []
    acc = 0
    for x in rows:
        r = src[x["id"]]
        user = D.messages(r)[-1]["content"]
        if task == "N":
            g = str(own_n_answer(user))
        else:
            g, depth = own_r_answer(user)
            if depth != r["depth"]:
                gm += 1
        if g != x["gold"]:
            gm += 1; ex.append(("gold", x["id"], g, x["gold"]))
        ok = own_norm(task, x["out"], cot) == g
        acc += ok
        if ok != x["correct"]:
            cm += 1
            if len(ex) < 8:
                ex.append(("correct", x["id"], x["out"][-80:], x["pred"], g))
    print(f"{path}: n={len(rows)} acc_main={sum(x['correct'] for x in rows) / len(rows):.4f} acc_check={acc / len(rows):.4f} "
          f"gold_mismatch={gm} correct_mismatch={cm}")
    for e in ex[:8]:
        print("   ", e)
    return gm, cm


if __name__ == "__main__":
    tot = [0, 0]
    for p in sys.argv[1:]:
        g, c = check(p)
        tot[0] += g; tot[1] += c
    print("TOTAL gold_mismatch", tot[0], "correct_mismatch", tot[1])
