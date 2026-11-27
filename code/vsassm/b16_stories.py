"""B16 (prereg/B16_prereg.md) step 1: Qwen3-8B writes free stories (no event list) for the reader to label.
python3 b16_stories.py --model <Qwen3-8B> --domain locations --n 1500 --seed 0 --out b16_data/stories_loc_0.jsonl
"""
import argparse
import json
import os
import random

from b13_distill import LM, SCENES

STYLES = ["a diary entry", "a few chat messages to a friend", "a short email to a colleague", "a matter-of-fact report",
          "a children's story", "a plain narrative", "a note left for a housemate", "a hurried voice memo transcript"]
WHAT = {
    "locations": "people put, move, hang, drop, pack, leave or swap everyday objects in or on places (drawers, shelves, "
                 "rooms, bags, tables, cars)",
    "containers": "people put everyday items inside containers, put those containers inside bigger ones (a ring in a "
                  "box in a bag in a suitcase), and take containers out of one another",
}


EXTRA17 = [  # B17: what each new story must also contain (one or two of these)
    "tell one fact out of order, after the event it came before (for example: \"Before that, the drawer had held the socks.\")",
    "have a person take, hold or be handed an object (for example: \"Ben grabbed the keys\", \"she gave the cup to Ana\")",
    "use passive voice and put the place first in some sentences (for example: \"Into the bag went the charger.\")",
    "describe a swap of everything in two places in an unusual way (for example: \"the two boxes traded contents\")",
]


def prompt(domain, rng, b17=False):
    n = rng.randint(5, 10)
    return (f"Write {rng.choice(STYLES)}, about {n} sentences long, set in {rng.choice(SCENES)}, in which {WHAT[domain]}. "
            f"Use {rng.randint(3, 6)} different objects. Make the writing natural and varied: refer back to things "
            f"with pronouns (it, them, there), say at least once where something already is, include an unrelated "
            f"action, a plan that does not happen, and a sentence with two moves."
            + (" Also " + " and ".join(rng.sample(EXTRA17, rng.randint(1, 2))) + "." if b17 else "")
            + " Output only the text.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--domain", choices=list(WHAT), required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--out", required=True)
    ap.add_argument("--b17", action="store_true")  # B17: flashbacks, holders, varied wording
    args = ap.parse_args()
    rng = random.Random(16_000 + args.seed)
    lm = LM(args.model)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = sum(1 for _ in open(args.out)) if os.path.exists(args.out) else 0
    with open(args.out, "a") as f:
        while done < args.n:
            ps = [prompt(args.domain, rng, args.b17) for _ in range(args.bs)]
            for p, txt in zip(ps, lm.gen(ps, sample=True, max_new=500)):
                txt = txt.strip().strip('"')
                if 80 < len(txt) < 2500:
                    f.write(json.dumps({"domain": args.domain, "text": txt, "prompt": p}) + "\n")
                    done += 1
            f.flush()
            print(f"{done}/{args.n}", flush=True)


if __name__ == "__main__":
    main()
