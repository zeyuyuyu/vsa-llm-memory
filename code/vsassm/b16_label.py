"""B16 step 2: the B15 reader (Qwen3-8B, prompt C, no merging) labels every sentence; one example per reader call.
python3 b16_label.py --model <Qwen3-8B> --stories b16_data/stories_loc_0.jsonl --out b16_data/labels_loc_0.jsonl
"""
import argparse
import json

import b14
from readable_memory import split_sentences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stories", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="C")  # B17: D
    args = ap.parse_args()
    b14.MERGE = "none"
    b14.B17 = args.prompt == "D"
    stories = [json.loads(l) for l in open(args.stories)]
    rec = []
    b14.run_stories(b14.Reader(args.model), [{"domain": s["domain"], "sentences": split_sentences(s["text"])}
                                             for s in stories], args.prompt, record=rec)
    with open(args.out, "w") as f:
        for r in rec:
            f.write(json.dumps(r) + "\n")
    print(len(rec), "examples")


if __name__ == "__main__":
    main()
