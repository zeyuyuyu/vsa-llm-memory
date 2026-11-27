import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

HEX = "0123456789ABCDEF"


def int_to_hex2(n: int) -> str:
    return HEX[(n >> 4) & 0xF] + HEX[n & 0xF]


def marker(kind: str, idx: int) -> str:
    # kind in {R,X,Y}
    return f"[{kind}{int_to_hex2(idx)}]"


def filler(n: int) -> str:
    return f"{{{n}}}"


@dataclass
class ChainConfig:
    num_roles: int = 64
    vocab_size: int = 256  # for X/Y symbols
    num_pairs: int = 16
    max_length: int = 512
    p_positive: float = 0.5


class SystematicChainDataset(Dataset):
    """3-hop retrieval dataset.

    Context contains:
      A: [Rrr][Xxx] for many roles
      B: [Xxx][Yyy]
      C: [Yyy]{f}
    Query:
      Q [Rrr]{cand} ?

    Answer requires 3 hops: R -> X -> Y -> f.
    """

    def __init__(self, tokenizer, num_samples: int, split: str, cfg: ChainConfig):
        assert split in ("train", "eval")
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.split = split
        self.cfg = cfg

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        cfg = self.cfg
        rng = random.Random(1000003 * idx + (42 if self.split == "train" else 123))

        # choose a subset of roles for this example to keep context size bounded
        roles = list(range(cfg.num_roles))
        rng.shuffle(roles)
        roles = roles[: cfg.num_pairs]

        xs = rng.sample(range(cfg.vocab_size), cfg.num_pairs)
        ys = rng.sample(range(cfg.vocab_size), cfg.num_pairs)
        fs = rng.sample(range(cfg.vocab_size), cfg.num_pairs)

        # build bijections
        A = list(zip(roles, xs))
        B = list(zip(xs, ys))
        C = list(zip(ys, fs))

        rng.shuffle(A)
        rng.shuffle(B)
        rng.shuffle(C)

        # pick query role
        q_role, q_x = rng.choice(A)
        # find y and f
        x_to_y = {x: y for x, y in B}
        y_to_f = {y: f_ for y, f_ in C}
        q_y = x_to_y[q_x]
        true_f = y_to_f[q_y]

        label = 1 if rng.random() < cfg.p_positive else 0
        if label == 1:
            cand = true_f
        else:
            # hard negative: choose a filler from another y
            other_fs = [f_ for f_ in fs if f_ != true_f]
            cand = rng.choice(other_fs) if other_fs else (true_f + 1) % cfg.vocab_size

        # render text (add spaces to avoid BPE merges like "][" and make parsing robust)
        a_txt = " ".join([marker("R", r) + " " + marker("X", x) for r, x in A])
        b_txt = " ".join([marker("X", x) + " " + marker("Y", y) for x, y in B])
        c_txt = " ".join([marker("Y", y) + " " + filler(f_) for y, f_ in C])
        q_txt = marker("R", q_role) + " " + filler(cand)

        text = "A " + a_txt + " B " + b_txt + " C " + c_txt + " Q " + q_txt + " ?"

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=cfg.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
