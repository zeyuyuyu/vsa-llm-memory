import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset

HEX = "0123456789ABCDEF"


def int_to_hex2(n: int) -> str:
    return HEX[(n >> 4) & 0xF] + HEX[n & 0xF]


def make_role_token(role_idx: int) -> str:
    return f"[R{int_to_hex2(role_idx)}]"


def make_filler_token(filler_idx: int) -> str:
    # numeric filler {123}
    return f"{{{filler_idx}}}"


@dataclass
class SystematicConfig:
    num_roles: int = 64
    num_fillers: int = 256
    max_length: int = 768
    holdout_mod: int = 4
    p_positive: float = 0.5


def is_heldout_pair(role_idx: int, filler_idx: int, cfg: SystematicConfig) -> bool:
    return (role_idx + filler_idx) % cfg.holdout_mod == 0


class SystematicRoleBindingDataset(Dataset):
    """Strict systematic split.

    Train: no heldout (role,filler) appears anywhere.
    Eval: query pair is heldout and appears in context.

    Format: [Rxx]{j} ... Q [Rqq]{cand} ?
    """

    def __init__(self, tokenizer, num_samples: int, split: str, cfg: SystematicConfig):
        assert split in ("train", "eval")
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.split = split
        self.cfg = cfg

        self.train_allowed: List[List[int]] = []
        self.eval_heldout: List[List[int]] = []
        for r in range(cfg.num_roles):
            allowed = [f for f in range(cfg.num_fillers) if not is_heldout_pair(r, f, cfg)]
            heldout = [f for f in range(cfg.num_fillers) if is_heldout_pair(r, f, cfg)]
            if not allowed or not heldout:
                raise ValueError(f"Bad split role={r} allowed={len(allowed)} heldout={len(heldout)}")
            self.train_allowed.append(allowed)
            self.eval_heldout.append(heldout)

    def __len__(self):
        return self.num_samples

    def _sample_context(self, rng: random.Random, query_role: int, query_filler: int) -> Dict[int, int]:
        cfg = self.cfg
        role_to_filler: Dict[int, int] = {}
        for r in range(cfg.num_roles):
            if self.split == "train":
                f = rng.choice(self.train_allowed[r])
            else:
                f = query_filler if r == query_role else rng.choice(self.train_allowed[r])
            role_to_filler[r] = f
        return role_to_filler

    def _render_example(self, rng: random.Random, role_to_filler: Dict[int, int], query_role: int, cand_filler: int) -> str:
        items = list(role_to_filler.items())
        rng.shuffle(items)
        bindings = " ".join([make_role_token(r) + make_filler_token(f) for r, f in items])
        query = make_role_token(query_role) + make_filler_token(cand_filler)
        return bindings + " Q " + query + " ?"

    def __getitem__(self, idx: int):
        cfg = self.cfg
        rng = random.Random(1000003 * idx + (42 if self.split == "train" else 123))

        query_role = rng.randrange(cfg.num_roles)
        true_filler = rng.choice(self.train_allowed[query_role] if self.split == "train" else self.eval_heldout[query_role])

        role_to_filler = self._sample_context(rng, query_role, true_filler)

        label = 1 if rng.random() < cfg.p_positive else 0
        if label == 1:
            cand = true_filler
        else:
            wrong_role = rng.choice([r for r in range(cfg.num_roles) if r != query_role])
            cand = role_to_filler[wrong_role]
            if cand == true_filler:
                cand = rng.choice([f for f in range(cfg.num_fillers) if f != true_filler])

        text = self._render_example(rng, role_to_filler, query_role, cand)

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
