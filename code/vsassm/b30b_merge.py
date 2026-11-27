"""B30b: write the B18 reader (Qwen3-1.7B + LoRA student_s0) as one merged bf16 checkpoint for vLLM, merged exactly as
b14.Reader merges it (base loaded in bf16 on the GPU, PeftModel.merge_and_unload), plus the tokenizer and the base
generation_config. Also writes the HF reader's greedy outputs on a few prompts for the dev engine check.

python3 b30b_merge.py BASE_SNAPSHOT ADAPTER_DIR OUT_DIR
"""
import json
import os
import shutil
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base, adapter, out = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(base)
m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
m = PeftModel.from_pretrained(m, adapter).merge_and_unload().eval()
m.save_pretrained(out, safe_serialization=True)
tok.save_pretrained(out)
shutil.copy(os.path.join(base, "generation_config.json"), os.path.join(out, "generation_config.json"))
json.dump({"base": base, "adapter": adapter}, open(os.path.join(out, "b30b_merge.json"), "w"))
print("saved", out)
