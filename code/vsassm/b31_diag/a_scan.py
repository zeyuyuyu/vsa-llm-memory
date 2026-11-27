# B31 pre-registration diagnostic (arena A train + dev only, CPU): where does the published exact-symbol ceiling (0.918) come from?
# Runs the ORIGINAL SymbolicVSAChainClassifier forward (published parser arm) at its initial readout (threshold 0.20) and
# counts instances the token-level scanner cannot parse; compares the scanner's markers with a regex parse of the text.
import sys, re, json, torch, numpy as np
sys.path.insert(0, "/path/to/work/b27")
import b27_common as C
M = C.load_arena_modules()
tok = C.tokenizer(); cfg = C.chain_cfg(M)
ECM = M["explicit_chain_models"]
out = {}
for split, n, off in (("train", 2000, 42), ("dev", 500, C.DEV_OFFSET)):
    torch.manual_seed(42)
    m = ECM.SymbolicVSAChainClassifier(vsa_dim=2048, num_roles=64, vocab_size=256, seed=42).eval()
    zero, acc, nm_bad, missing_pairs = 0, 0, 0, []
    for i in range(n):
        text, y = C.gen_chain(i, off, cfg)
        enc = tok(text, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
        ids = enc["input_ids"]
        with torch.no_grad():
            lg = m(ids, enc["attention_mask"])
        if float(lg.abs().sum()) == 0:
            zero += 1
        acc += int(int(lg.argmax(-1)) == y)
        ms = m._scan_markers(ids[0])
        nre = len(re.findall(r"\[[RXY][0-9A-F]{2}\]", text))
        if len(ms) != nre:
            nm_bad += 1
            if len(missing_pairs) < 5:
                toks = tok.convert_ids_to_tokens(ids[0].tolist())
                found = set(s for _, _, s, _ in ms)
                missing_pairs.append([t for t in toks[:200]][:0] or [(len(ms), nre)])
    out[split] = {"n": n, "unparsed_zero_logits": zero, "acc_at_init_readout": acc / n, "instances_marker_count_mismatch": nm_bad,
                  "examples": missing_pairs}
    print(split, json.dumps(out[split]), flush=True)
# which markers does the scanner miss? token strings around a missed marker (train, first 2000)
miss = {}
for i in range(300):
    text, y = C.gen_chain(i, 42, cfg)
    ids = tok(text)["input_ids"]
    toks = tok.convert_ids_to_tokens(ids)
    m2 = ECM.SymbolicVSAChainClassifier._scan_markers(type("o", (), {"tokenizer": tok})(), torch.tensor(ids))
    got = [(k, v) for k, v, _, _ in m2]
    want = [(k, int(h, 16)) for k, h in re.findall(r"\[([RXY])([0-9A-F]{2})\]", text)]
    if got != want:
        for k, v in want:
            if (k, v) not in got:
                key = "[%s%02X]" % (k, v)
                miss[key] = miss.get(key, 0) + 1
print("missed markers (first 300 train texts):", json.dumps(dict(sorted(miss.items(), key=lambda x: -x[1])[:30])), flush=True)
for key in list(miss)[:6]:
    print(key, tok.tokenize(" " + key + " "), flush=True)
