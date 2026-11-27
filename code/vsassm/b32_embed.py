"""B32: continuous codes of every entity label and alias, from a frozen encoder (the 'reader' of B32).

python3 b32_embed.py st  MODEL TAG            # sentence-transformers encoder, CPU
python3 b32_embed.py lm  MODEL_PATH TAG [--device cuda:0] [--layers 12,18,24,30] [--batch 64]
                                              # frozen causal LM: two poolings x several layers, all saved

Output: b32_data/emb_TAG.npy (float32, one row per string, L2-normalized) and b32_data/emb_strings.json (the string
list, shared by every encoder; built on the first call). For the LM, one file per (pooling, layer):
emb_TAG_mean_L{l}.npy (mean of the hidden states over the name's tokens in 'Name: {x}') and emb_TAG_eol_L{l}.npy
(the last token's state of 'This name: "{x}" means in one word:"', the PromptEOL recipe of Jiang et al. 2023).
"""
import json
import os
import sys

import numpy as np

ENT = 'b32_data/entities.json'
STR = 'b32_data/emb_strings.json'


def strings():
    if os.path.exists(STR):
        return json.load(open(STR))
    d = json.load(open(ENT))
    out, seen = [], set()
    for e in d['entities']:
        for s in [e['name']] + e['aliases']:
            if s not in seen:
                seen.add(s)
                out.append(s)
    json.dump(out, open(STR, 'w'), ensure_ascii=False)
    return out


def st(model, tag):
    sys.path.insert(0, '/path/to/HOME/venvs/b32_site')
    from sentence_transformers import SentenceTransformer
    S = strings()
    m = SentenceTransformer(model, device='cpu')
    pre = 'query: ' if 'e5' in model else ''
    X = m.encode([pre + s for s in S], batch_size=256, normalize_embeddings=True, show_progress_bar=False,
                 convert_to_numpy=True).astype(np.float32)
    np.save(f'b32_data/emb_{tag}.npy', X)
    print(tag, X.shape)


def lm(path, tag, device='cuda:0', layers=(12, 18, 24, 30), batch=64):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    S = strings()
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = 'right'
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()
    outs = {(p, l): np.zeros((len(S), model.config.hidden_size), np.float32) for p in ('mean', 'eol') for l in layers}
    with torch.no_grad():
        for b0 in range(0, len(S), batch):
            chunk = S[b0:b0 + batch]
            # mean pooling over the name's tokens in 'Name: {x}'
            pre = tok('Name:', add_special_tokens=False)['input_ids']
            ids = [pre + tok(' ' + s, add_special_tokens=False)['input_ids'] for s in chunk]
            L = max(map(len, ids))
            inp = torch.full((len(ids), L), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(ids), L), dtype=torch.long)
            msk = torch.zeros((len(ids), L), dtype=torch.float32)
            for i, x in enumerate(ids):
                inp[i, :len(x)] = torch.tensor(x)
                att[i, :len(x)] = 1
                msk[i, len(pre):len(x)] = 1
            hs = model(input_ids=inp.to(device), attention_mask=att.to(device), output_hidden_states=True).hidden_states
            m = msk.to(device).unsqueeze(-1)
            for l in layers:
                v = (hs[l].float() * m).sum(1) / m.sum(1)
                outs['mean', l][b0:b0 + len(chunk)] = v.cpu().numpy()
            # PromptEOL: last token of 'This name: "{x}" means in one word:"'
            ids = [tok(f'This name: "{s}" means in one word:"', add_special_tokens=False)['input_ids'] for s in chunk]
            L = max(map(len, ids))
            inp = torch.full((len(ids), L), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(ids), L), dtype=torch.long)
            for i, x in enumerate(ids):
                inp[i, :len(x)] = torch.tensor(x)
                att[i, :len(x)] = 1
            hs = model(input_ids=inp.to(device), attention_mask=att.to(device), output_hidden_states=True).hidden_states
            last = att.sum(1) - 1
            for l in layers:
                v = hs[l].float()[torch.arange(len(ids)), last.to(device)]
                outs['eol', l][b0:b0 + len(chunk)] = v.cpu().numpy()
            if b0 % (batch * 50) == 0:
                print(b0, len(S), flush=True)
    for (p, l), X in outs.items():
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        np.save(f'b32_data/emb_{tag}_{p}_L{l}.npy', X.astype(np.float32))
    print('done', tag, len(S))


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'st':
        st(a[1], a[2])
    elif a[0] == 'lm':
        kw = {}
        for k in ('--device', '--layers', '--batch'):
            if k in a:
                v = a[a.index(k) + 1]
                kw[k[2:]] = tuple(int(x) for x in v.split(',')) if k == '--layers' else (int(v) if k == '--batch' else v)
        lm(a[1], a[2], **kw)
