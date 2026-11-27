"""B46 (prereg/B46_second_llm_family_prereg.md): hidden-state codes of every entity label and alias from a frozen
non-Qwen LLM, for the encoder choice and for the replication of B36 on a second LLM family. Runs on the A800 box.

python b46_embed.py MODEL_PATH TAG --layers 8,16,24,32 [--device cuda:0] [--batch 64] [--out DIR]

Recipe = b32_embed.lm (B32's Qwen3-8B reader) with the two changes registered in the B46 pre-registration:
  1. the prompt is tokenized as one string with the tokenizer's default special tokens, so a model whose tokenizer
     prepends a BOS token by default (Llama 3.1, Mistral) reads it as it was trained to; Qwen3 has no BOS token, so its
     input is unchanged;
  2. the name's tokens are found by character offsets (tokens that start at or after the colon of 'Name:' and cover at
     least one character of the name), not by tokenizing ' ' + name separately; for byte-level BPE tokenizers this is
     the same token sequence, for SentencePiece tokenizers (Mistral) it avoids a spurious lone '_' token.
Two poolings x the given layers: mean of hidden_states[l] over the name's tokens in 'Name: {x}' ('mean'), and the last
token's hidden_states[l] of 'This name: "{x}" means in one word:"' ('eol', PromptEOL, Jiang et al. 2023);
hidden_states[0] is the embedding output, hidden_states[l] the output of block l. Rows are L2-normalized and saved as
float32 (DIR/emb_TAG_{mean,eol}_L{l}.npy); the string list is b32_data/emb_strings.json (B32's, unchanged).
"""
import json
import os
import sys
import time

import numpy as np


def load_tokenizer(path):
    """AutoTokenizer, as transformers loads the model's tokenizer by default. Fallback (registered): when AutoTokenizer
    insists on rebuilding the fast tokenizer from the SentencePiece model and sentencepiece is not installed (Mistral
    v0.3's tokenizer_config asks for this), the tokenizer.json shipped with the model is loaded directly, with the
    special tokens of special_tokens_map.json; its post-processor adds the BOS token as the model expects."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    try:
        return AutoTokenizer.from_pretrained(path), 'AutoTokenizer'
    except ValueError as e:
        if 'sentencepiece' not in str(e).lower() and 'slow' not in str(e).lower():
            raise
        sm = json.load(open(os.path.join(path, 'special_tokens_map.json')))
        kw = {k: (v['content'] if isinstance(v, dict) else v) for k, v in sm.items()
              if k in ('bos_token', 'eos_token', 'unk_token', 'pad_token')}
        return PreTrainedTokenizerFast(tokenizer_file=os.path.join(path, 'tokenizer.json'), **kw), 'tokenizer.json'


def main():
    a = sys.argv[1:]
    path, tag = a[0], a[1]
    layers = tuple(int(x) for x in a[a.index('--layers') + 1].split(','))
    device = a[a.index('--device') + 1] if '--device' in a else 'cuda:0'
    batch = int(a[a.index('--batch') + 1]) if '--batch' in a else 64
    out = a[a.index('--out') + 1] if '--out' in a else 'b32_data'
    import torch
    from transformers import AutoModelForCausalLM
    torch.backends.cuda.matmul.allow_tf32 = False
    S = json.load(open('b32_data/emb_strings.json'))
    tok, loader = load_tokenizer(path)
    probe = tok('a')['input_ids']
    adds_bos = tok.bos_token_id is not None and len(probe) > 0 and probe[0] == tok.bos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else (
        tok.eos_token_id if isinstance(tok.eos_token_id, int) else 0)
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()
    nl = model.config.num_hidden_layers
    assert max(layers) <= nl, (layers, nl)
    print(f'{tag}: {path}; layers {nl}, hidden {model.config.hidden_size}; tokenizer ({loader}) adds BOS: {adds_bos}; '
          f'pad {pad}',
          flush=True)
    outs = {(p, l): np.zeros((len(S), model.config.hidden_size), np.float32) for p in ('mean', 'eol') for l in layers}
    PRE = 'Name: '
    ntok = []
    t0 = time.time()

    def run(texts):
        enc = [tok(t, return_offsets_mapping=True) for t in texts]
        L = max(len(e['input_ids']) for e in enc)
        inp = torch.full((len(enc), L), pad, dtype=torch.long)
        att = torch.zeros((len(enc), L), dtype=torch.long)
        for i, e in enumerate(enc):
            inp[i, :len(e['input_ids'])] = torch.tensor(e['input_ids'])
            att[i, :len(e['input_ids'])] = 1
        hs = model(input_ids=inp.to(device), attention_mask=att.to(device), output_hidden_states=True).hidden_states
        return enc, att, hs

    with torch.no_grad():
        for b0 in range(0, len(S), batch):
            chunk = S[b0:b0 + batch]
            # mean pooling over the name's tokens in 'Name: {x}'
            enc, att, hs = run([PRE + s for s in chunk])
            msk = torch.zeros(att.shape, dtype=torch.float32)
            for i, e in enumerate(enc):
                for j, (st, en) in enumerate(e['offset_mapping']):
                    if st >= len(PRE) - 1 and en > len(PRE) and e['input_ids'][j] not in tok.all_special_ids:
                        msk[i, j] = 1
                n = int(msk[i].sum())
                assert n > 0, (chunk[i], e['input_ids'], e['offset_mapping'])
                ntok.append(n)
            m = msk.to(device).unsqueeze(-1)
            for l in layers:
                v = (hs[l].float() * m).sum(1) / m.sum(1)
                outs['mean', l][b0:b0 + len(chunk)] = v.cpu().numpy()
            # PromptEOL: last token of 'This name: "{x}" means in one word:"'
            enc, att, hs = run([f'This name: "{s}" means in one word:"' for s in chunk])
            last = att.sum(1) - 1
            for l in layers:
                v = hs[l].float()[torch.arange(len(chunk)), last.to(device)]
                outs['eol', l][b0:b0 + len(chunk)] = v.cpu().numpy()
            if b0 % (batch * 50) == 0:
                print(b0, len(S), f'{time.time() - t0:.0f}s', flush=True)
    os.makedirs(out, exist_ok=True)
    for (p, l), X in outs.items():
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        assert np.isfinite(X).all(), (p, l)
        np.save(f'{out}/emb_{tag}_{p}_L{l}.npy', X.astype(np.float32))
    # one example of the tokenization, for the record
    e = tok(PRE + S[0], return_offsets_mapping=True)
    info = {'tag': tag, 'path': path, 'tokenizer_loader': loader, 'layers': list(layers), 'num_hidden_layers': nl,
            'adds_bos': adds_bos,
            'pad': pad, 'name_tokens_mean': float(np.mean(ntok)), 'example': S[0],
            'example_tokens': tok.convert_ids_to_tokens(e['input_ids']), 'seconds': round(time.time() - t0, 1)}
    json.dump(info, open(f'{out}/embinfo_{tag}.json', 'w'), indent=1, ensure_ascii=False)
    print('done', json.dumps(info, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
