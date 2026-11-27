# B46 encoder record (before any memory comparison on the new codes)

Date: 2026-11-19, 06:55 box clock. Written after the encoder choice and before any B32 / B36 memory (VSA, index,
key-free families) has answered a dev or test query on the B46 codes.

## Encoding (A800 box; b46_box/b46_gpu.sh; logs copied to b46_out/box/)

- embedA, GPU 1, 22:21:05Z-22:25:51Z: Llama-3.1-8B-Instruct (AutoTokenizer, BOS added), Granite-3.3-8B-Instruct
  (AutoTokenizer, no BOS by default), Qwen3-8B recipe check (layer 9 only).
- embedM, GPU 7, 22:47:14Z-22:49:00Z: Mistral-7B-Instruct-v0.3 (download from ModelScope finished 06:35:59 box clock,
  all three shards' sha256 equal to ModelScope's listing; tokenizer.json loaded directly, BOS added, as registered).
- GPU time: 0.080 + 0.030 = 0.11 GPU-hours.

## Criterion (B32's b32_lm_select.py, unchanged; dev split; box CPU) for every registered variant

| model | mean L1/4 | mean L1/2 | mean L3/4 | mean L4/4 | eol L1/4 | eol L1/2 | eol L3/4 | eol L4/4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B-Instruct (8/16/24/32) | 0.5181 | 0.5929 | 0.6478 | 0.4452 | 0.2319 | 0.3058 | 0.6143 | 0.5609 |
| Mistral-7B-Instruct-v0.3 (8/16/24/32) | 0.5493 | 0.4805 | 0.5962 | 0.4693 | 0.3095 | 0.3485 | 0.6427 | 0.5804 |
| Granite-3.3-8B-Instruct (10/20/30/40) | 0.6199 | 0.6348 | **0.6538** | 0.3713 | 0.2495 | 0.2974 | 0.3072 | 0.4600 |
| (reference, B32) Qwen3-8B (9/18/27/36) | 0.7035 | 0.4744 | 0.6208 | 0.4094 | 0.2551 | 0.2337 | 0.4638 | 0.4419 |

**Choice (registered rule): Granite-3.3-8B-Instruct, mean pooling over the name's tokens, layer 30 of 40 (3/4 depth),
criterion 0.6538** (next: Llama-3.1-8B-Instruct mean L24 0.6478, Mistral eol L24 0.6427; no tie within 0.001).
Stored as float16 (b46_out/b32_data/emb_b46_granite33i_mean_L30_f16.npy, md5 c474cf933dc90f0d0f3eaff1b95e26a3;
criterion recomputed locally on the float16 file: 0.6538). Encoder tag b46. results/b46_encoder_choice.json.
Prediction P1 (Llama-3.1-8B-Instruct, mean pooling at 1/4 depth, 0.62-0.74) FAILED on the model and the layer (the
criterion range held). The B46 codes retrieve entities less well than Qwen3-8B's (0.654 vs 0.704) and e5's (0.698).

## Recipe check (b46_embed.py on Qwen3-8B, layer 9, mean)

Criterion 0.7035 (B32: 0.7035). Rows bit-identical to B32's codes: 99.09% (all strings whose name tokens are unchanged);
the 0.91% whose leading lone space token is now left out of the mean have row cosine >= 0.893. So the two registered
recipe changes leave Qwen3-8B's codes unchanged except for that token rule. Prediction P11's recipe part HELD.

## Code diagnostics (b46_diag.py; training split for preprocessing, dev split for the cosines; no memory)

| codes | max abs coordinate (raw / centred / whitened) | alias-label cosine median, centred / whitened | P(all k SimHash bits equal), centred, k = 2/4/8/16 |
| --- | --- | --- | --- |
| B46 Granite L30 mean | 0.592 / 0.336 / 0.192 (62.6% of raw rows have a coordinate > 0.5; 0% after centring) | 0.534 / 0.403 | 0.43 / 0.21 / 0.08 / 0.02 |
| Qwen3-8B L9 mean | 0.723 / 0.573 / 0.151 | 0.512 / 0.430 | 0.51 / 0.25 / 0.08 / 0.01 |
| e5-base-v2 | 0.203 / 0.200 / 0.206 | 0.599 / 0.499 | 0.46 / 0.26 / 0.12 / 0.04 |

The Granite codes have a few outlier dimensions in their raw form (rogue dimensions, as the literature check warned for
some families); centring, which every memory applies, removes them. After preprocessing their alias-label geometry is
close to Qwen3-8B's: aliases share about as little absolute similarity with their labels, and an alias's SimHash
signature matches its label's about as rarely (e5's codes match exact signatures more often at k >= 8, which is the
regime where B36's key-free families tied the store). This is descriptive and changes nothing in the protocol.

## Next (registered)

B32 dev grids (VSA, index; 4-64 and 1-2 bits per fact), B36 PQ codebooks and dev grid of the new families, choices
(b46_dev.sh), all on the dev split; choices committed before any test run.
