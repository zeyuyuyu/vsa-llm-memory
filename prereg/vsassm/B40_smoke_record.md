# B40 smoke record (training rows only; no dev or test accuracy of any arm seen)

Written 2026-11-19 after the pre-registration commit (4d15bba) and before any dev-selection run.

## Model files

- Source: ModelScope LLM-Research/Meta-Llama-3.1-8B-Instruct (Meta's release mirrored by ModelScope), downloaded to
  /path/to/work/b40/llama31_8b_instruct with ranged requests; the HF mirror (hf-mirror.com) stalled on its xet CDN.
- sha256 of the four shards (box, after download): 2b1879f356ae..., 09d433f65064..., fc1cdddd6bfa..., 92ecfe1a2414...
  (full values in results/b40_model_sha256_modelscope_etag.txt). They equal ModelScope's X-Linked-Etag and the LFS
  oids of unsloth/Llama-3.1-8B-Instruct on the Hub, shard for shard, and DIFFER from the current LFS oids of
  meta-llama/Llama-3.1-8B-Instruct (results/b40_model_sha256_meta_hub.txt), whose shard sizes are identical. So B40
  uses the release file set that ModelScope and unsloth carry, not byte-identical to the Hub's current meta-llama
  upload (most likely a re-upload; tensors not compared). Disclosed; it does not change the protocol.
- Download incident: the first concatenation of shard 1 contained HTTP error bodies (curl without -f); loading failed
  with a safetensors header error before any forward pass ran; shard 1 was re-downloaded cleanly (curl -f, half the
  ranges from the identical unsloth file through the box proxy) and verified against the etag (2b1879f356ae...).

## Pool check (b40_run.py pools) - one change before any run read the task

- Numbers 0-999 and every person and container name (with a leading space) are single tokens.
- The first place pool failed: kitchen, garden, bedroom, hallway, bathroom, garage, attic and cellar are NOT single
  tokens WITHOUT a leading space (the answer's form at the start of the assistant turn), so the answer's first token
  would be ambiguous (garden/garage share 'g') for the first-token training loss and dev metric. As the prereg
  prescribes ("fixed ... by changing the pool, not the protocol"), the pool is now ten places that are one token both
  with and without a leading space: office, library, studio, lobby, hall, park, school, church, hotel, farm; the R
  system prompts say "the name of the place" instead of "the name of the room". Nothing else changed.
- New task-R split hashes (sha256 of the json rows, first 16 hex): train 90585d9dd2f73427, dev 2986204f990b4d15,
  test 2d143c130d13d9a8, train_e2e a0855fc3539822fc. Task N unchanged (679603129fbaa6bb, c7323a009f6a7cba,
  f70af3c45c8a740d). The R caches made with the old pool were deleted before any reader used them.

## CPU check of the walk (no LLM)

With exact entity codes (no hidden-state read), the VSA walk of task R (d = 2,048, permuted binding, argmax clean-up)
returns the right place in every cell of the test split (1.000 at depth 1-5, small and large; computed on the first
place pool, before the pool change; the walk does not depend on the place words). This is the ceiling by construction,
like a dictionary; it reads labels, not an arm's output.

## Smoke runs (training rows only)

Every arm's code path ran on training rows only (dev metric computed on training rows via --dev_split train;
box logs logs/wS1-3.*, outputs out/smoke/): readers (5 epochs), write-back / LoRA / E2E training (20 steps), test-style
evaluation with generation (first 24-64 training rows), noisy read (rho 0.5), soft clean-up (beta 30), shuffled
pairings, CoT. All 25 commands returned 0. The second scorer (b40_check.py) agrees with the main scorer on all 10
smoke prediction files (0 gold and 0 correctness mismatches). Speeds: module write-back ~0.3 s/step (N), LoRA
~0.5 s/step (N) and ~1-2 s/step (R, gradient checkpointing).

Numbers seen (training rows; the first training rows of task N are multiplication, of task R depth 1, of
train_e2e depth 2): N BASE 0/64, N CoT 3/24 (multiplication), R BASE 64/64 (depth 1), R CoT 24/24 (depth 1); with
the layer-17 state halved (the E2E arms at initialisation, write-back ~0) the LLM still answers 29/32 depth-2
train_e2e rows. So the frozen LLM is strong on shallow task-R questions; whether it fails at depth 4-5 and on the
large size is what the test cells show. No design change follows from these numbers.

Code change during the smoke (no effect on any result): reader training keeps the cached states on the GPU instead of
gathering each batch on the CPU (speed only).
