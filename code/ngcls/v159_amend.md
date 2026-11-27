# V-159 AMENDMENT 1 — the Granite arm moves to the A800 (2026-10-15, before any Granite generation)

**Why.** The Spark's ngrok tunnel has refused connections since about 05:30Z. The Qwen2.5-14B arm passed its format
gate there and its registered seeds are running detached, but nothing on the Spark can be read. The Granite arm has
not started anywhere.

**Change.** The Granite-3.3-8B-Instruct arm runs on the A800 (`run_v159_granite_a800.sh`), which the prereg allows
("a model may run on an A800 instead").
- Same b3.py (md5 9d32c789), same seeds 0–2, same 4-cell grid, same per_cell 50, same frozen scorer v159_score.py
  (tag granite8b).
- Same format-only smoke and gate, on unregistered seed 99.
- Batch size is 8 (the A800 default) instead of the Spark's 16. That affects padding only, and the Spark/A800 smoke
  comparison showed byte-identical non-CoT outputs.
- Split by cell, as in V-157 amendment 1: each seed's N128_H4 is its own process on its own GPU, and the light
  cells run in a chain on one GPU. Every cell has its own RNG, so the instances are identical.

**The Spark's Granite stage.** If the Spark later runs it, it is a cross-hardware duplicate. It is reported as a
reproducibility check and is not scored.

**Seen before this amendment.** V-157 seed 0 in full; V-157 seeds 1–2 light cells. None of it bears on a hardware move.
