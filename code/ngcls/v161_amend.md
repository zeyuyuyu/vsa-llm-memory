# V-161 AMENDMENT 1 — re-running four jobs that crashed with CUDA out-of-memory (2026-10-16)

**What failed.** Four of the 36 registered local jobs exited with `torch.OutOfMemoryError` during prefill. Each was on a
GPU shared with another user's ~40 GB job; the tracebacks are in b_logs/v161_*.out.

- qwen7b-s1-h8
- granite8b-s0-h8
- qwen14b-s0-h4
- granite8b-s2-h4

The frozen scorer therefore found 100/150 in the bar cell N256_H8 for Qwen2.5-7B and Granite, and VOIDed both. That
is the registered outcome of the run as it stood.

**Change.** Re-run exactly those four jobs, with the same code, seeds and cells, at a smaller batch size so that they
fit beside the other jobs:
- Qwen2.5-7B and Granite at batch size 4;
- Qwen2.5-14B at batch size 2.

Batch size changes only left-padding. In the Spark/A800 comparison, the non-CoT outputs were byte-identical across
batch sizes 8 and 16.

**Determinism check, which comes free with this.** The crashed jobs had written their cells before the crash.
Specifically:
- N16_H8, N64_H8 and N128_H8 for the two h8 jobs;
- N16_H4 and N64_H4 / N128_H4 for the h4 jobs, as present.

These partial files are kept as `*_OOMPARTIAL.json`. They will be compared instance by instance with the re-run, and
the agreement rate reported. The scored file is the complete re-run.

**Seen before this amendment.** The frozen verdicts for all four local models (14B and 32B HOLD; 7B and Granite VOID
for the missing seed) and their per-cell counts. The re-run cannot be steered by them: the instances, code and arms
are fixed, and only the batch size changes.
