# B47 exploratory 1 (registered 2026-11-19 ~18:05 UTC, before it runs; no registered bar; train and dev only)

**Observation that prompts it (train + dev only).** Every generative answer-token fine-tune of Phi-2 (LoRA, rank 8 / 64 x
three rates; full fine-tune x three rates) stayed at chance on its training subset and on dev, at 800 and at 8,000
training inputs: the loss stays near ln 2, the free arg-max over the vocabulary is always " yes" or " no", and the
predicted-positive rate swings between 0 and 1 across evaluation points. The scratchpad LoRA, through the same data
and LoRA code, fits (dev 1.000). A pipeline bug in the answer-token read-out (wrong position, wrong target, frozen
weights) would look the same.

**Positive control.** The same read-outs and recipes (gen: rank 8, lr 1e-4; fullgen: lr 1e-5; 800 = B31's inputs, 8
epochs) with the label replaced by a one-step property of the same input: "the candidate is one of table F's fillers"
(candF; the shortcut B44 found in the Qwen3-8B fine-tune; positive rate 0.696 train / 0.702 dev, so the majority
baseline is 0.70). Dev scored with the same derived label. The test split is never read (b47_ft.py --label candF asserts
it).

**Reading (exploratory).** If both reach dev >= 0.90 on candF, the answer-token pipeline learns from these inputs and the
chance-level result on the real label is a property of the two-step task for these fine-tunes, not of the code. If
neither leaves 0.70, the read-out code is suspect and the gen / fullgen results are reported as not interpretable.
