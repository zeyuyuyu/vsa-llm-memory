# B47 exploratory 2 (registered 2026-11-20 ~02:45 UTC, before it runs; train and dev only; no registered bar)

Exploratory 1 failed its own criterion: on the candF label both answer-token fine-tunes stayed at the majority baseline
(dev 0.702 = always "yes"), so by its registered reading the gen / fullgen results are "not interpretable" unless the
read-out can be shown to learn at all. candF needs a membership test over 32 fillers mid-text, so it does not separate
a code fault from a hard label. **Control 2:** the same two runs (gen rank 8 lr 1e-4; fullgen lr 1e-5; 800 = B31's
inputs; 8 epochs) with label = "the last number in the input (the last table-F filler, the token just before
'\nAnswer:') is even" (lastpar), a property of the token next to the answer position. Dev scored with the same label;
the test split is never read.

Reading: lastpar dev >= 0.90 for an arm -> that arm's answer-token pipeline learns, and its chance-level results on the
task are a property of the task (reported as such, with exploratory 1's failure beside it); lastpar dev < 0.70 -> the
pipeline is faulty for that arm and its task results are withdrawn; in between -> reported as unresolved.
