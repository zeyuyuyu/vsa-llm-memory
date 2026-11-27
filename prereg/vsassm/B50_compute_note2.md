# B50 compute note 2: COT2048 task R moved to a second A800 GPU; a PACE fallback was started and cancelled before it produced anything

Written 2026-11-19, about 13:55 box time (05:55Z).

COT2048 task N runs on GPU 6 beside another user's job and our DISC lanes; its first batch (48 items, 1,787 steps) took
527 s (0.29 s per step, about six times B40's rate at 512 tokens), so running task R after it on the same lane would
take too long. The task-R command (unchanged) was moved from GPU 6's lane to a second worker (b50b) that waits for an
idle A800 GPU under the shared lock rule, and, as the prereg allows when the box is full, a PACE fallback was prepared:
the model file set fetched on PACE (unsloth/Llama-3.1-8B-Instruct; sha256 of all four shards equal B40's; the small
config and tokenizer files then replaced by B40's own copies from the box, md5-identical) and one GPU job queued
(smoke on 8 training rows, then the task-R test run).

Both started within four seconds of each other: b50b claimed GPU 5 at 05:50:38Z and began the task-R test run; the
PACE job started at 05:50:42Z with the smoke. I kept the A800 run (same GPU type as B40, so B40's 512-token outputs can
be compared token for token) and cancelled the PACE job at about 05:52:00Z, 13 s after its own test command had
started loading the model. Its prediction directory is empty: no PACE test output exists, so the task-R test run is
executed once. (The PACE smoke on training rows, 8 of 8 correct, 163 steps, read the unsloth config files, which had
not yet been replaced; it has no bearing on any result.) The worker queue line on the box was also replaced by STOP so
that no second copy can start.
