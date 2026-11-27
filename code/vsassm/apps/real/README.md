# Application scenarios

Four sites where the system of record goes stale because people move things without logging them. In each, a day's
log interleaves the system's own feed with human channels, and Readable Memory answers "where is it now?" from both.

| Scenario | System feed (read by a connector) | Human channels (read by the reader) | Typical question |
| --- | --- | --- | --- |
| Warehouse floor | WMS scanner transactions (RECEIVE, PUTAWAY, MOVE, CYCLECNT) | floor chat, end-of-shift handover note | "Where's 7469 now?" |
| Biobank custody | LIMS audit trail (TRANSFER, CHECKOUT, STORE, SHIP) | bench notebook, lab Slack, email | "Where is specimen S-26-03866 stored right now?" (box > rack > freezer) |
| Home robot | camera detections | what the robot hears people say | "Where did Grandpa leave the wallet?" |
| Cloud incident | deploy bot, Patroni | incident channel | "Which host is the orders-db primary on?" |

## How it works

- `world.py` plays out one day at a site. Every change is written once, either as a record in the system's own
  format or only as a human message (25-45% of changes, depending on the site, never reach the system of record).
  Human lines also include plans, failed searches, chatter and handover notes that restate locations.
- `gen.py` has Qwen3-32B write each human line in the channel's voice from the facts alone, and a second pass checks
  it against the intended change (a yes/no pass for plans and failed searches). A line that fails every attempt falls
  back to a template.
- `product.py` is the deployed system: regex connectors for the feeds, the site registry (IDs, aliases, digit
  suffixes) and `SiteMemory`, which holds every change to the message it came from (a name the message does not
  contain goes to the thread's last-mentioned item), treats two moves that trade two places as one swap, and writes
  into the 256 x 256 memory of `b14.Memory`.
- `run_real.py reader` runs the shipped reader (Qwen3-1.7B, distilled adapter `b16_models/student_s0`, one message
  at a time with the last three messages of the same channel), a one-word check that the change actually happened,
  and a forced choice when one message names several places.

## Protocol

| Split | Logs | Use |
| --- | --- | --- |
| `dev` | 2 per scenario (seeds 100+) | development |
| `test` | 10 per scenario (seeds 200+) | **retired**: an audit found that in 27/76 warehouse, 17/73 home and 15/78 ops follow-ups ("stuck that one in B-11-03") another item had been mentioned in between, so the label was not the reading a person would give. Used as development data after that (`tools/audit_followups.py`) |
| `test2` | 10 per scenario (seeds 300+), 414 questions | held-out: generated after the generator fix, with the product configuration frozen (commit `cd9de8d`), scored once |

The comparison models (Qwen3-1.7B / 8B / 32B, thinking mode, greedy, 8,192 reasoning tokens) read the whole log in
one prompt with the same asset list the product is given (`prep_llm --tag _assets`). Runs that hit the limit are
given their reasoning back and told to answer now (`force_answer.py`); both the strict and the forced score are kept.

## Results on test2

Share of questions answered exactly (box > rack > freezer chains must match in full). Comparison models: budget-forced
score, strict score (a run that hit the reasoning limit counts as no answer) in brackets.

| Scenario | Readable Memory | System of record alone | Qwen3-32B | Qwen3-8B | Qwen3-1.7B |
| --- | --- | --- | --- | --- | --- |
| Warehouse | **0.983** (118/120) | 0.375 | 0.525 (0.483) | 0.225 (0.100) | 0.092 (0.067) |
| Biobank | **0.915** (86/94) | 0.245 | 0.266 (0.266) | 0.064 (0.053) | 0.021 (0.021) |
| Home robot | **0.990** (99/100) | 0.510 | 0.750 (0.750) | 0.540 (0.360) | 0.220 (0.170) |
| Cloud incident | **0.940** (94/100) | 0.450 | 0.620 (0.620) | 0.460 (0.370) | 0.320 (0.290) |
| All | **0.959** (397/414) | 0.396 | 0.543 (0.531) | 0.321 (0.217) | 0.162 (0.135) |

Readable Memory reads 15-25 generated tokens per human message, once, and answers with no generated tokens in
~0.3 ms; the comparison models spend 2,981 (32B), 4,132 (1.7B) and 6,261 (8B) reasoning tokens per question on
average, re-reading a 6,800-9,900-token log each time. The 8B and 1.7B models hit the 8,192-token limit on 208 and 131 of
414 questions, mostly while still retelling the log line by line; the 32B model on 21. Scoring the whole final board
against the simulation's state (every item, not only the questions) gives the same picture: 0.994 / 0.968 / 0.973 /
0.945 against 0.550 / 0.764 / 0.582 / 0.573 for the system of record.

A first run of the comparison models without the asset list was discarded as unfair (a model cannot link "the Sony
pallet" in chat to the LPN on the scans without it); its outputs are kept outside the results as an ablation.

On the retired `test` set the same configuration scores 0.842 / 0.929 / 0.950 / 0.920; its labelling flaw counts
against every system.

## Known failure modes (from the 17 misses on test2)

- Chatter or a question that sounds like a location ("chat-gateway still flapping in staging", "did I leave my
  wallet on the desk?") can be written as a move.
- In the biobank, where boxes are both things and places, "moved it to box 10" can move the box instead of the
  specimen.
- A handover note that names three pallets can attach a place to the wrong pallet.
- Events told out of order are applied in the order they are read.

## Reproduce

```bash
python3 apps/real/gen.py --split test2 --model <Qwen3-32B> --scen warehouse          # one scenario's logs (PACE: pace/real_gen.sbatch)
python3 apps/real/run_real.py reader --split test2 --verify --tag _v4 \
    --model <Qwen3-1.7B> --adapter b16_models/student_s0                              # PACE: pace/real_reader.sbatch
python3 apps/real/run_real.py prep_llm --split test2 --scen warehouse --tag _assets  # then pace/real_llm.sbatch per log
python3 apps/real/run_real.py score --split test2 --reader_file apps/real/out/reader_test2_verify_v4.jsonl \
    --llm qwen3-1.7b qwen3-8b qwen3-32b --llm_dir apps/real/out/test2a
python3 apps/real/build_real.py --split test2 --reader_file out/reader_test2_verify_v4.jsonl \
    --llm_dir out/test2a --retired out/reader_test_verify_v4.jsonl                    # the showcase page, apps/real/page/
```

`results/` keeps the reader's answers and events on test2 and the comparison models' final and budget-forced
answers (reasoning text omitted), so every number above can be rescored without a GPU. The SDK
(`readable_memory.SiteMemory`) runs the same reading step as the evaluation (`run_real.read_messages`);
`tools/check_reader_equivalence.py` and `tools/check_sdk_equivalence.py` check that with a deterministic fake reader.

## Running it at a site

```python
from readable_memory import SiteMemory
mem = SiteMemory(registry=site_assets, connector="warehouse", reader_model=QWEN3_1_7B,
                 adapter="b16_models/student_s0", memory="sharded")   # "sharded": past 256 items or 256 places
mem.observe(lines)                      # system records and messages, in time order
mem.save("site.pt")                     # the whole memory, its size set by the site's items and places
mem = SiteMemory.load("site.pt", reader_model=QWEN3_1_7B, adapter="b16_models/student_s0",
                      registry=newer_assets)                          # optional: the asset list has grown
```

A loaded memory carries on exactly where the saved one stopped: the same reader prompts, answers, board and state
matrices as a run that never stopped (tested on all 40 test2 logs, with the single memory, the sharded one, and the
sharded one at tiny limits that shard and rebuild all day). It stores each channel's last three messages, which is
all the reader is shown, and none of the lines themselves. Loading reads plain data only (`torch.load` with
`weights_only=True`) and refuses a file saved with other memory settings.

Tests, no GPU or model needed (about 35 s):

```bash
python3 -m pytest -q apps/real/tests
```

They cover save and resume, a newer asset list at load, refusing other settings, that the saved size does not grow
with the lines read, every system record of every split through its connector, the registry's digit-suffix option, and
the sharded memory staying exact at 1,024 items and 1,024 places.
