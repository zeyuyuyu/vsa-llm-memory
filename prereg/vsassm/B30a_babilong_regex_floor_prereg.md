# B30a pre-registration: a BABILong floor with no language model (template parser + dict + B18's rules)

Date: 2026-11-18. Written, with its code (`b30a_floor.py`), before any B30a test output exists. Requested by a strict
referee of the VSA-LLM paper (r35): "a regex/keyword bAbI parser + Python dict + the same hand-written rules (no
language model), scored on the same BABILong test samples at every length". CPU only, no API, no training.

## Question

B18 reports 99.6-100% on QA1-5 from 0k to 10M tokens with a fine-tuned Qwen3-1.7B reader writing events into a
256 x 256 matrix memory, answered by fixed rules. The matrix is an exact lookup (orthogonal codes; SI says so). bAbI
facts are templated sentences. How much of the BABILong result needs a language model at all? A floor that replaces
the reader by a template parser and the matrix by a dict, and keeps B18's own rules, answers that.

## What has been seen before this file

- B18's banked per-question outputs (questions, targets, our predictions) and its verdicts; the frontier outputs.
- The local test files' sizes, sample counts and lengths, and (by accident, while counting) the first ~300 characters
  of one 0k QA1 sample of `b18_api_subset.jsonl.gz`. No floor output on any test sample has been produced or read.
- Gate G0 (below) may be run before the commit: it compares inputs, questions and targets only.

## Data

- Test: RMT-team/babilong (HF revision ee0d588794c7ac098062ee0d247c733d62e94fe2), `data/{qa1..qa5}/{0k,4k,16k,64k,
  128k,512k,1M}.json` downloaded whole on 2026-11-18 to `/path/to/HOME/bench_dl/babilong_hf/test/`; for 10M only the
  first 5 records of each `10M.json` are kept (`10M_first5.json`; streamed, the rest not downloaded).
- Development: RMT-team/babilong-train-5k-samples (revision b3513ef7c25c54ce706054530d47668c532019d6), `0k` and `4k`
  of qa1-qa5, to `/path/to/HOME/bench_dl/babilong_hf/train/`. The training split is the only data used to write and
  check the parser.
- **Gate G0 (data identity; run before scoring):** questions and targets equal B18's banked per-question outputs
  (`b23_ref/b18_s0`, 0k-128k, 100 per task; `b18_10M`), the inputs equal the ones sent to the frontier models
  (`b18_api_subset.jsonl.gz`, first 25 / 5 per task), and at 10M `b18.sentences()` yields exactly B18's reader-call
  count per sample. Any mismatch: no B30a number is reported until it is explained.

## System (no language model)

- **Parser** (`b30a_floor.parse`): the sentence splitter is B18's (`b18.sentences`). A sentence is tried against five
  templates written from the training stories: `the A is DIR of the B` (rel), `P GIVE the O to Q` (give),
  `P TAKE [up] the O [there]` (take), `P DROP the O [there]` (drop), `P GO [back] [to] the L` (go). The verb
  alternations are B18's cue lists (`b18.CUES`, written from general English in B18 amendment 2), not new lists.
  Because BABILong inserts a fact into book text without a break before or after it (found on the training split),
  the sentence is first cut at every full stop, and in each piece every suffix that starts at a capitalised word is
  tried, longest first. Pronouns are not accepted as names.
- **Vocabulary filter (arm F1):** a candidate is accepted only if every word is in the vocabulary of the 0k
  training stories and questions of qa1-qa5 (51 words) - the filter that labelled B18's student training data as
  bAbI facts (B18 amendment 2).
- **Memory:** a Python dict (`DictWorld`: `b18.World` with `_write` / `parent` replaced by a dict). `apply()` and
  `answer()` are B18's, imported unchanged, so the history log (qa3), the give log (qa5) and the relation log (qa4)
  and every answer rule are the paper's. B18's grounding and cue check (`b18.grounded`) is applied to every event.
- **Arms:** F1 (primary): vocabulary filter + templates -> dict. F1m (check): the same events into B18's
  256 x 256 matrix memory (`b18.World` unchanged). F2 (sensitivity): templates without the vocabulary filter.

## Development (training split only; before this file)

`python3 b30a_floor.py dev --lengths 0k,4k --n 5000` (code hash d17c0016ecf5aae7, the committed version): F1 and F2
answer 5,000 / 5,000 training samples of each of qa1-qa4 at 0k and at 4k, and 4,990 / 5,000 of qa5 at both lengths.
Every qa5 miss is B18's own rule (`answer` returns the latest give between the two people; the label names an
earlier one: "What did Bill give to Jeff?" apple / football), left unchanged because the rules are B18's. Two changes
were made on development data, both at 4k: (1) the first version, whole-sentence templates only, missed 4-22 of 300
per task, because BABILong glues a fact to the book text before it ("...but I The office is east of the garden."):
every suffix starting at a capitalised word is now tried; (2) the second version still missed 1-12 of 5,000 per
task, facts followed by book text with no space after the full stop ("John went to the bathroom.'s face",
"...the bedroom..0 8 1/2"): the sentence is now first cut at every full stop. Nothing else was changed, and the
5,000 samples of each task include the 300 used for the first change.

## Test

- Samples, identical to B18: the first 100 per task at 0k, 4k, 16k, 64k and 128k; the first 25 at 512k and 1M; the
  first 5 at 10M (n = 500 / 500 / 500 / 500 / 500 / 125 / 125 / 25 for QA1-5). Secondary: all 100 per task at 512k
  and 1M (B18 did not run 26-100; floor only).
- Metric: the official rule as in B18 (b18.correct_official logic, labels = every training-split answer); second
  scorer the verbatim `babilong/metrics.py` `compare_answers` (`babilong_metrics_official.py`); exact match reported.
- Comparison: B18's per-question outputs (`b23_ref/b18_s0` for 0k-128k; `b18_api/ours_subset.json` for the first 5
  at 512k and 1M, the other 20 being correct by B18's 25/25 verdict; `b18_10M`), paired, exact McNemar at 0k-128k
  (n = 500 per length). Frontier references on the first 25 / 5 per task (b18_api_verdict).

## Bars (registered reading)

- **E1 (the floor matches):** F1's QA1-5 mean >= ours - 1.0 point at every length. Holds: on BABILong, a template
  parser and a dict with B18's rules do what the 1.7B reader and the matrix memory do; the benchmark then measures
  exact state tracking outside a model, not the language model's reading.
- **E2 (the reader adds robustness):** F1 below ours by >= 5 points at some length. Holds: the language model adds
  accuracy there, against book text.
- Between the two (1-5 points): reported as a small reader advantage, no claim.
- **M (memory form):** F1m equals F1 on every question (the matrix is an exact lookup below its 256-code cap).

## Descriptive analyses registered with the test (no bars)

- **Code paths (referee item 1).** For each test question of QA1-5 at every length, the branch of `b18.World.answer`
  that produced the answer, which is determined by the question text alone: `where is X` reads the matrix memory
  (holder / place chain); `where was X before Y` reads the place-history log (filled from matrix reads at each event);
  `what is D of P` / `what is P D of` read the relation log (no matrix); the five give questions read the give log (no
  matrix); anything else returns no answer (rules alone). Reported as shares, with our and the floor's accuracy per
  path, and for `where is X` the number of matrix hops needed under the floor's parse of the story (1 when the
  object lies where it was dropped or the entity is a person, 2 when a person carries it).
- **Names per story (referee item 3).** Distinct names written into the dict (keys, values, both) per story under F1
  and F2 at every length; maximum against the 256-code cap of B18's memory. The reader's own codebooks were not
  logged; for them only the bound from banked counts is available (keys written <= events applied; 10M:
  `b18_10M` 'events').

## Predictions (stated so they can be wrong)

- G0 passes (it was run before this commit: 3,225 checks, 0 mismatches; the record is in the verdict).
- E1 holds at every length, with F1 >= 99% at every length (bAbI facts are templates; book sentences made only of
  the 51 bAbI words are rare).
- M holds (at most a few dozen names per story).
- F2 falls below F1 at the longest lengths (book sentences such as "Mary went to the door" become events), but stays
  above 90%.
- Code paths: 40% matrix (qa1, qa2), 20% history log (qa3), 40% relation and give logs (qa4, qa5), 0% rules alone.

## Consequence for the paper, fixed now

If E1 holds, the SI and the main text must say that on BABILong a template parser with a dict and the same rules
reaches the same accuracy as our 1.7B reader with the matrix memory, so the BABILong rows show what exact external
state gives, not what the language model or the vector form adds. If E2 holds, the length at which the reader adds
accuracy is reported.
