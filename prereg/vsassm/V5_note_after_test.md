# V5 note after the test: the two headline comparisons re-implemented by gpt-6.1-sol

Date: 2026-11-23, about 18:45 machine clock. Pre-registration: prereg/V5_gpt_reimplementation_prereg.md (committed and
OpenTimestamps-stamped before any code prompt). Verdict: results/v5_verdict.txt (.json), built by v5_verdict.py.

## Who did what

All V5 code was written by OpenAI's gpt-6.1-sol, called through the user's api_zheng gateway (never api.openai.com),
from v5_out/spec_packet.md: the paper's text for the two comparisons with every result number removed, the parameter
values that the text leaves unstated filled in from the preregistrations and marked, and the data paths and formats.
A Claude Opus 5.5 agent assembled the packet, ran the code and compared. GPT saw no pipeline code and no result; it
was sent only tracebacks, one environment fact (a thread-limit segfault reproduced independently of its code) and
resource facts (wall time against the limit). Every request and response, minus the token, is in
v5_out/gpt_log.jsonl. This is a different model family writing all of the code, under the same operator, from text
that the assistant drafted; it is not a person, and shared misreadings of the text can still be correlated.

## Results against the pre-registered criteria

(a) Equal bits, region R1, ideal accounting (store minus memory, points; V5 against pipeline):

| encoder | STORE - LAMc | STORE - NLMSc |
| --- | --- | --- |
| Qwen3-8B | -4.38 against -4.06 (met) | -3.15 against -2.11 (met) |
| Llama-3.1-8B | -6.97 against -5.56 (met, by 0.09) | -4.34 against -1.51 (NOT met: same sign, 2.83 apart) |

Criterion (a): NOT MET (3 of 4 checks). Every sign agreed, and every V5 interval lies below zero: the paper's
qualitative claims (corrected least squares, and the one-pass delta rule, ahead of the store's final design over the
region) were reproduced on both encoders. The disagreements make least squares' and the delta rule's lead larger,
not smaller.

(b) Equal cells, cells per stored pair for >= 99.5% exact, 30 days, i.i.d. cell errors (V5 against pipeline):

| memory | x1 | x8 | x16 |
| --- | --- | --- | --- |
| dense store, last-value read | 76 / 76 | 128 / 128 | 431 / 431 |
| BCH static table | 19 / 19 | 108 / 108 | 512 / 431 (one step) |
| sparse-block store | not final / 32 | not final / 91 | not final / 304 |
| cuckoo insertable table (better of O-BCH, O-LDPC) | not final / 32 | not final / 91 | not final / 304 |
| secondary: O-BCH alone | 38 / 32 | 215 / 181 | 1,024 / 724 |

GPT's own estimate of the device model's 30-day cell-error rates: 0, 0.12258 and 0.27631 at x1, x8, x16 (pipeline 0,
0.12263 and 0.27601), from the equations alone. Criterion (b): NOT MET, because two of the four memories have no final
V5 value; for the two that finished, every requirement was within one grid step (five of six identical) and no pair
was reversed (dense and BCH tie at x16 in the pipeline; V5 puts BCH one step above).

## Discrepancies and what is known about them

1. Llama-3.1-8B, STORE - NLMSc (unresolved). Over R1, V5's store is 0.96 points below the pipeline's, its NLMSc 1.87
   above (LAMc +0.44). Most of the gap sits at V = 16: the pipeline's R1-V16 STORE - NLMSc is +0.27, V5's -6.18.
   Two localized patterns: (i) the pipeline's dev-chosen NLMSc configurations lost 7.7 to 10.1 points from dev to test
   in four Llama V = 16 conditions (256/16/4, 256/16/16, 1024/16/4, 1024/16/8) against 1.4 to 6.1 for V5's choices,
   which points at selection noise among NLMSc configurations (step size, averaging) on six or three dev stores;
   (ii) V5's store is 3 to 6 points below ours at M = 1,024, V = 16 on Llama (also on dev; on Qwen3-8B by about 1),
   not traced to a code difference within the time box. Stores were drawn independently (GPT's own seeds), so part of
   any single-condition difference is store sampling; Qwen3-8B agreed within 1.1 points on both differences.
2. BCH table at x16, one step above (512 against 431). Within the criterion; GPT noted a conservative codeword-failure
   expectation in its closed-form (n, t) choice.
3. O-BCH one to two steps above. GPT chose a power-of-two bucket count for the XOR partial-key cuckoo hashing (the
   packet said the smallest count with n <= 0.95 x 4 x buckets), so its tables carry up to twice the slots; the likely
   cause, not tested.
4. Not finished: the sparse-block store and O-LDPC. GPT's scripts swept every grid budget (8 to 4,096) on the
   development stores, as the packet literally said, and the local machine was shared from about 18:05 with another
   job; the A800 box could not be used (ssh link too slow to copy the data). Four of the six iterations of part B were
   spent on speed. Recorded in v5_out/cut_note.md before any comparison.

## Text gaps found (the paper's text alone did not define these; filled from the preregistrations)

The store's stored shares (1, 1/2, 1/4, 1/8); NLMSc's step sizes and averaging options; the drift-exponent mean and
spread, nu_Q(G) and the 1/f exponent of the device model (the text says "the measured mean and spread"); the dense
store's pair code (key x position x value, x LAST) and read loop bound; the sparse-block store's block-length rule and
block-load grid; the test-budget window rule. Supplementary Table "Implementation details" should state them.

## Procedure notes

- The gateway sits behind a Cloudflare quick tunnel that ends any request after about 125 s (HTTP 524), streamed or
  not, so reasoning effort was lowered from "high" to "low" and long scripts were requested in consecutive pieces;
  fixes came back as whole top-level functions, substituted by name by v5_patch.py with no other edit.
- Iterations: part A 2 of 6 (one segfault traceback); part B 6 of 6 (one pickling traceback, four resource facts;
  iteration 6 was received but not run).
- Token usage (13 completed calls, 2 of them pings): 160,247 prompt tokens (104,146 cached) and 33,349 completion
  tokens (1,914 reasoning). 160 HTTP 429 replies (not processed) and 3 HTTP 524 timeouts, whose usage the gateway did
  not report.
- Local runs only; the GPT scripts ran unchanged (v5_gpt_a.py = part A iteration 2; v5_gpt_b.py = part B iteration
  5; the dense and sparse-block processes ran iteration 3 and the BCH process iteration 4, identical to iteration 5 on
  their code paths).

## Plain sentences for the paper (Supplementary Note "Second implementations")

A further second implementation was written entirely by a model of a different family, OpenAI's gpt-6.1-sol, from a
specification that the assistant assembled from this paper's text with every result removed; it ran under the same
operator, is not a person, and worked from text drafted by the assistant that wrote the pipeline. Over the
pre-specified region it reproduced the lead of corrected least squares over the store's final design on Qwen3-8B and
Llama-3.1-8B (-4.4 and -7.0 points, against our -4.1 and -5.6) and of the one-pass delta rule (-3.2 and -4.3,
against -2.1 and -1.5; the Llama value fell outside the 1.5-point agreement margin set in advance). On the simulated
devices it reproduced the dense store's requirement exactly (76, 128 and 431 cells per pair at one, eight and sixteen
times the noise) and the BCH table's within one grid step (19, 108 and 512, against 19, 108 and 431); its sparse-block
store and insertable table did not finish in the time allotted.
