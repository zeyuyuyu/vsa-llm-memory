# V5 pre-registration: the two headline comparisons re-implemented by a different model family (gpt-6.1-sol)

Date: 2026-11-23, about 16:50 machine clock (Asia/Shanghai), 03:50 US Eastern. Written and committed, then
OpenTimestamps-stamped, before any V5 prompt that asks for code is sent and before any V5 code exists. The paper is not
edited by V5.

## Why

Referees have asked three times for verification that does not come only from Claude agents ("by a person or a clearly
independent toolchain"). The earlier second implementations (V1 and the others in Supplementary Note "Second
implementations and a defect in the least-squares memory") were written by Claude agents, the same model family as the
assistant that wrote the pipeline. V5 has the code written by a different model family: OpenAI's gpt-6.1-sol, called
through the user's api_zheng gateway (an Azure bridge; never api.openai.com, no OpenAI key file). A Claude Opus 5.5 agent
orchestrates: it builds the specification packet from the paper's text, sends it, runs GPT's code, and compares.

What V5 is NOT, stated now so that the paper cannot overclaim: it is not a person; it runs under the same operator; and
the text GPT works from was drafted by the assistant that wrote the pipeline (and the packet was assembled by a Claude
agent), so shared misreadings of the text can still be correlated. It is a different model family writing all of the
code from text.

## Disclosure (what the orchestrator has seen)

The orchestrating agent has read the paper, which prints the pipeline's B60 / B62 / B59 / B61 / B63 numbers (for
example Supplementary Tables lamfix, b60percond, ecc, b61, b63). It has read the preregistrations B38, B60, B61, B62,
B63 and short parts of the pipeline code (b32_common.py, b32_run.py, b38_eq.py, b34_inmem.py) to find data paths,
formats and the parameter values that the paper's text leaves unstated. It has NOT opened results/b60_verdict.txt,
results/b62_*, results/b59_*, results/b61_* or results/b63_* in this task, and will not until GPT's outputs are final
and committed. GPT sees no pipeline code, no result number of the paper or of the ledger, and no ledger file.

## Scope (fixed now)

**(a) Equal bits.** The store's final design (MAP binding, antipodal value codes for two values, Sylvester-Hadamard
codes for sixteen, matched-filter read, 72 configurations), the corrected least-squares memory (LAMc, B60's options,
288 configurations) and the one-pass normalised delta rule (NLMSc, B62, 480 configurations), each chosen per condition
on development stores by the paper's rule (highest mean dev accuracy, ties within 0.1 points to the first in grid
order) and scored once on test stores. Encoders: Qwen3-8B (layer-9 mean codes) and Llama-3.1-8B (layer-24 mean codes).
Region R1: V = 2 or 16, 4, 8 or 16 bits per fact, M = 256 or 1,024 (12 conditions per encoder), ideal accounting. GPT
draws its own stores (its own seeds) by the paper's protocol (6 / 3 dev stores and 20 / 5 test stores per (M, V);
entities without replacement from the split; values uniform; every alias of a stored entity is a query).

**(b) Equal cells.** Cells per stored pair for at least 99.5% of the 4,776 probed keys of the 50 RULER stores read
exactly, under independent cell errors at the device model's 30-day error rate, at x1, x8 and x16 noise, value
vocabulary uncharged (free to every memory), for four memories: the dense bipolar store (each key's last value marked,
no threshold: the paper's best store read), the sparse-block store (H-SBC), the best BCH static table (minimal
perfect hash, shortened BCH, repetition; E-BCH) and the cuckoo insertable table (the better of its BCH and LDPC
versions; O-BCH / O-LDPC). Budget grid 2^(i/4), i = 12 .. 48 (8 to 4,096 cells per pair; one step = 19%). GPT derives
the cell-error rates itself from the device model's parameters (they are not given to it). Designs chosen on the 12
synthetic development stores, then the test stores scored once over a window of budgets.

## Text gaps filled from the preregistrations (parameters and definitions only; no result numbers)

Where the paper's text names a choice without its values, the packet fills it in from the preregistration and marks it
"[filled from prereg]". These fills are themselves a finding (the paper's text is incomplete there) and are listed in
the verdict. Planned fills: the store's stored shares {1, 1/2, 1/4, 1/8}; NLMSc's step sizes {1, 0.5, 0.25, 0.1, 0.03}
and output (last iterate or Polyak average); Max's Gaussian-optimal step constants (literature values); the RULER
stores' file format and probe rule; the dense store's pair code (key x position x value, x LAST on a key's last value)
and its read loop bound; the sparse-block store's block-length rule and block-load grid {0.35, 0.5, 0.7, 1.0}; the
test-budget window rule (4 grid steps around the dev requirement, extended until bracketed). Anything further found
missing during the iterations is answered only with a definition or parameter, logged in v5_out/packet_addenda.md
before the answer is sent.

## Agreement criteria (fixed now)

**(a)** For each encoder and each of the two differences STORE - LAMc and STORE - NLMSc (region mean over the 12 R1
conditions of the per-condition test-accuracy difference, points, ideal accounting): the V5 value and the pipeline's
(B60 for LAMc, B62 for NLMSc; results/b60_verdict.txt, results/b62_*) have the same sign AND differ by at most 1.5
points. Four checks per encoder pair; (a) is met if all are met on every encoder in scope. Reported beside it, not part
of the criterion: the R1-V2 and R1-V16 halves, per-condition accuracies, GPT's 95% intervals.

**(b)** For each memory and noise level in scope: |grid index (V5) - grid index (pipeline)| <= 1 (one 19% step); a
requirement never reached by 4,096 counts as one step beyond 4,096. Pipeline values from results for B59 (dense
uncoded store; BCH static table), B61 (H-SBC) and B63 (best insertable table). And the same ordering of memories at
each noise level: no pair of memories ordered strictly in opposite directions by the two implementations (a pair tied
in one implementation and strictly ordered in the other is not a reversal). Strict identity of the orderings (ties
included) is reported beside it. (b) is met if every requirement and every ordering check is met.

A part whose final script does not produce a requirement or accuracy for some cell counts as not met for that cell.

## Iterations and feedback (fixed now)

- At most 6 GPT code iterations per part (the first script counts as iteration 1). Each later prompt carries only:
  Python tracebacks, shape or type errors, a resource fact (wall time or memory exceeded, with the limits), or an
  answer to a question GPT asked that is a definition, format or parameter already in the paper or a preregistration.
- Never: any accuracy, difference, requirement, ratio or other result number of the pipeline; any statement that
  GPT's output looks right or wrong, high or low; any pipeline code.
- If GPT's code runs to completion but disagrees with the pipeline, it is NOT told; the disagreement is recorded.
- The final script of a part is the last iteration that runs to completion and writes its output file; its output is
  committed (v5_gpt_a.py / v5_gpt_b.py verbatim, every iteration under v5_out/iter*/, outputs under v5_out/) before any
  pipeline result file named above is opened.

## Procedure and logging

- Gateway: POST {base}/v1/chat/completions, model gpt-6.1-sol, reasoning effort high, User-Agent research-client;
  credentials from apps/real/api_reader.zheng_creds() (parsed, never executed, never printed or logged). Every request
  and response, minus the token, with its token usage, is appended to v5_out/gpt_log.jsonl (committed). One attempt per
  call; on a connection error the run stops and is reported.
- Runs on the local machine (16 cores) or the A800 box's CPUs (ssh cavin; background commands only, exact PIDs only,
  no other user's processes touched). Libraries: numpy, scipy, the standard library and packages already installed.
- After the outputs are final and committed: results/v5_verdict.txt (tables and criteria) and
  prereg/V5_note_after_test.md (what was found, plus two or three plain sentences for the paper that say precisely
  that the code was written by a different model family under the same operator, from text the assistant drafted, and
  that it is not a person).

## Time box and cut rule

Three hours from about 16:15 machine clock. If time is short: (a) on Qwen3-8B only and (b) at x1 and x8 only, written
into a dated note (v5_out/cut_note.md) before any comparison with the pipeline is made.

## Files

v5_client.py (gateway call and log), v5_out/spec_packet.md (the packet, committed before the first code prompt),
v5_out/gpt_log.jsonl, v5_gpt_a.py, v5_gpt_b.py, v5_out/iter*/, v5_out/a_out/, v5_out/b_out/, results/v5_verdict.txt,
prereg/V5_note_after_test.md.
