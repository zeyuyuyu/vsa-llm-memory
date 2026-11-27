# Frontier arms, amendment 2: answer segments glued by the stream reassembly (POST HOC)

Date: 2026-11-11. POST HOC: found while inspecting the NSR5 K = 8 frontier no-answers, AFTER the NSR1 and NSR3
frontier scores and the K = 8 score were seen. The correction moves every frontier score UP, against the pipeline
hypothesis. Both parses are reported everywhere; bars are evaluated on the registered parse, and the repaired parse
is reported next to them.

## What happened
The gateway streams gpt-6.1-sol's output as several segments: one or more short progress notes ("I'll trace each
specimen's final box..."), then the answer. nsr_api.py joined the deltas with no separator, giving text such as
"...against that list.Answer: S-26-01225, ...". The registered parser takes the last line that starts with
"Answer:", so when every "Answer:" was glued to the end of a progress note, the answer counted as unparseable.
The model did write a separate answer segment. The failure is in the reassembly, not in the model.
Counts (answers with no line starting with "Answer:" at all): NSR1 5/230, NSR3 6/460, NSR5 K = 8 4/69, K = 4 4/93
so far. Earlier I attributed NSR3's 6 unparseable answers to underscore IDs. That was wrong: these 6 are the glued ones.

## Repair (nsr1_llm.resolve, flag --repair)
Applied ONLY to answers with no "Answer:" line: a line break is restored before each "Answer:" that directly follows
non-whitespace, and the registered rule (last Answer line) is then applied. Without --repair the parse is byte-identical
to the registered one (checked on K = 8). All 5 NSR1 cases were read by hand: each repaired parse is exactly the
model's final answer segment.
The Claude runner joined text blocks with no separator as well; it now joins them with a line break (amendment 2), and
its parse uses --repair too.
