# Frontier arms, amendment 1: dropped streams are re-run (NSR1 addendum, NSR3 T2-F, NSR5)

Date: 2026-11-11. Written while the NSR3 and NSR5 frontier runs were in progress, BEFORE any NSR3 or NSR5 frontier
answer was parsed or scored. (I read only record metadata, finish reasons and token counts, plus the last characters
of 18 NSR5 records to check the answer format; no answer was compared with gold.)

## What happened

The Sidrune gateway sometimes closes the stream early without an HTTP error, a finish_reason or a usage chunk.
nsr/nsr_api.py stored such calls as finished, with empty or truncated text. Counts so far: NSR1 1/230 (already noted in the
NSR1 addendum as "1 stream without a finish reason" and scored as no answer), NSR3 8/210, NSR5 K=8 4/18.

## Rule

1. A record with no finish_reason and no usage is a transport failure, not a model answer. nsr_api.py now raises on it
   and retries, up to 4 attempts. On resume it is not counted as done, so it is re-run. The answer of record per
   question is the last valid record (nsr_api.final).
2. Each question is re-run until it gives a valid record. If a question gives none after 3 resume passes, it counts as
   no answer and is reported.
3. A finished call whose reasoning used the whole budget (completion tokens >= max_completion_tokens) and gave no
   answer counts as hit_limit. The gateway reports "stop" for these. As registered, it is scored as no answer
   (32,000-token budget). Sensitivity analysis, not used for any bar: those questions are re-run with a 64,000-token
   budget.
4. NSR1 is corrected the same way: the one dropped call was re-run. The registered score is reported both before
   and after the re-run.

Why: the bars compare model accuracy. A gateway disconnect is neither a model error nor a pipeline advantage, and
scoring it as no answer would bias every frontier bar toward the pipeline.
