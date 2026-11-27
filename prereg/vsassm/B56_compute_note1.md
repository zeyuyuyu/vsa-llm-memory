# B56 compute note 1 (during dev; no test store built)

2026-11-20 15:48-15:50 (machine clock). With six B56 processes (three sparse dev parts, three engine dev jobs on the
LLMs at M = 1,024) and the machine's other load, available memory fell to 0.7 GB (swap 7.7 of 8 GB already used by
other sessions). I stopped one sparse dev part (qwen8 center:tern64) after 12 of its 18 dev stores; the xargs wrapper
logged its stop as "rc=0" (the line in b56_out/logs/steps.log at 15:48:56 is that stop, not a completed part). Its
finished stores are kept and the remaining stores are run afterwards with the same seeds (the part's output does not
depend on how it is split into runs: every store is computed independently from registered seeds). I then lowered the
sparse-part queue's parallelism from 3 to 1 (SIGUSR2 to xargs) and raised it again when the LLM M = 1,024 engine jobs
had finished. No design, grid, seed or rule changed; dev results were not looked at before this note.
