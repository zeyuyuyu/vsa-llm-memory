# B74 compute note 4 (2026-11-23, 23:54 machine clock; written before any test output is read)

1. **Tests launched once** after the dev choices were committed (d08695a, 23:40): locally 23:41:07 for the three
   sentence encoders (plan_test_sent.jsonl, 1,041 jobs, 6 workers), on the box 23:42:06 for the four language models
   (plan_test_llm.jsonl, md5 bf0b4c7a... on both ends, 1,393 jobs, 48 workers, PID 4151736). The box is shared with
   another agent's B75 jobs (about 45 single-thread processes); load about 96 of 128 cores.
2. **Failure and its repair (no test output read).** 20 local label jobs of the one-pass delta-rule engine (LABN: e5
   M = 1,024 whiten, bge M = 1,024 V = 2) raised FileNotFoundError: b62_run reads its RLS training-norm files
   (b62_out/trainnorm_{enc}.npz, training split only) from its output directory, which B74 redirects. A job that raises
   writes no output. The seven files were copied unchanged (md5 equal to b62_out's, locally and to ~/b62_run's on the
   box) into b74_out/lab62 and b74_out/nl72 locally at 23:48 and on the box at 23:48, before the box reached any
   LABN job (its log had 0 failures at that time). After the local run ended (23:52), the same plan was run once more;
   only the 20 missing jobs ran (23:53, 0 failures). Every local test job now has its output (1,041 of 1,041).
3. No cut: (A), (B) and (C) run in full on all seven encoders. The language-model part is still running; its
   completion is recorded in compute note 5 before any of its output is read.
