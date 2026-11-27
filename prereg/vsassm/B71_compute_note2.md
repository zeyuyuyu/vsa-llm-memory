# B71 compute note 2 (during dev; no dev or test output of any B71 arm read)

Date: 2026-11-23, about 17:45 (machine clock).

1. **Local memory.** With the 8 local workers (about 1.5 GB each) and other sessions' processes, available memory fell to
   about 2 GB with swap full. At 17:40 I stopped the local runner by its exact PIDs (8 finished jobs, all VSAD kind P, kept;
   jobs in flight discarded, no partial file) and re-split the remaining dev work without changing any job:
   VSAD kind R with the audit (171 jobs; plan_dev_local2.jsonl, md5 4aef4b97...) locally with 6 workers, because the audit
   must reproduce B53's and B56's dev accuracies, which were computed on this machine (B70 compute note 2 found float32
   differences between the machines in near-ties); NLMSF (315 jobs; plan_dev_box2.jsonl, md5 3517a322..., started 17:39) and
   VSAD kind P (144 jobs; plan_dev_box3.jsonl, md5 7e1fba16..., started 17:41) on the box, each made on the box by a
   filter of the registered plan_dev.jsonl and md5-checked against the same filter here. Where a job exists on both
   machines (the 8 local VSAD kind P jobs), the local file is used. VSAPR continues on the box.
2. **D2's dev part started (no test run).** results/b71_d2_lam_choice.json: LAMc keeps its chosen configuration in 72 of 84
   R1 cells (the store whitens there, so the principal basis is derivable from the store's own whitening matrix); in the 12
   cells where the store's chosen preparation is centring, LAMc is re-chosen on its existing dev rows among the 72
   configurations with no charged excess (centre preparation, random features). SDMt with its hard locations capped by the
   first design's own side information (N_max = 2,049 language models, 385 e5, 193 bge / MiniLM where the first design
   whitens; 1 where it centres): 297 dev jobs on the box (b71_side.py d2sdm_run; b64_run.py md5 as the repository's,
   4e7708d0), plan b71_out/d2sdm/plan_dev.jsonl. These are dev rows only.
