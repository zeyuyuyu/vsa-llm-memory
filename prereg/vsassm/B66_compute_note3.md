# B66 compute note 3 (before (D)'s output is read)

Date: 2026-11-23, 13:55 (machine clock).

- **(D) ran once, locally, 13:42-13:52** (b66_storeboot.py run, two processes: qwen8 / b46 / llama / mistral and e5 /
  bge / minilm; one thread each, nice 15), on the saved test predictions only, exactly the registered contrasts and
  regions; 10,000 draws ideal, 2,000 real. Each part logged "missing 12": the NLMSC arm has no real-accounting run
  (B62 ran ideal only), so STORE - NLMSC under real accounting is reported as missing (registered: "where the arm
  exists"). No part file has been opened. Next: `b66_storeboot.py panel` and `check`, then the D table is read.
- **Status of the other parts (no output read):** A1 clean-room dev phase 16 of 24 conditions done (local); SDMTCR test
  350 / 350 jobs done on the box and pulled back (plan md5 44e7221a, the same file as the local plan); SDMTm dev done,
  choices committed (7a64e5b), test started on the box (plans md5 372dfbd4 / 164d4f3b / 0014913e; a first start
  truncated the uploaded plans because b66_box.sh re-made them without the choice files on the box: 0 jobs ran, the
  plans were re-uploaded and checked by md5, then run with b66_run.py runb64 directly); VSARm dev about half done on
  the box; LSfc dev moved to the box (43 jobs done locally, copied; the rest with 24 workers); (C) SDMT / LAMC test jobs
  not started yet (box plan md5 a718a567 checked).
- Uploads to the box are slow (a few KB/s); downloads are fast. Inputs are now uploaded gzip-compressed and checked by
  md5 on the box before use.
