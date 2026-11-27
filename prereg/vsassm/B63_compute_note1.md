# B63 compute note 1 (02:36 machine clock; dev pass 1 running; no dev or test output has been read)

- **Header time.** The pre-registration's header says "about 02:35" machine clock; it was committed at 02:28 (d4cea81)
  and stamped, the proof committed at 02:28 (a3d40f2). The self-test record's "02:24" is the self-test run (02:2x).
- **Box.** ~/b63_run on the A800 box; b63_online.py, b63_box.sh, b38_eq.py, b34_inmem.py, b25_pcm.py, ruler_vsa.py,
  ruler_rm.py, b22_robust.py, b26b_pcm_table.py and b34_out/dev_stores.json.gz MD5-identical to the local commit d4cea81.
  The test stores' copy was still uploading (slow link) when dev pass 1 started; dev does not read them; their MD5 is
  checked before the test (compute note 2).
- **Dev pass 1** started 02:33:38 (b63_box.sh dev1, 60 workers, nice 10). The box's load average was 83 before the
  start (other processes, not touched), so the box is oversubscribed while B63 runs; this changes only the wall time.
- Verdict and check code (b63_verdict.py, b63_check.py) committed with this note, before any test output exists. The
  verdict's comparator rows reproduce the registered numbers with no B63 data (H-SBC 32 / 32 / 45 / 54 / 91 / 304 /
  1,448; U 76 / 76 / 76 / 91 / 128 / 431 / 1,722; E-BCH, E-LDPC, T-rep, E-CAM as B38). The check's independent reader
  (own hash code, bit-by-bit slot extraction, explicit candidate loops, targets from the store's own pairs) agrees with
  the committed reader and with b63_online.evaluate on 8 / 8 synthetic noisy cells (O-BCH and O-LDPC, f 12 / 14, k4-k16).
