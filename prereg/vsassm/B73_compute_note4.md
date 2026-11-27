# B73 (equal bits) compute note 4 (about 21:31 machine clock; no language-model test output exists yet)

- **Upload.** The link to the box ran at 66 % packet loss and about 320 ms round trip (ping at 21:20); transfers above
  a few kB failed in most attempts. The fresh entity file (xz) was sent in 40 kB, then 8 kB, then 4 kB pieces, each
  written to its own file on the box (each piece sent once per explicit pass; idempotent), 21:00-21:29. On the box the
  pieces were joined and checked before anything ran: the tar MD5 (85565fd5...), entities.json MD5 b06ac059... (= this
  repository's), and the string list rebuilt by b32_embed.strings() MD5 774a3a13... (= this repository's).
- **Launch** at 21:29:51 (box clock), once, guarded so that it cannot start twice: b73_box_all.sh (committed 1ab4015)
  with the registered clock cut (no unit started after 22:25:00, B73_DEADLINE 1795443900). GPUs at launch: 1, 4, 5, 6, 7
  idle (4 MiB used), 2 in use by another user, 0 and 3 not used by rule. The chain encodes Qwen3-8B, Llama-3.1-8B,
  Mistral-7B and Granite-3.3-8B (one idle GPU each, b73_box.sh embed), then runs the test (48 workers, plan order:
  core arms on Qwen3-8B / Llama / Mistral, then Granite, VSAR, SDMT), then assembles, computes the verdict's
  per-encoder parts and the checks on the box, and packs only the small result files for transfer.
- **Sentence encoders read** (allowed by compute note 3's clock rule): their verdict part files, the panel for the
  three, the plain-loop scorer (99 region statistics, max |difference| 1.4e-17) and the e5-base-v2 sample reading were
  computed locally after 20:55. Nothing in the language-model run depends on them.
