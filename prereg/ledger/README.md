# Excerpts of the experiment ledger

Eight rows of Supplementary Table `tab:prereg` (the table of pre-specified tests) are marked "ledger": Arenas; Code
coherence; three bAbI rows; Arena C at five times the budget; 2WikiMultihopQA; MuSiQue. Their pre-specifications were
not written as separate files, as the others in `prereg/` were. They are entries in `ledger.md`, an append-only
experiment ledger kept in a private repository (`ngcls`) on our workstation. That ledger is shared by several studies
and runs to 26,903 lines, most of them about other studies, so it is not deposited whole. This folder holds verbatim
excerpts of the entries that pre-specified these eight rows, so that each row can be checked against a deposited text.

## What each file holds

| File | Row of `tab:prereg` | Ledger entries |
| --- | --- | --- |
| `arenas.md` | Arenas | V-13 (rule that chose arena C), its verdicts, the V-13R re-run |
| `code-coherence.md` | Code coherence | V-95 (tolerance first defined), V-163 (re-registered under the fitted rule) |
| `babi-naming-head-vs-lora.md` | bAbI, naming head against a LoRA fine-tune | V-50, V-131 (separate file), V-132, V-152 |
| `babi-screening.md` | bAbI, screening of the twenty tasks | V-51, V-52, V-56 Part A, V-58 |
| `babi-length-contrast.md` | bAbI, length contrast on task 1 | V-45, V-46, a voided duplicate, V-47 |
| `arena-c-lora-five-times-budget.md` | Arena C, LoRA at five times the budget | V-15 (context), V-15b |
| `2wiki-inference.md` | 2WikiMultihopQA, inference-type questions | V-65 (context), V-69 |
| `musique-pooled.md` | MuSiQue, store under pooled facts | V-79, V-79a, V-131 Part C (separate file) |

Each file opens with a header naming the source (`ledger.md`, the repository commit it was read at, the line range of
each part) and, for each part, the commit that added those lines and its date. Next come notes by the depositor that
compare the entries with the row as the Supplementary Information states it, including every difference found. The
excerpts follow, each in a fenced block exactly as it stands in the ledger, with the SHA-256 of its lines.

## How the excerpts were made

- Source: `ledger.md` at commit `0732e13` (tag `archive-2026-11-11`) of `ngcls`, the HEAD of the workstation's clone
  on 2026-11-21; the copy read is byte-identical to that commit's file (MD5 `6843647bad4782ac58fcd341448bf95c`).
- Line ranges run from an entry's heading to its last non-blank line. Nothing inside a range is changed, added or
  dropped. Entries between two excerpted entries are left out, and an omitted entry is never silently joined to a
  neighbour. Each part is its own block.
- Commits and dates: `git blame` on `ledger.md` at that commit gives, for every line, the commit that last changed it.
  In every excerpt each part was added by a single commit and never changed afterwards, and `git log -S` on each
  pre-registration heading names the same commit as the first to add it. Dates are author dates on the workstation's
  clock (UTC+08:00). These are records in our own repository, not independent timestamps (Methods).
- Two parts come from a separate file of the same repository, `v131_prereg.md` (commit `6a2a52d`, unchanged since),
  because that pre-specification was written as a file rather than in the ledger. The files that include it say so.
- Redaction: in `v131_prereg.md`, the name of the venue of a separate manuscript of ours, then under review, is
  replaced by "[venue redacted]" in two places. No password, key or personal detail occurs in any excerpt. Process ids,
  GPU numbers and the user's instructions quoted in the ledger are left as written.
- The headings, wording and first person are the ledger's own: its entries were drafted by the AI assistant that
  ran the work (Methods, "Use of AI tools").

These files were written for the package; they are listed in `SHA256SUMS` and in `docs/PROVENANCE.tsv` (source `curated`).
