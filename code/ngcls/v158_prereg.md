# V-158 PREREG — B1 v2: V-155 with every B1 review finding fixed (same 300 rows)

Registered 2026-10-15, before any V-158 generation and before V-155's verdict was seen. Runner `b1v2.py`, frozen
scorer `v158_score.py`, independent second scorer `v158_indep.py`, launcher `run_v158.sh`, all in this commit.

## Why

The independent code review (`review_B.md`) found that V-155 (registered and running) cannot be audited per row and
that its secondary bars do not test what they claim:

- **F0.** The artifact saves no raw generations and no predictions for the symbolic arms.
- **B1-1.** EM undercounts the LM arms: of the answers scored wrong, 29% of direct and 38% of CoT are right by
  meaning.
- **B1-2.** P3 (VSA = table) compares two arms that differ in three ways besides superposition:
  - the HRR never abstains;
  - multi-valued (subject, relation) pairs are resolved differently;
  - the HRR returns a different surface string.
- **B1-3.** P4 (LM identity beats exact strings) compares a norm()-ed exact match with un-normed embedding identity.
- **B1-5.** Embeddings depend on batch padding.
- **B1-6/7.** The bars use float thresholds, and P4 has no CI.
- **B1-9.** An empty start would NaN a whole row's identity.

V-155 runs to completion and is scored verbatim as registered. V-158 is the corrected instrument on the **same rows
50–349 with the same prompts and generation settings**. Its direct and CoT arms are therefore also a determinism
re-run of V-155.

## Changes (all in b1v2.py; prompts, generate(), parsers of LM output and rows unchanged)

- Raw logging per row: every generation, the decomposition, the triples and every arm's prediction.
- Containment (the normalised gold is a token-bounded substring of the normalised prediction) and F1 are logged
  beside EM for every arm.
- Table and HRR are built from ONE store, `build_store`: (subject key, relation key) → object key, first triple wins.
  They share one abstention rule (the (current, relation) pair is not stored → no answer) and one returned string
  (the first-seen surface form of the object key). The HRR superposes exactly the store's bindings, with the value
  role-marked by the same permutation as V-155.
  - CPU test: the review's three demos now agree between table and HRR (abstain / first value / same string).
  - On random stores at B1 load (40–160 triples), table == HRR on 298/300 two-hop queries.
- Identity: mentions are grouped into norm()-classes first. Each class is embedded by its first-seen surface form,
  ALONE (no padding), and the greedy clustering at τ = .70 (fixed from the pilot) runs over classes. LM identity is
  therefore a superset of exact identity. Empty phrases are dropped.
- HRR codes come from a per-row seeded RNG (1000 + row idx), independent of call order.
- "trivial_qent" is renamed `one_hop_guess`. It is the first hop's object, not the majority baseline promised in
  B_DESIGN (review B1-8).

Not changed, disclosed (review B1-4): the extraction and decomposition few-shots use Wikidata relation labels. 58% of
gold hop relations appear verbatim in the prompts. This favours the symbolic arms, so P1/P2 FAILs stay conservative.

## Bars (frozen scorer; counts out of 300; registered metric EM, also printed under containment and F1)

- **P1** vsa_ident − direct ≥ +15 (= .05) and paired bootstrap 95% CI > 0.
- **P2** vsa_ident − cot ≥ 0 and CI > 0.
- **P3** |vsa_ident − dict_ident| ≤ 9, AND the number of rows where the two disagree on EM ≤ 9.
- **P4** dict_ident − dict_exact ≥ +6 and CI > 0.
- **Determinism (descriptive):** per-row direct and CoT EM, and the direct text, compared with V-155 row by row.

## Predictions, written before the run

- **P1 FAILS, P2 FAILS** under EM and under containment. The pilot had symbolic ≈ .06 against direct .24 and CoT
  .32. Lenient scoring raises the LM arms more than the symbolic ones, because the symbolic arms return extracted
  surface strings.
- **P3 HOLDS.** Table and HRR now share every rule except retrieval, and retrieval agrees on ~99% of queries at
  this load.
- **P4 FAILS.** I expect it to fail more likely than not: the review's probe shows mean-pooled isolated-phrase
  embeddings are a weak alias detector (Paris / "Paris, France" cosine .39) that also merges distinct entities.
- **Determinism:** direct and CoT identical to V-155 in ≥ 297/300 rows (same GPU type, same batches; bf16 batch
  kernels are not guaranteed bitwise stable across runs).

## Multi-angle plan (a verdict is a conclusion only after these)

1. `v158_indep.py`:
   - independent normaliser and parsers;
   - my own exact-string walk;
   - an outcome decomposition of where the symbolic route loses;
   - exact McNemar for P1 and P2.
2. Manual reading of 30 random rows' raw outputs (direct, CoT, decomposition, triples, symbolic prediction).
3. Same pipeline on a second model, and on a second dataset (MuSiQue 2-hop). Each is registered separately before
   it runs.
