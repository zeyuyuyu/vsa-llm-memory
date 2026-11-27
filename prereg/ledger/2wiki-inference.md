# Ledger excerpt: Store on 2WikiMultihopQA inference-type questions

Supplementary Table `tab:prereg`, row "2Wiki (store on inference-type questions)" (marked "ledger" in the Supplementary Information).

**Source.** `ledger.md` of the experiment repository `ngcls` (private; see `README.md` in this folder), at commit `0732e13` (`0732e13ef2322f308a99dcac3bf0bd2e25a0da10`, tag `archive-2026-11-11`); the copy read has MD5 `6843647bad4782ac58fcd341448bf95c`, identical to that commit's working tree. Line numbers refer to that file. For each part, the commit is the one that added those lines, from `git blame` and checked with `git log -S` on the heading; dates are commit author dates in UTC+08:00 (the machine's clock). Parts not from `ledger.md` say so.

| # | Part | Ledger lines | Heading date | Added in commit | Commit date |
|---|---|---|---|---|---|
| 1 | Context: PRE-REGISTRATION V-65 (the protocol V-69 reuses) | 14227-14236 | none in heading | `679939f` | 2026-08-24T20:41:22+08:00 |
| 2 | Context: verdict V-65 | 14238-14242 | none in heading | `9efa172` | 2026-08-24T21:20:54+08:00 |
| 3 | PRE-REGISTRATION: V-69 | 14320-14323 | none in heading | `c0722ee` | 2026-08-25T04:16:59+08:00 |
| 4 | Verdict: V-69 | 14330-14333 | none in heading | `6e6a0c9` | 2026-08-25T04:29:28+08:00 |

**Depositor's notes** (written when this excerpt was deposited, 2026-11-21; not part of the ledger).

- **Row in the paper.** "prediction: store >= 0.90"; outcome "failed (0.762)". The Supplementary Note adds a cause: "because the chain follows the evidence order, which on this question type is not always the chain order".
- **Matching registration.** V-69 (part 3) runs V-65's three arms (part 1) on type=inference: P1 ordering hrr_gold >= prompt_ev >= prompt in 3/3, P2 hrr_gold - prompt >= 0.20, P3 hrr_gold >= 0.90. Verdict (part 4): P1 hit, P2 hit (+0.669), P3 missed (0.7622). The bar and the outcome match the row.
- **Differences from the row.** (1) The row reports only P3; P1 and P2 were registered too and held. (2) The scorer printed the wrong P3 label; the verdict was re-derived by hand against the registered bars (stated in part 4). (3) The ledger states the cause only as a hypothesis, "to be checked against the failure split before any mechanism sentence enters the paper"; no later ledger entry found in a search records that check. The paper states it as the reason.
- V-69's registration was committed together with V-70's (`c0722ee`).

---

## Part 1: Context: PRE-REGISTRATION V-65 (the protocol V-69 reuses) (ledger.md lines 14227-14236)

SHA-256 of the lines below, each followed by a newline: `a7b33f290d0541433f7520587318c3bd373470dc0814b4a02b0c66f02be6eb7b`.

````text
## V-65 prereg, before any code: the mainstream-dataset test (2WikiMultihopQA)
Purpose: measure the paper's claim where readers live — on a mainstream multi-hop QA benchmark (Ho et al., COLING 2020), as an error-rate reduction under the frozen-deployment condition. Dataset: framolfese/2WikiMultihopQA via hf-mirror; SUBSET: type=compositional (pure two-hop bridge chains), dev split, N=300; model phi-2 frozen throughout (no fine-tune in this experiment).
Arms, all seeing the same instances:
  (1) prompt      — few-shot phi-2, context passages (truncated to fit 2048: supporting sentences first, then distractor paragraphs until budget; truncation stats reported), generate, exact-match after standard normalisation. This is the model's single pass.
  (2) prompt_ev   — few-shot phi-2, the GOLD evidence triples verbalised in the prompt plus the question. Equal information, in-context route.
  (3) hrr_gold    — HRR store over the same gold triples with STRING-IDENTITY symbol codes (fixed random vector per distinct entity/relation string; supplying identity externally, which is the paper's stated requirement): key = subject (x) relation, value = object; two unbind+cleanup hops following the question's relation chain read from the gold evidence order; answer = retrieved entity string, same EM.
Seeds 42/123/456 (few-shot sampling and codebook init). Declared diagnostics: answer-coverage of gold triples (fraction of instances whose answer string appears as a triple object); context-truncation rate; per-arm EM on the covered slice AND the full set (primary metric: covered slice, declared now).
Condition check, declared: if arm (1) >= .85 EM, the single pass does not fail here and no gain is claimable — reported as such.
Predictions, registered so they can fail: (P1) ordering hrr_gold >= prompt_ev >= prompt in 3/3 seeds; (P2) hrr_gold - prompt >= .20 EM on the covered slice; (P3) prompt_ev sits strictly between — the model inherits evidence but errs at composition (V-59's inheritance result predicts this).
Negative readings, committed now: if hrr_gold <= prompt_ev, the algebra adds nothing over in-context evidence on natural QA and that boundary enters the paper; if answer-coverage < .80 the evidence route is ill-posed on this subset and coverage itself is the reported result. Gold triples are supervision arms (1)-(2) also receive in usable form; an extraction-parser arm (no gold) is a REGISTERED FOLLOW-UP (V-66), not smuggled into this verdict.
````

## Part 2: Context: verdict V-65 (ledger.md lines 14238-14242)

SHA-256 of the lines below, each followed by a newline: `6312fac934a46bb84c5fb5a35f952a9a75c0b7428ea2860ab645e9b1a55fbe30`.

````text
## V-65 verdict (prereg 679939f; artifact v65_logs/v65_BANKED.json)
300 compositional dev instances, answer-coverage 1.000 (covered slice = full set), context-truncation rate 0.0 (the scoring line printed "trunc -1.000" because a falsy 0.0 fell through an `or -1` display default — display bug in my scorer, value in the banked JSON is 0.0).
prompt .2533 (identical across seeds: greedy decode, fixed shots — expected), prompt_ev .9500 (identical, same reason), hrr_gold .9700/.9767/.9767 mean .9744 (codebook-init noise only).
All three registered predictions HIT: P1 ordering hrr_gold >= prompt_ev >= prompt in 3/3; P2 gain vs single pass .721 >= .20; P3 prompt_ev strictly between.
Honest reading, transcribed with the verdict: most of the gain is EVIDENCE ACCESS (+70 points of it available in-prompt); the algebra's own contribution over the same evidence is halving the residual error, 5.0% -> 2.6% — the model errs at composition even with the facts in front of it (V-59's inheritance result predicted exactly this), the algebra composes exactly and its residue is cleanup noise. Gold triples are supervision ALL arms receive in usable form; the no-gold extraction arm is V-66, registered follow-up, not claimed here.
````

## Part 3: PRE-REGISTRATION: V-69 (ledger.md lines 14320-14323)

SHA-256 of the lines below, each followed by a newline: `4d75615a845ca6cbfb347cacd10304403dfb4b6b884be943c3462958e39723fa`.

````text
## V-69 prereg, before any code: breadth across question types (2WikiMultihopQA, inference type)
V-65 measured one question type (compositional). V-69 runs the identical three arms on type=inference (two-hop chains through a logical rule), 300 dev instances, same models, same EM, same declared run-once greedy protocol. Floor/parser refs not applicable; the three arms are self-contained as in V-65.
Predictions, registered so they can fail: (P1) ordering hrr_gold >= prompt_ev >= prompt in 3/3 codebook seeds; (P2) hrr_gold - prompt >= .20; (P3) hrr_gold >= .90 (gold triples + string-identity codes should be near-exact if the type is really a two-hop chain).
Negative reading, committed now: if the gain does not replicate on this type, the mainstream claim stays scoped to compositional and the paper says which structural property of the type broke it (diagnosed from the failure split, then stated, not guessed).
````

## Part 4: Verdict: V-69 (ledger.md lines 14330-14333)

SHA-256 of the lines below, each followed by a newline: `2ab80d2c00f5fd0c0bac5a18c64e0c18c3bbc4940e64e0e732d6767b8ba48267`.

````text
## V-69 verdict (prereg c0722ee; artifact v69_logs/v69_BANKED.json; verdict re-derived against the REGISTERED bars)
Scorer fault, recorded first: v69.py inherited v65's P3 label ("strict middle") instead of the registered P3 (hrr_gold >= .90); the printed verdict line is therefore mislabelled and the verdict below is re-derived externally against c0722ee. The numbers themselves are unaffected.
Inference type, 300 dev instances, coverage 1.000: prompt .0933, prompt_ev .6833, hrr_gold .7567/.7433/.7867 mean .7622.
(P1) ordering in 3/3: HIT. (P2) gain vs single pass .669 >= .20: HIT — the mainstream gain REPLICATES on a second question type, and is larger (+67 points; the model with the facts in front of it reaches only .68 on this type, and the store still beats that by +8). (P3) hrr_gold >= .90: MISSED (.7622). The algebra is not near-exact here; hypothesis for the miss, stated as hypothesis: the chain follows gold-evidence order, and inference-type evidence order need not be chain order — to be checked against the failure split before any mechanism sentence enters the paper.
````
