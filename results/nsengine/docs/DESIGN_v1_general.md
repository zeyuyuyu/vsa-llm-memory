# Design v1: a general reasoning engine (2026-11-15)

User direction 2026-11-15: the goal is a **general** reasoning engine, not a tool for one domain. v0 (docs/DESIGN.md)
built one DSL and one solver per family (SRE, ZebraLogic). v1 replaces both with one modelling language, one solver
layer and one certification layer that does not know the problem type.

## What carries over from v0 (our own results)
- The solver was never wrong; every error was reading or modelling (NSR1-NSR7b).
- On large search a frontier model alone fails, and fails confidently: NSR6b 130-180 hosts, gpt-6.1-sol direct 0.30;
  with 128k thinking 0.575, with 17 of 40 answers confidently wrong. Frontier + code is good but not safe: 0.975, with
  one silent miss (it dropped 1 of 299 constraint pairs).
- Agreement of readings by one model, checked only at answer level, is not a certificate on unseen wordings: NSR7b
  certified 20 wrong in 300. Two things helped:
  - agreement on *why* an answer holds ("impossible" certified only if its minimal core holds under every reading;
    dev 4 -> 1 wrong);
  - readings of a different *kind* (consequence probes caught 92% of the shared misreads).
- Win condition (He & Wang, arXiv 2512.03272): search-heavy problems whose text is easy to read. On shallow deduction
  the frontier model ties or wins.

## Architecture
1. **One modelling language.** The language model writes a declarative constraint model in Python against
   **CPMpy** (int/bool variables, global constraints, sums, objectives, solveAll, MUS tools; OR-Tools and Z3
   backends). Why CPMpy: CP-Bench (Michailidis, Tsouros & Guns, arXiv 2506.06052) found high-level Python modelling
   easiest for LLMs, and an API the models already know needs no grammar in the prompt. The model must:
   - number-tag every constraint with the sentence(s) it encodes, `add(c, "S3")`, from a numbered copy of the problem
     text;
   - expose the answer as named variables following the question's answer schema (`answer({...})`);
   - declare the question kind: `find` (any solution), `unique` (the solution, claimed unique), `all` (every solution),
     `optimize` (min/max), `check` (is it possible?).
   It runs in the no-network Docker sandbox, under a time limit.
2. **Solver layer** (the harness, not LM code), primitives over any submitted model:
   - solve;
   - solve with the answer variables fixed to a given assignment (does this answer satisfy that model?);
   - block a solution and re-solve (uniqueness);
   - enumerate up to a cap;
   - optimum value;
   - a minimal unsatisfiable subset of the tagged constraints, mapped back to sentence ids.
3. **Certification layer** (type-agnostic). K independent formalisations, from different prompts, samples and model
   families. The primary answer A is certified only if every formalisation M_i:
   - **find:** accepts A (A fixed -> satisfiable).
   - **unique / all / optimize:** gives the same answer set or optimum value.
   - **check = impossible:** is unsatisfiable, *and* the minimal cores cite the same sentences. This is the general
     form of the NSR7b core rule: agreement on the reason, not only on the verdict.
   - **Scenario probes** (the general form of NSR7b's P): where two formalisations disagree on a concrete assignment,
     or on the primary answer with one tagged constraint dropped, the solver writes the scenario out and a reader of
     another family judges it against the text. Any unresolved disagreement means abstain.

   Output: a certified answer with its certificate (the answer, the models that accept it, the sentences the deciding
   constraints come from), or an abstention that names the sentences in dispute.
4. **Translators.** A frontier model (strong tier). Local models of two families, Qwen3-8B and IBM Granite 3.3 8B
   (cheap tier, NSR7c), plus Qwen3-32B where it fits.
5. **Explainer.** A certificate in plain language (from v0's engine/certify.explain, generalised to sentence ids).

## Prior art (searched 2026-11-15; read from abstracts, to be read in full before any claim)
- CP-Bench (arXiv 2506.06052): 101 combinatorial problems with runnable CP models. Python frameworks do best; up to 70%
  with repeated sampling and self-verification.
- Text2Zinc (Singirikonda, Kadioglu & Uppuluri 2025): 110 problems across 11 domains, NL -> MiniZinc.
- Uncertainty for SMT autoformalisation (arXiv 2505.20047): consistency across several SMT generations strongly
  predicts correctness (the agree-or-abstain signal).
- Draft-and-Prune (arXiv 2603.17233): sample formalisations, prune contradictory or ambiguous ones, aggregate. The
  authors state there is no guarantee of semantic correctness.
- Round-trip verification and repair (arXiv 2604.25031): re-formalise through back-translation and check SMT
  equivalence.
- Also LINC (vote over FOL translations), Logic-LM, SatLM, Logic.py, NATURAL PLAN (arXiv 2406.04520).

Not new: LM + solver, sampling several formalisations, consistency as a confidence signal. What we can own, if the
data supports it:
- (a) a certify-or-abstain contract with a **preregistered error bar measured on held-out problem families**;
- (b) agreement on the reason (cores mapped to sentences) and other-family scenario probes as the certificate;
- (c) scale where frontier models fail confidently (NSR6b);
- (d) a cheap local tier.

## Evaluation plan
Families: ZebraLogic (external; we have it), SRE (ours, now written in the general language with its rules in the
text, no SRE DSL), CP-Bench (external), Text2Zinc (external), NATURAL PLAN calendar/meeting (external), and new
generated families with a size knob. Development uses some families; the test uses families the translators' examples
and our prompt work never touched. Arms:
- frontier direct;
- frontier + code (one CPMpy model, no checks; the CP-Bench setting);
- engine certified, with frontier and local translators.

Measures: certified-slice error (the bar), coverage, and accuracy against size.

## Milestones
- **G0:** modelling harness and solver primitives (above). Gold re-encodings of ZebraLogic and SRE written once by
  hand in CPMpy and solved to gold = 1.000 (expressiveness sanity). CP-Bench reference models run through the harness.
- **G1:** one generic translator prompt (examples from a separate pool, none from a test family). Development
  measurement on ZebraLogic + SRE + a CP-Bench dev split, frontier and local.
- **G2:** certification layer; development measurement of certified error and coverage.
- **G3 (NSR8 prereg):** held-out families, bars frozen before the items are opened.

## What happens to NSR7c
Parked at development. The Granite readings on the burned SRE sets finish; their dev numbers are written down. Its two
ideas (the core / reason check and other-family readers) become part of the certification layer above, tested there on
held-out families rather than on a fourth SRE wording set.
