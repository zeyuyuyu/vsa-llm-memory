## V-140 prereg — does fine-tuning confer robustness to distraction, or only the store's structure?

Registered before the run. Sharpened from V-138/V-139, which together say: fine-tuning succeeds where the
evidence is already short and clean in the input (bAbI task 2 standard 0.9722 and held-out combinations 1.000;
2Wiki with gold triples ~0.967) and fails where the model must locate or bind the evidence itself (arena C at
the frozen-probe floor; 2Wiki raw passages 0.283 against a 0.253 untrained single pass). This experiment holds
the TASK fixed and varies only the amount of material the model must bind through -- the axis the manuscript
already measures for prompting and for the store, and the one arm it never ran.

Banked references on the same 300 compositional rows (V-65, V-72b):
    gold facts in the prompt, no distractors      prompt_ev  0.950
    gold facts pooled with ~150 others' facts      prompt_ev  0.353
    store, no distractors                          hrr_gold   0.974
    store at the 150-fact pool                     hrr_gold   0.953

New arm: LoRA (rank 16, 6 epochs, lr 1e-4, three seeds) fine-tuned and evaluated with the SAME context the
prompt arm sees at each distractor level, N in {0, 50, 150}; distractor triples drawn from other instances,
as in V-72b. Training rows carry their own distractors at the same N.

BARS (fixed now):
  P1 (primary) fine-tune at N=150 relative to its own N=0:
      drop >= 30 points while the store drops <= 5  -> EVIDENCE: fine-tuning does not buy robustness to
            distraction on real text; what does is the store's structure, not more training.
      drop <= 10 points                              -> REFUTED: fine-tuning fixes the collapse that prompting
            suffers, and the pooled-evidence result needs the fine-tune reported beside it as a competitor.
      otherwise -> partial, reported as measured.
  P2 (control) the fine-tune at N=0 must land within 5 points of the 0.967 smoke value; otherwise the setting
      has changed and P1 is not interpretable.
  P3 (gate) evaluation rows identical to V-65's 300; train/eval question overlap must be zero.

Disclosure carried from V-138: the store is handed gold triples and the fine-tune is handed the same triples
here, so at last both arms see the same evidence -- this IS an apples-to-apples comparison, unlike V-138.
