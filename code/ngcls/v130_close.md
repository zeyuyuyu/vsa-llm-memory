
## 2026-09-11 — V-130 CLOSED (scorer verbatim, raw banked 70bcbaa before scoring): P0a/P0b EXACT; **P1 FAILS 0/3, P2 FAILS 0/3 — the user's hypothesis is right for an 8B reader and WRONG for every frontier reader we can reach. V-106's evidence-scale advantage is a small-model finding.**

| K | store | Qwen3-8B | deepseek-v4-flash | glm-5.3 | kimi-k3 |
|---|---|---|---|---|---|
| 8 | .9361 | .9500 | .9750 | 1.0000 | 1.0000 |
| 64 | .9333 | .9083 | .9667 | 1.0000 | 1.0000 |
| **100** | .9250 | **.8250** | **.9750** | **1.0000** | **1.0000** |
| 128 | .9194 | .8500 | .9667 | .9917 | .9917 |
| 512 | .9194 | .5750 | .9750 | .9917 | 1.0000 |
| 1024 | .9083 | .5417 | .9500 | 1.0000 | 1.0000 |

All 24 reader cells 120/120 (glm-5.3 and kimi-k3 both complete this time; no empty-content failures).

**Gates.** P0a: the store reproduces V-106 **to the digit** at all five shared K. P0b: the re-run 8B arm
reproduces V-106 to the digit at four of five K and within one row at K=512 (.5750 vs .5833). The renderer
copy, the fact pool and the store path are certified; what follows is about the readers.

**P1 (PRIMARY) FAILS, 0/3.** At K=1024 store − reader is **negative** for every frontier reader:
deepseek −.0417 [−.1056, +.0222], glm-5.3 −.0917 [−.1417, −.0472], kimi-k3 −.0917 [−.1417, −.0472]. Two
of the three CIs exclude zero *on the wrong side*: the reader beats the store. The 8B reference on the same
rows is +.3667 [+.2667, +.4667].

**P2 FAILS, 0/3.** At the user's K=100: deepseek −.0500 [−.1056, +.0028], glm-5.3 and kimi-k3 −.0750
[−.1222, −.0333]. 8B: +.1000 [+.0167, +.1833] — so the user's claim is confirmed at 8B and refuted at
frontier on the same 120 rows and the same 100 facts.

**P3.** The store never leads a frontier reader by .02 at any K. Drop from K=8 to K=1024: 8B −.4083,
deepseek −.0250, glm-5.3 and kimi-k3 **0.0000**. Two frontier models read 1024 pipe-delimited facts
(~11.5k tokens) with no measurable loss.

**P4 (what survives).** At K=1024 the store answers in .086 s against 11.2–11.5k prompt tokens and
4.0–7.1 s for the frontier readers (1.9 s for local 8B). The cost claim stands; the accuracy claim does
not.

**Reading, as registered: reported first.** The collapse of structured plain text at scale is a property
of the 8B reader, not of reading structured text. It joins V-108's "algebra halves the residual" as a
small-model claim. **Consequence for the ICLR paper, applied in the same commit that reports this:** Sec
3.1's evidence-scale subsection is restricted to the 8B reader and states the frontier result; the store's
advantage on this axis is *cost*, not accuracy. My registered expectation (P1 and P2 fail) held; the user's
did not, and that is recorded as such.

**What this does NOT touch.** Every arm received gold triples. The acquisition limit (V-98/V-129-void) is
unchanged. The K≈8 in-instance ladder (V-108/V-124, single_pass .49–.55 vs store .978) is a different
axis — there the model reads *prose* and must compose; here it reads *already-structured* lines. Put
together, the two axes say the same thing the paper already says: the lever is structuring the evidence,
and once the evidence is structured a frontier reader needs no algebra to use it, at any scale we tested.

**Literature check (mandate).** Consistent with Liu et al. 2023 ("Lost in the Middle"), which found
position-dependent degradation in 2023-era models over multi-document contexts, and with the 2024–26
needle-in-a-haystack results on which frontier models saturate for clean retrieval targets; no conflicting
result found for pipe-delimited fact lists specifically. Recorded as "consistent, extends to the structured
case".
