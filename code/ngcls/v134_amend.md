# V-134 AMENDMENT — an instrument error of mine that ran AGAINST the algebra, and how it is fixed

Written after two pre-data diagnostics and before the registered V-134 run. Labelled as an after-the-fact
amendment, per the V-132 and cf5d8d4 precedents.

## The error

Both V-133 and the V-134 smoke used `v133_store.answer_chain`, which superposes every fact into **one**
vector. The store the manuscript actually describes and deploys **shards**: V-102 (prereg cc5ae08) measured
512 facts at `d = 2048` reading 0.039 in one store and 0.919 across eight, because capacity is set by facts
per shard rather than by the total, and cross-shard selection costs about a point.

Running the unsharded version above roughly a hundred facts therefore understates the algebra **against the
paper's own published remedy**. That is my error, and it is the direction that made my own side look worse.

## Diagnostic 1 — why the V-134 knowledge base is hard for the store

Fact count and dimension held constant at 160 facts and `d = 4096`; only the entity pool changes:

| entities x edges | facts | single-hop readout |
| --- | ---: | ---: |
| 40 x 4 | 160 | 0.785 |
| 80 x 2 | 160 | 0.935 |
| 160 x 1 | 160 | 0.910 |

Re-using an entity vector across many bindings produces **structured** rather than random interference, and
it costs about fifteen points. Dimension does not buy it back: 0.785 at `d = 4096`, 0.800 at 8192, 0.800 at
16384 — saturated, consistent with the manuscript's own report that `k*` stalls at 256 facts beyond 4096.

Pushing further, at `d = 4096` with a 400-entity pool: 400 facts read 0.555, 1600 facts read 0.110, and
4000 facts read 0.010. **An unsharded store cannot hold a knowledge base of even modest size.**

This is kept as a finding, not engineered away. Real knowledge graphs have high-degree nodes, so degradation
under entity re-use is a property of the approach that belongs in the record.

## Diagnostic 2 — what sharding changes

Gate first: at `S = 1` the sharded implementation reproduces the unsharded one to the digit at `k = 2, 4, 6`,
so the wrapper did not change the path.

On the **V-133** instances (120 facts, `S = 2` by the manuscript's ~64-facts-per-shard rule), `d = 4096`,
three codebook seeds, with the language-model arms untouched:

| k | store, registered | store, sharded | cot | gap, registered | gap, sharded |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.9600 | 0.9889 | 0.9200 | +0.0400 | +0.0689 |
| 3 | 0.8956 | 0.9489 | 0.9667 | −0.0711 | −0.0178 |
| 4 | 0.7978 | 0.9089 | 0.9333 | −0.1356 | −0.0244 |
| 5 | 0.7622 | 0.8644 | 0.9467 | −0.1844 | −0.0822 |
| 6 | 0.7267 | 0.8511 | 0.9200 | −0.1933 | −0.0689 |

**V-133's conclusion survives but its magnitude was overstated by me.** `gap(6) − gap(2)` moves from
−0.2333 to −0.1378, so P2 still fails against its `+0.20` bar and "the gap widens with depth" is still
refuted. What is *not* true, and what I wrote to the user before this check, is that the algebra loses badly
at depth: with the paper's own implementation it stays within seven points of a 32B reader with a
scratchpad at every depth from two to six, while making no model call at all. Both columns are banked
(`v133a.json`, `v133a_sharded.json`) and both are reported.

## The amendment to V-134

1. **The store arm uses the sharded implementation.** Shard count follows the manuscript's published rule,
   about 64 facts per shard, which at 160 facts gives `S = 3`. That rule is fixed here, before the data, and
   is not chosen by looking at accuracy. The full `S ∈ {1, 2, 3, 4, 8}` curve is recorded as descriptive so
   the choice is visible: at `S = 8` the store reads higher still, and I am *not* taking that number as
   primary precisely because selecting it would be choosing a configuration on the outcome.
2. **LoRA epochs rise from 2 to the registered 12, and further if the memorisation gate still fails.** The
   smoke reached only 0.375 at one hop after 2 epochs against a P0(a) bar of 0.80. As the prereg already
   states, strengthening the fine-tune is a concession to the opponent, not a thumb on the scale.
3. Bars P0, P1, P2 and P3 are carried over **byte-identical**. The frozen `v134_score.py` is not edited.

## Net effect on who this helps

Change 1 helps my side and change 2 helps the opponent, so the pair is not directionally self-serving. The
justification for change 1 is independent of any outcome: the sharded store is the system the manuscript
describes, its remedy was published in V-102 before this experiment existed, and the `S = 1` gate proves the
two implementations agree where they should.
