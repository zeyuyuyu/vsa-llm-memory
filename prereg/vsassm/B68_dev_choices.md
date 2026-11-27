# B68 dev choices (committed before the finals of the cells they cover; dev split only)

Rule (prereg): binding-loss lr in {3e-4, 1e-3} by best-epoch dev, ties within 0.002 to 3e-4; per family the grid
configuration with the highest best-epoch dev accuracy (perm/named), ties within 0.002 to the first listed. Dev
outputs are in out/tune/ and out/g1/ on the box; selection files are copied to b68_out/.

Procedure note: the registered batches are per arena; to keep the GPUs busy, a batch is released in sub-batches as soon
as all 64 dev runs of each of its cells exist. Every sub-batch's choices are committed before its finals are queued; no
cell's finals run before its own commit.

## Batch A1 (07:43 UTC): qwen8L0_A, qwen8L4_A, qwen8L2_A (b68_out/select_A1.json)

select.json md5 5e61809fe423487529958ae4c1fe1cf4; complete cells 3, incomplete {"phi2L0_A": 4, "phi2L4_A": 30, "phi2L2_A": 64, "qwen8L0_B": 64, "qwen8L4_B": 64, "qwen8L2_B": 64, "phi2L0_B": 64, "phi2L4_B": 64, "phi2L2_B": 64, "qwen8L0_C": 64, "qwen8L4_C": 64, "qwen8L2_C": 64, "phi2L0_C": 64, "phi2L4_C": 64, "phi2L2_C": 64}

| cell | bind lr (dev 3e-4 / 1e-3) | vq | dkvb | gumbel | st | cw |
|---|---|---|---|---|---|---|
| qwen8L0_A | 0.0003 (0.534 / 0.524) | 0.544 ep 10 (cleanup hard, inw 1, lr 0.0003, mult 4) | 0.984 ep 1 (cleanup hard, heads 1, lr 0.001, mult 1) | 0.542 ep 7 (cleanup hard, div 0.1, lr 0.001) | 0.550 ep 6 (cleanup hard, div 0, lr 0.001) | 0.550 ep 10 (cleanup hard, code center, inw 1, lr 0.001) |
| qwen8L4_A | 0.0003 (0.540 / 0.526) | 0.630 ep 1 (cleanup hard, inw 1, lr 0.0003, mult 1) | 0.568 ep 1 (cleanup st, heads 1, lr 0.001, mult 4) | 0.554 ep 3 (cleanup st, div 0, lr 0.001) | 0.560 ep 7 (cleanup st, div 0.1, lr 0.0003) | 0.562 ep 7 (cleanup hard, code zca, inw 0, lr 0.001) |
| qwen8L2_A | 0.0003 (0.514 / 0.514) | 0.820 ep 1 (cleanup st, inw 1, lr 0.001, mult 1) | 0.560 ep 1 (cleanup st, heads 8, lr 0.001, mult 4) | 0.542 ep 2 (cleanup st, div 0, lr 0.001) | 0.552 ep 10 (cleanup hard, div 1, lr 0.0003) | 0.554 ep 3 (cleanup hard, code zca, inw 1, lr 0.001) |

## Batch A2 (07:57 UTC): phi2L0_A (b68_out/select_A2.json)

select.json md5 efd9d16ea137b0d74fd6d8946a469b0d; complete cells 4, incomplete {"phi2L4_A": 1, "phi2L2_A": 28, "qwen8L0_B": 64, "qwen8L4_B": 64, "qwen8L2_B": 64, "phi2L0_B": 64, "phi2L4_B": 64, "phi2L2_B": 64, "qwen8L0_C": 64, "qwen8L4_C": 64, "qwen8L2_C": 64, "phi2L0_C": 64, "phi2L4_C": 64, "phi2L2_C": 64}

| cell | bind lr (dev 3e-4 / 1e-3) | vq | dkvb | gumbel | st | cw |
|---|---|---|---|---|---|---|
| phi2L0_A | 0.0003 (0.542 / 0.538) | 0.988 ep 3 (cleanup hard, inw 1, lr 0.001, mult 1) | 0.994 ep 2 (cleanup hard, heads 1, lr 0.001, mult 4) | 0.542 ep 9 (cleanup hard, div 1, lr 0.0003) | 0.554 ep 10 (cleanup hard, div 0.1, lr 0.0003) | 0.986 ep 2 (cleanup hard, code center, inw 1, lr 0.0003) |

## Batch A3 (07:58 UTC): phi2L4_A (b68_out/select_A3.json)

select.json md5 cdb06da15617803f5f4ff987ba16dd73; complete cells 5, incomplete {"phi2L2_A": 14, "qwen8L0_B": 62, "qwen8L4_B": 64, "qwen8L2_B": 64, "phi2L0_B": 64, "phi2L4_B": 64, "phi2L2_B": 64, "qwen8L0_C": 64, "qwen8L4_C": 64, "qwen8L2_C": 64, "phi2L0_C": 64, "phi2L4_C": 64, "phi2L2_C": 64}

| cell | bind lr (dev 3e-4 / 1e-3) | vq | dkvb | gumbel | st | cw |
|---|---|---|---|---|---|---|
| phi2L4_A | 0.001 (0.544 / 0.550) | 0.964 ep 6 (cleanup hard, inw 1, lr 0.0003, mult 1) | 0.556 ep 5 (cleanup hard, heads 1, lr 0.001, mult 1) | 0.556 ep 1 (cleanup hard, div 0, lr 0.0003) | 0.552 ep 10 (cleanup st, div 0, lr 0.0003) | 0.876 ep 2 (cleanup hard, code zca, inw 0, lr 0.0003) |

## Batch A4 (08:14 UTC): phi2L2_A (b68_out/select_A4.json)

select.json md5 febc84f952a48b2c89267c4c9837b434; complete cells 7, incomplete {"qwen8L4_B": 4, "qwen8L2_B": 41, "phi2L0_B": 64, "phi2L4_B": 64, "phi2L2_B": 64, "qwen8L0_C": 64, "qwen8L4_C": 64, "qwen8L2_C": 64, "phi2L0_C": 64, "phi2L4_C": 64, "phi2L2_C": 64}

| cell | bind lr (dev 3e-4 / 1e-3) | vq | dkvb | gumbel | st | cw |
|---|---|---|---|---|---|---|
| phi2L2_A | 0.001 (0.558 / 0.564) | 0.976 ep 10 (cleanup st, inw 1, lr 0.0003, mult 1) | 0.558 ep 8 (cleanup hard, heads 8, lr 0.001, mult 1) | 0.550 ep 2 (cleanup hard, div 1, lr 0.0003) | 0.540 ep 5 (cleanup hard, div 1, lr 0.0003) | 0.828 ep 5 (cleanup hard, code zca, inw 0, lr 0.0003) |

## Batch B1 (08:15 UTC): qwen8L0_B (b68_out/select_B1.json)

select.json md5 2dfc02379e78017fd87d041b0041fcd2; complete cells 7, incomplete {"qwen8L4_B": 4, "qwen8L2_B": 33, "phi2L0_B": 64, "phi2L4_B": 64, "phi2L2_B": 64, "qwen8L0_C": 64, "qwen8L4_C": 64, "qwen8L2_C": 64, "phi2L0_C": 64, "phi2L4_C": 64, "phi2L2_C": 64}

| cell | bind lr (dev 3e-4 / 1e-3) | vq | dkvb | gumbel | st | cw |
|---|---|---|---|---|---|---|
| qwen8L0_B | 0.0003 (0.550 / 0.526) | 0.702 ep 10 (cleanup st, inw 1, lr 0.0003, mult 1) | 0.992 ep 5 (cleanup hard, heads 1, lr 0.001, mult 1) | 0.934 ep 9 (cleanup hard, div 1, lr 0.001) | 0.932 ep 8 (cleanup hard, div 1, lr 0.001) | 0.638 ep 5 (cleanup hard, code inst, inw 1, lr 0.0003) |

## Batch B2 (08:19 UTC): qwen8L4_B (b68_out/select_B2.json)

select.json md5 53c786a1d2067478c58b2126550f35a2; complete cells 8, incomplete {"qwen8L2_B": 3, "phi2L0_B": 59, "phi2L4_B": 64, "phi2L2_B": 64, "qwen8L0_C": 64, "qwen8L4_C": 64, "qwen8L2_C": 64, "phi2L0_C": 64, "phi2L4_C": 64, "phi2L2_C": 64}

| cell | bind lr (dev 3e-4 / 1e-3) | vq | dkvb | gumbel | st | cw |
|---|---|---|---|---|---|---|
| qwen8L4_B | 0.001 (0.582 / 0.586) | 0.602 ep 5 (cleanup hard, inw 0, lr 0.001, mult 1) | 0.574 ep 3 (cleanup hard, heads 1, lr 0.001, mult 1) | 0.538 ep 1 (cleanup hard, div 0.1, lr 0.001) | 0.634 ep 10 (cleanup hard, div 1, lr 0.001) | 0.584 ep 4 (cleanup hard, code inst, inw 0, lr 0.0003) |

## Batch B3 (08:24 UTC): qwen8L2_B (b68_out/select_B3.json)

select.json md5 a21af3ddacf9965d488c275d47ee12cb; complete cells 9, incomplete {"phi2L0_B": 10, "phi2L4_B": 64, "phi2L2_B": 64, "qwen8L0_C": 64, "qwen8L4_C": 64, "qwen8L2_C": 64, "phi2L0_C": 64, "phi2L4_C": 64, "phi2L2_C": 64}

| cell | bind lr (dev 3e-4 / 1e-3) | vq | dkvb | gumbel | st | cw |
|---|---|---|---|---|---|---|
| qwen8L2_B | 0.0003 (0.516 / 0.518) | 0.950 ep 3 (cleanup hard, inw 1, lr 0.0003, mult 1) | 0.848 ep 1 (cleanup hard, heads 1, lr 0.001, mult 1) | 0.650 ep 8 (cleanup hard, div 1, lr 0.001) | 0.766 ep 10 (cleanup hard, div 1, lr 0.001) | 0.566 ep 1 (cleanup hard, code zca, inw 0, lr 0.0003) |

## Batch B4 (08:25 UTC): phi2L0_B (b68_out/select_B4.json)

select.json md5 633c61c2b0338b00d2fa6ced7d2d4a0a; complete cells 10, incomplete {"phi2L4_B": 64, "phi2L2_B": 64, "qwen8L0_C": 64, "qwen8L4_C": 64, "qwen8L2_C": 64, "phi2L0_C": 64, "phi2L4_C": 64, "phi2L2_C": 64}

| cell | bind lr (dev 3e-4 / 1e-3) | vq | dkvb | gumbel | st | cw |
|---|---|---|---|---|---|---|
| phi2L0_B | 0.0003 (0.518 / 0.496) | 0.978 ep 1 (cleanup hard, inw 1, lr 0.0003, mult 1) | 1.000 ep 2 (cleanup hard, heads 1, lr 0.001, mult 4) | 0.950 ep 10 (cleanup hard, div 1, lr 0.001) | 0.970 ep 9 (cleanup hard, div 1, lr 0.0003) | 0.820 ep 1 (cleanup hard, code inst, inw 1, lr 0.0003) |
