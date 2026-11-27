
## 2026-09-11 — CORRECTION to the 2026-09-10 defect entry (795059b): v102.sharded_two_hop is NOT an elementwise self-inverse store and does NOT clean up over 200,003 rows

Reading `v102.py` line by line while auditing the npj Methods' "every measurement uses circular-convolution
HRR" sentence: `bind(a, b) = irfft(rfft(a) * rfft(b))` — **circular convolution**, the same operator as
`v65.hrr_answer`; `unbind(ab, a) = bind(involution(a), ab)` — correlation; `codebook()` returns **unit-norm
Gaussian** rows, not bipolar; and `sharded_two_hop` cleans up with `cosine_similarity(OV, hat)` where `OV`
is the stack of **object vectors present in that shard** — the cleanup candidate set is the K stored objects
(the whole pool at the pooled settings), not the 200,003-row codebook, and there is no abstain floor.

So the 795059b entry was right that the ICLR Setup paragraph described the wrong module, and right about
the remedy (re-base on v102 — the measured .9767/.9833/.9733 stand), but wrong about **what v102 is**:
it attributed to v102 an elementwise product, a 200,003-way cleanup and an abstain floor that exist in
neither implementation. Those three specifics were the manuscript's own text, and I repeated them into the
ledger as if they were the code. What actually separates v102 from v65: a fixed 200,003-row codebook
indexed by a hash of the normalised string (v65: fresh per-instance vectors), d=2048 (v65: 4096), relation
discarded (v65: bound and queried), sharding. The npj manuscript, which uses v65 and says "HRR, circular
convolution", is unaffected.

ICLR Setup paragraph rewritten to match the code (paper repo, this date; no number changes). **Rule
reinforced:** a method description is gate-checked against the module by READING THE MODULE, not by
reading the manuscript's account of it — 795059b failed its own new rule in the same entry that stated it.
