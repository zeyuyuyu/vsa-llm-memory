"""V-85 VSA module (prereg 85461e8): a differentiable HRR store read from the LLM's own hidden states and
trained JOINTLY with the model.

This is the component the whole experiment turns on, so what it does and does not assume is spelled out.

The frozen route the paper reports works like this: gold (subject, relation, object) triples are supplied,
each string gets a codebook row, binding and unbinding are exact, and the readout is a nearest-neighbour
cleanup. Its measured limitation is that the model cannot supply those triples itself -- self-extraction
collapses to 0.0178-0.0311.

Here nothing is supplied. Two trainable projections read the passage's hidden states and emit a role vector
and a filler vector at every token position. Those are bound by circular convolution and superposed into one
memory. The question's hidden state emits a query role, which unbinds the memory, and the result is scored
against the filler vectors of the passage itself -- so the "cleanup" ranges over what the passage actually
contains rather than over a supplied codebook. The whole path is differentiable and trains with the answer
loss, which is exactly the gradient the frozen protocol denied it.

The readout is POOLED-equivalent: the answer distribution is over passage token positions, not over one
designated position, which is the wiring V-82 established as necessary for this operator.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def circ_conv(a, b, n):
    """Circular convolution via FFT. torch.fft has no bfloat16 kernel, so the transform is taken in float32
    and the result cast back to the caller's dtype. This is a precision detail of the implementation, not a
    change to the algebra: binding and unbinding are unaffected."""
    dt = a.dtype
    out = torch.fft.irfft(torch.fft.rfft(a.float(), n=n) * torch.fft.rfft(b.float(), n=n), n=n)
    return out.to(dt)


def involution(x):
    """The HRR approximate inverse: reverse every component but the first."""
    return torch.cat([x[..., :1], torch.flip(x[..., 1:], dims=[-1])], dim=-1)


class VSAStore(nn.Module):
    """Reads hidden states, binds role to filler, superposes, unbinds with a question-derived query, and
    scores passage positions. Trained jointly with the LLM through LoRA."""

    def __init__(self, d_model, d_vsa=512, p=0.1):
        super().__init__()
        self.d = d_vsa
        self.role = nn.Sequential(nn.Linear(d_model, d_vsa), nn.LayerNorm(d_vsa))
        self.fill = nn.Sequential(nn.Linear(d_model, d_vsa), nn.LayerNorm(d_vsa))
        self.query = nn.Sequential(nn.Linear(d_model, d_vsa), nn.LayerNorm(d_vsa))
        self.do = nn.Dropout(p)
        # temperature on the retrieval scores, learned rather than fixed so the module is not handed a
        # tuned constant
        self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden, mask, q_index):
        """hidden [B,T,H] ; mask [B,T] true on real tokens ; q_index [B] position the question ends at.
        Returns scores [B,T] over passage positions."""
        r = F.normalize(self.do(self.role(hidden)), dim=-1)
        f = F.normalize(self.do(self.fill(hidden)), dim=-1)
        mf = mask.unsqueeze(-1).to(r.dtype)

        bound = circ_conv(r, f, self.d) * mf              # bind role to filler at every position
        memory = bound.sum(dim=1, keepdim=True)           # superpose the whole passage into one vector

        qh = hidden[torch.arange(hidden.shape[0], device=hidden.device), q_index]
        q = F.normalize(self.query(qh), dim=-1).unsqueeze(1)
        retrieved = circ_conv(involution(q), memory, self.d)   # unbind

        # cleanup: score every passage position's filler against what was retrieved
        scores = F.cosine_similarity(f, retrieved, dim=-1) * self.temp.clamp(0.1, 20.0)
        return scores.masked_fill(~mask, -1e4)


class VSAHead(nn.Module):
    """Turns the store's position scores into a distribution over the vocabulary, by scattering each
    position's score onto the token id that sits there. This keeps the answer space the model's own
    vocabulary, so the arm is comparable with generation-based arms rather than scored on a private space."""

    def __init__(self, d_model, vocab_size, d_vsa=512, p=0.1):
        super().__init__()
        self.store = VSAStore(d_model, d_vsa, p)
        self.vocab_size = vocab_size

    def forward(self, hidden, mask, q_index, input_ids):
        scores = self.store(hidden, mask, q_index)                       # [B,T]
        logits = hidden.new_full((hidden.shape[0], self.vocab_size), -1e4)
        logits = logits.scatter_reduce(1, input_ids, scores, reduce="amax", include_self=True)
        return logits
