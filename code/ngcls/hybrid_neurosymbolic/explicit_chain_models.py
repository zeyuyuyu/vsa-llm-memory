import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType

from vsa import VSAModule, VSAProjection


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class LoRAClassifier(nn.Module):
    def __init__(self, model_name: str = "microsoft/phi-2", lora_rank: int = 8):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "dense"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.llm = get_peft_model(self.llm, lora_config)

        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.llm_dim = cfg.hidden_size
        self.classifier = nn.Linear(self.llm_dim, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1].float()
        pooled = mean_pool(h, attention_mask)
        return self.classifier(pooled)


class LoRAQPoolClassifier(nn.Module):
    """LoRA baseline with query-aware pooling.

    Instead of mean-pooling the whole sequence, we find the last occurrence of token 'Q' (or ' Q')
    in the input and pool a small window of hidden states around it.

    This is a stronger baseline for retrieval-style tasks where the answer is conditioned on a query segment.
    """

    def __init__(self, model_name: str = "microsoft/phi-2", lora_rank: int = 8, window: int = 8):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "dense"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.llm = get_peft_model(self.llm, lora_config)

        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.llm_dim = cfg.hidden_size
        self.classifier = nn.Linear(self.llm_dim, 2)

        self.window = int(window)

        # token ids for query marker
        # Note: Phi-2 BPE distinguishes 'Q' vs 'ĠQ'. We handle both.
        q_ids = self.tokenizer.encode("Q", add_special_tokens=False)
        qsp_ids = self.tokenizer.encode(" Q", add_special_tokens=False)
        self.q_token_ids = set(q_ids + qsp_ids)

    def _qpool(self, h: torch.Tensor, attention_mask: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """h: (B,T,D) float, attention_mask: (B,T), input_ids: (B,T)"""
        B, T, D = h.shape
        pooled = torch.zeros(B, D, device=h.device, dtype=h.dtype)

        for b in range(B):
            ids = input_ids[b].tolist()
            q_pos = None
            for i in range(T - 1, -1, -1):
                if ids[i] in self.q_token_ids:
                    q_pos = i
                    break

            if q_pos is None:
                # fallback to mean pool
                pooled[b] = mean_pool(h[b : b + 1], attention_mask[b : b + 1]).squeeze(0)
                continue

            left = max(0, q_pos - self.window)
            right = min(T, q_pos + self.window + 1)

            mask = attention_mask[b, left:right].unsqueeze(-1).float()  # (L,1)
            seg = h[b, left:right, :]  # (L,D)
            pooled[b] = (seg * mask).sum(dim=0) / mask.sum().clamp(min=1e-9)

        return pooled

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1].float()
        pooled = self._qpool(h, attention_mask, input_ids)
        return self.classifier(pooled)


class FrozenLLMClassifier(nn.Module):
    def __init__(self, model_name: str = "microsoft/phi-2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        for p in self.llm.parameters():
            p.requires_grad = False

        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.llm_dim = cfg.hidden_size
        self.classifier = nn.Linear(self.llm_dim, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1].float()
        pooled = mean_pool(h, attention_mask)
        return self.classifier(pooled)


class VSAChainClassifier(nn.Module):
    """VSA model that explicitly performs 3-hop retrieval with cleanup memory.

    memA = Σ bind(X_vec, role_vec)
    memB = Σ bind(Y_vec, X_vec)
    memC = Σ bind(F_vec, Y_vec)

    Query: role -> X_hat -> Y_hat -> F_hat -> compare with candidate.

    Key trick: cleanup step snaps X_hat/Y_hat to nearest symbol key from the context,
    reducing error accumulation across hops (classic VSA cleanup memory).
    """

    def __init__(self, model_name: str = "microsoft/phi-2", vsa_dim: int = 2048, num_roles: int = 64):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        for p in self.llm.parameters():
            p.requires_grad = False

        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.llm_dim = cfg.hidden_size

        self.vsa = VSAModule(dim=vsa_dim, num_role_vectors=num_roles)
        self.proj = VSAProjection(self.llm_dim, vsa_dim)

        # token ids
        self.id_lbr = self.tokenizer.encode("[", add_special_tokens=False)[0]
        self.id_rbr = self.tokenizer.encode("]", add_special_tokens=False)[0]
        self.id_lc = self.tokenizer.encode("{", add_special_tokens=False)[0]
        self.id_rc = self.tokenizer.encode("}", add_special_tokens=False)[0]
        self.id_R = self.tokenizer.encode("R", add_special_tokens=False)[0]
        self.id_X = self.tokenizer.encode("X", add_special_tokens=False)[0]
        self.id_Y = self.tokenizer.encode("Y", add_special_tokens=False)[0]

        self.hex_id_to_val = {}
        for i, ch in enumerate("0123456789ABCDEF"):
            tid = self.tokenizer.encode(ch, add_special_tokens=False)
            if len(tid) != 1:
                raise ValueError("hex not single token")
            self.hex_id_to_val[tid[0]] = i

        self.num_roles = num_roles

        self.classifier = nn.Sequential(
            nn.Linear(self.llm_dim + 1, self.llm_dim),
            nn.GELU(),
            nn.Linear(self.llm_dim, 2),
        )

    def _scan_markers(self, ids_1d: torch.Tensor):
        # Robust under BPE tokenization: parse token strings (e.g. 'Ġ[' vs '['; '00' as one token)
        # Returns: list[(kind, idx_val, start, end_excl)]
        ids = ids_1d.tolist()
        toks = self.tokenizer.convert_ids_to_tokens(ids)
        markers = []  # (kind, idx, start, end_excl)

        HEX = set("0123456789ABCDEF")
        i = 0
        n = len(ids)
        while i < n:
            if toks[i].lstrip("Ġ") == "[":
                if i + 2 >= n:
                    i += 1
                    continue

                kind = toks[i + 1].lstrip("Ġ")
                if kind not in ("R", "X", "Y"):
                    i += 1
                    continue

                j = i + 2
                idx_str = ""
                while j < n:
                    t = toks[j].lstrip("Ġ")
                    if t == "]":
                        j += 1
                        break
                    # Rare: token may contain trailing ']'
                    if t.endswith("]") and len(t) > 1:
                        idx_str += t[:-1]
                        j += 1
                        break
                    idx_str += t
                    j += 1

                if len(idx_str) == 2 and all(ch in HEX for ch in idx_str):
                    idx_val = int(idx_str, 16)
                    markers.append((kind, idx_val, i, j))
                    i = j
                    continue

            i += 1
        return markers


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1].float()

        B, T, D = h.shape
        device = h.device

        pooled = mean_pool(h, attention_mask)
        sim_feat = torch.zeros(B, 1, device=device)

        for b in range(B):
            markers = self._scan_markers(input_ids[b])
            ids = input_ids[b].tolist()
            toks = self.tokenizer.convert_ids_to_tokens(ids)

            # candidate filler span: last {...} (brace token can be '{' or 'Ġ{')
            last_lc = None
            for i in range(len(toks) - 1, -1, -1):
                if toks[i].lstrip('Ġ') == '{':
                    last_lc = i
                    break
            if last_lc is None:
                continue
            j = last_lc + 1
            while j < len(toks) and toks[j].lstrip('Ġ') != '}':
                j += 1
            if j >= len(toks) or j <= last_lc + 1:
                continue
            cand_vec = h[b, last_lc + 1 : j, :].mean(dim=0)

            # map marker to llm vec
            mvec = {}
            for kind, midx, s, e in markers:
                mvec[(kind, midx, s)] = h[b, s:e, :].mean(dim=0)

            markers_sorted = sorted(markers, key=lambda x: x[2])

            # query role: last R marker
            q_role = None
            for kind, midx, s, e in reversed(markers_sorted):
                if kind == "R" and midx < self.num_roles:
                    q_role = midx
                    break
            if q_role is None:
                continue

            # Build A/B/C pairs
            A_pairs = []  # (role_idx, x_llm_vec)
            B_pairs = []  # (x_llm_vec, y_llm_vec)
            C_pairs = []  # (y_llm_vec, f_llm_vec)

            for i in range(len(markers_sorted) - 1):
                k1, idx1, s1, e1 = markers_sorted[i]
                k2, idx2, s2, e2 = markers_sorted[i + 1]
                v1 = mvec[(k1, idx1, s1)]
                v2 = mvec[(k2, idx2, s2)]
                if k1 == "R" and k2 == "X" and idx1 < self.num_roles:
                    A_pairs.append((idx1, v2))
                elif k1 == "X" and k2 == "Y":
                    B_pairs.append((v1, v2))

            for kind, midx, s, e in markers_sorted:
                if kind != "Y":
                    continue
                pos = e
                if pos < len(toks) and toks[pos].lstrip('Ġ') == '{':
                    jj = pos + 1
                    while jj < len(toks) and toks[jj].lstrip('Ġ') != '}':
                        jj += 1
                    if jj < len(toks) and jj > pos + 1:
                        y_llm = mvec[(kind, midx, s)]
                        f_llm = h[b, pos + 1 : jj, :].mean(dim=0)
                        C_pairs.append((y_llm, f_llm))

            if not A_pairs or not B_pairs or not C_pairs:
                continue

            # Project & normalize
            role_vec = self.vsa.role_vectors[q_role].unsqueeze(0)

            # A memory
            x_llm = torch.stack([p[1] for p in A_pairs], dim=0).unsqueeze(0)
            x_vsa = l2_normalize(self.proj.project_to_vsa(x_llm))
            roles = torch.tensor([p[0] for p in A_pairs], device=device, dtype=torch.long).unsqueeze(0)
            memA = self.vsa(x_vsa, roles)
            x_hat = l2_normalize(self.vsa.unbind(memA, role_vec))

            # B memory
            x_keys_llm = torch.stack([p[0] for p in B_pairs], dim=0).unsqueeze(0)
            y_vals_llm = torch.stack([p[1] for p in B_pairs], dim=0).unsqueeze(0)
            xk = l2_normalize(self.proj.project_to_vsa(x_keys_llm))
            yv = l2_normalize(self.proj.project_to_vsa(y_vals_llm))
            boundB = self.vsa.bind(yv, xk)
            memB = self.vsa.bundle(boundB.transpose(0, 1))

            # cleanup X_hat to nearest X key
            sims_x = self.vsa.similarity(x_hat.expand_as(xk.squeeze(0)), xk.squeeze(0))
            best_x = sims_x.argmax(dim=0)
            x_hat_clean = xk[:, best_x, :]

            y_hat = l2_normalize(self.vsa.unbind(memB, x_hat_clean))

            # C memory
            y_keys_llm = torch.stack([p[0] for p in C_pairs], dim=0).unsqueeze(0)
            f_vals_llm = torch.stack([p[1] for p in C_pairs], dim=0).unsqueeze(0)
            yk = l2_normalize(self.proj.project_to_vsa(y_keys_llm))
            fv = l2_normalize(self.proj.project_to_vsa(f_vals_llm))
            boundC = self.vsa.bind(fv, yk)
            memC = self.vsa.bundle(boundC.transpose(0, 1))

            # cleanup Y_hat to nearest Y key
            sims_y = self.vsa.similarity(y_hat.expand_as(yk.squeeze(0)), yk.squeeze(0))
            best_y = sims_y.argmax(dim=0)
            y_hat_clean = yk[:, best_y, :]

            f_hat = l2_normalize(self.vsa.unbind(memC, y_hat_clean))

            cand_vsa = l2_normalize(self.proj.project_to_vsa(cand_vec.view(1, 1, -1)).squeeze(1))
            sim = self.vsa.similarity(f_hat, cand_vsa).clamp(-1, 1)
            sim_feat[b, 0] = sim
            pooled[b] = cand_vec

        feats = torch.cat([pooled, sim_feat], dim=-1)
        return self.classifier(feats)



def _init_random_unit_vectors(n: int, dim: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    v = torch.randn(n, dim, generator=g)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    return v


class SymbolicVSAChainClassifier(nn.Module):
    """A faster + more VSA-faithful chain model.

    Key idea: parse marker indices ([R..]/[X..]/[Y..]) and filler ids ({...}) explicitly,
    then perform HRR binding/unbinding in VSA space with fixed random symbol vectors.

    This avoids relying on frozen LLM hidden states to preserve symbol identity.
    """

    def __init__(self, model_name: str = "microsoft/phi-2", vsa_dim: int = 2048, num_roles: int = 64, vocab_size: int = 256, seed: int = 0):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.vsa_dim = vsa_dim
        self.num_roles = num_roles
        self.vocab_size = vocab_size

        # HRR ops
        self.vsa = VSAModule(dim=vsa_dim, num_role_vectors=num_roles)
        # Freeze role vectors to keep them random/approximately orthogonal (classic VSA setting)
        self.vsa.role_vectors.requires_grad_(False)

        # Fixed random symbol vectors for X/Y/filler ids
        self.register_buffer('x_vectors', _init_random_unit_vectors(vocab_size, vsa_dim, seed=seed + 11))
        self.register_buffer('y_vectors', _init_random_unit_vectors(vocab_size, vsa_dim, seed=seed + 23))
        self.register_buffer('f_vectors', _init_random_unit_vectors(vocab_size, vsa_dim, seed=seed + 37))

        # Learnable calibration for similarity thresholding
        # Initialize to roughly threshold at sim≈0.2 (since we often observe pos sim ~0.23, neg ~0.0)
        self.sim_scale = nn.Parameter(torch.tensor(20.0))
        self.sim_bias = nn.Parameter(torch.tensor(-4.0))

    def _scan_markers(self, ids_1d: torch.Tensor):
        ids = ids_1d.tolist()
        toks = self.tokenizer.convert_ids_to_tokens(ids)
        markers = []  # (kind, idx, start, end_excl)

        HEX = set("0123456789ABCDEF")
        i = 0
        n = len(ids)
        while i < n:
            if toks[i].lstrip("Ġ") == "[":
                if i + 2 >= n:
                    i += 1
                    continue

                kind = toks[i + 1].lstrip("Ġ")
                if kind not in ("R", "X", "Y"):
                    i += 1
                    continue

                j = i + 2
                idx_str = ""
                while j < n:
                    t = toks[j].lstrip("Ġ")
                    if t == "]":
                        j += 1
                        break
                    if t.endswith("]") and len(t) > 1:
                        idx_str += t[:-1]
                        j += 1
                        break
                    idx_str += t
                    j += 1

                if len(idx_str) == 2 and all(ch in HEX for ch in idx_str):
                    idx_val = int(idx_str, 16)
                    markers.append((kind, idx_val, i, j))
                    i = j
                    continue

            i += 1

        return markers

    def _parse_brace_int_at(self, toks, start_idx: int):
        # Accept '{' or 'Ġ{' tokens
        if start_idx < 0 or start_idx >= len(toks) or toks[start_idx].lstrip('Ġ') != '{':
            return None, start_idx

        j = start_idx + 1
        num_str = ""
        while j < len(toks):
            t = toks[j].lstrip('Ġ')
            if t == '}':
                break
            if t.endswith('}') and t != '}':
                num_str += t[:-1]
                break
            num_str += t
            j += 1

        try:
            val = int(num_str)
        except Exception:
            val = None

        # move j to token after '}' if present
        while j < len(toks) and toks[j].lstrip('Ġ') != '}':
            j += 1
        if j < len(toks) and toks[j].lstrip('Ġ') == '}':
            j += 1

        return val, j

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        device = input_ids.device
        B, T = input_ids.shape

        logits = torch.zeros(B, 2, device=device)

        for b in range(B):
            ids = input_ids[b].tolist()
            toks = self.tokenizer.convert_ids_to_tokens(ids)
            markers = self._scan_markers(input_ids[b])
            if not markers:
                continue

            # candidate filler: last {...}
            last_lc = None
            for i in range(len(toks) - 1, -1, -1):
                if toks[i].lstrip('Ġ') == '{':
                    last_lc = i
                    break
            if last_lc is None:
                continue
            cand, _ = self._parse_brace_int_at(toks, last_lc)
            if cand is None or cand < 0 or cand >= self.vocab_size:
                continue

            markers_sorted = sorted(markers, key=lambda x: x[2])

            # query role: last R marker
            q_role = None
            for kind, midx, s, e in reversed(markers_sorted):
                if kind == 'R' and midx < self.num_roles:
                    q_role = midx
                    break
            if q_role is None:
                continue

            # Build A/B/C pairs in index space
            A_pairs = []  # (role_idx, x_id)
            B_pairs = []  # (x_id, y_id)
            C_pairs = []  # (y_id, f_id)

            for i in range(len(markers_sorted) - 1):
                k1, idx1, s1, e1 = markers_sorted[i]
                k2, idx2, s2, e2 = markers_sorted[i + 1]
                if k1 == 'R' and k2 == 'X' and idx1 < self.num_roles and idx2 < self.vocab_size:
                    A_pairs.append((idx1, idx2))
                elif k1 == 'X' and k2 == 'Y' and idx1 < self.vocab_size and idx2 < self.vocab_size:
                    B_pairs.append((idx1, idx2))

            for kind, midx, s, e in markers_sorted:
                if kind != 'Y' or midx >= self.vocab_size:
                    continue
                pos = e
                if pos < len(toks) and toks[pos].lstrip('Ġ') == '{':
                    f_id, _ = self._parse_brace_int_at(toks, pos)
                    if f_id is None or f_id < 0 or f_id >= self.vocab_size:
                        continue
                    C_pairs.append((midx, f_id))

            if not A_pairs or not B_pairs or not C_pairs:
                continue

            # ---- HRR retrieval: R -> X -> Y -> f ----
            role_q = self.vsa.role_vectors[q_role].to(device)

            # memA = Σ bind(X[x], R[r])
            boundA = []
            x_ids = []
            for r, x in A_pairs:
                boundA.append(self.vsa.bind(self.x_vectors[x].to(device), self.vsa.role_vectors[r].to(device)))
                x_ids.append(x)
            memA = self.vsa.bundle(torch.stack(boundA, dim=0))
            x_hat = l2_normalize(self.vsa.unbind(memA, role_q))

            # cleanup X
            x_cands = self.x_vectors[torch.tensor(x_ids, device=device, dtype=torch.long)]
            sims_x = self.vsa.similarity(x_cands, x_hat.unsqueeze(0).expand_as(x_cands))
            x_clean = x_cands[int(sims_x.argmax().item())]

            # memB = Σ bind(Y[y], X[x])
            boundB = []
            y_ids = []
            for x, y in B_pairs:
                boundB.append(self.vsa.bind(self.y_vectors[y].to(device), self.x_vectors[x].to(device)))
                y_ids.append(y)
            memB = self.vsa.bundle(torch.stack(boundB, dim=0))
            y_hat = l2_normalize(self.vsa.unbind(memB, x_clean))

            # cleanup Y
            y_cands = self.y_vectors[torch.tensor(y_ids, device=device, dtype=torch.long)]
            sims_y = self.vsa.similarity(y_cands, y_hat.unsqueeze(0).expand_as(y_cands))
            y_clean = y_cands[int(sims_y.argmax().item())]

            # memC = Σ bind(F[f], Y[y])
            boundC = []
            f_ids = []
            for y, f in C_pairs:
                boundC.append(self.vsa.bind(self.f_vectors[f].to(device), self.y_vectors[y].to(device)))
                f_ids.append(f)
            memC = self.vsa.bundle(torch.stack(boundC, dim=0))
            f_hat = l2_normalize(self.vsa.unbind(memC, y_clean))

            cand_v = self.f_vectors[int(cand)].to(device)
            sim = self.vsa.similarity(f_hat, cand_v).clamp(-1, 1)

            logit_pos = sim * self.sim_scale + self.sim_bias
            logits[b, 1] = logit_pos
            logits[b, 0] = -logit_pos

        return logits
