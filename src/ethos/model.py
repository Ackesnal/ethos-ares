import math
from collections import namedtuple
from functools import lru_cache

import torch
import torch.nn as nn
import transformers.activations
from torch.nn import functional as F
from transformers import GPT2Config

ModelOutput = namedtuple("ModelOutput", ["loss", "logits", "moe_loss"])


class CausalSelfAttention(nn.Module):
    def __init__(self, config, attention_weights: list | None = None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.attn_pdrop
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash or attention_weights is not None:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.n_positions, config.n_positions)).view(
                    1, 1, config.n_positions, config.n_positions
                ),
                persistent=False,
            )
        self.attention_weights = attention_weights

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the
        # batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash and self.attention_weights is None:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            self.attention_weights.append(att.detach().cpu())
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.activation = transformers.activations.get_activation(config.activation_function)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GLU(nn.Module):
    """Gated Linear Unit FFN: uses a gating mechanism where one linear projection
    controls the gate and another provides the value, followed by a down-projection."""

    def __init__(self, config):
        super().__init__()
        self.w_gate = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.w_up = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.activation = transformers.activations.get_activation(config.activation_function)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        gate = self.activation(self.w_gate(x))
        up = self.w_up(x)
        x = gate * up
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


def _build_ffn(config):
    """Build the FFN module based on config.ffn_type ('mlp' or 'glu')."""
    ffn_type = getattr(config, "ffn_type", "mlp")
    if ffn_type == "glu":
        return GLU(config)
    return MLP(config)


class TopKRouter(nn.Module):
    """Top-k gating router for Mixture-of-Experts.

    Computes a softmax over expert logits and selects the top-k experts per token.
    Also computes an auxiliary load-balancing loss (Switch Transformer style).
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args:
            x: (num_tokens, hidden_size)
        Returns:
            top_k_weights: (num_tokens, top_k)  — normalized routing weights
            top_k_indices: (num_tokens, top_k)  — expert indices
            aux_loss: scalar — load-balancing auxiliary loss
        """
        # (num_tokens, num_experts)
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)

        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        # Re-normalize the selected weights so they sum to 1
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # --- Auxiliary load-balancing loss (Switch Transformer, Eq. 4) ---
        # f_i = fraction of tokens routed to expert i (based on argmax / top-1)
        # P_i = mean probability assigned to expert i
        # loss = num_experts * sum(f_i * P_i)
        num_tokens = x.size(0)
        # Count how many tokens each expert is the top-1 choice for
        top1_indices = top_k_indices[:, 0]  # (num_tokens,)
        tokens_per_expert = torch.zeros(self.num_experts, device=x.device)
        tokens_per_expert.scatter_add_(
            0, top1_indices, torch.ones(num_tokens, device=x.device)
        )
        f = tokens_per_expert / num_tokens  # (num_experts,)
        P = probs.mean(dim=0)               # (num_experts,)
        aux_loss = self.num_experts * (f * P).sum()

        return top_k_weights, top_k_indices, aux_loss


class SparseMoE(nn.Module):
    """Sparse Mixture-of-Experts layer (pure PyTorch, no DeepSpeed dependency).

    Each expert is an independent FFN. A top-k router selects which experts process
    each token. The outputs are combined via the gating weights.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = getattr(config, "num_experts_total", 1)
        self.top_k = getattr(config, "num_experts_activated", 1)
        self.hidden_size = config.n_embd

        self.router = TopKRouter(self.hidden_size, self.num_experts, self.top_k)
        self.experts = nn.ModuleList([_build_ffn(config) for _ in range(self.num_experts)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            output: (batch, seq_len, hidden_size)
            aux_loss: scalar
        """
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)  # (B*T, H)

        # Route
        top_k_weights, top_k_indices, aux_loss = self.router(x_flat)  # (B*T, k), (B*T, k)

        # Compute expert outputs — gather tokens for each expert for efficiency
        output = torch.zeros_like(x_flat)  # (B*T, H)
        for expert_idx in range(self.num_experts):
            # Mask: which (token, slot) pairs selected this expert
            mask = (top_k_indices == expert_idx)  # (B*T, k)
            if not mask.any():
                continue
            # Get the token indices that route to this expert (any of the k slots)
            token_mask = mask.any(dim=-1)  # (B*T,)
            expert_input = x_flat[token_mask]  # (n, H)
            expert_output = self.experts[expert_idx](expert_input)  # (n, H)
            # Weight by the routing weights for this expert across all slots
            # Sum the weights across slots that selected this expert
            weights = (top_k_weights * mask.float()).sum(dim=-1)  # (B*T,)
            expert_weights = weights[token_mask].unsqueeze(-1)    # (n, 1)
            output[token_mask] += expert_output * expert_weights

        output = output.view(batch_size, seq_len, hidden_size)
        return output, aux_loss


class Block(nn.Module):
    def __init__(self, config, attention_weights: list | None = None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, attention_weights=attention_weights)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)

        num_experts = getattr(config, "num_experts_total", 1)

        if num_experts > 1:
            self.moe = SparseMoE(config)
            self.use_moe = True
        else:
            self.mlp = _build_ffn(config)
            self.use_moe = False

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        residual = x
        x = self.ln_2(x)
        if self.use_moe:
            x, moe_loss = self.moe(x)
            x = residual + x
            return x, moe_loss
        else:
            x = residual + self.mlp(x)
            return x, torch.tensor(0.0, device=x.device)


class GPT2LMNoBiasModel(nn.Module):
    def __init__(
        self,
        config: GPT2Config,
        return_attention=False,
    ):
        super().__init__()
        self.config = config
        self.use_moe = getattr(config, "num_experts_total", 1) > 1

        self.return_attention = return_attention
        self.attention_weights = [] if return_attention else None

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.n_positions, config.n_embd),
                drop=nn.Dropout(config.embd_pdrop),
                h=nn.ModuleList(
                    [Block(config, self.attention_weights) for _ in range(config.n_layer)]
                ),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # Continuous relative-time embedding: log(1 + Δt) → n_embd
        self.time_emb = nn.Sequential(
            nn.Linear(1, config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd, bias=False),
        )

        # Vocab-derived buffers are registered lazily via register_vocab_info()
        self._vocab_registered = False

        # init all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        pos = torch.arange(0, config.n_positions, dtype=torch.long)
        self.register_buffer("pos", pos, persistent=False)

    # ------------------------------------------------------------------
    # Vocabulary-aware buffers (token type masks, admission IDs)
    # ------------------------------------------------------------------
    def register_vocab_info(self, vocab) -> None:
        """Register token-type classification buffers derived from *vocab*.

        Must be called once before the first forward pass that uses
        ``times``.  The buffers are persisted inside ``state_dict`` so
        they survive checkpoint round-trips.
        """
        from .vocabulary import TokenType

        token_types = vocab.get_token_type_tensor()          # (vocab_size,)
        admission_ids = vocab.get_admission_token_ids()      # (num_admission_tokens,)
        # Boolean masks indexed by token ID
        is_value = (token_types == TokenType.VALUE)          # (vocab_size,)
        is_code = (token_types == TokenType.CODE)            # (vocab_size,)
        is_admission = (token_types == TokenType.ADMISSION)  # (vocab_size,)

        self.register_buffer("_token_types", token_types, persistent=True)
        self.register_buffer("_admission_ids", admission_ids, persistent=True)
        self.register_buffer("_is_value", is_value, persistent=True)
        self.register_buffer("_is_code", is_code, persistent=True)
        self.register_buffer("_is_admission", is_admission, persistent=True)
        self._vocab_registered = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_relative_times(tokens, times, is_admission_mask):
        """Compute per-token relative time (seconds) to the nearest preceding admission.

        Parameters
        ----------
        tokens : (B, T) long tensor of token IDs
        times : (B, T) long/float tensor of absolute timestamps in *microseconds*
        is_admission_mask : (V,) bool tensor indexed by token ID

        Returns
        -------
        rel_times : (B, T) float tensor — relative time in **seconds**
        """
        is_adm = is_admission_mask[tokens]  # (B, T)
        adm_times = torch.where(is_adm, times.float(), float("-inf"))  # (B, T)
        # cummax along time axis gives the most recent admission time at each position
        nearest_adm_time, _ = adm_times.cummax(dim=1)  # (B, T)
        # Relative time in microseconds → seconds.  Before any admission the
        # value is -inf; we clamp to 0 so log(1 + 0) = 0.
        rel_us = (times.float() - nearest_adm_time).clamp(min=0.0)
        return rel_us / 1e6  # → seconds

    def _fuse_embeddings(self, input_ids, times, tok_emb):
        """Fuse code, value, and relative-time embeddings.

        For every *value* token (Q1 … Q10) that is immediately preceded
        by a *code* token, the value embedding is enriched::

            fused_value = value_emb + code_emb + time_emb

        Absorbed code positions are zeroed out and their labels should
        be set to ``-100`` so they do not contribute to the loss.
        All surviving tokens also receive the relative-time embedding.

        The implementation is fully vectorised (no ``.item()``,
        ``.nonzero()``, or Python loops) so it is ``torch.compile``
        friendly.

        Parameters
        ----------
        input_ids : (B, T)
        times : (B, T)
        tok_emb : (B, T, C)  — raw token embeddings from ``wte``

        Returns
        -------
        fused_emb : (B, T, C)
        keep_mask : (B, T)   — False at absorbed-code positions
        """
        B, T, C = tok_emb.shape
        device = tok_emb.device

        # --- Relative-time embedding for every token -----------------------
        rel_times = self._compute_relative_times(
            input_ids, times, self._is_admission
        )  # (B, T) in seconds
        log_rt = torch.log1p(rel_times).unsqueeze(-1)  # (B, T, 1)
        time_emb = self.time_emb(log_rt)  # (B, T, C)

        # --- Identify code-before-value pairs ------------------------------
        is_val = self._is_value[input_ids]                    # (B, T)
        is_cd = self._is_code[input_ids]                      # (B, T)

        # A code token at position t is "absorbed" if position t+1 is a value token
        absorbed_code = torch.zeros(B, T, dtype=torch.bool, device=device)
        absorbed_code[:, :-1] = is_cd[:, :-1] & is_val[:, 1:]

        # --- Build fused embeddings (compile-friendly, no boolean indexing) -
        # Value positions preceded by an absorbed code token
        val_with_code = torch.zeros(B, T, dtype=torch.bool, device=device)
        val_with_code[:, 1:] = absorbed_code[:, :-1]

        # Preceding-token embeddings via a simple shift
        prev_tok_emb = torch.cat([tok_emb[:, :1, :], tok_emb[:, :-1, :]], dim=1)
        # Add preceding code embedding only where a value follows a code
        code_contrib = torch.where(
            val_with_code.unsqueeze(-1), prev_tok_emb, torch.zeros_like(tok_emb)
        )

        # Fuse: all tokens get time_emb; value-after-code tokens also get code_emb
        fused = tok_emb + time_emb + code_contrib  # (B, T, C)

        # Zero out absorbed code positions so they contribute minimally
        fused = torch.where(absorbed_code.unsqueeze(-1), torch.zeros_like(fused), fused)

        keep_mask = ~absorbed_code  # (B, T)
        return fused, keep_mask

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @lru_cache
    def num_parameters(self, exclude_embeddings=True):
        """Total number of parameters (all experts counted)."""
        n_params = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    @lru_cache
    def num_active_parameters(self, exclude_embeddings=True):
        """Number of parameters activated per token (for MoE, only the activated experts count).

        For non-MoE models this equals num_parameters(). For MoE models, each layer's FFN
        contribution is scaled by (num_experts_activated / num_experts_total)."""
        if not self.use_moe:
            return self.num_parameters(exclude_embeddings)

        num_experts = getattr(self.config, "num_experts_total", 1)
        num_activated = getattr(self.config, "num_experts_activated", 1)

        # Separate expert parameters from non-expert parameters
        expert_params_total = 0
        non_expert_params = 0
        for name, p in self.named_parameters():
            if ".moe.experts." in name:
                expert_params_total += p.numel()
            else:
                non_expert_params += p.numel()

        if exclude_embeddings:
            non_expert_params -= self.transformer.wpe.weight.numel()

        # expert_params_total counts ALL expert params across ALL layers.
        # Each layer has num_experts copies; active = total / num_experts * num_activated
        active_expert_params = expert_params_total // num_experts * num_activated

        return non_expert_params + active_expert_params

    def forward(self, input_ids, labels=None, times=None) -> ModelOutput:
        """Forward pass with optional fused (code, value, relative-time) embeddings.

        Parameters
        ----------
        input_ids : (B, T) long tensor of token IDs
        labels : (B, T) long tensor of target token IDs (optional)
        times : (B, T) long tensor of absolute timestamps in microseconds
            (optional).  When provided **and** ``register_vocab_info`` has
            been called, the model fuses code + value + log-relative-time
            embeddings and removes absorbed code tokens from the sequence.
        """
        _, t = input_ids.size()
        if self.return_attention:
            self.attention_weights.clear()

        tok_emb = self.transformer.wte(input_ids)

        use_fusion = times is not None and self._vocab_registered
        if use_fusion:
            tok_emb, keep_mask = self._fuse_embeddings(input_ids, times, tok_emb)
            t_eff = tok_emb.size(1)

            # Mask labels at absorbed-code positions so they don't affect the loss
            if labels is not None:
                labels = torch.where(keep_mask, labels, -100)
        else:
            t_eff = t

        pos_emb = self.transformer.wpe(self.pos[:t_eff])
        x = self.transformer.drop(tok_emb + pos_emb)

        total_moe_loss = torch.tensor(0.0, device=input_ids.device)
        for block in self.transformer.h:
            x, moe_loss = block(x)
            total_moe_loss = total_moe_loss + moe_loss

        x = self.transformer.ln_f(x)

        if labels is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return ModelOutput(loss=loss, logits=logits, moe_loss=total_moe_loss)

    @torch.no_grad()
    def get_next_token(
        self,
        x: torch.Tensor,
        times: torch.Tensor | None = None,
        return_probs: bool = False,
        top_k: int | None = None,
    ):
        logits = self(x, times=times).logits
        logits = logits[:, -1, :]
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        if return_probs:
            return next_token, probs
        return next_token
