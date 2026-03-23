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


class Block(nn.Module):
    def __init__(self, config, attention_weights: list | None = None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, attention_weights=attention_weights)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)

        num_experts = getattr(config, "num_experts_total", 1)
        num_experts_activated = getattr(config, "num_experts_activated", 1)

        if num_experts > 1:
            from deepspeed.moe.layer import MoE

            expert = _build_ffn(config)
            self.moe = MoE(
                hidden_size=config.n_embd,
                expert=expert,
                num_experts=num_experts,
                ep_size=1,  # will be overridden by DeepSpeed engine when distributed
                k=num_experts_activated,
                use_residual=False,
                capacity_factor=1.0,
                eval_capacity_factor=1.0,
                min_capacity=4,
                drop_tokens=True,
                use_tutel=False,
                noisy_gate_policy=None,
            )
            self.use_moe = True
        else:
            self.mlp = _build_ffn(config)
            self.use_moe = False

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        residual = x
        x = self.ln_2(x)
        if self.use_moe:
            # DeepSpeed MoE expects (batch*seq, hidden) or (batch, seq, hidden)
            # It returns (output, moe_loss, _)
            x, moe_loss, _ = self.moe(x)
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

        # init all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        pos = torch.arange(0, config.n_positions, dtype=torch.long)
        self.register_buffer("pos", pos, persistent=False)

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

        # Sum non-expert parameters
        expert_params_per_layer = 0
        non_expert_params = 0
        for name, p in self.named_parameters():
            if ".moe." in name and ".deepspeed_moe.experts." in name:
                # This is an expert parameter; count per-layer once
                expert_params_per_layer += p.numel()
            else:
                non_expert_params += p.numel()

        if exclude_embeddings:
            non_expert_params -= self.transformer.wpe.weight.numel()

        # expert_params_per_layer counts ALL expert params across ALL layers
        # Each layer has num_experts copies; activated params = total_expert / num_experts * num_activated
        active_expert_params = expert_params_per_layer // num_experts * num_activated

        return non_expert_params + active_expert_params

    def forward(self, input_ids, labels=None) -> ModelOutput:
        _, t = input_ids.size()
        if self.return_attention:
            self.attention_weights.clear()

        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(self.pos[:t])
        x = self.transformer.drop(tok_emb + pos_emb)

        total_moe_loss = torch.tensor(0.0, device=input_ids.device)
        for block in self.transformer.h:
            x, moe_loss = block(x)
            total_moe_loss = total_moe_loss + moe_loss

        x = self.transformer.ln_f(x)

        if labels is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return ModelOutput(loss=loss, logits=logits, moe_loss=total_moe_loss)

    @torch.no_grad()
    def get_next_token(self, x: torch.Tensor, return_probs: bool = False, top_k: int | None = None):
        logits = self(x).logits
        logits = logits[:, -1, :]
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        if return_probs:
            return next_token, probs
        return next_token
