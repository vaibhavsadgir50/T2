"""
T2 — GPT-2 / DialoGPT-shaped LM: world-model loop write → think → predict per block step.

Each layer’s **conv_rsm** writes the token into slots, runs **ERG think** for
``erg_n_steps`` over slots (and optional domain context), then **attention read**
from the ERG conclusion, **out_proj**, and MLP.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C
from .geometric_memory import GeometricResonantStateMemory


@dataclass
class GPT2RSMConfig:
    vocab_size: int = C.VOCAB_SIZE
    n_embd: int = C.N_EMBD
    n_layer: int = C.N_LAYER
    n_head: int = C.N_HEAD
    n_inner: int = C.N_INNER
    n_positions: int = C.N_POSITIONS
    n_slots: int = C.RSM_N_SLOTS
    erg_n_steps: int = C.ERG_N_STEPS
    erg_ffn_dim: Optional[int] = None
    slot_ema: Optional[float] = None
    read_logit_scale: Optional[float] = None

    @property
    def domain_head_dim(self) -> int:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        return self.n_embd // self.n_head


class GPT2RSMBlock(nn.Module):
    """
    **Flow:** ``write_step`` → optional domain reads → padded ``domain_proj`` →
    ``think`` (ERG over slots + domain node) → ``read`` (attention from conclusion) →
    ``out_proj`` → MLP residual.

    **V1:** no domain RSMs; ERG **think** uses conv slots only.
    """

    def __init__(self, cfg: GPT2RSMConfig) -> None:
        super().__init__()
        C.resolve_rsm_d_model(cfg.n_embd)
        hd = cfg.domain_head_dim
        erg_ffn_conv = (
            cfg.erg_ffn_dim
            if cfg.erg_ffn_dim is not None
            else C.erg_ffn_hidden_dim(cfg.n_embd)
        )
        self.n_embd = cfg.n_embd
        self.domain_head_dim = hd
        self.think_n_steps = cfg.erg_n_steps

        self.conv_rsm = GeometricResonantStateMemory(
            d_model=cfg.n_embd,
            n_slots=cfg.n_slots,
            erg_ffn_dim=erg_ffn_conv,
            erg_n_steps=cfg.erg_n_steps,
            slot_ema=cfg.slot_ema,
            read_logit_scale=cfg.read_logit_scale,
        )
        self.domain_rsms: nn.ModuleList = nn.ModuleList()
        self.domain_proj: Optional[nn.Linear] = None

        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.n_inner),
            nn.GELU(approximate="tanh"),
            nn.Linear(cfg.n_inner, cfg.n_embd),
        )

    def attach_domain(self, domain_rsm: GeometricResonantStateMemory) -> None:
        if domain_rsm.d_model != self.domain_head_dim:
            raise ValueError(
                f"domain_rsm.d_model={domain_rsm.d_model} != domain_head_dim={self.domain_head_dim}"
            )
        n_next = len(self.domain_rsms) + 1
        if n_next * self.domain_head_dim > self.n_embd:
            raise ValueError(
                "not enough n_embd: at most n_head domain RSMs for disjoint x slices"
            )
        for p in domain_rsm.parameters():
            p.requires_grad_(False)
        domain_rsm.eval()
        self.domain_rsms.append(domain_rsm)

        if self.domain_proj is None:
            self.domain_proj = nn.Linear(self.n_embd, self.n_embd)
            nn.init.xavier_uniform_(self.domain_proj.weight)
            nn.init.zeros_(self.domain_proj.bias)

    def forward_step(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        domain_states: List[torch.Tensor],
        return_aux: bool = False,
        extra_nodes: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        List[torch.Tensor],
        Optional[Dict[str, Any]],
    ]:
        if len(domain_states) != len(self.domain_rsms):
            raise ValueError(
                f"domain_states length {len(domain_states)} != "
                f"domain_rsms {len(self.domain_rsms)}"
            )

        conv_state = self.conv_rsm.write_step(x, conv_state)

        if extra_nodes is not None:
            extra = extra_nodes
        elif len(self.domain_rsms) == 0:
            extra = None
        else:
            assert self.domain_proj is not None
            B = x.shape[0]
            r_pad = x.new_zeros(B, self.n_embd)
            off = 0
            for i, drsm in enumerate(self.domain_rsms):
                sl = slice(i * self.domain_head_dim, (i + 1) * self.domain_head_dim)
                r_i = drsm.read_from_x(x[:, sl], domain_states[i])
                d_i = r_i.shape[-1]
                r_pad[:, off : off + d_i] = r_i
                off += d_i
            r_combined = self.domain_proj(r_pad)
            extra = r_combined.unsqueeze(1)

        aux: Optional[Dict[str, Any]] = None
        if return_aux:
            q_erg, think_traj = self.conv_rsm.think(
                x,
                conv_state,
                extra_nodes=extra,
                n_steps=self.think_n_steps,
                return_trajectory=True,
            )
            ctx, q_for_coh = self.conv_rsm.read_with_query(q_erg, conv_state)
            aux = {
                "think_traj": think_traj,
                "q_erg": q_for_coh,
                "ctx": ctx,
                "slots": conv_state,
            }
        else:
            q_erg = self.conv_rsm.think(
                x,
                conv_state,
                extra_nodes=extra,
                n_steps=self.think_n_steps,
            )
            ctx = self.conv_rsm.read(q_erg, conv_state)

        x = x + self.out_proj(ctx)
        x = x + self.mlp(self.ln_2(x))
        return x, conv_state, domain_states, aux


class GPT2RSMModel(nn.Module):
    """
    Causal LM with per-layer **write → think → predict** (attention read) loop.

    **ERG** runs ``erg_n_steps`` refinement passes **before** slot attention; more
    steps mean deeper reasoning before the read that feeds the MLP and LM head.

    **V1:** no domain RSMs; **think** uses conversation slots only.

    **V2+:** ``add_domain_rsm`` attaches frozen readers; domain reads are packed into
    the first ``k * domain_head_dim`` dims of a padded ``(B, n_embd)`` vector, then
    ``domain_proj: Linear(n_embd, n_embd)`` feeds a single extra ERG node in **think**,
    grounding reasoning in world knowledge without a separate scalar gate.

    This is the **world-model loop**: write memory → ERG think → attention predict.
    """

    def __init__(self, cfg: GPT2RSMConfig, tie_word_embeddings: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList(
            [GPT2RSMBlock(cfg) for _ in range(cfg.n_layer)]
        )
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=1e-5)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.wte.weight

    def init_states(
        self, B: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        conv_states: List[torch.Tensor] = []
        domain_states: List[List[torch.Tensor]] = []
        for block in self.blocks:
            conv_states.append(
                block.conv_rsm.init_batched_memory_state(B, device, dtype)
            )
            ds: List[torch.Tensor] = []
            for dr in block.domain_rsms:
                ds.append(dr.init_batched_memory_state(B, device, dtype))
            domain_states.append(ds)
        return conv_states, domain_states

    def add_domain_rsm(
        self,
        domain_rsm: GeometricResonantStateMemory,
        layer_idx: Optional[int] = None,
    ) -> None:
        targets = [layer_idx] if layer_idx is not None else list(range(len(self.blocks)))
        for li in targets:
            dom = copy.deepcopy(domain_rsm)
            self.blocks[li].attach_domain(dom)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        compute_aux_losses: bool = False,
        extra_nodes: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        B, T = input_ids.shape
        if T > self.cfg.n_positions:
            raise ValueError(
                f"sequence length {T} > n_positions {self.cfg.n_positions}"
            )
        device = input_ids.device
        dtype = self.wte.weight.dtype
        pos = torch.arange(T, device=device, dtype=torch.long)
        h = self.wte(input_ids) + self.wpe(pos)
        h = self.drop(h)

        conv_states, domain_states = self.init_states(B, device, dtype)
        out = torch.empty_like(h)

        sum_consistency = h.new_zeros(())
        sum_self_corr = h.new_zeros(())
        sum_coherence = h.new_zeros(())
        n_consistency_pairs = 0
        n_self_corr = 0
        n_coherence = 0

        for t in range(T):
            x = h[:, t, :]
            for i, block in enumerate(self.blocks):
                x, conv_states[i], domain_states[i], aux = block.forward_step(
                    x,
                    conv_states[i],
                    domain_states[i],
                    return_aux=compute_aux_losses,
                    extra_nodes=extra_nodes,
                )
                if aux is not None:
                    traj = aux["think_traj"]
                    for j in range(len(traj) - 1):
                        cos = F.cosine_similarity(traj[j], traj[j + 1], dim=-1)
                        sum_consistency = sum_consistency + (-cos).sum()
                        n_consistency_pairs += int(cos.numel())

                    q_erg = aux["q_erg"]
                    slots = aux["slots"]
                    dots = torch.einsum("bd,bnd->bn", q_erg, slots)
                    idx = dots.argmax(dim=-1)
                    rows = torch.arange(B, device=slots.device, dtype=torch.long)
                    best_slot = slots[rows, idx]
                    cos_mem = F.cosine_similarity(q_erg, best_slot, dim=-1)
                    sum_self_corr = sum_self_corr + F.relu(-cos_mem).sum()
                    n_self_corr += int(cos_mem.numel())

                    ctx = aux["ctx"]
                    cos_c = F.cosine_similarity(ctx, q_erg, dim=-1)
                    sum_coherence = sum_coherence + (1.0 - cos_c).sum()
                    n_coherence += int(cos_c.numel())

            out[:, t, :] = x

        h_out = self.ln_f(out)
        logits = self.lm_head(h_out)
        result: dict[str, Any] = {"logits": logits}
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss
        if compute_aux_losses and n_consistency_pairs > 0:
            result["loss_consistency"] = sum_consistency / n_consistency_pairs
            result["loss_self_correction"] = sum_self_corr / n_self_corr
            result["loss_coherence"] = sum_coherence / n_coherence
        return result

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_gpt2_rsm_model(
    cfg: Optional[GPT2RSMConfig] = None,
    tie_word_embeddings: bool = True,
    device: Optional[str] = None,
) -> GPT2RSMModel:
    m = GPT2RSMModel(cfg or GPT2RSMConfig(), tie_word_embeddings=tie_word_embeddings)
    if device:
        m = m.to(device)
    return m


def load_dialogpt_small_non_attention_weights(
    model: GPT2RSMModel,
    pretrained_model_name: str = "microsoft/DialoGPT-small",
) -> None:
    """
    Load HF ``GPT2LMHeadModel``: embeddings, ``ln_f``, LM head (if untied);
    per block ``ln_1`` → ``conv_rsm.norm1``, ``ln_2``, MLP ``c_fc`` / ``c_proj``;
    ``attn.c_proj`` → ``out_proj`` when ``(n_embd, n_embd)``. Skips ``c_attn``.
    """
    try:
        from transformers import GPT2LMHeadModel
    except ImportError as e:
        raise ImportError(
            "load_dialogpt_small_non_attention_weights requires `transformers`."
        ) from e

    hf = GPT2LMHeadModel.from_pretrained(pretrained_model_name)
    t = hf.transformer
    ne = model.cfg.n_embd

    model.wte.weight.data.copy_(t.wte.weight.data)
    n_ctx_hf = t.wpe.weight.shape[0]
    n_ctx_our = model.wpe.weight.shape[0]
    if n_ctx_hf >= n_ctx_our:
        model.wpe.weight.data.copy_(t.wpe.weight.data[:n_ctx_our])
    else:
        model.wpe.weight.data[:n_ctx_hf].copy_(t.wpe.weight.data)

    model.ln_f.weight.data.copy_(t.ln_f.weight.data)
    model.ln_f.bias.data.copy_(t.ln_f.bias.data)

    for i, hf_b in enumerate(t.h):
        b = model.blocks[i]
        b.conv_rsm.norm1.weight.data.copy_(hf_b.ln_1.weight.data)
        b.conv_rsm.norm1.bias.data.copy_(hf_b.ln_1.bias.data)
        b.ln_2.weight.data.copy_(hf_b.ln_2.weight.data)
        b.ln_2.bias.data.copy_(hf_b.ln_2.bias.data)
        b.mlp[0].weight.data.copy_(hf_b.mlp.c_fc.weight.data.t())
        b.mlp[0].bias.data.copy_(hf_b.mlp.c_fc.bias.data)
        b.mlp[2].weight.data.copy_(hf_b.mlp.c_proj.weight.data.t())
        b.mlp[2].bias.data.copy_(hf_b.mlp.c_proj.bias.data)

        acp = hf_b.attn.c_proj
        w = acp.weight.data
        if tuple(w.shape) == (ne, ne):
            b.out_proj.weight.data.copy_(w.t())
            if acp.bias is not None:
                b.out_proj.bias.data.copy_(acp.bias.data)

    if model.lm_head.weight.data_ptr() != model.wte.weight.data_ptr():
        model.lm_head.weight.data.copy_(hf.lm_head.weight.data)

    del hf
