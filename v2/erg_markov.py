"""
Markov-chain reasoning over RSM slots (NTM-style controller).

Softmax attention from the anchor onto slots is the transition distribution; each
step gated-mixes the anchor toward the expected slot context and refines with an FFN.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

class MarkovERG(nn.Module):
    """
    Reasoning engine: ``n_steps`` iterations of transition → context → gate → FFN.

    The Markov transition is ``softmax(q @ slots^T / sqrt(d))`` with
    ``q = transition_proj(norm1(anchor))``, i.e. a learned query direction in slot space.
    """

    def __init__(
        self,
        d_model: int,
        n_steps: int = 6,
        ffn_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if ffn_dim is None:
            ffn_dim = max(1, 4 * d_model)
        self.d_model = d_model
        self.n_steps = int(n_steps)
        self.ffn_dim = int(ffn_dim)

        self.transition_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.ffn_dim, d_model),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.transition_proj.weight)
        nn.init.zeros_(self.transition_proj.bias)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        for m in self.ffn:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        anchor: torch.Tensor,
        slots: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            anchor: ``(B, d_model)`` query / token state.
            slots: ``(B, n_slots, d_model)`` RSM memory (not modified).

        Returns:
            ``(B, d_model)`` anchor after ``n_steps`` reasoning steps.
        """
        if anchor.dim() != 2 or slots.dim() != 3:
            raise ValueError(
                f"anchor must be (B, d), slots (B, S, d); got {anchor.shape}, {slots.shape}"
            )
        B, d_a = anchor.shape
        Bs, S, d_s = slots.shape
        if B != Bs or d_a != self.d_model or d_s != self.d_model:
            raise ValueError(
                f"shape mismatch: anchor {anchor.shape}, slots {slots.shape}, "
                f"d_model={self.d_model}"
            )

        steps = self.n_steps if n_steps is None else int(n_steps)
        scale = self.d_model ** -0.5
        h = anchor

        for _ in range(steps):
            q = self.transition_proj(self.norm1(h))
            logits = torch.bmm(q.unsqueeze(1), slots.transpose(1, 2)).squeeze(1) * scale
            scores = F.softmax(logits, dim=-1)
            context = torch.bmm(scores.unsqueeze(1), slots).squeeze(1)

            gate_in = torch.cat([h, context], dim=-1)
            g = torch.sigmoid(self.gate(gate_in))
            h = h * (1.0 - g) + context * g
            h = h + self.ffn(self.norm2(h))

        return h


def build_markov_erg(
    d_model: int,
    n_steps: int = 6,
    ffn_dim: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> MarkovERG:
    m = MarkovERG(d_model, n_steps=n_steps, ffn_dim=ffn_dim)
    if device is not None:
        m = m.to(device)
    return m


def erg_anchor_slots_forward(
    erg: MarkovERG,
    nodes: torch.Tensor,
    n_steps: Optional[int] = None,
) -> torch.Tensor:
    """
    Compatibility wrapper: ``nodes[:, 0]`` = anchor, ``nodes[:, 1:]`` = slots.

    Args:
        erg: ``MarkovERG`` instance.
        nodes: ``(B, 1 + n_slots, d_model)``.
        n_steps: if ``None``, uses ``erg.n_steps``; else overrides step count for this call.
    """
    if not isinstance(erg, MarkovERG):
        raise TypeError(f"erg must be MarkovERG, got {type(erg).__name__}")
    if nodes.dim() != 3:
        raise ValueError(f"nodes must be (B, N, d), got shape {tuple(nodes.shape)}")
    B, N, d_model = nodes.shape
    if N < 2:
        raise ValueError(f"need N >= 2 (anchor + slots), got N={N}")
    if d_model != erg.d_model:
        raise ValueError(f"nodes d_model {d_model} != erg.d_model {erg.d_model}")

    anchor = nodes[:, 0, :].contiguous()
    slots = nodes[:, 1:, :].contiguous()
    return erg.forward(anchor, slots, n_steps=n_steps)
