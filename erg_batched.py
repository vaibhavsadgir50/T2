"""
O(B × N) ERG-style geometric pass: anchor token + all slot vectors.

Full ``ExplicitReasoningGraph`` uses all node pairs (O(k²)). Here node 0 is
the anchor (incoming token); nodes 1..N-1 are RSM slots. For each slot we form
one (receiver=anchor, sender=slot) pair, apply the same five ERG primitives and
router, then aggregate slot messages with softmax attention from the anchor.

Input:  (B, N, d_model) with N = 1 + n_slots — [:,0] token, [:,1:] slots.
Output: (B, d_model) — refined token after ``ERG_N_STEPS`` anchor iterations.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from dpnn.erg_reasoning import ExplicitReasoningGraph

from .config import ERG_N_STEPS


def _erg_anchor_slot_step(
    erg: ExplicitReasoningGraph,
    h0: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    """
    One refinement step: O(B × S) in slot count S = N - 1.

    Args:
        erg:   shared ``ExplicitReasoningGraph`` modules (pair_proj, ops, …).
        h0:    (B, d_model) anchor (token).
        slots: (B, S, d_model) slot rows (not mutated).

    Returns:
        (B, d_model) updated anchor.
    """
    B, S, d_model = slots.shape
    # Receiver = anchor, sender = slot — matches ERG pair layout cat(h_recv, h_send).
    h_recv = h0.unsqueeze(1).expand(-1, S, -1)
    pairs = torch.cat([h_recv, slots], dim=-1)

    pair_feat = erg.pair_proj(pairs)
    op_w = F.softmax(erg.op_scorer(pair_feat), dim=-1)

    m0 = erg.op_compose(pairs)
    m1 = erg.op_compare(pairs)
    m2 = erg.op_negate(h_recv)
    m3 = erg.op_generalise(pairs)
    m4 = erg.op_instantiate(pairs)
    ops = torch.stack([m0, m1, m2, m3, m4], dim=2)
    msgs = (op_w.unsqueeze(-1) * ops).sum(dim=2)

    scale = d_model ** -0.5
    attn_logits = (h0.unsqueeze(1) * slots).sum(dim=-1) * scale
    attn = F.softmax(attn_logits, dim=-1)
    msg_agg = (attn.unsqueeze(-1) * msgs).sum(dim=1)

    h_new = erg.update_proj(torch.cat([h0, msg_agg], dim=-1))
    h0 = erg.norm1(h_new) + h0
    h0 = h0 + erg.ffn(erg.norm2(h0))
    return h0


def erg_anchor_slots_forward(
    erg: ExplicitReasoningGraph,
    nodes: torch.Tensor,
    n_steps: Optional[int] = None,
) -> torch.Tensor:
    """
    Geometric pass: anchor token with full slot context, linear in slot count.

    Args:
        erg:    ``ExplicitReasoningGraph`` (``d_model`` must match ``nodes``).
        nodes:  (B, N, d_model) with ``N == 1 + n_slots``.
                ``nodes[:, 0]`` — new token; ``nodes[:, 1:]`` — current slots.
        n_steps: override default from :data:`T2.config.ERG_N_STEPS`.

    Returns:
        (B, d_model) transformed token (node 0) after all steps.

    Complexity: ``O(n_steps × B × N × d_model²)`` from linears; **linear in N**,
    not quadratic in N (no all-pairs graph).
    """
    if nodes.dim() != 3:
        raise ValueError(f"nodes must be (B, N, d), got shape {tuple(nodes.shape)}")
    B, N, d_model = nodes.shape
    if N < 2:
        raise ValueError(
            f"need N >= 2 (token + at least one slot), got N={N}"
        )
    steps = ERG_N_STEPS if n_steps is None else int(n_steps)

    h0 = nodes[:, 0, :].contiguous()
    slots = nodes[:, 1:, :].contiguous()

    for _ in range(steps):
        h0 = _erg_anchor_slot_step(erg, h0, slots)

    return h0
