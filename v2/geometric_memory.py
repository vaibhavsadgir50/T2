"""
T2 — Conversation RSM: write → think (ERG) → read (attention).

**think** runs multiple ERG anchor passes over slots (and optional domain nodes),
refining a conclusion vector before **read** applies softmax attention over slots only.

Parallel path (write_parallel / think_parallel / read_parallel):
  Replaces the per-token sequential loop with a chunked prefix-scan EMA write and
  batched ERG think/read over the full sequence. This eliminates the O(T) sequential
  CUDA-kernel launches that cause OOM on T4 / A100 for long sequences.

  Write recurrence (per slot s):
      state[t, s] = (1-α) * state[t-1, s] + α * w[t, s] * v[t]
  where w[t, s] = softmax(query_proj(norm1(h[t])) @ state0.T)[s].
  Routing uses the *initial* state (state0) to break the sequential dependency.
  The recurrence has a constant decay coefficient a = 1-α, enabling a parallel
  prefix-scan via cumsum within chunks of size ≤ 128 tokens (safe in float32).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C
from .erg_markov import MarkovERG, build_markov_erg, erg_anchor_slots_forward  # noqa: F401

_NORMALIZE_EPS: float = 1e-8  # guard for all F.normalize calls


def _resolve_slot_ema(explicit: Optional[float]) -> float:
    if explicit is not None:
        return float(explicit)
    b = float(C.RSM_BETA)
    if 0.0 < b <= 1.0:
        return b
    return float(C.RSM_SLOT_EMA)


class GeometricResonantStateMemory(nn.Module):
    """
    **Write:** ERG routes value into slots (EMA).

    **Think:** ``n_steps`` times, refine a hidden vector against conv slots (+ optional
    extra ERG nodes); each step is one anchor–slot ERG pass.

    **Read:** softmax attention ``q @ slots`` only (no ERG); ``q`` is typically the
    conclusion from **think**. Use **output_proj** only in **memory_step_batched** legacy path.
    """

    def __init__(
        self,
        d_model: Optional[int] = None,
        n_slots: Optional[int] = None,
        erg_ffn_dim: Optional[int] = None,
        erg_n_steps: Optional[int] = None,
        slot_ema: Optional[float] = None,
        read_logit_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        rd = C.resolve_rsm_d_model(d_model)
        ns = n_slots if n_slots is not None else C.RSM_N_SLOTS
        efd = erg_ffn_dim if erg_ffn_dim is not None else C.erg_ffn_hidden_dim(rd)
        est = erg_n_steps if erg_n_steps is not None else C.ERG_N_STEPS
        ema = _resolve_slot_ema(slot_ema)
        if not (0.0 < ema <= 1.0):
            raise ValueError(f"slot_ema must be in (0, 1], got {ema}")
        rscale = float(
            read_logit_scale if read_logit_scale is not None else C.RSM_BETA
        )

        self.d_model = rd
        self.n_slots = ns
        self.erg_n_steps = est
        self.slot_ema = ema
        self.read_logit_scale = rscale

        self.norm1 = nn.LayerNorm(rd)
        self.query_proj = nn.Linear(rd, rd)
        self.value_proj = nn.Linear(rd, rd)
        self.output_proj = nn.Linear(rd, rd)
        self.erg = build_markov_erg(
            d_model=rd,
            n_steps=est,
            ffn_dim=efd,
        )
        self.register_buffer("state", torch.zeros(ns, rd))
        self._init_weights()

    def _init_weights(self) -> None:
        for m in (self.query_proj, self.value_proj, self.output_proj):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def reset_state(self) -> None:
        g = torch.Generator()
        g.manual_seed(42)
        fresh = torch.randn(
            self.n_slots, self.d_model, generator=g, device=self.state.device
        ).mul_(0.01)
        self.state.copy_(fresh)

    def init_batched_memory_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        g = torch.Generator()
        g.manual_seed(42)
        base = torch.randn(
            self.n_slots, self.d_model, generator=g, dtype=dtype, device="cpu"
        ).mul_(0.01)
        base = base.to(device=device, dtype=dtype)
        return base.unsqueeze(0).expand(batch_size, -1, -1).contiguous().clone()

    def _slot_logits(
        self, vec: torch.Tensor, slots: torch.Tensor
    ) -> torch.Tensor:
        d = self.d_model
        scale = self.read_logit_scale * (d ** -0.5)
        return torch.bmm(vec.unsqueeze(1), slots.transpose(1, 2)).squeeze(1) * scale

    def write_step(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """ERG-routed write into slots; returns ``new_state``."""
        x_norm = self.norm1(x)
        slots0 = state.detach()
        B = x.shape[0]

        v_new = self.value_proj(x_norm)
        nodes_w = torch.cat([v_new.unsqueeze(1), slots0], dim=1)
        v_geo = erg_anchor_slots_forward(self.erg, nodes_w, self.erg_n_steps)
        scores_w = self._slot_logits(v_geo, slots0)
        idx = scores_w.argmax(dim=-1)
        ema = self.slot_ema
        new_state = state.clone()
        with torch.no_grad():
            rows = torch.arange(B, device=state.device, dtype=torch.long)
            old = new_state[rows, idx].clone()
            new_state[rows, idx] = (1.0 - ema) * old + ema * v_geo.detach()
        return new_state

    def think(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        extra_nodes: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        ``n_steps`` ERG refinements: each step runs one anchor–slot ERG pass over
        ``[h, conv_slots, *extra_nodes]``; ``h`` starts as ``query_proj(norm1(x))``.

        Returns:
            ``q_erg`` of shape ``(B, d_model)``, or ``(q_erg, trajectory)`` if
            ``return_trajectory`` — trajectory is ``[h_0, h_1, …, h_N]`` (length
            ``n_steps + 1``) for consistency losses.
        """
        steps = int(self.erg_n_steps if n_steps is None else n_steps)
        slots = state.detach()
        h = self.query_proj(self.norm1(x))
        trajectory: Optional[List[torch.Tensor]] = [h] if return_trajectory else None

        for _ in range(steps):
            parts = [h.unsqueeze(1), slots]
            if extra_nodes is not None:
                if extra_nodes.shape[0] != x.shape[0]:
                    raise ValueError("extra_nodes batch mismatch")
                if extra_nodes.shape[-1] != self.d_model:
                    raise ValueError(
                        f"extra_nodes dim {extra_nodes.shape[-1]} != d_model {self.d_model}"
                    )
                parts.append(extra_nodes)
            nodes = torch.cat(parts, dim=1)
            h = erg_anchor_slots_forward(self.erg, nodes, n_steps=1)
            if trajectory is not None:
                trajectory.append(h)

        if return_trajectory:
            assert trajectory is not None
            return h, trajectory
        return h

    def read(self, q: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Attention over conv slots only: ``softmax(scores) @ slots`` → ``(B, d_model)``.
        No ERG. ``q`` is usually the **think** conclusion; see **read_with_query** for
        ``(ctx, q)`` used in coherence losses.
        """
        slots = state.detach()
        scores_r = self._slot_logits(q, slots)
        attn_r = F.softmax(scores_r, dim=-1)
        return torch.bmm(attn_r.unsqueeze(1), slots).squeeze(1)

    def read_with_query(
        self, q: torch.Tensor, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(ctx, q)`` — same ``q`` tensor as passed in (for coherence vs. ERG conclusion)."""
        return self.read(q, state), q

    def read_from_x(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Domain-style read from token slice: ``read(query_proj(norm1(x)), state)``."""
        q = self.query_proj(self.norm1(x))
        return self.read(q, state)

    # ------------------------------------------------------------------
    # Parallel path — processes full (B, T, d) sequence at once
    # ------------------------------------------------------------------

    def write_parallel(
        self,
        h: torch.Tensor,
        state0: torch.Tensor,
        chunk_size: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Soft-routed parallel EMA write over a full token sequence.

        Recurrence per slot s:
            state[t, s] = (1-α) * state[t-1, s] + α * w[t, s] * v[t]
        Routing weights w are computed from token queries against ``state0`` (the
        initial state), breaking the sequential state dependency.  The constant
        decay coefficient a = 1-α enables a parallel prefix-scan via torch.cumsum
        within chunks of at most ``chunk_size`` tokens.

        Args:
            h:          ``(B, T, d_model)`` — token embeddings (pre-embedded).
            state0:     ``(B, n_slots, d_model)`` — initial slot state.
            chunk_size: tokens per chunk for the prefix scan (≤128 is safe in
                        float32; larger values risk a^(-t) overflow).

        Returns:
            states:      ``(B, T, n_slots, d_model)`` — slot state *after* each write.
            final_state: ``(B, n_slots, d_model)`` — state after the last token.
        """
        B, T, d = h.shape
        S = self.n_slots
        alpha = self.slot_ema
        a = 1.0 - alpha

        # --- value and routing query — fully parallel ---
        h_flat = h.reshape(B * T, d)
        h_norm = self.norm1(h_flat).reshape(B, T, d)
        v = self.value_proj(h_norm)   # (B, T, d)
        q = self.query_proj(h_norm)   # (B, T, d)

        # Routing weights: softmax(q @ state0^T / scale)   shape: (B, T, S)
        scale = self.read_logit_scale * (d ** -0.5)
        scores = torch.bmm(q, state0.transpose(1, 2)) * scale  # (B, T, S)
        w = F.softmax(scores, dim=-1)                           # (B, T, S)

        # Per-slot write input: u[t, s] = α * w[t, s] * v[t]   shape: (B, T, S, d)
        u = alpha * w.unsqueeze(-1) * v.unsqueeze(2)

        # --- chunked cumsum prefix scan ---
        # Within chunk of length C, with constant decay a:
        #   state[t] = a^(t+1) * state_prev + a^t * cumsum(u[k] * a^(-k))[t]
        # chunk_size ≤ 128 keeps a^(-127) ≤ ~1.7e5 for a=0.9, safe in float32.
        all_chunks: List[torch.Tensor] = []
        state = state0

        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            u_c = u[:, start:end]            # (B, C, S, d)
            C = end - start

            t_idx = torch.arange(C, device=h.device, dtype=h.dtype)
            decay = a ** t_idx               # (C,) — a^0 … a^(C-1)
            inv_decay = 1.0 / decay.clamp(min=1e-30)

            # Scale, cumsum, then rescale
            cs = torch.cumsum(u_c * inv_decay.view(1, C, 1, 1), dim=1)  # (B, C, S, d)

            # state[t] = a^t * cs[t] + a^(t+1) * state_prev
            t1_decay = a ** (t_idx + 1)      # a^1 … a^C
            chunk_states = (
                decay.view(1, C, 1, 1) * cs
                + t1_decay.view(1, C, 1, 1) * state.unsqueeze(1)
            )                                # (B, C, S, d)

            all_chunks.append(chunk_states)
            state = chunk_states[:, -1]      # carry state to next chunk

        states = torch.cat(all_chunks, dim=1)   # (B, T, S, d)
        return states, state

    def think_parallel(
        self,
        h: torch.Tensor,
        states: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Batched ERG think over a full sequence.

        Args:
            h:      ``(B, T, d_model)`` — token embeddings.
            states: ``(B, T, n_slots, d_model)`` — per-token slot states
                    (e.g. from ``write_parallel``).

        Returns:
            ``(B, T, d_model)`` — ERG-refined query vectors.
        """
        B, T, d = h.shape
        BT = B * T
        h_flat = h.reshape(BT, d)
        anchors = self.query_proj(self.norm1(h_flat))   # (B*T, d)
        slots = states.reshape(BT, self.n_slots, d)
        steps = self.erg_n_steps if n_steps is None else int(n_steps)
        q_erg = self.erg.forward(anchors, slots, n_steps=steps)  # (B*T, d)
        return q_erg.reshape(B, T, d)

    def read_parallel(
        self,
        q: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched softmax slot-attention read over a full sequence.

        Args:
            q:      ``(B, T, d_model)`` — query vectors (e.g. from ``think_parallel``).
            states: ``(B, T, n_slots, d_model)`` — per-token slot states.

        Returns:
            ``(B, T, d_model)`` — context vectors.
        """
        B, T, d = q.shape
        BT = B * T
        slots = states.reshape(BT, self.n_slots, d)
        scale = self.read_logit_scale * (d ** -0.5)
        logits = torch.bmm(q.reshape(BT, 1, d), slots.transpose(1, 2)).squeeze(1) * scale
        attn = F.softmax(logits, dim=-1)                    # (B*T, S)
        ctx = torch.bmm(attn.unsqueeze(1), slots).squeeze(1)  # (B*T, d)
        return ctx.reshape(B, T, d)

    # ------------------------------------------------------------------
    # Legacy single-step path
    # ------------------------------------------------------------------

    def memory_step_batched(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``write_step`` → ``think`` → ``read``; returns ``(x + output_proj(ctx), state)``."""
        st = self.write_step(x, state)
        q_erg = self.think(x, st, extra_nodes=None, n_steps=self.erg_n_steps)
        ctx = self.read(q_erg, st)
        return x + self.output_proj(ctx), st

    def forward(self, x: torch.Tensor, sequential: bool = True) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"x must be (N, d), got {tuple(x.shape)}")
        N, d = x.shape
        if d != self.d_model:
            raise ValueError(f"d_model mismatch: x has {d}, module has {self.d_model}")

        if sequential:
            st = self.init_batched_memory_state(1, x.device, x.dtype)
            outs = []
            for i in range(N):
                o, st = self.memory_step_batched(x[i : i + 1], st)
                outs.append(o)
            self.state.copy_(st.squeeze(0))
            return torch.cat(outs, dim=0)

        st = self.init_batched_memory_state(N, x.device, x.dtype)
        out, _ = self.memory_step_batched(x, st)
        return out


def build_geometric_memory(
    d_model: Optional[int] = None,
    n_slots: Optional[int] = None,
    erg_ffn_dim: Optional[int] = None,
    erg_n_steps: Optional[int] = None,
    slot_ema: Optional[float] = None,
    read_logit_scale: Optional[float] = None,
    device: Optional[str] = None,
) -> GeometricResonantStateMemory:
    m = GeometricResonantStateMemory(
        d_model=d_model,
        n_slots=n_slots,
        erg_ffn_dim=erg_ffn_dim,
        erg_n_steps=erg_n_steps if erg_n_steps is not None else C.ERG_N_STEPS,
        slot_ema=slot_ema,
        read_logit_scale=read_logit_scale,
    )
    if device:
        m = m.to(device)
    return m
