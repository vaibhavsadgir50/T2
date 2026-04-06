"""
T2 — DialoGPT-small backbone constants + RSM/ERG configuration.

Backbone sizes are fixed for the default LM (GPT-2 small / DialoGPT-small).
RSM/ERG width is derived at runtime from the backbone ``d_model``; no FFN size
is tied to a single hardcoded embedding dim except via helpers below.
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Hugging Face / OpenAI GPT-2 small — same as microsoft/DialoGPT-small
# ---------------------------------------------------------------------------

N_EMBD: int = 768
N_HEAD: int = 12
N_LAYER: int = 12
N_POSITIONS: int = 1024
VOCAB_SIZE: int = 50257  # GPT-2 tokenizer

N_INNER: int = N_EMBD * 4  # 3072 — GPT-2 block MLP hidden (also 4× n_embd)

# ---------------------------------------------------------------------------
# RSM — slot memory (count is fixed; vector dim comes from backbone at runtime)
# ---------------------------------------------------------------------------

RSM_N_SLOTS: int = 1024

# Set by the trainer / model builder to override implicit backbone default.
RSM_D_MODEL: Optional[int] = None

RSM_MAX_D_MODEL: int = 16384

# Scales slot–token dot logits (read softmax and write slot pick) after 1/sqrt(d).
RSM_BETA: float = 4.0

# V1 single-slot EMA when RSM_BETA is not in (0, 1] (e.g. 4.0 used as logit scale only).
RSM_SLOT_EMA: float = 0.1

# Legacy name: RSM FFN width multiplier if other code builds RSM-internal FFNs.
RSM_FFN_MULT: int = 4

# ---------------------------------------------------------------------------
# ERG — graph width tied to slot count for spec / future wiring
# ---------------------------------------------------------------------------

# Mirrors slot cardinality (importable separately from total ERG node count).
ERG_N_SLOTS: int = RSM_N_SLOTS

# Full ERG input graph size: one token (or summary) node + one node per slot.
ERG_N_INPUT_NODES: int = 1 + RSM_N_SLOTS

ERG_N_STEPS: int = 4

# ERG FFN hidden size = d_model × (ERG_REF_FFN / ERG_REF_D_MODEL); refs are scale only.
ERG_REF_D_MODEL: int = 512
ERG_REF_FFN: int = 557


def resolve_rsm_d_model(d_model: Optional[int] = None) -> int:
    """
    Effective RSM/ERG embedding width.

    Priority: explicit ``d_model`` argument → :data:`RSM_D_MODEL` if set →
    :data:`N_EMBD` (DialoGPT-small default backbone).
    """
    d = d_model if d_model is not None else RSM_D_MODEL
    if d is None:
        d = N_EMBD
    if d <= 0 or d > RSM_MAX_D_MODEL:
        raise ValueError(
            f"d_model must be in (0, {RSM_MAX_D_MODEL}], got {d}"
        )
    return int(d)


def rsm_ffn_hidden_dim(d_model: Optional[int] = None) -> int:
    """RSM block FFN hidden size from resolved ``d_model``."""
    d = resolve_rsm_d_model(d_model)
    return int(d * RSM_FFN_MULT)


def erg_ffn_hidden_dim(d_model: Optional[int] = None) -> int:
    """ERG internal FFN hidden size, proportional to resolved ``d_model``."""
    d = resolve_rsm_d_model(d_model)
    return max(1, int(round(d * (ERG_REF_FFN / float(ERG_REF_D_MODEL)))))
