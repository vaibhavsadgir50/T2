# V2 config — d_model=300 matches knowledge graph vectors
# No bridge needed — graph plugs directly into RSM slots
# Not tied to DialoGPT or any pretrained model

"""
T2 V2 — backbone constants + RSM/ERG configuration.

RSM/ERG width is derived at runtime from the backbone ``d_model``; FFN sizes
use helpers below.
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Backbone (V2 test — not a pretrained checkpoint)
# ---------------------------------------------------------------------------

N_EMBD: int = 300
N_HEAD: int = 4
N_LAYER: int = 2
N_POSITIONS: int = 512
VOCAB_SIZE: int = 50257

N_INNER: int = N_EMBD * 4  # 1200 — block MLP hidden (4× n_embd)

# ---------------------------------------------------------------------------
# RSM — slot memory (count is fixed; vector dim comes from backbone at runtime)
# ---------------------------------------------------------------------------

RSM_N_SLOTS: int = 64

T2_TRAIN_MAX_LENGTH: int = 512

RSM_D_MODEL: Optional[int] = None

RSM_MAX_D_MODEL: int = 300

# Scales slot–token dot logits (read softmax and write slot pick) after 1/sqrt(d).
RSM_BETA: float = 4.0

# V1 single-slot EMA when RSM_BETA is not in (0, 1] (e.g. 4.0 used as logit scale only).
RSM_SLOT_EMA: float = 0.1

# Legacy name: RSM FFN width multiplier if other code builds RSM-internal FFNs.
RSM_FFN_MULT: int = 4

# ---------------------------------------------------------------------------
# ERG — graph width tied to slot count for spec / future wiring
# ---------------------------------------------------------------------------

ERG_N_SLOTS: int = RSM_N_SLOTS

ERG_N_INPUT_NODES: int = 1 + RSM_N_SLOTS

ERG_N_STEPS: int = 6

# ERG FFN hidden size = d_model × (ERG_REF_FFN / ERG_REF_D_MODEL); refs are scale only.
ERG_REF_D_MODEL: int = 300
ERG_REF_FFN: int = 557


def resolve_rsm_d_model(d_model: Optional[int] = None) -> int:
    """
    Effective RSM/ERG embedding width.

    Priority: explicit ``d_model`` argument → :data:`RSM_D_MODEL` if set →
    :data:`N_EMBD`.
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
