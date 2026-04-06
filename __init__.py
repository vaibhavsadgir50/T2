"""
T2 V1 — RSM+ERG memory in place of transformer attention (DialoGPT-small scale).

Training: use ``train_colab.ipynb`` (all training code is in the notebook; no ``train.py``).
"""

from .config import (
    ERG_N_INPUT_NODES,
    ERG_N_SLOTS,
    ERG_N_STEPS,
    N_EMBD,
    N_HEAD,
    N_INNER,
    N_LAYER,
    N_POSITIONS,
    RSM_BETA,
    RSM_D_MODEL,
    RSM_MAX_D_MODEL,
    RSM_N_SLOTS,
    RSM_SLOT_EMA,
    T2_TRAIN_MAX_LENGTH,
    VOCAB_SIZE,
    erg_ffn_hidden_dim,
    resolve_rsm_d_model,
    rsm_ffn_hidden_dim,
)
from .erg_batched import erg_anchor_slots_forward
from .geometric_memory import GeometricResonantStateMemory, build_geometric_memory
from .model import (
    GPT2RSMBlock,
    GPT2RSMConfig,
    GPT2RSMModel,
    build_gpt2_rsm_model,
    load_dialogpt_small_non_attention_weights,
)

__all__ = [
    "N_EMBD",
    "N_HEAD",
    "N_LAYER",
    "N_POSITIONS",
    "N_INNER",
    "VOCAB_SIZE",
    "RSM_N_SLOTS",
    "T2_TRAIN_MAX_LENGTH",
    "RSM_D_MODEL",
    "RSM_MAX_D_MODEL",
    "RSM_BETA",
    "RSM_SLOT_EMA",
    "ERG_N_SLOTS",
    "ERG_N_INPUT_NODES",
    "ERG_N_STEPS",
    "resolve_rsm_d_model",
    "rsm_ffn_hidden_dim",
    "erg_ffn_hidden_dim",
    "erg_anchor_slots_forward",
    "GeometricResonantStateMemory",
    "build_geometric_memory",
    "GPT2RSMConfig",
    "GPT2RSMBlock",
    "GPT2RSMModel",
    "build_gpt2_rsm_model",
    "load_dialogpt_small_non_attention_weights",
]
