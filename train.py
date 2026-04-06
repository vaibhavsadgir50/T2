"""
T2 V1 — world-model training (write → think → predict) with auxiliary ERG losses.

Colab — cell 0 (run before this script)
======================================
.. code-block:: python

    import os, sys, subprocess
    assert torch.cuda.is_available(), "GPU required"
    from google.colab import drive
    drive.mount("/content/drive")

    token = os.environ.get("GITHUB_TOKEN", "")
    subprocess.run(
        ["git", "clone", f"https://{token}@github.com/vaibhavsadgir50/uni.git", "/content/uni"],
        check=True,
    )
    sys.path.insert(0, "/content/uni")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "datasets", "torch"], check=True)

Run training (from repo root or with ``sys.path`` as above)::

    python -m T2.train

Default checkpoint: ``/content/drive/MyDrive/dpnn_g/checkpoints/t2_v1_best.pt``
(``t2_v1_epoch_*`` files use the same directory). On Colab, mount Drive first;
``train_colab.ipynb`` sets ``T2_CHECKPOINT`` and creates that folder. Override
with ``--checkpoint_path`` or ``T2_CHECKPOINT`` for local runs.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from . import config as C
from .model import (
    GPT2RSMConfig,
    GPT2RSMModel,
    load_dialogpt_small_non_attention_weights,
)

AUX_WEIGHT = 0.1
DEFAULT_CHECKPOINT = "/content/drive/MyDrive/dpnn_g/checkpoints/t2_v1_best.pt"


def _format_duration(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(round(seconds % 60))
    return f"{m}m {s}s"


def _build_checkpoint_payload(
    model: GPT2RSMModel,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: Optional[GradScaler],
    cfg: GPT2RSMConfig,
    epoch: int,
    val_loss: float,
    train_metrics: Dict[str, float],
    best_val_loss: float,
    best_epoch: int,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "train_loss": train_metrics["total"],
        "train_ce": train_metrics["ce"],
        "loss_consistency": train_metrics["consistency"],
        "loss_self_correction": train_metrics["self_correction"],
        "loss_coherence": train_metrics["coherence"],
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "cfg": dataclasses.asdict(cfg),
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    return payload


def _plot_training_curves(
    epochs_1based: List[int],
    train_ce: List[float],
    val_ce: List[float],
    ppl: List[float],
    consistency: List[float],
    self_corr: List[float],
    coherence: List[float],
    lrs: List[float],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].plot(epochs_1based, train_ce, "b-", marker="o", markersize=3)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Train CE")
    axes[0, 0].set_title("Train CE loss")
    axes[0, 0].grid(True, alpha=0.3)

    ax_v = axes[0, 1]
    ax_v.plot(epochs_1based, val_ce, "g-", marker="o", markersize=3, label="Val CE")
    ax_v.set_xlabel("Epoch")
    ax_v.set_ylabel("Val CE", color="g")
    ax_v.tick_params(axis="y", labelcolor="g")
    ax_p = ax_v.twinx()
    ppl_plot = [min(x, 1e10) if math.isfinite(x) else 1e10 for x in ppl]
    ax_p.plot(epochs_1based, ppl_plot, "r--", marker="s", markersize=3, label="Perplexity")
    ax_p.set_ylabel("Perplexity", color="r")
    ax_p.tick_params(axis="y", labelcolor="r")
    ax_v.set_title("Val CE & perplexity")
    ax_v.grid(True, alpha=0.3)

    axes[1, 0].plot(
        epochs_1based, consistency, label="consistency", marker="o", markersize=2
    )
    axes[1, 0].plot(
        epochs_1based, self_corr, label="self_correction", marker="o", markersize=2
    )
    axes[1, 0].plot(
        epochs_1based, coherence, label="coherence", marker="o", markersize=2
    )
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Aux loss")
    axes[1, 0].set_title("Auxiliary losses (train)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs_1based, lrs, "m-", marker="o", markersize=3)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning rate")
    axes[1, 1].set_title("LR schedule")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def freeze_t2_v1_pretrained(model: GPT2RSMModel) -> None:
    """DialoGPT-small knowledge: embeddings, LM head, every LayerNorm frozen."""
    model.wte.weight.requires_grad_(False)
    model.wpe.weight.requires_grad_(False)
    if model.lm_head.weight.data_ptr() != model.wte.weight.data_ptr():
        model.lm_head.weight.requires_grad_(False)
    for mod in model.modules():
        if isinstance(mod, nn.LayerNorm):
            for p in mod.parameters():
                p.requires_grad_(False)


def dialogue_to_string(dialog: List[str], eos: str) -> str:
    parts = [u.strip() for u in dialog if u and str(u).strip()]
    if not parts:
        return ""
    return eos.join(parts) + eos


class DailyDialogDataset(Dataset):
    """DailyDialog turns concatenated with EOS between them; max length ``n_positions``."""

    def __init__(
        self,
        split_rows: List[Dict[str, Any]],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.samples: List[Dict[str, torch.Tensor]] = []
        eos = tokenizer.eos_token or "<|endoftext|>"
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id

        for row in split_rows:
            dialog = row.get("dialog")
            if dialog is None:
                continue
            text = dialogue_to_string(list(dialog), eos)
            if not text:
                continue
            enc = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attn = enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attn == 0] = -100
            self.samples.append({"input_ids": input_ids.long(), "labels": labels.long()})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return self.samples[i]


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "labels": torch.stack([b["labels"] for b in batch], dim=0),
    }


def load_daily_dialog_splits(
    tokenizer: Any,
    max_length: int,
    seed: int = 42,
) -> tuple[DailyDialogDataset, DailyDialogDataset]:
    from datasets import load_dataset

    # Canonical script dataset ``daily_dialog`` is often blocked on modern ``datasets``;
    # ``OpenRL/daily_dialog`` is parquet-backed with the same ``dialog`` list field.
    last_err: Optional[Exception] = None
    raw = None
    for name in ("OpenRL/daily_dialog", "daily_dialog"):
        try:
            kwargs: Dict[str, Any] = {}
            if name == "daily_dialog":
                kwargs["trust_remote_code"] = True
            raw = load_dataset(name, **kwargs)
            break
        except Exception as e:
            last_err = e
    if raw is None:
        raise RuntimeError(
            "Could not load DailyDialog-style dataset (tried OpenRL/daily_dialog, daily_dialog)."
        ) from last_err

    train_hf = raw["train"]
    split = train_hf.train_test_split(test_size=0.1, seed=seed)
    return (
        DailyDialogDataset(split["train"], tokenizer, max_length),
        DailyDialogDataset(split["test"], tokenizer, max_length),
    )


def train_one_epoch(
    model: GPT2RSMModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: Optional[GradScaler],
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    sums = {
        "total": 0.0,
        "ce": 0.0,
        "consistency": 0.0,
        "self_correction": 0.0,
        "coherence": 0.0,
    }
    n_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        Bsz = input_ids.shape[0]
        mdtype = model.wte.weight.dtype
        inject = (torch.rand(Bsz, device=device) < 0.5).float().view(Bsz, 1, 1).to(
            mdtype
        )
        batch_extra_nodes = (
            torch.randn(Bsz, 1, C.N_EMBD, device=device, dtype=mdtype) * 0.1 * inject
        )

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            out = model(
                input_ids,
                labels=labels,
                compute_aux_losses=True,
                extra_nodes=batch_extra_nodes,
            )
            ce = out["loss"]
            lc = out["loss_consistency"]
            ls = out["loss_self_correction"]
            lco = out["loss_coherence"]
            total = ce + AUX_WEIGHT * (lc + ls + lco)

        if use_amp and scaler is not None:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        sums["total"] += float(total.detach())
        sums["ce"] += float(ce.detach())
        sums["consistency"] += float(lc.detach())
        sums["self_correction"] += float(ls.detach())
        sums["coherence"] += float(lco.detach())
        n_batches += 1

    inv = 1.0 / max(n_batches, 1)
    return {k: v * inv for k, v in sums.items()}


@torch.no_grad()
def eval_ce_only(
    model: GPT2RSMModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with autocast(enabled=use_amp):
            out = model(input_ids, labels=labels, compute_aux_losses=False)
        total += float(out["loss"].detach())
        n += 1
    return total / max(n, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="T2 V1 world-model training")
    p.add_argument(
        "--checkpoint_path",
        type=str,
        default=os.environ.get("T2_CHECKPOINT", DEFAULT_CHECKPOINT),
    )
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--eta_min", type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--pretrained", type=str, default="microsoft/DialoGPT-small")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    abs_ckpt = os.path.abspath(args.checkpoint_path)
    if abs_ckpt.startswith("/content/drive/") and not os.path.isdir(
        "/content/drive/MyDrive"
    ):
        print(
            "Checkpoint path is under /content/drive but Google Drive is not mounted "
            "(missing /content/drive/MyDrive). In Colab run drive.mount('/content/drive') "
            "first, or set --checkpoint_path / T2_CHECKPOINT to a writable path.",
            file=sys.stderr,
        )
        sys.exit(1)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = GPT2RSMConfig(
        vocab_size=len(tokenizer),
        n_embd=C.N_EMBD,
        n_layer=C.N_LAYER,
        n_head=C.N_HEAD,
        n_inner=C.N_INNER,
        n_positions=C.N_POSITIONS,
        n_slots=C.RSM_N_SLOTS,
        erg_n_steps=C.ERG_N_STEPS,
    )

    train_ds, val_ds = load_daily_dialog_splits(
        tokenizer, max_length=cfg.n_positions, seed=args.seed
    )
    if len(train_ds) == 0:
        print(
            "DailyDialog dataset is empty after parsing. "
            "Check `datasets` version / dataset schema (expected `dialog` list field).",
            file=sys.stderr,
        )
        sys.exit(1)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )

    model = GPT2RSMModel(cfg, tie_word_embeddings=True)
    load_dialogpt_small_non_attention_weights(model, pretrained_model_name=args.pretrained)
    freeze_t2_v1_pretrained(model)
    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = args.epochs * steps_per_epoch
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.eta_min)
    scaler: Optional[GradScaler] = GradScaler(enabled=use_amp) if use_amp else None

    start_epoch = 0
    best_val = math.inf
    ckpt_dir = os.path.dirname(args.checkpoint_path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    best_epoch = -1
    if os.path.isfile(args.checkpoint_path):
        print(f"Resuming from {args.checkpoint_path}")
        try:
            ckpt = torch.load(
                args.checkpoint_path, map_location=device, weights_only=False
            )
        except TypeError:
            ckpt = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val = float(ckpt.get("best_val_loss", math.inf))
        best_epoch = int(ckpt.get("best_epoch", -1))

    no_improve = 0
    t0_all = time.perf_counter()

    hist_ep: List[int] = []
    hist_train_ce: List[float] = []
    hist_val_ce: List[float] = []
    hist_ppl: List[float] = []
    hist_cons: List[float] = []
    hist_self: List[float] = []
    hist_coh: List[float] = []
    hist_lr: List[float] = []

    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        tr = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, use_amp
        )
        val_ce = eval_ce_only(model, val_loader, device, use_amp)
        elapsed = time.perf_counter() - t0
        try:
            ppl = math.exp(val_ce)
        except OverflowError:
            ppl = float("inf")

        lr_now = optimizer.param_groups[0]["lr"]
        improved = val_ce < best_val - 1e-6
        if improved:
            best_val = val_ce
            best_epoch = epoch
            no_improve = 0
            best_payload = _build_checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                cfg,
                epoch,
                val_ce,
                tr,
                best_val,
                best_epoch,
            )
            torch.save(best_payload, args.checkpoint_path)
        else:
            no_improve += 1

        ep_display = epoch + 1
        hist_ep.append(ep_display)
        hist_train_ce.append(tr["ce"])
        hist_val_ce.append(val_ce)
        hist_ppl.append(ppl if math.isfinite(ppl) else float("nan"))
        hist_cons.append(tr["consistency"])
        hist_self.append(tr["self_correction"])
        hist_coh.append(tr["coherence"])
        hist_lr.append(lr_now)

        epoch_ckpt = _build_checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            cfg,
            epoch,
            val_ce,
            tr,
            best_val,
            best_epoch,
        )
        save_dir = ckpt_dir if ckpt_dir else "."
        epoch_name = f"t2_v1_epoch_{ep_display:02d}_val{val_ce:.4f}.pt"
        torch.save(epoch_ckpt, os.path.join(save_dir, epoch_name))

        if improved:
            status = "✓ improved"
        else:
            status = f"⚠ no improvement ({no_improve}/3)"

        if best_epoch >= 0:
            best_line = f"  Best:   epoch {best_epoch + 1:02d} val={best_val:.4f}"
        else:
            best_line = "  Best:   (none yet)"

        try:
            from IPython.display import clear_output

            clear_output(wait=True)
        except ImportError:
            pass

        print("--------------------------------------------------")
        print(f"Epoch {ep_display:02d}/{args.epochs}")
        print(
            f"  Train:  total={tr['total']:.4f}  ce={tr['ce']:.4f}  "
            f"consistency={tr['consistency']:.4f}  "
        )
        print(
            f"          self_correction={tr['self_correction']:.4f}  "
            f"coherence={tr['coherence']:.4f}"
        )
        ppl_str = f"{ppl:.2f}" if math.isfinite(ppl) else "inf"
        print(f"  Val:    ce={val_ce:.4f}  perplexity={ppl_str}")
        print(f"  LR:     {lr_now:.6f}")
        print(f"  Time:   {_format_duration(elapsed)}")
        print(best_line)
        print(f"  Status: {status}")
        print("--------------------------------------------------")

        _plot_training_curves(
            hist_ep,
            hist_train_ce,
            hist_val_ce,
            hist_ppl,
            hist_cons,
            hist_self,
            hist_coh,
            hist_lr,
        )

    print(f"Done in {time.perf_counter() - t0_all:.1f}s. Best val CE: {best_val:.4f}")


if __name__ == "__main__":
    main()
