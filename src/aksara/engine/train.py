"""Training / evaluation loop shared by every model in the matrix.

One loop for all architectures is a deliberate constraint: if ResNet and ViT
were trained by different code, an accuracy gap between them would confound
architecture with training-recipe differences.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from . import metrics as metrics_mod


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    label_smoothing: float = 0.0
    # Stops when val macro-F1 hasn't improved for this many epochs. 0 disables.
    early_stopping_patience: int = 8
    mixed_precision: bool = True
    num_workers: int = 2
    grad_clip: float | None = 1.0
    # "auto" preloads when the decoded array fits in preload_max_gb, which is
    # the usual case at 64px (~0.4 GB for the full corpus) and never at 224px
    # (~4.9 GB). "always"/"never" override the decision.
    preload: str = "auto"
    preload_max_gb: float = 2.0


@dataclass
class TrainResult:
    best_val_metrics: dict
    test_metrics: dict
    history: list[dict] = field(default_factory=list)
    epochs_run: int = 0
    train_seconds: float = 0.0
    best_epoch: int = 0
    num_params: int = 0


def _build_optimizer(model: nn.Module, cfg: TrainConfig):
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay, nesterov=True)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}")


def _build_scheduler(optimizer, cfg: TrainConfig, steps_per_epoch: int):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(cfg.epochs // 3, 1), gamma=0.1)
    if cfg.scheduler == "none":
        return None
    raise ValueError(f"Unknown scheduler: {cfg.scheduler!r}")


@torch.no_grad()
def evaluate(model, loader, device, class_names) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_targets = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        all_logits.append(outputs.float().cpu().numpy())
        all_targets.append(targets.numpy())

    logits = np.concatenate(all_logits)
    targets = np.concatenate(all_targets)
    return metrics_mod.compute_metrics(logits, targets, class_names), logits, targets


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_names: list[str],
    cfg: TrainConfig,
    device: torch.device,
    progress: bool = True,
) -> tuple[TrainResult, np.ndarray, np.ndarray]:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg, len(train_loader))

    use_amp = cfg.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    best_val_metrics: dict = {}
    epochs_without_improvement = 0
    history: list[dict] = []
    start = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        running_loss, seen = 0.0, 0

        iterator = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.epochs}", leave=False, disable=not progress)
        for images, targets in iterator:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(images), targets)

            scaler.scale(loss).backward()
            if cfg.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            seen += images.size(0)

        if scheduler is not None:
            scheduler.step()

        val_metrics, _, _ = evaluate(model, val_loader, device, class_names)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": running_loss / max(seen, 1),
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        # Model selection on val macro-F1. The test set is touched exactly once,
        # after this loop, using the selected weights.
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_val_metrics = val_metrics
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if cfg.early_stopping_patience and epochs_without_improvement >= cfg.early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, test_logits, test_targets = evaluate(model, test_loader, device, class_names)

    result = TrainResult(
        best_val_metrics=best_val_metrics,
        test_metrics=test_metrics,
        history=history,
        epochs_run=len(history),
        train_seconds=time.time() - start,
        best_epoch=best_epoch,
        num_params=sum(p.numel() for p in model.parameters()),
    )
    return result, test_logits, test_targets
