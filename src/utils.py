"""Utility functions: seeding, checkpointing, scheduling, logging."""

import random
import os
import json
import logging
from typing import Optional
from pathlib import Path

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_dir: str, name: str = "train") -> logging.Logger:
    """Configure logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[LambdaLR],
    epoch: int,
    best_metric: float,
    checkpoint_dir: str,
    filename: str = "checkpoint.pt",
) -> None:
    """Save a training checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[Optimizer],
    checkpoint_path: str,
    scheduler: Optional[LambdaLR] = None,
) -> dict:
    """Load a training checkpoint. Returns checkpoint metadata."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """Linear learning rate schedule with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


def save_metrics(metrics: dict, output_path: str) -> None:
    """Save evaluation metrics to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {output_path}")


def compute_class_weights(labels: list[int], num_classes: int = 3) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for c in range(num_classes):
        count = counts.get(c, 1)
        weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)
