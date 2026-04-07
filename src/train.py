"""Main training loop for the Tenglish sentiment model."""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from model import TenglishModel
from dataset import TenglishDataset, create_splits
from losses import CombinedLoss
from utils import (
    set_seed,
    setup_logging,
    save_checkpoint,
    load_checkpoint,
    get_linear_schedule_with_warmup,
    compute_class_weights,
    save_metrics,
)
from transliterate import transliterate_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Train Tenglish sentiment model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--data_csv", type=str, default="data/processed/combined.csv")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    logger,
    max_grad_norm: float = 1.0,
    accumulation_steps: int = 2,
) -> dict[str, float]:
    model.train()
    total_loss, total_scl, total_ce = 0.0, 0.0, 0.0
    correct, total = 0, 0

    # Ensure gradients are zeroed before we start the loop
    optimizer.zero_grad()
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    for step, batch in pbar:
        view1_ids = batch["view1_input_ids"].to(device)
        view1_mask = batch["view1_attention_mask"].to(device)
        view2_ids = batch["view2_input_ids"].to(device)
        view2_mask = batch["view2_attention_mask"].to(device)
        labels = batch["label"].to(device)

        # FIXED: device_type changed from hardcoded "cpu" to dynamic device.type
        with autocast(device_type=device.type):
            out = model(
                view1_input_ids=view1_ids,
                view1_attention_mask=view1_mask,
                view2_input_ids=view2_ids,
                view2_attention_mask=view2_mask,
                return_embeddings=True,
            )
            logits = out["logits"]
            z1 = out["z1"]
            z2 = out["z2"]

            loss, scl_loss, ce_loss = criterion(z1, z2, labels, logits)
            
            # Normalize the loss to account for gradient accumulation
            loss = loss / accumulation_steps

        # Accumulate the gradients
        scaler.scale(loss).backward()

        # Only update weights when we reach the accumulation target or the end of the data
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            
            # Reset gradients only after taking a step
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        # Track metrics (multiply loss back by accumulation_steps for accurate reporting)
        total_loss += (loss.item() * accumulation_steps)
        total_scl += scl_loss.item()
        total_ce += ce_loss.item()

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{(loss.item() * accumulation_steps):.4f}",
            "scl": f"{scl_loss.item():.4f}",
            "ce": f"{ce_loss.item():.4f}",
            "acc": f"{correct / total:.3f}",
        })

    n = len(dataloader)
    return {
        "train_loss": total_loss / n,
        "train_scl": total_scl / n,
        "train_ce": total_ce / n,
        "train_acc": correct / total,
    }

@torch.no_grad()
def evaluate(model, dataloader, criterion, device) -> dict[str, float]:
    model.eval()
    total_loss, total_scl, total_ce = 0.0, 0.0, 0.0
    correct, total = 0, 0

    for batch in dataloader:
        view1_ids = batch["view1_input_ids"].to(device)
        view1_mask = batch["view1_attention_mask"].to(device)
        view2_ids = batch["view2_input_ids"].to(device)
        view2_mask = batch["view2_attention_mask"].to(device)
        labels = batch["label"].to(device)

        with autocast(device_type=device.type):
            out = model(
                view1_input_ids=view1_ids,
                view1_attention_mask=view1_mask,
                view2_input_ids=view2_ids,
                view2_attention_mask=view2_mask,
                return_embeddings=True,
            )
            logits = out["logits"]
            z1 = out["z1"]
            z2 = out["z2"]
            loss, scl_loss, ce_loss = criterion(z1, z2, labels, logits)

        total_loss += loss.item()
        total_scl += scl_loss.item()
        total_ce += ce_loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    n = len(dataloader)
    return {
        "val_loss": total_loss / n,
        "val_scl": total_scl / n,
        "val_ce": total_ce / n,
        "val_acc": correct / total,
    }


def main():
    import yaml

    args = parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    paths = cfg["paths"]

    set_seed(train_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = setup_logging(paths["log_dir"])
    logger.info(f"Device: {device}")
    logger.info(f"Config: {cfg}")

    # --- Prepare data ---
    if not os.path.exists(args.data_csv):
        raise FileNotFoundError(
            f"Data CSV not found at {args.data_csv}. "
            "Run data/download_data.sh and transliterate first."
        )

    # Check for processed splits
    train_csv = f"{paths['processed_dir']}/train.csv"
    val_csv = f"{paths['processed_dir']}/val.csv"
    test_csv = f"{paths['processed_dir']}/test.csv"

    if not os.path.exists(train_csv):
        logger.info("Creating train/val/test splits...")
        create_splits(
            csv_path=args.data_csv,
            output_dir=paths["processed_dir"],
            train_split=cfg["data"]["train_split"],
            val_split=cfg["data"]["val_split"],
            test_split=cfg["data"]["test_split"],
            stratify=cfg["data"]["stratify"],
            seed=train_cfg["seed"],
        )

    train_dataset = TenglishDataset(train_csv, max_seq_len=model_cfg["max_seq_len"])
    val_dataset = TenglishDataset(val_csv, max_seq_len=model_cfg["max_seq_len"])

    train_loader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg["batch_size"], shuffle=False)

    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # --- Model ---
    model = TenglishModel(
        base_model_name=model_cfg["base_model"],
        projection_dim=model_cfg["projection_dim"],
        num_classes=model_cfg["num_classes"],
        lora_r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        lora_target_modules=lora_cfg["target_modules"],
    ).to(device)

    logger.info(f"Trainable params: {model.get_trainable_params():,}")

    # --- Loss & Optimizer ---
    all_labels = train_dataset.df["label"].map({"positive": 0, "negative": 1, "neutral": 2}).tolist()
    class_weights = compute_class_weights(all_labels).to(device)

    criterion = CombinedLoss(
        scl_weight=train_cfg["scl_weight"],
        temperature=train_cfg["temperature"],
        class_weights=class_weights,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    total_steps = len(train_loader) * train_cfg["epochs"]
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = GradScaler(device.type)

    # --- Training loop ---
    best_f1 = 0.0
    patience_counter = 0
    start_epoch = 0

    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from {args.resume}")
        ckpt = load_checkpoint(model, optimizer, args.resume, scheduler)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_f1 = ckpt.get("best_metric", 0.0)

    for epoch in range(start_epoch, train_cfg["epochs"]):
        logger.info(f"\n=== Epoch {epoch + 1}/{train_cfg['epochs']} ===")
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, logger,
            max_grad_norm=train_cfg["max_grad_norm"],
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        logger.info(
            f"Train — loss: {train_metrics['train_loss']:.4f} | "
            f"scl: {train_metrics['train_scl']:.4f} | "
            f"ce: {train_metrics['train_ce']:.4f} | "
            f"acc: {train_metrics['train_acc']:.4f}"
        )
        logger.info(
            f"Val   — loss: {val_metrics['val_loss']:.4f} | "
            f"scl: {val_metrics['val_scl']:.4f} | "
            f"ce: {val_metrics['val_ce']:.4f} | "
            f"acc: {val_metrics['val_acc']:.4f}"
        )

        # Simple best-model tracking based on val accuracy (F1 approximation)
        current_f1 = val_metrics["val_acc"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_f1, paths["checkpoint_dir"],
                filename="best_model.pt",
            )
            logger.info(f"New best model saved (acc={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= train_cfg.get("early_stopping_patience", 3):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        save_checkpoint(
            model, optimizer, scheduler, epoch, best_f1, paths["checkpoint_dir"],
            filename=f"epoch_{epoch+1}.pt",
        )

    logger.info(f"Training complete. Best val acc: {best_f1:.4f}")


if __name__ == "__main__":
    main()
 