"""Evaluate the trained Tenglish sentiment model on the test set."""

import argparse
import os
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

from model import TenglishModel
from dataset import TenglishDataset
from utils import load_checkpoint, save_metrics

LABEL_NAMES = ["positive", "negative", "neutral"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent

@torch.no_grad()
def predict(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for batch in dataloader:
        view1_ids = batch["view1_input_ids"].to(device)
        view1_mask = batch["view1_attention_mask"].to(device)
        view2_ids = batch["view2_input_ids"].to(device)
        view2_mask = batch["view2_attention_mask"].to(device)
        labels = batch["label"].cpu().numpy()

        out = model(
            view1_input_ids=view1_ids, view1_attention_mask=view1_mask,
            view2_input_ids=view2_ids, view2_attention_mask=view2_mask,
        )
        logits = out["logits"]

        probs = torch.softmax(logits.float(), dim=1)
        preds = probs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)
        all_probs.append(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.vstack(all_probs)

def plot_confusion_matrix(y_true, y_pred, label_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
    )
    plt.title("Confusion Matrix — Tenglish Sentiment")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Tenglish sentiment model")
    
    # Absolute path handling
    default_ckpt = PROJECT_ROOT / "outputs" / "checkpoints" / "best_model_lora_only.pt"
    default_test = PROJECT_ROOT / "data" / "processed" / "test.csv"
    default_config = PROJECT_ROOT / "configs" / "config.yaml"
    default_out = PROJECT_ROOT / "outputs" / "results"
    
    parser.add_argument("--checkpoint", type=str, default=str(default_ckpt))
    parser.add_argument("--test_csv", type=str, default=str(default_test))
    parser.add_argument("--config", type=str, default=str(default_config))
    parser.add_argument("--output_dir", type=str, default=str(default_out))
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    model = TenglishModel(
        base_model_name=cfg["model"]["base_model"], projection_dim=cfg["model"]["projection_dim"],
        num_classes=cfg["model"]["num_classes"], lora_r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"],
        lora_target_modules=cfg["lora"]["target_modules"],
    ).to(device)

    ckpt = load_checkpoint(model, None, args.checkpoint)
    
    # Better print formatting for evaluation output
    epoch_num = ckpt.get("epoch", "?")
    best_val = ckpt.get("best_metric", "?")
    if isinstance(best_val, float):
        print(f"Loaded checkpoint from epoch {epoch_num}, best metric (Val Acc): {best_val:.4f}")
    else:
        print(f"Loaded checkpoint from epoch {epoch_num}, best metric: {best_val}")

    model.merge_and_unload()

    test_dataset = TenglishDataset(args.test_csv, max_seq_len=cfg["model"]["max_seq_len"])
    test_loader = DataLoader(test_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)

    preds, labels, logits = predict(model, test_loader, device)

    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    per_class_f1 = f1_score(labels, preds, average=None)
    per_class_f1_dict = {name: float(f1) for name, f1 in zip(LABEL_NAMES, per_class_f1)}
    report = classification_report(labels, preds, target_names=LABEL_NAMES, digits=4)

    print("\n" + "=" * 50)
    print("TEST SET RESULTS")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")

    print("\nPer-class F1:")
    for name, f1 in per_class_f1_dict.items():
        print(f"  {name:10s}: {f1:.4f}")

    print("\nFull Classification Report:")
    print(report)

    os.makedirs(args.output_dir, exist_ok=True)
    plot_confusion_matrix(labels, preds, LABEL_NAMES, save_path=os.path.join(args.output_dir, "confusion_matrix.png"))

    metrics = {
        "accuracy": float(accuracy), "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1_dict, "trainable_params": model.get_trainable_params(),
    }
    save_metrics(metrics, os.path.join(args.output_dir, "metrics.json"))

if __name__ == "__main__":
    main()