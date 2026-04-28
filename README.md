# Tenglish Sentiment Analysis

**Parameter-Efficient Contrastive Learning for Code-Mixed Telugu-English Sentiment Analysis**

---

## Overview

This project builds a highly optimized sentiment classification system for code-mixed Telugu-English (Tenglish) text. Instead of a standard classification pipeline, this repository implements a dual-view architecture using:

- **XLM-RoBERTa** (base) as the shared transformer backbone.
- **LoRA** (Low-Rank Adaptation) for parameter-efficient fine-tuning (only ~394k trainable parameters).
- **Supervised Contrastive Learning (SCL)** to mathematically align the embedding spaces of phonetic Roman and native Telugu scripts.
- **IndicXlit** for programmatic Roman → Telugu transliteration to create the dual-view data.

### ✨ Key Engineering Features
* **Hardware Optimized:** Seamlessly supports CUDA, CPU, and Apple Silicon (`mps`) with PyTorch mixed-precision (`autocast`) for accelerated training.
* **Execution Robustness:** Built using `pathlib` absolute path resolution. Scripts can be executed securely from any directory without throwing path or missing file errors.
* **Lightweight Submission:** The evaluation pipeline maps a tiny 2MB LoRA checkpoint onto the dynamically downloaded HF base model, negating the need to transfer gigabytes of weights.

---

## 🚀 Interactive Inference (New)

You can test the model interactively directly from your terminal. The inference script features color-coded outputs, automatic transliteration, and confidence scoring.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the interactive CLI
python src/inference.py

# Or run batch predictions directly:
python src/inference.py --sentence "movie chala bagundi bro" --sentence "acting assalu nachaledu"
```

## **Quick Evaluation (For Graders)**
```bash
# Evaluate the lightweight checkpoint on the hold-out test set
python src/evaluate.py --checkpoint outputs/checkpoints/best_model_lora_only.pt
**Commands:** `quit`/`exit` to stop, `clear` to clear screen.
```
---

## Training Pipeline

```bash
# 1. Download raw CMTET dataset
bash data/download_data.sh

# 2. Preprocess and create train/val/test splits
python src/prepare_data.py

# 3. Train the model
python src/train.py --config configs/config.yaml

# 4. Extract LoRA weights
python src/shrink_checkpoint.py
```

---

## Architecture

```
Input: (Roman Tenglish, Telugu Script)
         │                │
    XLM-RoBERTa      XLM-RoBERTa      ← Shared frozen backbone + LoRA adapters
         │                │
      [CLS] emb        [CLS] emb
         │                │
    Projection Head   Projection Head  (768 → 256, L2-normed)
         │                │
         z1               z2
         └──────┬──────────┘
                │
    ┌───────────┴───────────┐
    │                       │
SCL Loss              Classifier
(z1 ↔ z2 alignment)    (z_avg → 3 classes)
    │                       │
    └───────────┬───────────┘
                │
         Combined Loss = λ·SCL + (1-λ)·CE
```

**Key Components:**
- **Backbone:** XLM-RoBERTa base (frozen)
- **Adaptation:** LoRA (r=16, α=32) on query, key, value, dense layers
- **Contrastive:** Supervised NT-Xent loss aligns dual-view embeddings
- **Classification:** Averaged embeddings → 3-way sentiment

---

## Project Structure

```
tenglish-sentiment/
├── src/
│   ├── transliterate.py    # IndicXlit Roman → Telugu script wrapper
│   ├── dataset.py          # TenglishDataset (dual-view PyTorch loader)
│   ├── model.py            # XLM-R + LoRA + projection heads + classifier
│   ├── losses.py           # Supervised Contrastive Loss (NT-Xent) + Combined Loss
│   ├── train.py            # Main training loop with mixed precision
│   ├── evaluate.py         # Inference, LoRA merging, and metrics
│   ├── inference.py        # Interactive CLI for custom predictions
│   ├── shrink_checkpoint.py# Utility to strip frozen weights
│   └── utils.py            # Checkpointing, scheduling, and class weight helpers
├── notebooks/              # EDA, Transliteration, Training Curves, and Error Analysis
├── configs/config.yaml     # Hyperparameters and absolute path configurations
├── data/
│   ├── raw/                # Raw CMTET dataset
│   └── processed/          # train.csv, val.csv, test.csv
└── outputs/
    ├── checkpoints/        # best_model_lora_only.pt (Submission weight file)
    └── results/            # metrics.json, confusion_matrix.png
```

---

## Dataset

The **CMTET (Code-Mixed Telugu-English)** dataset contains ~19,869 sentences labeled for sentiment:

| Split | Samples | Percentage |
|-------|---------|------------|
| Train | 13,907  | 70% |
| Val   | 2,980   | 15% |
| Test  | 2,981   | 15% |

**Label Distribution:** Positive, Negative, Neutral (imbalanced)

---

## Configuration

Edit `configs/config.yaml` to modify:

```yaml
model:
  base_model: "xlm-roberta-base"
  projection_dim: 256
  max_seq_len: 128

lora:
  r: 16
  alpha: 32
  dropout: 0.2
  target_modules: ["query", "key", "value", "dense"]

training:
  epochs: 30
  batch_size: 32
  learning_rate: 1.0e-4
  scl_weight: 0.5        # λ: SCL loss weight
  temperature: 0.1       # τ: Contrastive temperature
  early_stopping_patience: 3
```

---

## Results

**Test Set Performance:**

| Metric | Value |
|--------|-------|
| Accuracy | 81.38% |
| Macro F1 | 79.97% |
| Positive F1 | 84.35% |
| Negative F1 | 86.11% |
| Neutral F1 | 69.46% |

---

## References

1. Conneau et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale*. XLM-RoBERTa.
2. Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*.
3. Khosla et al. (2020). *Supervised Contrastive Learning*.
4. AI4Bharat. *IndicXlit Transliteration*.
5. CMTET Dataset: https://github.com/ksubbu199/cmtet-sentiment

---

## License

MIT License
