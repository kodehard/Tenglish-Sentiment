# Tenglish Sentiment Analysis

**Parameter-Efficient Contrastive Learning for Code-Mixed Telugu-English Sentiment Classification**

*M.Tech Deep Learning Project @ IITM*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

A dual-view sentiment classification system for code-mixed Telugu-English (Tenglish) text using **Supervised Contrastive Learning (SCL)** and **LoRA** parameter-efficient fine-tuning.

| Model | Trainable Params | Accuracy | Macro F1 |
|-------|-----------------|----------|----------|
| XLM-R Full Fine-tune (Baseline) | ~278M | ~72.0% | ~70.0% |
| XLM-R + LoRA + CE | ~394K | ~74.0% | ~72.0% |
| **XLM-R + LoRA + SCL + CE (Ours)** | **~394K** | **81.38%** | **79.97%** |

---

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Evaluate Pretrained Model

```bash
python src/evaluate.py
```

### Interactive Inference

Test custom Tenglish sentences:

```bash
# Interactive mode
python src/inference.py

# Batch mode
python src/inference.py -s "movie chala bagundi" -s "acting nachaledu"
```

**Commands:** `quit`/`exit` to stop, `clear` to clear screen.

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
│   ├── inference.py        # Interactive sentiment prediction
│   ├── evaluate.py         # Test set evaluation
│   ├── train.py            # Training loop with mixed precision
│   ├── model.py            # XLM-R + LoRA + projection heads
│   ├── losses.py           # SCL + CrossEntropy loss
│   ├── dataset.py          # Dual-view PyTorch Dataset
│   ├── transliterate.py    # Roman → Telugu transliteration
│   ├── prepare_data.py     # Data preprocessing pipeline
│   └── utils.py            # Checkpointing, scheduling, logging
├── configs/
│   └── config.yaml         # Hyperparameters and paths
├── data/
│   ├── raw/                # Raw CMTET dataset
│   └── processed/          # Train/val/test CSV splits
├── outputs/
│   ├── checkpoints/        # Model weights
│   ├── logs/               # Training logs
│   └── results/            # Metrics and visualizations
├── notebooks/              # EDA, training curves, error analysis
└── requirements.txt        # Python dependencies
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
