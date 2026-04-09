# Tenglish Sentiment Analysis

**Parameter-Efficient Contrastive Learning for Code-Mixed Telugu-English Sentiment Analysis**
*M.Tech Deep Learning @ IITM*

---

## Overview

This project builds a highly optimized sentiment classification system for code-mixed Telugu-English (Tenglish) text. Instead of a standard classification pipeline, this repository implements a dual-view architecture using:

- **XLM-RoBERTa** (base) as the shared transformer backbone.
- **LoRA** (Low-Rank Adaptation) for parameter-efficient fine-tuning (only ~394k trainable parameters).
- **Supervised Contrastive Learning (SCL)** to mathematically align the embedding spaces of Roman and Telugu scripts.
- **IndicXlit** for programmatic Roman → Telugu transliteration to create the dual-view data.

---

## 🚀 Quick Evaluation (For Graders)

You can verify the model's performance **without retraining** or downloading massive checkpoint files. The project uses a lightweight LoRA parameter extraction strategy (`best_model_lora_only.pt`, ~2MB) which seamlessly maps onto the frozen Hugging Face XLM-RoBERTa base model at runtime.

*Note: The codebase uses absolute path resolution via `pathlib`, meaning you can safely execute these commands from any directory.*

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Evaluate the lightweight checkpoint on the test set
python src/evaluate.py --checkpoint outputs/checkpoints/best_model_lora_only.pt


🛠️ Full Training Pipeline
If you wish to replicate the entire data preparation and training process from scratch:

# 1. Download raw CMTET dataset
bash data/download_data.sh

# 2. Prepare data (Parses raw txt → Transliterates → Stratified Train/Val/Test splits)
python src/prepare_data.py

# 3. Train the model (Supports CUDA, Apple Silicon/MPS, and CPU)
python src/train.py --config configs/config.yaml

# 4. (Optional) Compress checkpoint for submission
python shrink_checkpoint.py


📊 Results
The dual-view SCL approach successfully outperforms baseline fine-tuning. The current metrics on the hold-out test set are:

Model Configuration	Trainable Params	Accuracy	Macro F1
XLM-R (Baseline full fine-tune)	~278M	~0.7200	~0.7000
XLM-R + LoRA + CE	~394k	~0.7400	~0.7200
XLM-R + LoRA + SCL + CE (Proposed)	~394k	0.8138	0.7997


Input: (Roman Tenglish, Telugu Script)
         │                │
    XLM-RoBERTa      XLM-RoBERTa      ← Shared frozen backbone + Trainable LoRA adapters
         │                │
      [CLS] emb        [CLS] emb
         │                │
    Projection Head   Projection Head  (768 → 256, L2-normed)
         │                │
         z1               z2
         └──────┬──────────┘
                │
    SCL Loss + Cross-Entropy Loss
             (λ = 0.5)


tenglish-sentiment/
├── src/
│   ├── transliterate.py    # IndicXlit Roman → Telugu script wrapper
│   ├── dataset.py          # TenglishDataset (dual-view PyTorch loader)
│   ├── model.py            # XLM-R + LoRA + projection heads + classifier
│   ├── losses.py           # Supervised Contrastive Loss (NT-Xent) + Combined Loss
│   ├── train.py            # Main training loop with mixed precision
│   ├── evaluate.py         # Inference, LoRA merging, and metrics
│   └── utils.py            # Checkpointing, scheduling, and class weight helpers
├── configs/config.yaml     # Hyperparameters and path configurations
├── data/
│   ├── raw/                # Raw CMTET dataset
│   └── processed/          # train.csv, val.csv, test.csv
├── outputs/
│   ├── checkpoints/        # best_model_lora_only.pt (Submission weight file)
│   └── results/            # metrics.json, confusion_matrix.png
└── shrink_checkpoint.py    # Utility to strip frozen weights for academic submission


Key References
Conneau et al. (2020) — Unsupervised Cross-lingual Representation Learning at Scale (XLM-RoBERTa)

Hu et al. (2022) — LoRA: Low-Rank Adaptation of Large Language Models

Khosla et al. (2020) — Supervised Contrastive Learning

AI4Bharat IndicXlit — Aksharantar transliteration library

CMTET dataset — https://github.com/ksubbu199/cmtet-sentiment
