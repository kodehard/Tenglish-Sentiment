"""TenglishDataset — dual-view PyTorch Dataset for Roman + Telugu sentiment data."""

from typing import Optional
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer


LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class TenglishDataset(Dataset):
    """
    Dual-view contrastive dataset: each sample has a Roman-script view
    and a Telugu-script view, both encoded by XLM-RoBERTa.

    Returns a dict with:
        view1_input_ids, view1_attention_mask,
        view2_input_ids, view2_attention_mask,
        label (int tensor)
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer_name: str = "xlm-roberta-base",
        max_seq_len: int = 128,
        label_col: str = "label",
        roman_col: str = "text_roman",
        telugu_col: str = "text_telugu",
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # XLM-RoBERTa has no pad token by default; use eos as pad
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_seq_len = max_seq_len
        self.label_col = label_col
        self.roman_col = roman_col
        self.telugu_col = telugu_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        text_roman = str(row[self.roman_col]) if pd.notna(row[self.roman_col]) else ""
        text_telugu = str(row[self.telugu_col]) if pd.notna(row[self.telugu_col]) else ""
        label_str = str(row[self.label_col])
        label = LABEL2ID.get(label_str, -1)

        # Tokenize Roman view
        view1 = self.tokenizer(
            text_roman,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize Telugu view
        view2 = self.tokenizer(
            text_telugu,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "view1_input_ids": view1["input_ids"].squeeze(0),
            "view1_attention_mask": view1["attention_mask"].squeeze(0),
            "view2_input_ids": view2["input_ids"].squeeze(0),
            "view2_attention_mask": view2["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def create_splits(
    csv_path: str,
    output_dir: str,
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    stratify: bool = True,
    seed: int = 42,
) -> None:
    """
    Split a CSV into train/val/test CSVs and save them.

    Args:
        csv_path: Path to combined CSV with text_roman, text_telugu, label.
        output_dir: Directory to save train.csv, val.csv, test.csv.
        train_split: Fraction for training.
        val_split: Fraction for validation.
        test_split: Fraction for test.
        stratify: Use stratified split by label.
        seed: Random seed.
    """
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path)
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6

    stratify_col = df["label"] if stratify else None

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_split + test_split),
        stratify=stratify_col,
        random_state=seed,
    )
    val_frac = val_split / (val_split + test_split)
    stratify_temp = temp_df["label"] if stratify else None

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_frac),
        stratify=stratify_temp,
        random_state=seed,
    )

    import os
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    print(f"Splits saved to {output_dir}/:")
    print(f"  train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")
