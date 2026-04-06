"""Prepare the CMTET dataset: download, transliterate, and split into train/val/test."""

import os
import argparse

import pandas as pd

from transliterate import transliterate_batch


LABEL_COLORS = {"positive": 0, "negative": 1, "neutral": 2}


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw CMTET file.

    Format: each entry spans 2 lines
      Line 1: "POS: <text>" | "NEG: <text>" | "NTL: <text>"
      Line 2: language codes (en/te/univ/ne etc.)
      Line 3: blank separator

    Labels map: POS → positive, NEG → negative, NTL → neutral
    """
    label_map = {"POS": "positive", "NEG": "negative", "NTL": "neutral"}
    records = []
    with open(path, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f.readlines()]

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue  # skip blank separator lines
        # Format: "POS: text" / "NEG: text" / "NTL: text"
        if ": " not in line:
            i += 1  # skip malformed line + its lang-code line
            continue
        label_str, text = line.split(": ", 1)
        if label_str not in label_map:
            i += 1  # skip lang-code line
            continue
        records.append({"text_roman": text.strip(), "label": label_map[label_str]})
        i += 1  # skip the language-code line (line 2 of entry)

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Prepare Tenglish dataset")
    parser.add_argument("--raw_data", default="data/raw/codemix_sentiment_data.txt")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--transliterate", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load raw data
    print(f"Loading raw data from {args.raw_data}...")
    df = load_raw_data(args.raw_data)
    print(f"Loaded {len(df)} sentences")
    print(df["label"].value_counts())

    # Transliterate
    combined_csv = os.path.join(args.output_dir, "combined.csv")
    if args.transliterate and not os.path.exists(combined_csv):
        print("Transliterating Roman → Telugu script...")
        telugu_texts = transliterate_batch(
            df["text_roman"].tolist(),
            cache_path=os.path.join(args.output_dir, "_transliteration_cache.csv"),
        )
        df["text_telugu"] = telugu_texts
    else:
        df["text_telugu"] = ""

    df.to_csv(combined_csv, index=False)
    print(f"Saved combined CSV: {combined_csv}")

    # Create splits
    from dataset import create_splits
    create_splits(combined_csv, args.output_dir, seed=42)
    print("Data preparation complete!")


if __name__ == "__main__":
    main()
