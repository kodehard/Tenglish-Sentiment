"""Roman-script Tenglish → Telugu script transliteration using indic_transliteration (sanscript)."""

import os
from functools import lru_cache
from typing import Optional

from indic_transliteration.sanscript import transliterate as _transliterate

import pandas as pd


# Roman → Telugu: itrans_dravidian (Roman) → telugu (Telugu script)
_ROMAN_SCHEME = "itrans_dravidian"
_TELUGU_SCHEME = "telugu"


@lru_cache(maxsize=1)
def _get_scheme_map():
    """Return a pre-computed scheme map for fast repeated transliteration."""
    from indic_transliteration.sanscript import SchemeMap, SCHEMES
    return SchemeMap(SCHEMES[_ROMAN_SCHEME], SCHEMES[_TELUGU_SCHEME])


def transliterate_batch(texts: list[str], cache_path: Optional[str] = None) -> list[str]:
    """
    Transliterate a batch of Roman-script Tenglish sentences to Telugu script.

    Uses indic_transliteration.sanscript with itrans_dravidian → telugu scheme.
    Falls back to the original token on any error.

    Args:
        texts: List of Roman-script Tenglish sentences.
        cache_path: Optional path to cache file for results.

    Returns:
        List of Telugu-script sentences (same order as input).
    """
    if cache_path and os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        if len(cached) == len(texts):
            return cached["text_telugu"].tolist()

    try:
        scheme_map = _get_scheme_map()
    except Exception:
        return texts

    results = []
    for text in texts:
        if not text or not isinstance(text, str):
            results.append("")
            continue

        tokens = text.split()
        transliterated_tokens = []
        for token in tokens:
            try:
                tele = _transliterate(token, scheme_map=scheme_map)
                # If transliteration produced Telugu characters, use it
                # Otherwise fall back to original (handles punctuation, numbers)
                if any("\u0C00" <= c <= "\u0C7F" for c in tele):
                    transliterated_tokens.append(tele)
                else:
                    transliterated_tokens.append(token)
            except Exception:
                transliterated_tokens.append(token)

        results.append(" ".join(transliterated_tokens))

    if cache_path:
        pd.DataFrame({"text_roman": texts, "text_telugu": results}).to_csv(
            cache_path, index=False
        )

    return results


def transliterate_csv(
    input_csv: str,
    output_csv: str,
    roman_col: str = "text_roman",
    telugu_col: str = "text_telugu",
) -> None:
    """
    Read a CSV, transliterate the Roman column, save to a new CSV.

    Args:
        input_csv: Path to input CSV with Roman-script text.
        output_csv: Path to output CSV with added Telugu column.
        roman_col: Name of the column containing Roman-script text.
        telugu_col: Name of the new column to create with Telugu text.
    """
    df = pd.read_csv(input_csv)
    cache_path = output_csv.replace(".csv", "_transliteration_cache.csv")

    texts = df[roman_col].fillna("").tolist()
    telugu_texts = transliterate_batch(texts, cache_path=cache_path)

    df[telugu_col] = telugu_texts
    df.to_csv(output_csv, index=False)
    print(f"Saved transliterated CSV to {output_csv}")


if __name__ == "__main__":
    samples = [
        "movie chala bagundi bro",
        "acting assalu nachaledu",
        "story ok ok undi",
        "I love this movie",
        "2024 elections",
    ]
    for roman, telugu in zip(samples, transliterate_batch(samples)):
        print(f"Roman: {roman}")
        print(f"Telugu: {telugu}")
        print()
