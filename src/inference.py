"""Interactive inference script for Tenglish Sentiment Analysis.

Allows users to input custom Telugu-English (Tenglish) sentences
and get sentiment predictions (positive/negative/neutral).
"""

import argparse
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

from model import TenglishModel
from transliterate import transliterate_batch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABEL_NAMES = ["positive", "negative", "neutral"]

# ANSI color codes for pretty terminal output
COLORS = {
    "positive": "\033[92m",   # Green
    "negative": "\033[91m",   # Red
    "neutral": "\033[93m",    # Yellow
    "reset": "\033[0m",
    "bold": "\033[1m",
}


def load_model(checkpoint_path: str, config_path: str, device: torch.device) -> TenglishModel:
    """Load the trained model from checkpoint."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model = TenglishModel(
        base_model_name=cfg["model"]["base_model"],
        projection_dim=cfg["model"]["projection_dim"],
        num_classes=cfg["model"]["num_classes"],
        lora_r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        lora_target_modules=cfg["lora"]["target_modules"],
    ).to(device)

    # Load checkpoint (strict=False because base model weights are loaded from pretrained)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.merge_and_unload()
    model.eval()

    return model


def predict_sentiment(
    model: TenglishModel,
    tokenizer: AutoTokenizer,
    text_roman: str,
    device: torch.device,
    max_seq_len: int = 128,
) -> dict:
    """Predict sentiment for a single Tenglish sentence.

    Args:
        model: Trained TenglishModel
        tokenizer: XLM-RoBERTa tokenizer
        text_roman: Input sentence in Roman-script Tenglish
        device: Torch device (cuda/mps/cpu)
        max_seq_len: Maximum sequence length

    Returns:
        Dict with predicted_label, confidence, and all_probabilities
    """
    # Transliterate to Telugu script
    text_telugu = transliterate_batch([text_roman])[0]

    # Tokenize both views
    view1 = tokenizer(
        text_roman,
        max_length=max_seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    view2 = tokenizer(
        text_telugu,
        max_length=max_seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Move to device
    view1_ids = view1["input_ids"].to(device)
    view1_mask = view1["attention_mask"].to(device)
    view2_ids = view2["input_ids"].to(device)
    view2_mask = view2["attention_mask"].to(device)

    # Forward pass
    with torch.no_grad():
        out = model(
            view1_input_ids=view1_ids,
            view1_attention_mask=view1_mask,
            view2_input_ids=view2_ids,
            view2_attention_mask=view2_mask,
        )
        logits = out["logits"]
        probs = torch.softmax(logits.float(), dim=1).squeeze(0).cpu().numpy()

    predicted_idx = probs.argmax()
    predicted_label = LABEL_NAMES[predicted_idx]
    confidence = probs[predicted_idx]

    return {
        "input_roman": text_roman,
        "input_telugu": text_telugu,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "all_probabilities": dict(zip(LABEL_NAMES, probs)),
    }


def format_result(result: dict) -> str:
    """Format prediction result as colored string."""
    label = result["predicted_label"]
    conf = result["confidence"]
    color = COLORS.get(label, "")
    reset = COLORS["reset"]
    bold = COLORS["bold"]

    lines = [
        f"\n{bold}Input (Roman):{reset} {result['input_roman']}",
        f"{bold}Input (Telugu):{reset} {result['input_telugu']}",
        f"\n{bold}Prediction:{reset} {color}{label.upper()}{reset}",
        f"{bold}Confidence:{reset} {conf:.2%}",
        f"\n{bold}All Probabilities:{reset}",
    ]

    for lbl, prob in result["all_probabilities"].items():
        lbl_color = COLORS.get(lbl, "")
        lines.append(f"  {lbl_color}{lbl:10s}{reset}: {prob:6.2%}")

    lines.append("")
    return "\n".join(lines)


def interactive_mode(model: TenglishModel, tokenizer: AutoTokenizer, device: torch.device, max_seq_len: int):
    """Run interactive CLI for sentiment prediction."""
    print("\n" + "=" * 60)
    print(f"{COLORS['bold']}Tenglish Sentiment Analysis - Interactive Mode{COLORS['reset']}")
    print("=" * 60)
    print("\nEnter Telugu-English (Tenglish) sentences to analyze sentiment.")
    print("Commands:")
    print("  - Type 'quit' or 'exit' to stop")
    print("  - Type 'clear' to clear screen")
    print("\nExamples:")
    print("  - 'movie chala bagundi bro'")
    print("  - 'acting assalu nachaledu'")
    print("  - 'food taste ga undi'")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n> Enter sentence: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                print("\nGoodbye!\n")
                break

            if user_input.lower() == "clear":
                print("\033[2J\033[H", end="")
                continue

            result = predict_sentiment(model, tokenizer, user_input, device, max_seq_len)
            print(format_result(result))

        except KeyboardInterrupt:
            print("\n\nGoodbye!\n")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def batch_mode(
    model: TenglishModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_seq_len: int,
    sentences: list[str],
):
    """Run predictions on a list of sentences."""
    print("\n" + "=" * 60)
    print(f"{COLORS['bold']}Tenglish Sentiment Analysis - Batch Mode{COLORS['reset']}")
    print("=" * 60)

    for sentence in sentences:
        result = predict_sentiment(model, tokenizer, sentence, device, max_seq_len)
        print(format_result(result))


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Tenglish Sentiment Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python src/inference.py

  # Interactive mode with custom checkpoint
  python src/inference.py --checkpoint outputs/checkpoints/best_model_lora_only.pt

  # Batch mode (predict on multiple sentences)
  python src/inference.py --sentence "movie bagundi" --sentence "acting nachaledu"
        """,
    )

    default_ckpt = PROJECT_ROOT / "outputs" / "checkpoints" / "best_model_lora_only.pt"
    default_config = PROJECT_ROOT / "configs" / "config.yaml"

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(default_ckpt),
        help=f"Path to model checkpoint (default: {default_ckpt})",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help=f"Path to config file (default: {default_config})",
    )
    parser.add_argument(
        "--sentence",
        "-s",
        type=str,
        action="append",
        help="Sentence to predict (can be repeated for batch mode). If not provided, runs interactive mode.",
    )

    args = parser.parse_args()

    # Validate checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please train the model first or provide a valid checkpoint path.")
        return

    # Set up device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Loading model on {device}...")

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load model and tokenizer
    model = load_model(args.checkpoint, args.config, device)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded successfully!\n")

    # Run in appropriate mode
    if args.sentence:
        batch_mode(model, tokenizer, device, cfg["model"]["max_seq_len"], args.sentence)
    else:
        interactive_mode(model, tokenizer, device, cfg["model"]["max_seq_len"])


if __name__ == "__main__":
    main()
