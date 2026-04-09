# shrink_checkpoint.py
import torch
import yaml
from model import TenglishModel

def main():
    print("Loading config...")
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    print("Initializing model architecture...")
    model = TenglishModel(
        base_model_name=cfg["model"]["base_model"],
        projection_dim=cfg["model"]["projection_dim"],
        num_classes=cfg["model"]["num_classes"],
        lora_r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        lora_target_modules=cfg["lora"]["target_modules"],
    )

    print("Loading massive checkpoint (this might take a moment)...")
    original_ckpt_path = "outputs/checkpoints/best_model.pt"
    checkpoint = torch.load(original_ckpt_path, map_location="cpu")
    
    # Load all weights into the model
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Extracting only the trainable LoRA and Head parameters...")
    # This is the magic step: filter for requires_grad == True
    trainable_state_dict = {
        name: param for name, param in model.named_parameters() if param.requires_grad
    }

    # Create the new, lightweight dictionary
    small_checkpoint = {
        "epoch": checkpoint.get("epoch", 0),
        "best_metric": checkpoint.get("best_metric", 0.0),
        "model_state_dict": trainable_state_dict
    }

    new_ckpt_path = "outputs/checkpoints/best_model_lora_only.pt"
    torch.save(small_checkpoint, new_ckpt_path)
    
    print(f"Success! Shrunk checkpoint saved to {new_ckpt_path}")
    print(f"Original keys: {len(checkpoint['model_state_dict'])}")
    print(f"New keys: {len(trainable_state_dict)}")

if __name__ == "__main__":
    main()