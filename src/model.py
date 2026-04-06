"""XLM-RoBERTa + LoRA + Projection Head + Classifier for Tenglish Sentiment."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoConfig


class TenglishModel(nn.Module):
    """
    Dual-view model: shared XLM-RoBERTa backbone with LoRA adapters,
    separate projection heads per view, and a shared classifier head.

    Architecture:
      Input (Roman or Telugu) → XLM-RoBERTa + LoRA → [CLS] token
                                                        ↓
                                              Projection Head (768 → 256)
                                                        ↓
                                                   z1 or z2 (L2-normed)
                                                        ↓
                                          ┌───────────┴───────────┐
                                          │  SCL Loss (z1, z2)    │
                                          │  Classifier (z_avg)   │
                                          └───────────────────────┘
    """

    def __init__(
        self,
        base_model_name: str = "xlm-roberta-base",
        projection_dim: int = 256,
        num_classes: int = 3,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: list[str] | None = None,
    ) -> None:
        super().__init__()

        if lora_target_modules is None:
            lora_target_modules = ["query", "value"]

        # --- Backbone ---
        self.backbone = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.backbone.config.hidden_size  # 768 for xlm-roberta-base

        # LoRA config and apply
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            task_type=None,  # FEATURE_EXTRACTION equivalent
        )
        self.backbone = get_peft_model(self.backbone, lora_config)

        # --- Projection heads (one per view, tied is also OK but separate is fine) ---
        self.projection1 = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.ReLU(),
            nn.Dropout(lora_dropout),
        )
        self.projection2 = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.ReLU(),
            nn.Dropout(lora_dropout),
        )

        # --- Classifier head ---
        self.classifier = nn.Linear(projection_dim, num_classes)

        self.projection_dim = projection_dim

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract [CLS] embedding from the backbone."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # XLM-RoBERTa uses pooler_output (first token, run through linear)
        # Fall back to [CLS] (last_hidden_state[:, 0])
        cls_emb = out.last_hidden_state[:, 0]
        return cls_emb

    def project(self, cls_emb: torch.Tensor, head: nn.Module) -> torch.Tensor:
        """Project to projection_dim and L2-normalize."""
        z = head(cls_emb)
        z = F.normalize(z, p=2, dim=1)
        return z

    def forward(
        self,
        view1_input_ids: torch.Tensor,
        view1_attention_mask: torch.Tensor,
        view2_input_ids: torch.Tensor,
        view2_attention_mask: torch.Tensor,
        return_embeddings: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with both views.

        Args:
            view1_input_ids: Roman view token IDs.
            view1_attention_mask: Roman view attention mask.
            view2_input_ids: Telugu view token IDs.
            view2_attention_mask: Telugu view attention mask.
            return_embeddings: If True, also return projected embeddings for SCL.

        Returns:
            Dict with:
                - logits: classifier output (B, num_classes)
                - z1: projected view1 (B, projection_dim), L2-normed
                - z2: projected view2 (B, projection_dim), L2-normed
                - z_avg: mean of z1 and z2 (B, projection_dim)
        """
        # Encode both views
        cls1 = self.encode(view1_input_ids, view1_attention_mask)
        cls2 = self.encode(view2_input_ids, view2_attention_mask)

        # Project both views
        z1 = self.project(cls1, self.projection1)
        z2 = self.project(cls2, self.projection2)

        # Average embeddings for classification
        z_avg = (z1 + z2) / 2.0

        logits = self.classifier(z_avg)

        if return_embeddings:
            return {"logits": logits, "z1": z1, "z2": z2, "z_avg": z_avg}
        return {"logits": logits}

    def get_trainable_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def merge_and_unload(self) -> None:
        """Merge LoRA adapters into the backbone for inference."""
        self.backbone.merge_and_unload()
