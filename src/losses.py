"""Supervised Contrastive Loss (SCL) for dual-view Tenglish sentiment model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (NT-Xent) over a batch.

    For each sample, positives are:
      - Both views (Roman + Telugu) of the same sentence (z1, z2)
      - All other samples in the batch that share the same label

    Args:
        temperature: Scaling factor for logits (τ). Default: 0.07.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.tau = temperature

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SCL loss.

        Args:
            z1: Embeddings from view 1, shape (batch_size, projection_dim), L2-normalized.
            z2: Embeddings from view 2, shape (batch_size, projection_dim), L2-normalized.
            labels: Sentiment labels, shape (batch_size,).

        Returns:
            Scalar SCL loss.
        """
        batch_size = z1.size(0)
        device = z1.device

        # Stack both views: shape (2*batch_size, projection_dim)
        embeddings = torch.cat([z1, z2], dim=0)
        # Corresponding labels repeated twice
        all_labels = torch.cat([labels, labels], dim=0)

        # Similarity matrix: (2B, 2B) — all pairwise cosine similarities
        sim_matrix = embeddings @ embeddings.T / self.tau  # already L2-normed

        # Create mask for valid positives (same label, exclude self)
        labels_match = all_labels.unsqueeze(1) == all_labels.unsqueeze(0)  # (2B, 2B)
        # Exclude diagonal (self-self pairs)
        identity = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        labels_match = labels_match & ~identity

        # For each sample, compute denominator: sum of exp(sim) over all positives
        # Numerator: exp(sim) for positive pairs
        # We use the log-sum-exp trick for numerical stability
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max  # subtract max for stability

        # Masked logsumexp: only over positive pairs
        # Set non-positive entries to -inf before exp
        exp_logits = torch.exp(logits)
        exp_logits = exp_logits * labels_match.float()

        # Denominator: sum of exp(sim) over ALL pairs (positives + negatives)
        denom = exp_logits.sum(dim=1)  # (2B,)

        # Numerator: exp(sim) for positive pairs only
        pos_exp = exp_logits  # already masked to positives only
        # For each i, we want sum over j where labels_match[i,j] is True
        numerator = pos_exp.sum(dim=1)  # (2B,)

        # Compute per-sample loss: -log(numerator/denom)
        # Add small epsilon to avoid log(0)
        loss_per_sample = -torch.log(numerator / (denom + 1e-8) + 1e-8)
        loss = loss_per_sample.mean()

        return loss


class CombinedLoss(nn.Module):
    """
    Combined SCL + Cross-Entropy loss.

    Total = lambda * SCL + (1 - lambda) * CE
    """

    def __init__(
        self,
        scl_weight: float = 0.5,
        temperature: float = 0.07,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.scl_weight = scl_weight
        self.ce_weight = 1.0 - scl_weight
        self.scl_loss = SupervisedContrastiveLoss(temperature=temperature)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            z1: View 1 embeddings (B, projection_dim), L2-normalized.
            z2: View 2 embeddings (B, projection_dim), L2-normalized.
            labels: Ground-truth labels (B,).
            logits: Classifier output (B, num_classes).

        Returns:
            Tuple of (total_loss, scl_loss, ce_loss).
        """
        scl = self.scl_loss(z1, z2, labels)
        ce = self.ce_loss(logits, labels)
        total = self.scl_weight * scl + self.ce_weight * ce
        return total, scl, ce
