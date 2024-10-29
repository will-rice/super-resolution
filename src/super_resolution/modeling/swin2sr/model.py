"""Swin2SR model."""

import torch
from torch import nn
from transformers import AutoConfig, Swin2SRForImageSuperResolution


class Swin2SR(nn.Module):
    """Swin2SR model."""

    def __init__(self, gradient_checkpointing: bool = True) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
            attention_probs_dropout_prob=0.1,
            hidden_dropout_prob=0.1,
        )
        self.model = Swin2SRForImageSuperResolution.from_pretrained(
            "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"
        )
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x, return_dict=False)[0]
