"""Swin2SR model."""

import torch
from torch import nn
from transformers import Swin2SRForImageSuperResolution


class Swin2SR(nn.Module):
    """Swin2SR model."""

    def __init__(self) -> None:
        super().__init__()
        self.model = Swin2SRForImageSuperResolution.from_pretrained(
            "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x, return_dict=False)[0].clip(0, 1)
