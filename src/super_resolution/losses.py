"""Super resolution losses."""

import torch
from torch import nn
from torchmetrics.functional.image.lpips import (
    learned_perceptual_image_patch_similarity,
)
from torchvision.transforms.functional import gaussian_blur


class PerceptualLoss(nn.Module):
    """Perceptual loss based on LPIPS."""

    def __init__(self, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize

    def forward(self, preds: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return learned_perceptual_image_patch_similarity(
            preds, images, normalize=self.normalize
        )


class AuxiliaryLoss(nn.Module):
    """Auxiliary loss based on L1 loss between downsampled images."""

    def __init__(self, scale_factor=4) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.loss_fn = nn.L1Loss()

    def forward(self, preds: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        preds_downsampled = torch.nn.functional.interpolate(
            preds, scale_factor=1 / self.scale_factor, mode="bicubic"
        )
        images_downsampled = torch.nn.functional.interpolate(
            images, scale_factor=1 / self.scale_factor, mode="bicubic"
        )
        return self.loss_fn(preds_downsampled, images_downsampled)


class HighFrequencyLoss(nn.Module):
    """High frequency loss based on L1 loss between high frequency components."""

    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, preds: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        preds = preds - (preds * gaussian_blur(preds, kernel_size=[5, 5], sigma=[10.0]))
        images = images - (
            images * gaussian_blur(images, kernel_size=[5, 5], sigma=[10.0])
        )
        return self.loss_fn(preds, images)
