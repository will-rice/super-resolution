"""LAION HD subset dataset for super-resolution."""

from typing import Any

import requests
import torchvision
from datasets import load_dataset
from PIL import Image

from super_resolution.datasets import Batch
from super_resolution.datasets.base import BaseDataset


class LaionHDDataset(BaseDataset):
    """LAION HD dataset for super-resolution."""

    def __init__(
        self, patch_size: tuple[int, int] = (64, 64), scale_factor: int = 4
    ) -> None:
        super().__init__(patch_size=patch_size, scale_factor=scale_factor)
        self.dataset = load_dataset("laion/laion-high-resolution")["train"]
        self.scale_factor = scale_factor

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Batch | None:
        """Return item at index."""
        url = self.dataset[idx]["URL"]

        image = self.download_image(url)
        while image is None:
            idx = (idx + 1) % len(self.dataset)
            url = self.dataset[idx]["URL"]
            image = self.download_image(url)

        height, width = image.size
        pad_height = max(self.patch_size[0] * self.scale_factor - height, 0)
        pad_width = max(self.patch_size[1] * self.scale_factor - width, 0)
        padding = (0, 0, pad_height, pad_width)
        image = torchvision.transforms.functional.pad(
            image, padding, fill=255, padding_mode="constant"
        )
        patch = self.transforms(image)
        low_res = self.decimate(patch.clone())
        patch = self.uint8_to_float32(patch)

        return Batch(source=low_res, target=patch)

    @staticmethod
    def download_image(url: str) -> Image.Image | None:
        """Download image from URL."""
        try:
            response: Any = requests.get(url, stream=True, timeout=1)
            image = Image.open(response.raw).convert("RGBA").convert("RGB")
        except Exception:
            image = None
        return image
