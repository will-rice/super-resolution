"""General image dataset."""

from pathlib import Path

from PIL import Image, PngImagePlugin

from super_resolution.datasets import Batch
from super_resolution.datasets.base import BaseDataset

LARGE_ENOUGH_NUMBER = 1000
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)
Image.MAX_IMAGE_PIXELS = 102658048


class ImageDataset(BaseDataset):
    """General image dataset."""

    def __init__(
        self, root: Path, patch_size: tuple[int, int] = (64, 64), scale_factor: int = 4
    ) -> None:
        super().__init__(patch_size=patch_size, scale_factor=scale_factor)
        self.paths = list(root.glob("**/*.png"))

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.paths)

    def __getitem__(self, idx: int) -> Batch:
        """Return item at index."""
        path = self.paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            patch = self.transforms(image)
        except Exception:
            return self.__getitem__((idx + 1) % len(self.paths))
        low_res = self.decimate(patch.clone())
        patch = self.uint8_to_float32(patch).clip(0.0, 1.0)
        return Batch(source=low_res, target=patch)
