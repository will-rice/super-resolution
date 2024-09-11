"""Base image dataset."""

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from super_resolution.datasets import Batch


class BaseDataset(Dataset):
    """Base image dataset."""

    def __init__(
        self,
        image_size: tuple[int, int] = (64, 64),
        scale_factor: int = 4,
    ) -> None:
        super().__init__()
        self.transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.RandomCrop(
                    (image_size[0] * scale_factor, image_size[1] * scale_factor)
                ),
                v2.RandomHorizontalFlip(0.5),
                v2.RandomVerticalFlip(0.5),
                v2.RandomGrayscale(0.1),
            ]
        )
        self.decimate = v2.Compose(
            [
                v2.Resize(image_size),
                v2.RandomChoice([v2.JPEG(quality=(10, 40)), v2.GaussianBlur(3)]),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomApply([v2.GaussianNoise()], p=0.1),
            ]
        )
        self.uint8_to_float32 = v2.ToDtype(torch.float32, scale=True)

    def __len__(self) -> int:
        """Return length of dataset."""
        return 0

    def __getitem__(self, idx: int) -> Batch | None:
        """Return item at index."""
        raise NotImplementedError
