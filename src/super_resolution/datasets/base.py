"""Base image dataset."""

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from super_resolution.datasets import Batch


class BaseDataset(Dataset):
    """Base image dataset."""

    def __init__(
        self, patch_size: tuple[int, int] = (64, 64), scale_factor: int = 4
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.RandomCrop(
                    (patch_size[0] * scale_factor, patch_size[1] * scale_factor)
                ),
                v2.RandomHorizontalFlip(0.5),
                v2.RandomVerticalFlip(0.5),
                v2.RandomGrayscale(0.1),
            ]
        )
        self.decimate = v2.Compose(
            [
                v2.RandomApply([v2.GaussianBlur(3)], p=0.25),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomApply([v2.GaussianNoise()], p=0.25),
                v2.ToDtype(torch.uint8, scale=True),
                v2.RandomApply([v2.JPEG((5, 40))], p=0.9),
                v2.Resize(patch_size, interpolation=v2.InterpolationMode.BICUBIC),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        self.uint8_to_float32 = v2.ToDtype(torch.float32, scale=True)

    def __len__(self) -> int:
        """Return length of dataset."""
        return 0

    def __getitem__(self, idx: int) -> Batch | None:
        """Return item at index."""
        raise NotImplementedError
