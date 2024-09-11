"""Module for dataset classes."""

from typing import NamedTuple

import torch


class Batch(NamedTuple):
    """Batch of inputs."""

    source: torch.Tensor
    target: torch.Tensor
