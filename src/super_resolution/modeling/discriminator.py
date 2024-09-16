"""Discriminators for super-resolution models."""

import functools

import numpy as np
from torch import nn
from torch.nn.utils import spectral_norm


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator, receptive field = 70x70 if n_layers = 3.

    Args:
        input_nc: number of input channels
        ndf: base channel number
        n_layers: number of conv layer with stride 2
        norm_type:  'batch', 'instance', 'spectral', 'batchspectral', instancespectral'

    Returns:
    -------
        tensor: score
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_type="spectral"):
        super().__init__()
        self.n_layers = n_layers
        norm_layer = self.get_norm_layer(norm_type=norm_type)

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [
            [
                self.use_spectral_norm(
                    nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    norm_type,
                ),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        nf = ndf
        for _ in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [
                [
                    self.use_spectral_norm(
                        nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                        norm_type,
                    ),
                    norm_layer(nf),
                    nn.LeakyReLU(0.2, True),
                ]
            ]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [
            [
                self.use_spectral_norm(
                    nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                    norm_type,
                ),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        sequence += [
            [
                self.use_spectral_norm(
                    nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw), norm_type
                )
            ]
        ]

        self.model = nn.Sequential()
        for n in range(len(sequence)):
            self.model.add_module("child" + str(n), nn.Sequential(*sequence[n]))

        self.model.apply(self._init_weights)

    def use_spectral_norm(self, module, norm_type="spectral"):
        """Apply spectral norm to module if specified."""
        if "spectral" in norm_type:
            return spectral_norm(module)
        return module

    def get_norm_layer(self, norm_type="instance"):
        """Return normalization layer."""
        if "batch" in norm_type:
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif "instance" in norm_type:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        else:
            norm_layer = functools.partial(nn.Identity)
        return norm_layer

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)

    def forward(self, x):
        """Forward pass."""
        return self.model(x)
