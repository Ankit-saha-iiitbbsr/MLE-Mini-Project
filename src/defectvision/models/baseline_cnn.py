"""Baseline CNN trained from scratch -- the control arm of the M3 comparison.

The architecture is small and deliberate rather than a stack of arbitrary
layers. Two choices carry the design:

**Strided stem.** A 5x5 stride-2 convolution at the input quarters the spatial
work of every later layer. On CPU (the target environment for this project)
that is roughly a 3x wall-clock saving, and at 128x128 the defects of interest
are still 5-12 px after the stride.

**Average + max pooling head.** This is the choice that makes the model work at
all. A defect occupies well under 1% of the frame, so a global *average* pool
divides that evidence across the whole feature map and the signal is swamped by
the casting body. Max pooling asks the question the task actually poses --
"is there a defect-like response *anywhere*?" -- and concatenating both keeps
the average branch's information about overall surface texture. Measured on the
validation split, swapping GAP for avg+max moved the model off chance
(~0.50 accuracy) to a learning trajectory; it is not a cosmetic detail.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class AvgMaxPool(nn.Module):
    """Concatenate global average and global max pooling over the spatial dims.

    Output width is ``2 * C``: the average branch answers "how much defect-like
    texture overall", the max branch answers "how strong is the single most
    defect-like location".
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1).flatten(1)
        mx = F.adaptive_max_pool2d(x, 1).flatten(1)
        return torch.cat([avg, mx], dim=1)


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU -> MaxPool.

    BatchNorm before the activation lets the block tolerate the wide brightness
    and contrast range the augmentation pipeline introduces without the loss
    scale wandering.
    """

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.norm(self.conv(x))))


class BaselineCNN(nn.Module):
    """Compact defect classifier trained from scratch.

    Parameters
    ----------
    in_channels:
        1 for the grayscale inspection images.
    channels:
        Width of each convolutional stage after the stem.
    dropout:
        Applied to the pooled feature vector before the classifier.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, ...] | list[int] = (32, 64, 128),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        channels = tuple(int(c) for c in channels)
        if not channels:
            raise ValueError("`channels` must list at least one stage width")

        stem_width = channels[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_width, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(inplace=True),
        )

        blocks: list[nn.Module] = []
        prev = stem_width
        for width in channels:
            blocks.append(ConvBlock(prev, width))
            prev = width
        # One extra stage at constant width deepens the receptive field without
        # widening the head.
        blocks.append(ConvBlock(prev, prev))
        self.features = nn.Sequential(*blocks)

        self.pool = AvgMaxPool()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(prev * 2, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """He initialisation for ReLU stacks; zero-init the final bias."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(batch,)``."""
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x).squeeze(-1)
