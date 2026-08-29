"""Transfer-learning arms of the M3 comparison (ResNet-18, MobileNetV3-Small).

Two engineering decisions worth stating explicitly, because both are places
where a naive port of an ImageNet recipe would cost accuracy or throughput:

**Grayscale stem adaptation.** The obvious way to feed a 1-channel image to an
ImageNet backbone is to repeat the channel three times. That works, but it
triples the stem's compute for no new information. Instead the first
convolution is rebuilt with ``in_channels=1`` and its pretrained weights are
*summed* across the RGB axis. For a grayscale input this reproduces the exact
response the original filter would have given on a grey RGB image, so no
pretrained signal is lost -- the filters keep their learned edge and blob
selectivity from epoch zero.

**Head replacement with avg+max pooling.** The stock backbones end in global
average pooling, which dilutes the small, localised evidence this task depends
on (see :mod:`defectvision.models.baseline_cnn`). Both backbones therefore get
the same ``AvgMaxPool`` head as the baseline, which keeps the comparison a
comparison *of backbones* rather than of pooling strategies.

Pretrained weight download is best-effort: if there is no network the model
falls back to random initialisation and says so loudly, so an offline run still
completes and the log makes clear that the "transfer" arm was not actually
transferring.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..logging_utils import get_logger
from .baseline_cnn import AvgMaxPool

log = get_logger(__name__)

SUPPORTED_BACKBONES = ("resnet18", "resnet34", "mobilenet_v3_small")


def _load_backbone(arch: str, pretrained: bool) -> tuple[nn.Module, bool]:
    """Instantiate a torchvision backbone. Returns ``(model, weights_loaded)``."""
    from torchvision import models as tv

    specs: dict[str, tuple[Any, Any]] = {
        "resnet18": (tv.resnet18, tv.ResNet18_Weights.IMAGENET1K_V1),
        "resnet34": (tv.resnet34, tv.ResNet34_Weights.IMAGENET1K_V1),
        "mobilenet_v3_small": (
            tv.mobilenet_v3_small,
            tv.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        ),
    }
    if arch not in specs:
        raise ValueError(f"Unsupported backbone {arch!r}; expected one of {SUPPORTED_BACKBONES}")

    ctor, weights_enum = specs[arch]
    if not pretrained:
        return ctor(weights=None), False

    try:
        model = ctor(weights=weights_enum)
        log.info("Loaded pretrained ImageNet weights for %s", arch)
        return model, True
    except Exception as exc:  # network failure, cache miss, checksum error
        log.warning(
            "Could not download pretrained weights for %s (%s: %s). "
            "Falling back to RANDOM initialisation -- this run is not a true "
            "transfer-learning result.",
            arch, type(exc).__name__, exc,
        )
        return ctor(weights=None), False


def _adapt_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Rebuild *conv* for ``in_channels`` inputs, folding pretrained RGB weights.

    Summing across the input-channel axis is the correct reduction: a grayscale
    pixel replicated to (r, g, b) would produce ``w_r*v + w_g*v + w_b*v``, which
    is exactly ``(w_r + w_g + w_b) * v``. The adapted filter is therefore
    response-identical to the channel-repeat approach at a third of the cost.
    """
    if conv.in_channels == in_channels:
        return conv

    adapted = nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,  # type: ignore[arg-type]
        stride=conv.stride,            # type: ignore[arg-type]
        padding=conv.padding,          # type: ignore[arg-type]
        dilation=conv.dilation,        # type: ignore[arg-type]
        groups=conv.groups,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        w = conv.weight  # (out, 3, kh, kw)
        if in_channels == 1:
            adapted.weight.copy_(w.sum(dim=1, keepdim=True))
        else:
            # Tile/trim to the requested width, rescaling to preserve gain.
            reps = (in_channels + w.shape[1] - 1) // w.shape[1]
            tiled = w.repeat(1, reps, 1, 1)[:, :in_channels]
            adapted.weight.copy_(tiled * (w.shape[1] / in_channels))
        if conv.bias is not None:
            adapted.bias.copy_(conv.bias)
    return adapted


class TransferClassifier(nn.Module):
    """ImageNet backbone + avg/max pooling head, emitting one logit per image."""

    def __init__(
        self,
        arch: str = "resnet18",
        in_channels: int = 1,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        backbone, self.pretrained_loaded = _load_backbone(arch, pretrained)
        self.arch = arch

        if arch.startswith("resnet"):
            backbone.conv1 = _adapt_first_conv(backbone.conv1, in_channels)
            feature_dim = backbone.fc.in_features
            # Strip the stock pooling + fc; keep everything up to layer4.
            backbone.avgpool = nn.Identity()
            backbone.fc = nn.Identity()
            self.features = nn.Sequential(
                backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
                backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            )
        elif arch == "mobilenet_v3_small":
            first_conv = backbone.features[0][0]
            backbone.features[0][0] = _adapt_first_conv(first_conv, in_channels)
            feature_dim = backbone.classifier[0].in_features
            self.features = backbone.features
        else:  # pragma: no cover - guarded by _load_backbone
            raise ValueError(f"Unsupported backbone {arch!r}")

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False
            # BatchNorm running statistics would otherwise keep updating on the
            # new domain even with frozen weights, which silently changes a
            # "frozen" backbone. Put them in eval mode and keep them there.
            self.features.eval()
        self.freeze_backbone = freeze_backbone

        self.pool = AvgMaxPool()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, 1),
        )
        nn.init.normal_(self.classifier[1].weight, std=0.01)
        nn.init.zeros_(self.classifier[1].bias)

    def train(self, mode: bool = True):  # noqa: D102 - see freeze note above
        super().train(mode)
        if self.freeze_backbone:
            self.features.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(batch,)``."""
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x).squeeze(-1)


def build_transfer_model(
    arch: str = "resnet18",
    *,
    in_channels: int = 1,
    pretrained: bool = True,
    dropout: float = 0.2,
    freeze_backbone: bool = False,
) -> TransferClassifier:
    """Construct a :class:`TransferClassifier`."""
    return TransferClassifier(
        arch=arch,
        in_channels=in_channels,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
