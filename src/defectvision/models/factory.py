"""One entry point for turning a config block into a model.

Keeping construction behind a factory means a run is described entirely by the
``train.models.<name>`` block in ``params.yaml``. Nothing else has to change to
add an experiment arm, and the same dict that built the model is logged to
MLflow -- which is what lets a run be rebuilt from its logged configuration.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from ..logging_utils import get_logger
from .baseline_cnn import BaselineCNN
from .transfer import SUPPORTED_BACKBONES, build_transfer_model

log = get_logger(__name__)

#: Architectures the factory can build. ``logreg_hog`` is handled separately by
#: :mod:`defectvision.models.classical` because it is not a torch module.
ARCHITECTURES: tuple[str, ...] = ("baseline_cnn", *SUPPORTED_BACKBONES, "logreg_hog")


def build_model(cfg: dict[str, Any], in_channels: int = 1) -> nn.Module:
    """Build a torch model from a ``train.models.<name>`` config block.

    The block must carry an ``arch`` key; remaining keys are architecture
    hyperparameters. Keys that belong to the *training loop* rather than the
    model (``learning_rate``, ``epochs``, ...) are ignored here.
    """
    arch = str(cfg.get("arch", "baseline_cnn"))

    if arch == "baseline_cnn":
        return BaselineCNN(
            in_channels=in_channels,
            channels=cfg.get("channels", (32, 64, 128)),
            dropout=float(cfg.get("dropout", 0.3)),
        )

    if arch in SUPPORTED_BACKBONES:
        return build_transfer_model(
            arch,
            in_channels=in_channels,
            pretrained=bool(cfg.get("pretrained", True)),
            dropout=float(cfg.get("dropout", 0.2)),
            freeze_backbone=bool(cfg.get("freeze_backbone", False)),
        )

    if arch == "logreg_hog":
        raise ValueError(
            "logreg_hog is a scikit-learn pipeline, not a torch module. "
            "Use defectvision.models.classical.build_classical_model()."
        )

    raise ValueError(f"Unknown architecture {arch!r}; expected one of {ARCHITECTURES}")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(total_parameters, trainable_parameters)``."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def describe_model(model: nn.Module) -> dict[str, Any]:
    """Summary logged to MLflow alongside the metrics."""
    total, trainable = count_parameters(model)
    return {
        "model_class": type(model).__name__,
        "arch": getattr(model, "arch", type(model).__name__),
        "total_params": total,
        "trainable_params": trainable,
        # float32 parameter footprint; a first-order proxy for the memory the
        # serving container will need.
        "param_size_mb": round(total * 4 / (1024 ** 2), 3),
        "pretrained_loaded": bool(getattr(model, "pretrained_loaded", False)),
        "frozen_backbone": bool(getattr(model, "freeze_backbone", False)),
    }
