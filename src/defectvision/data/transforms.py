"""Preprocessing and augmentation, split along the training-serving seam.

The single most common way an image pipeline fails in production is that the
service resizes, converts or normalises an image even slightly differently from
the training job, and accuracy quietly drops. The defence here is structural:

* :class:`PreprocessSpec` holds *only* the deterministic steps -- resize,
  grayscale, tensor conversion, normalisation. This is the exact transform used
  at inference, and it is **serialised into the model bundle** rather than read
  from ``params.yaml`` at serving time. Editing ``params.yaml`` after a model
  ships therefore cannot change how that model sees an image; the two can only
  disagree if someone rebuilds the bundle.
* Augmentation is stochastic and train-only, and is built as a *wrapper around*
  the same spec. There is no second code path that could drift.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from PIL import Image
from torchvision import transforms as T

from ..config import get

# ---------------------------------------------------------------------------
# Custom ops
# ---------------------------------------------------------------------------


class AddGaussianNoise(torch.nn.Module):
    """Additive sensor noise, applied post-normalisation.

    Models the read noise of a line-scan camera. Included because sensor noise
    is also one of the simulated drift scenarios in M5 -- training under mild
    noise is what makes the model robust to the milder end of that shift.
    """

    def __init__(self, std: float = 0.02) -> None:
        super().__init__()
        self.std = float(std)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std <= 0.0:
            return tensor
        return tensor + torch.randn_like(tensor) * self.std

    def __repr__(self) -> str:
        return f"{type(self).__name__}(std={self.std})"


# ---------------------------------------------------------------------------
# Deterministic preprocessing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessSpec:
    """Deterministic image -> tensor contract. Travels with the model bundle."""

    resize: int = 128
    grayscale: bool = True
    mean: tuple[float, ...] = (0.485,)
    std: tuple[float, ...] = (0.229,)

    @property
    def channels(self) -> int:
        return 1 if self.grayscale else 3

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> PreprocessSpec:
        cfg = get(params, "preprocess")
        return cls(
            resize=int(cfg["resize"]),
            grayscale=bool(cfg["grayscale"]),
            mean=tuple(float(v) for v in cfg["normalize_mean"]),
            std=tuple(float(v) for v in cfg["normalize_std"]),
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PreprocessSpec:
        return cls(
            resize=int(d["resize"]),
            grayscale=bool(d["grayscale"]),
            mean=tuple(float(v) for v in d["mean"]),
            std=tuple(float(v) for v in d["std"]),
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mean"] = list(self.mean)
        d["std"] = list(self.std)
        return d

    def build(self) -> T.Compose:
        """The inference-time transform: PIL.Image -> normalised float tensor."""
        ops: list[Any] = []
        if self.grayscale:
            ops.append(T.Grayscale(num_output_channels=1))
        else:
            ops.append(T.Lambda(lambda im: im.convert("RGB")))
        # A square resize (not resize-shortest-side + centre crop) because a
        # defect near the rim would be cropped away by the latter.
        ops.append(T.Resize((self.resize, self.resize),
                            interpolation=T.InterpolationMode.BILINEAR,
                            antialias=True))
        ops.append(T.ToTensor())
        ops.append(T.Normalize(mean=list(self.mean), std=list(self.std)))
        return T.Compose(ops)

    def apply(self, image: Image.Image) -> torch.Tensor:
        """Convenience: transform a single PIL image."""
        return self.build()(image)

    def expand_channels(self, tensor: torch.Tensor) -> torch.Tensor:
        """Repeat a 1-channel tensor to 3 channels for ImageNet backbones."""
        if tensor.shape[-3] == 1:
            return tensor.repeat(*([1] * (tensor.dim() - 3)), 3, 1, 1)
        return tensor


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


@dataclass
class AugmentSpec:
    """Stochastic, train-only augmentation."""

    enabled: bool = True
    random_rotation_deg: float = 12.0
    random_translate: float = 0.06
    random_scale: tuple[float, float] = (0.92, 1.08)
    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.5
    brightness_jitter: float = 0.22
    contrast_jitter: float = 0.22
    gaussian_blur_p: float = 0.20
    gaussian_noise_std: float = 0.02
    random_erasing_p: float = 0.0
    _extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> AugmentSpec:
        cfg = dict(get(params, "augment"))
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            random_rotation_deg=float(cfg.get("random_rotation_deg", 0.0)),
            random_translate=float(cfg.get("random_translate", 0.0)),
            random_scale=tuple(float(v) for v in cfg.get("random_scale", (1.0, 1.0))),
            horizontal_flip_p=float(cfg.get("horizontal_flip_p", 0.0)),
            vertical_flip_p=float(cfg.get("vertical_flip_p", 0.0)),
            brightness_jitter=float(cfg.get("brightness_jitter", 0.0)),
            contrast_jitter=float(cfg.get("contrast_jitter", 0.0)),
            gaussian_blur_p=float(cfg.get("gaussian_blur_p", 0.0)),
            gaussian_noise_std=float(cfg.get("gaussian_noise_std", 0.0)),
            random_erasing_p=float(cfg.get("random_erasing_p", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_extra", None)
        d["random_scale"] = list(self.random_scale)
        return d


def build_train_transform(pre: PreprocessSpec, aug: AugmentSpec) -> T.Compose:
    """Augmented transform for the training split.

    The augmentation policy is chosen to mirror the nuisance factors that a real
    inspection cell actually produces -- part placement jitter, lamp drift,
    focus wander, sensor noise -- so it doubles as pre-emptive hardening against
    the M5 drift scenarios. Corruptions the camera cannot produce (colour
    shifts, perspective warps) are deliberately absent.
    """
    if not aug.enabled:
        return pre.build()

    ops: list[Any] = []
    ops.append(T.Grayscale(num_output_channels=1) if pre.grayscale
               else T.Lambda(lambda im: im.convert("RGB")))
    ops.append(T.Resize((pre.resize, pre.resize),
                        interpolation=T.InterpolationMode.BILINEAR, antialias=True))

    # --- geometry: part placement jitter in the fixture -------------------
    if aug.random_rotation_deg > 0 or aug.random_translate > 0 or aug.random_scale != (1.0, 1.0):
        ops.append(T.RandomAffine(
            degrees=aug.random_rotation_deg,
            translate=(aug.random_translate, aug.random_translate),
            scale=tuple(aug.random_scale),
            interpolation=T.InterpolationMode.BILINEAR,
            fill=0,
        ))
    # An impeller is rotationally symmetric, so flips produce valid parts and
    # do not change the label.
    if aug.horizontal_flip_p > 0:
        ops.append(T.RandomHorizontalFlip(p=aug.horizontal_flip_p))
    if aug.vertical_flip_p > 0:
        ops.append(T.RandomVerticalFlip(p=aug.vertical_flip_p))

    # --- photometry: lamp drift and focus wander --------------------------
    if aug.brightness_jitter > 0 or aug.contrast_jitter > 0:
        ops.append(T.ColorJitter(brightness=aug.brightness_jitter,
                                 contrast=aug.contrast_jitter))
    if aug.gaussian_blur_p > 0:
        ops.append(T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
                                 p=aug.gaussian_blur_p))

    ops.append(T.ToTensor())
    ops.append(T.Normalize(mean=list(pre.mean), std=list(pre.std)))

    if aug.gaussian_noise_std > 0:
        ops.append(AddGaussianNoise(aug.gaussian_noise_std))

    # NOTE: random erasing is available but defaults to 0.0. On this task a
    # defect can occupy fewer than 100 px; erasing a patch can remove the only
    # evidence of the defect while the label still says "defect", which injects
    # label noise instead of regularisation. Enable only with a small area.
    if aug.random_erasing_p > 0:
        ops.append(T.RandomErasing(p=aug.random_erasing_p, scale=(0.01, 0.04), value=0.0))

    return T.Compose(ops)


def build_transforms(params: dict[str, Any]) -> tuple[T.Compose, T.Compose, PreprocessSpec]:
    """Return ``(train_transform, eval_transform, spec)`` for the current config."""
    pre = PreprocessSpec.from_params(params)
    aug = AugmentSpec.from_params(params)
    return build_train_transform(pre, aug), pre.build(), pre


def denormalize(tensor: torch.Tensor, spec: PreprocessSpec) -> torch.Tensor:
    """Invert normalisation for visualisation. Returns values clamped to [0, 1]."""
    mean = torch.tensor(spec.mean, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(spec.std, device=tensor.device).view(-1, 1, 1)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)
