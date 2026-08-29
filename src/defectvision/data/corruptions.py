"""Physically-motivated image corruptions for the M5 drift simulation.

Each operator corresponds to something that actually goes wrong on an
inspection cell, which is the point: a drift simulation built from arbitrary
noise proves the detector reacts to *something*, not that it reacts to the
failures the line will really produce.

======================  ==================================================
operator                real-world cause
======================  ==================================================
``brightness``          lamp ageing, ambient light leaking into the booth
``contrast``            diffuser fogging, dusty lens
``rotation`` / ``shear``camera remount, fixture knocked out of alignment
``scale``               working distance changed after maintenance
``blur``                focus drift, vibration
``noise``               sensor gain raised to compensate for a dim lamp
``jpeg_quality``        capture pipeline reconfigured to save bandwidth
======================  ==================================================

Operators act on PIL images and compose in a fixed order -- geometry, then
photometry, then sensor effects -- mirroring the real optical path, so a
combined scenario looks like a plausible camera rather than a stack of filters.
"""

from __future__ import annotations

import io
import math
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


def adjust_brightness(image: Image.Image, delta: float) -> Image.Image:
    """Shift brightness. *delta* is a fraction of full scale (+0.2 = +20%)."""
    if abs(delta) < 1e-9:
        return image
    arr = np.asarray(image, dtype=np.float32) + delta * 255.0
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode=image.mode)


def adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
    """Scale contrast about the image's own mean (1.0 = unchanged)."""
    if abs(factor - 1.0) < 1e-9:
        return image
    arr = np.asarray(image, dtype=np.float32)
    pivot = float(arr.mean())
    return Image.fromarray(
        np.clip((arr - pivot) * factor + pivot, 0, 255).astype(np.uint8), mode=image.mode
    )


def apply_geometry(
    image: Image.Image,
    rotation_deg: float = 0.0,
    shear_deg: float = 0.0,
    scale: float = 1.0,
    translate: tuple[float, float] = (0.0, 0.0),
) -> Image.Image:
    """Rotation + shear + scale + translation as a single affine warp.

    Composing into one transform matters: applying three separate warps would
    resample three times and blur the image, which would then also register as
    focus drift and confound the scenario.
    """
    if (abs(rotation_deg) < 1e-9 and abs(shear_deg) < 1e-9
            and abs(scale - 1.0) < 1e-9 and translate == (0.0, 0.0)):
        return image

    w, h = image.size
    cx, cy = w / 2.0, h / 2.0
    theta = math.radians(rotation_deg)
    shear = math.tan(math.radians(shear_deg))
    s = 1.0 / max(scale, 1e-6)  # PIL's matrix maps output -> input

    # Rotation composed with shear, then uniform scale.
    a = math.cos(theta) * s
    b = (math.sin(theta) + shear * math.cos(theta)) * s
    d = -math.sin(theta) * s
    e = (math.cos(theta) - shear * math.sin(theta)) * s

    tx = cx - (a * cx + b * cy) + translate[0] * w
    ty = cy - (d * cx + e * cy) + translate[1] * h

    # Fill with the border median so the warp does not introduce a black frame
    # that would itself look like a huge brightness shift.
    arr = np.asarray(image, dtype=np.uint8)
    border = np.concatenate([arr[0, :].ravel(), arr[-1, :].ravel(),
                             arr[:, 0].ravel(), arr[:, -1].ravel()])
    fill = int(np.median(border))

    return image.transform((w, h), Image.Transform.AFFINE, (a, b, tx, d, e, ty),
                           resample=Image.Resampling.BILINEAR, fillcolor=fill)


def apply_blur(image: Image.Image, sigma: float) -> Image.Image:
    """Gaussian defocus."""
    if sigma <= 1e-6:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def add_noise(image: Image.Image, std: float, seed: int | None = None) -> Image.Image:
    """Additive Gaussian sensor noise. *std* is a fraction of full scale."""
    if std <= 1e-9:
        return image
    rng = np.random.default_rng(seed)
    arr = np.asarray(image, dtype=np.float32)
    arr = arr + rng.normal(0.0, std * 255.0, size=arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode=image.mode)


def apply_jpeg(image: Image.Image, quality: int) -> Image.Image:
    """Re-encode as JPEG to introduce compression artifacts."""
    if quality >= 95:
        return image
    buffer = io.BytesIO()
    image.convert("L").save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    with Image.open(buffer) as reopened:
        return reopened.convert(image.mode).copy()


#: Scenario key -> (function, keyword name). Keeps ``params.yaml`` declarative.
_OPERATORS: dict[str, str] = {
    "brightness": "brightness",
    "contrast": "contrast",
    "rotation_deg": "geometry",
    "shear_deg": "geometry",
    "scale": "geometry",
    "translate": "geometry",
    "blur_sigma": "blur",
    "noise_std": "noise",
    "jpeg_quality": "jpeg",
}


def apply_scenario(
    image: Image.Image,
    scenario: dict[str, Any],
    seed: int | None = None,
) -> Image.Image:
    """Apply a scenario dict from ``params.yaml`` to one image.

    Unknown keys raise rather than being ignored, so a typo in a scenario name
    fails loudly instead of silently producing an uncorrupted image that would
    then be reported as "no drift detected".
    """
    unknown = set(scenario) - set(_OPERATORS)
    if unknown:
        raise ValueError(
            f"Unknown drift scenario parameter(s): {sorted(unknown)}. "
            f"Supported: {sorted(_OPERATORS)}"
        )

    out = image

    # 1. Geometry (single composed warp).
    out = apply_geometry(
        out,
        rotation_deg=float(scenario.get("rotation_deg", 0.0)),
        shear_deg=float(scenario.get("shear_deg", 0.0)),
        scale=float(scenario.get("scale", 1.0)),
        translate=tuple(scenario.get("translate", (0.0, 0.0))),  # type: ignore[arg-type]
    )
    # 2. Photometry.
    out = adjust_contrast(out, float(scenario.get("contrast", 1.0)))
    out = adjust_brightness(out, float(scenario.get("brightness", 0.0)))
    # 3. Optics and sensor.
    out = apply_blur(out, float(scenario.get("blur_sigma", 0.0)))
    out = add_noise(out, float(scenario.get("noise_std", 0.0)), seed=seed)
    if "jpeg_quality" in scenario:
        out = apply_jpeg(out, int(scenario["jpeg_quality"]))
    return out


def describe_scenario(scenario: dict[str, Any]) -> str:
    """Human-readable one-liner for reports."""
    parts = []
    labels = {
        "brightness": lambda v: f"brightness {v:+.0%}",
        "contrast": lambda v: f"contrast x{v:.2f}",
        "rotation_deg": lambda v: f"rotated {v:+.0f} deg",
        "shear_deg": lambda v: f"sheared {v:+.0f} deg",
        "scale": lambda v: f"scaled x{v:.2f}",
        "blur_sigma": lambda v: f"defocus sigma={v:.1f}px",
        "noise_std": lambda v: f"sensor noise sigma={v:.0%}",
        "jpeg_quality": lambda v: f"JPEG q={int(v)}",
    }
    for key, value in scenario.items():
        formatter = labels.get(key)
        if formatter:
            parts.append(formatter(float(value)))
    return ", ".join(parts) if parts else "unchanged"
