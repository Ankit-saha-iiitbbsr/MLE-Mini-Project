"""Cheap, interpretable image statistics used as the drift feature vector.

Why not embeddings? A CNN penultimate layer would detect drift too, but it is
expensive at request time, its dimensions have no physical meaning, and when it
fires nobody can say *what* changed. These seven scalars each map to a concrete
failure on a production line:

===================  ======================================================
statistic            what a shift in it means
===================  ======================================================
``mean_intensity``   line lighting got brighter or dimmer
``std_intensity``    contrast collapsed (fogged lens, diffuser change)
``p05_intensity``    shadows lifted or crushed
``p95_intensity``    highlights blown out (new lamp, specular glare)
``edge_density``     part geometry changed, or the image went soft
``laplacian_var``    focus drifted -- the classic blur detector
``entropy``          overall information content dropped (occlusion, washout)
===================  ======================================================

All are computed on the *preprocessed* grayscale image so the offline reference
distribution and the online production distribution are directly comparable.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

#: Canonical order. Anything that builds a feature matrix must use this order.
FEATURE_NAMES: tuple[str, ...] = (
    "mean_intensity",
    "std_intensity",
    "p05_intensity",
    "p95_intensity",
    "edge_density",
    "laplacian_var",
    "entropy",
)

# Sobel kernels (separable, but written out for clarity).
_SOBEL_X = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float64)
_SOBEL_Y = _SOBEL_X.T
_LAPLACIAN = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)

#: Gradient magnitude above which a pixel counts as an edge, on a [0, 1] image.
EDGE_THRESHOLD = 0.12

#: ITU-R BT.601 luma coefficients. These are the weights PIL's "L" conversion
#: uses, so a colour input reduces to exactly the grayscale image the rest of
#: the pipeline would have seen.
_BT601_LUMA = np.array([0.299, 0.587, 0.114])


def as_gray_float(image: Any) -> np.ndarray:
    """Coerce PIL image / ndarray / nested sequence to a 2-D float array in [0, 1]."""
    if hasattr(image, "convert"):  # PIL.Image
        arr = np.asarray(image.convert("L"), dtype=np.float64)
    else:
        arr = np.asarray(image, dtype=np.float64)

    if arr.ndim == 3:
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = arr.transpose(1, 2, 0)  # CHW -> HWC
        arr = arr[..., 0] if arr.shape[-1] == 1 else arr[..., :3] @ _BT601_LUMA
    elif arr.ndim != 2:
        raise ValueError(f"Expected a 2-D or 3-D image, got shape {arr.shape}")

    if arr.size == 0:
        raise ValueError("Cannot compute statistics on an empty image")

    # uint8 inputs are 0-255; already-normalised float inputs are left alone.
    if arr.max() > 1.0 + 1e-9:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _convolve2d_valid(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """3x3 'valid' correlation via stride tricks -- avoids a SciPy dependency."""
    kh, kw = kernel.shape
    h, w = arr.shape
    if h < kh or w < kw:
        return np.zeros((0, 0), dtype=np.float64)
    windows = np.lib.stride_tricks.sliding_window_view(arr, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel)


def image_statistics(image: Any) -> dict[str, float]:
    """Compute the full statistic set for one image. Keys follow :data:`FEATURE_NAMES`."""
    arr = as_gray_float(image)

    mean = float(arr.mean())
    std = float(arr.std())
    p05, p95 = (float(v) for v in np.percentile(arr, [5, 95]))

    gx = _convolve2d_valid(arr, _SOBEL_X)
    gy = _convolve2d_valid(arr, _SOBEL_Y)
    if gx.size:
        magnitude = np.hypot(gx, gy) / 4.0  # /4 normalises Sobel gain to ~[0, 1]
        edge_density = float((magnitude > EDGE_THRESHOLD).mean())
    else:
        edge_density = 0.0

    lap = _convolve2d_valid(arr, _LAPLACIAN)
    laplacian_var = float(lap.var()) if lap.size else 0.0

    # Shannon entropy over a 64-bin histogram: coarse enough to be stable on
    # small crops, fine enough to register a washout.
    hist, _ = np.histogram(arr, bins=64, range=(0.0, 1.0))
    total = hist.sum()
    if total > 0:
        p = hist[hist > 0] / total
        entropy = float(-(p * np.log2(p)).sum())
    else:  # pragma: no cover - unreachable for a non-empty image
        entropy = 0.0

    return {
        "mean_intensity": mean,
        "std_intensity": std,
        "p05_intensity": p05,
        "p95_intensity": p95,
        "edge_density": edge_density,
        "laplacian_var": laplacian_var,
        "entropy": entropy,
    }


def to_vector(stats: dict[str, float], names: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
    """Project a statistics dict onto a fixed-order float vector."""
    return np.array([float(stats[n]) for n in names], dtype=np.float64)


def statistics_frame(images: Iterable[Any], names: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
    """Stack statistics for many images into an ``(n_images, n_features)`` matrix."""
    rows = [to_vector(image_statistics(img), names) for img in images]
    if not rows:
        return np.empty((0, len(names)), dtype=np.float64)
    return np.vstack(rows)


def summarise(matrix: np.ndarray, names: Sequence[str] = FEATURE_NAMES) -> dict[str, dict[str, float]]:
    """Per-feature descriptive summary, used in validation and drift reports."""
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(names):
        col = matrix[:, i]
        col = col[np.isfinite(col)]
        if col.size == 0:
            out[name] = dict.fromkeys(("mean", "std", "min", "p25", "p50", "p75", "max"), math.nan)
            continue
        q25, q50, q75 = (float(v) for v in np.percentile(col, [25, 50, 75]))
        out[name] = {
            "mean": float(col.mean()),
            "std": float(col.std()),
            "min": float(col.min()),
            "p25": q25,
            "p50": q50,
            "p75": q75,
            "max": float(col.max()),
        }
    return out
