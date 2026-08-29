"""Classical control arm: Histogram of Oriented Gradients + logistic regression.

Included because "we used a CNN" is only a justified decision if something
simpler was measured and found wanting. HOG is the right classical baseline for
this task -- it is a texture/edge descriptor, and surface defects *are* local
texture anomalies, so it is a genuine attempt rather than a straw man.

HOG is implemented here in NumPy rather than pulled from scikit-image: it keeps
the dependency list short, and the same descriptor has to be computable in the
serving container if this arm were ever promoted.

What the comparison is expected to show: HOG pools orientation statistics over
fairly coarse cells, so a defect covering a handful of pixels contributes a
small perturbation to a high-dimensional descriptor that is dominated by the
blade edges. A CNN can learn a filter that responds to the defect specifically.
The gap between the two arms is the quantitative argument for deep learning.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# HOG descriptor
# --------------------------------------------------------------------------

DEFAULT_CELL = 16
DEFAULT_BINS = 9
DEFAULT_BLOCK = 2


def _gradients(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference gradients; returns (magnitude, unsigned angle in degrees)."""
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    magnitude = np.hypot(gx, gy)
    # Unsigned orientation: a dark-to-light edge and its reverse describe the
    # same structure, so angles wrap at 180 rather than 360.
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    return magnitude, angle


def hog_features(
    image: np.ndarray,
    cell: int = DEFAULT_CELL,
    bins: int = DEFAULT_BINS,
    block: int = DEFAULT_BLOCK,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute a HOG descriptor for a 2-D grayscale array in [0, 1]."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"hog_features expects a 2-D array, got shape {img.shape}")
    if img.max() > 1.0 + 1e-9:
        img = img / 255.0

    h, w = img.shape
    n_cy, n_cx = h // cell, w // cell
    if n_cy < block or n_cx < block:
        raise ValueError(
            f"Image {h}x{w} is too small for cell={cell}, block={block} "
            f"(needs at least {block * cell} px per side)"
        )
    img = img[: n_cy * cell, : n_cx * cell]

    magnitude, angle = _gradients(img)

    # --- soft (linearly interpolated) orientation binning -----------------
    bin_width = 180.0 / bins
    pos = angle / bin_width - 0.5
    lower = np.floor(pos).astype(np.int64)
    frac = pos - lower
    b0 = lower % bins
    b1 = (lower + 1) % bins
    w0 = magnitude * (1.0 - frac)
    w1 = magnitude * frac

    yy, xx = np.mgrid[0 : n_cy * cell, 0 : n_cx * cell]
    cy = (yy // cell).ravel()
    cx = (xx // cell).ravel()

    hist = np.zeros((n_cy, n_cx, bins), dtype=np.float64)
    np.add.at(hist, (cy, cx, b0.ravel()), w0.ravel())
    np.add.at(hist, (cy, cx, b1.ravel()), w1.ravel())

    # --- block normalisation (L2-Hys) -------------------------------------
    # Overlapping blocks make the descriptor robust to local contrast changes,
    # which matters here because lighting varies frame to frame.
    n_by, n_bx = n_cy - block + 1, n_cx - block + 1
    out = np.empty((n_by, n_bx, block * block * bins), dtype=np.float64)
    for by in range(n_by):
        for bx in range(n_bx):
            v = hist[by : by + block, bx : bx + block].ravel()
            v = v / np.sqrt((v * v).sum() + eps * eps)
            v = np.clip(v, 0.0, 0.2)          # Hys clipping
            v = v / np.sqrt((v * v).sum() + eps * eps)
            out[by, bx] = v
    return out.ravel()


def hog_feature_dim(image_size: int, cell: int = DEFAULT_CELL,
                    bins: int = DEFAULT_BINS, block: int = DEFAULT_BLOCK) -> int:
    """Descriptor length for a square image -- useful for logging."""
    n_c = image_size // cell
    n_b = n_c - block + 1
    return n_b * n_b * block * block * bins


def hog_matrix(images: list[np.ndarray], **kwargs: Any) -> np.ndarray:
    """Stack HOG descriptors into an ``(n_samples, n_features)`` matrix."""
    if not images:
        return np.empty((0, 0), dtype=np.float64)
    return np.vstack([hog_features(img, **kwargs) for img in images])


# --------------------------------------------------------------------------
# Estimator
# --------------------------------------------------------------------------


def build_classical_model(cfg: dict[str, Any] | None = None):
    """Standardiser + L2 logistic regression over HOG descriptors.

    ``liblinear`` is chosen over the default solver because the descriptor is
    wide (~1.7k dims) relative to the sample count, where it converges more
    reliably. ``class_weight="balanced"`` mirrors the weighted loss used by the
    deep arms so the comparison is like-for-like.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    cfg = cfg or {}
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=float(cfg.get("C", 1.0)),
                    max_iter=int(cfg.get("max_iter", 2000)),
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=int(cfg.get("random_state", 42)),
                ),
            ),
        ]
    )
