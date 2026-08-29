"""Procedural generator for casting-inspection images.

The reference dataset for this flavor is the Kaggle *Casting Product Image Data
for Quality Inspection* set (submersible-pump impellers, grayscale, one
``ok_front`` and one ``def_front`` class). Downloading it requires Kaggle
credentials, which makes a graded pipeline awkward to reproduce, so this module
renders a stand-in with the same structure: a light impeller on a dark
background, photographed under varying light and orientation, where the defect
class carries small localised surface flaws.

Two properties matter for the generator to be a *useful* stand-in:

1. **Nuisance factors are class-independent.** Brightness, contrast, rotation,
   scale, blur and sensor noise are drawn from the same distribution for both
   classes. A model cannot reach high accuracy through global image statistics;
   it has to localise the flaw. (:mod:`defectvision.data.validate` and the drift
   detectors exploit exactly those global statistics, which is why they must
   *not* be class-informative here.)
2. **Defect salience is tunable.** ``difficulty`` interpolates defect size and
   contrast. At the committed value (0.62) the flaws are a few pixels across at
   128x128 and only ~25 grey levels from the surrounding surface, which leaves
   the task hard enough that model choice actually changes the score.

Every image is a pure function of ``(seed, index)``, so the dataset is
regenerable byte-for-byte from ``params.yaml`` alone -- that is what the DVC
stage hashes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Geometry is drawn at SS x resolution and box-downsampled, which anti-aliases
# the blade edges. Without it the rim shows staircase artifacts that a CNN can
# latch onto as a shortcut feature.
SUPERSAMPLE = 3

DEFECT_KINDS: tuple[str, ...] = (
    "blowhole",        # gas porosity: round dark cavity on the surface
    "pinhole_cluster", # scattered micro-porosity
    "crack",           # hairline fracture
    "chipped_rim",     # material missing from the outer edge
    "burr",            # excess material protruding past the rim
    "scratch",         # linear surface score mark
)


@dataclass
class RenderMeta:
    """Provenance for one rendered frame (kept alongside the image in the manifest)."""

    index: int
    label: int
    defect_kinds: list[str] = field(default_factory=list)
    n_defects: int = 0
    brightness: float = 0.0
    contrast: float = 1.0
    rotation_deg: float = 0.0
    scale: float = 1.0
    blur_sigma: float = 0.0
    noise_std: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "defect_kinds": "|".join(self.defect_kinds),
            "n_defects": self.n_defects,
            "gen_brightness": round(self.brightness, 4),
            "gen_contrast": round(self.contrast, 4),
            "gen_rotation_deg": round(self.rotation_deg, 3),
            "gen_scale": round(self.scale, 4),
            "gen_blur_sigma": round(self.blur_sigma, 4),
            "gen_noise_std": round(self.noise_std, 4),
        }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _spiral(cx: float, cy: float, r0: float, r1: float, theta0: float,
            curve: float, n: int = 18) -> list[tuple[float, float]]:
    """Points along a curve sweeping from radius *r0* to *r1* while rotating by *curve*."""
    ts = np.linspace(0.0, 1.0, n)
    radii = r0 + (r1 - r0) * ts
    angles = theta0 + curve * ts
    return [(cx + r * math.cos(a), cy + r * math.sin(a)) for r, a in zip(radii, angles, strict=True)]


def _blade_polygon(cx: float, cy: float, r0: float, r1: float, theta: float,
                   half_width: float, curve: float) -> list[tuple[float, float]]:
    """A curved impeller vane, bounded by two spiral arcs."""
    leading = _spiral(cx, cy, r0, r1, theta - half_width, curve)
    trailing = _spiral(cx, cy, r0, r1, theta + half_width, curve)
    return leading + trailing[::-1]


def _disc(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float,
          fill: int | None = None, outline: int | None = None, width: int = 1) -> None:
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=width)


# ---------------------------------------------------------------------------
# Defect rendering
# ---------------------------------------------------------------------------


def _defect_amplitude(difficulty: float) -> float:
    """Grey-level separation between a flaw and the surrounding surface."""
    return float(np.interp(difficulty, [0.0, 1.0], [85.0, 26.0]))


def _defect_scale(difficulty: float) -> float:
    """Size multiplier for flaws.

    Calibrated against the reference Kaggle set, where a blowhole spans roughly
    3-8% of the image width. At the committed difficulty this puts flaws at
    5-12 px on a 128 px frame: small enough to need a convolutional receptive
    field, large enough to survive the resize.
    """
    return float(np.interp(difficulty, [0.0, 1.0], [1.7, 0.60]))


def _surface_point(rng: np.random.Generator, cx: float, cy: float,
                   r_hub: float, r_rim: float) -> tuple[float, float]:
    """Sample a point on the impeller face (between the hub and the rim)."""
    # sqrt keeps the sample uniform over area rather than clustered at the hub.
    t = math.sqrt(rng.uniform(0.0, 1.0))
    r = r_hub + (r_rim - r_hub) * (0.12 + 0.80 * t)
    a = rng.uniform(0.0, 2.0 * math.pi)
    return cx + r * math.cos(a), cy + r * math.sin(a)


def _draw_defect(draw: ImageDraw.ImageDraw, kind: str, rng: np.random.Generator,
                 cx: float, cy: float, r_hub: float, r_rim: float,
                 body: int, bg: int, difficulty: float, ss: int) -> None:
    """Paint one flaw of *kind* onto the casting face."""
    amp = _defect_amplitude(difficulty)
    scl = _defect_scale(difficulty)
    dark = int(np.clip(body - amp, 0, 255))
    bright = int(np.clip(body + amp * 0.85, 0, 255))
    unit = r_rim * 0.075 * scl  # base flaw radius

    if kind == "blowhole":
        px, py = _surface_point(rng, cx, cy, r_hub, r_rim)
        rad = unit * rng.uniform(0.7, 1.5)
        _disc(draw, px, py, rad, fill=dark)
        # A real cavity catches light on one edge; the highlight is what makes
        # this distinguishable from a plain shadow.
        _disc(draw, px - rad * 0.30, py - rad * 0.30, rad * 0.42,
              fill=int(np.clip(body - amp * 0.25, 0, 255)))

    elif kind == "pinhole_cluster":
        px, py = _surface_point(rng, cx, cy, r_hub, r_rim)
        spread = unit * 2.6
        for _ in range(int(rng.integers(5, 12))):
            ox, oy = rng.normal(0.0, spread, size=2)
            _disc(draw, px + ox, py + oy, unit * rng.uniform(0.22, 0.45), fill=dark)

    elif kind == "crack":
        px, py = _surface_point(rng, cx, cy, r_hub, r_rim)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        seg = unit * 1.8
        pts = [(px, py)]
        for _ in range(int(rng.integers(4, 8))):
            angle += rng.normal(0.0, 0.45)  # random walk -> jagged, not straight
            px += seg * math.cos(angle)
            py += seg * math.sin(angle)
            pts.append((px, py))
        draw.line(pts, fill=dark, width=max(1, int(unit * 0.42)), joint="curve")

    elif kind == "chipped_rim":
        a = rng.uniform(0.0, 2.0 * math.pi)
        px, py = cx + r_rim * math.cos(a), cy + r_rim * math.sin(a)
        rad = unit * rng.uniform(1.6, 2.8)
        # Punch background through the rim: material is missing, not shadowed.
        _disc(draw, px, py, rad, fill=bg)

    elif kind == "burr":
        a = rng.uniform(0.0, 2.0 * math.pi)
        rad = unit * rng.uniform(1.0, 1.9)
        px = cx + (r_rim + rad * 0.55) * math.cos(a)
        py = cy + (r_rim + rad * 0.55) * math.sin(a)
        _disc(draw, px, py, rad, fill=bright)

    elif kind == "scratch":
        px, py = _surface_point(rng, cx, cy, r_hub, r_rim)
        a = rng.uniform(0.0, 2.0 * math.pi)
        length = r_rim * rng.uniform(0.30, 0.62) * scl
        ex, ey = px + length * math.cos(a), py + length * math.sin(a)
        draw.line([(px, py), (ex, ey)], fill=bright if rng.random() < 0.4 else dark,
                  width=max(1, int(unit * 0.30)))

    else:  # pragma: no cover - guarded by DEFECT_KINDS
        raise ValueError(f"Unknown defect kind: {kind!r}")

    del ss  # geometry already scaled through r_rim


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------


def render_casting(
    index: int,
    *,
    seed: int = 42,
    size: int = 128,
    difficulty: float = 0.62,
    defective: bool = False,
    n_blades: int = 8,
) -> tuple[np.ndarray, RenderMeta]:
    """Render a single grayscale casting frame.

    Returns the ``uint8`` image of shape ``(size, size)`` and its provenance.
    Deterministic in ``(seed, index)``.
    """
    # Large odd multiplier decorrelates neighbouring indices.
    rng = np.random.default_rng(seed * 1_000_003 + index)

    ss = SUPERSAMPLE
    canvas = size * ss

    # --- palette ----------------------------------------------------------
    bg = int(rng.integers(8, 26))
    body = int(rng.integers(118, 152))
    rim = int(np.clip(body + rng.integers(22, 46), 0, 255))
    blade = int(np.clip(body - rng.integers(16, 34), 0, 255))
    hub = int(np.clip(body - rng.integers(34, 58), 0, 255))

    # --- pose -------------------------------------------------------------
    scale = float(rng.uniform(0.90, 1.08))
    cx = canvas / 2 + rng.normal(0.0, canvas * 0.012)
    cy = canvas / 2 + rng.normal(0.0, canvas * 0.012)
    r_rim = canvas * 0.40 * scale
    r_hub = r_rim * 0.20
    phase = float(rng.uniform(0.0, 2.0 * math.pi))

    img = Image.new("L", (canvas, canvas), color=bg)
    draw = ImageDraw.Draw(img)

    # --- casting body -----------------------------------------------------
    _disc(draw, cx, cy, r_rim, fill=body)
    _disc(draw, cx, cy, r_rim, outline=rim, width=max(1, int(r_rim * 0.045)))

    curve = float(rng.uniform(0.42, 0.66)) * (1 if rng.random() < 0.5 else -1)
    half_width = (math.pi / n_blades) * 0.46
    for i in range(n_blades):
        theta = phase + 2.0 * math.pi * i / n_blades
        draw.polygon(
            _blade_polygon(cx, cy, r_hub * 1.05, r_rim * 0.94, theta, half_width, curve),
            fill=blade,
        )

    # --- hub and bore -----------------------------------------------------
    _disc(draw, cx, cy, r_hub, fill=hub)
    _disc(draw, cx, cy, r_hub * 0.92, outline=int(np.clip(body + 18, 0, 255)),
          width=max(1, int(r_hub * 0.12)))
    _disc(draw, cx, cy, r_hub * 0.40, fill=int(np.clip(bg + 12, 0, 255)))

    # --- defects (before global nuisance, so they share the same optics) ---
    meta_kinds: list[str] = []
    if defective:
        n_defects = int(rng.integers(1, 4))
        for _ in range(n_defects):
            kind = str(rng.choice(DEFECT_KINDS))
            _draw_defect(draw, kind, rng, cx, cy, r_hub, r_rim, body, bg, difficulty, ss)
            meta_kinds.append(kind)

    # --- downsample (anti-alias) -----------------------------------------
    img = img.resize((size, size), Image.Resampling.BOX)
    arr = np.asarray(img, dtype=np.float32)

    # --- illumination: off-axis falloff + vignette ------------------------
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    lx, ly = rng.uniform(-1.0, 1.0, size=2)
    gradient = 1.0 + 0.13 * (lx * (xx / size - 0.5) + ly * (yy / size - 0.5)) * 2.0
    rr = np.sqrt(((xx - size / 2) / (size / 2)) ** 2 + ((yy - size / 2) / (size / 2)) ** 2)
    vignette = 1.0 - 0.16 * np.clip(rr, 0.0, 1.4) ** 2
    arr *= gradient * vignette

    # --- pose jitter ------------------------------------------------------
    # +/-22 deg, not a full turn: parts arrive in a fixture, so orientation is
    # constrained in practice. Free rotation would make a 6 px flaw effectively
    # unlocalisable from a few thousand samples and is not what the real
    # capture rig produces. Blade phase (above) already supplies rotational
    # variety in the *appearance* without destroying the pose prior.
    rotation = float(rng.uniform(-22.0, 22.0))
    arr = np.asarray(
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).rotate(
            rotation, resample=Image.Resampling.BILINEAR, fillcolor=bg
        ),
        dtype=np.float32,
    )

    # --- camera response --------------------------------------------------
    brightness = float(rng.normal(0.0, 0.10))
    contrast = float(rng.normal(1.0, 0.10))
    arr = (arr - 128.0) * contrast + 128.0 + brightness * 255.0

    blur_sigma = float(abs(rng.normal(0.0, 0.55)))
    if blur_sigma > 0.15:
        arr = np.asarray(
            Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(radius=blur_sigma)
            ),
            dtype=np.float32,
        )

    noise_std = float(rng.uniform(1.5, 6.0))
    arr += rng.normal(0.0, noise_std, size=arr.shape)

    out = np.clip(arr, 0, 255).astype(np.uint8)

    meta = RenderMeta(
        index=index,
        label=1 if defective else 0,
        defect_kinds=meta_kinds,
        n_defects=len(meta_kinds),
        brightness=brightness,
        contrast=contrast,
        rotation_deg=rotation,
        scale=scale,
        blur_sigma=blur_sigma,
        noise_std=noise_std / 255.0,
    )
    return out, meta


def generate_dataset(
    n_images: int,
    *,
    seed: int = 42,
    size: int = 128,
    difficulty: float = 0.62,
    defect_ratio: float = 0.5,
    n_blades: int = 8,
):
    """Yield ``(image, meta)`` for a balanced-by-construction synthetic corpus.

    The label sequence is shuffled with its own generator so that changing
    ``defect_ratio`` does not reshuffle which *index* produces which pose.
    """
    label_rng = np.random.default_rng(seed)
    n_defect = int(round(n_images * defect_ratio))
    labels = np.array([1] * n_defect + [0] * (n_images - n_defect), dtype=np.int64)
    label_rng.shuffle(labels)

    for i, label in enumerate(labels):
        img, meta = render_casting(
            i, seed=seed, size=size, difficulty=difficulty,
            defective=bool(label), n_blades=n_blades,
        )
        yield img, meta
