"""Shared fixtures.

Tests run against a **tiny synthetic corpus generated into a temp directory**,
never against the real dataset or the developer's ``data/`` tree. That keeps the
suite fast (seconds, not minutes), hermetic, and runnable in CI where no Kaggle
credentials exist -- which is precisely why the synthetic generator was kept
after the real data arrived.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from defectvision.config import load_params
from defectvision.data.synth import render_casting


@pytest.fixture(scope="session")
def base_params() -> dict:
    """The committed ``params.yaml``, loaded once."""
    return load_params()


@pytest.fixture
def tiny_params(tmp_path, base_params) -> dict:
    """A shrunk config pointing every path at a temp directory."""
    from defectvision.config import apply_overrides

    return apply_overrides(base_params, {
        "seed": 7,
        "data.source": "synthetic",
        "data.raw_dir": str(tmp_path / "raw"),
        "data.interim_dir": str(tmp_path / "interim"),
        "data.processed_dir": str(tmp_path / "processed"),
        "data.image_size": 64,
        "data.synthetic.n_images": 80,
        "data.synthetic.difficulty": 0.2,
        "preprocess.resize": 64,
        "validation.min_images_per_class": 10,
        "train.batch_size": 8,
        "train.num_workers": 0,
        # Keeps a test run from overwriting the real, submitted reports.
        "reports_dir": str(tmp_path / "reports"),
        "monitoring.db_path": str(tmp_path / "predictions.db"),
        "monitoring.reference_stats": str(tmp_path / "reference_stats.json"),
        "monitoring.min_samples_for_drift": 10,
    })


@pytest.fixture
def sample_image() -> Image.Image:
    """One rendered defective casting."""
    array, _ = render_casting(0, size=64, difficulty=0.2, defective=True)
    return Image.fromarray(array, mode="L")


@pytest.fixture
def ok_image() -> Image.Image:
    """One rendered good casting."""
    array, _ = render_casting(1, size=64, difficulty=0.2, defective=False)
    return Image.fromarray(array, mode="L")


@pytest.fixture
def image_bytes(sample_image) -> bytes:
    """PNG-encoded upload payload."""
    import io

    buffer = io.BytesIO()
    sample_image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
