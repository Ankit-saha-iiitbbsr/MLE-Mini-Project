"""Stage 1 of M2 - get raw images onto disk under ``data/raw/<class>/``.

Three interchangeable sources, selected by ``data.source`` in ``params.yaml``:

``kaggle``
    Download the reference dataset (*Casting Product Image Data for Quality
    Inspection*). Authentication uses a **Kaggle API token**, never an account
    password -- see :func:`kaggle_credentials_status`.
``synthetic``
    Render the stand-in corpus from :mod:`defectvision.data.synth`. This is the
    committed default so that ``dvc repro`` works on a clean machine with no
    credentials.
``local``
    Trust whatever is already in ``data/raw/<class>/``. Useful when a real
    production dump is dropped in by hand.

Whichever source runs, the output contract is identical: one directory per
class name in ``data.classes``, plus a ``_source.json`` provenance record. Every
downstream stage depends only on that contract.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..config import class_names, ensure_dir, get, resolve
from ..logging_utils import get_logger
from .synth import generate_dataset

log = get_logger(__name__)

#: Kaggle folder names -> our canonical class names.
KAGGLE_CLASS_MAP = {
    "ok_front": "ok",
    "def_front": "defect",
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def kaggle_credentials_status() -> dict[str, Any]:
    """Report whether a usable Kaggle API token is present, without reading secrets.

    Kaggle's public API authenticates with a **username + API key**, issued from
    <https://www.kaggle.com/settings> -> *API* -> *Create New Token*. It does not
    accept an account email/password, and this project never asks for one.

    The token is picked up from either:

    * ``~/.kaggle/kaggle.json`` (the standard location; the file the browser
      downloads, moved into place as-is), or
    * the ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` environment variables.

    Only presence is reported here -- the values are left for the ``kaggle``
    client to read, so they never pass through this project's logs.
    """
    env_ok = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    token = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    return {
        "env_vars_set": env_ok,
        "token_file": str(token),
        "token_file_present": token.is_file(),
        "authenticated": env_ok or token.is_file(),
    }


_CREDENTIALS_HELP = """
Kaggle API credentials were not found.

Kaggle authenticates with an API TOKEN, not your account email and password.
To create one:

  1. Sign in at https://www.kaggle.com and open https://www.kaggle.com/settings
  2. Under 'API', click 'Create New Token'. A file named kaggle.json downloads.
  3. Move that file to:
        Windows : %USERPROFILE%\\.kaggle\\kaggle.json
        Linux   : ~/.kaggle/kaggle.json   (then: chmod 600 ~/.kaggle/kaggle.json)
  4. Accept the dataset rules once, at:
        https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product

Alternatively export KAGGLE_USERNAME and KAGGLE_KEY in your shell.

No credentials are needed to run this project: leave data.source = synthetic
in params.yaml and the full pipeline reproduces from the generator instead.
""".strip()


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def _write_provenance(raw_dir: Path, record: dict[str, Any]) -> None:
    record["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (raw_dir / "_source.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def acquire_synthetic(params: dict[str, Any]) -> dict[str, Any]:
    """Render the synthetic corpus into ``data/raw/<class>/``."""
    raw_dir = ensure_dir(get(params, "data.raw_dir"))
    classes = class_names(params)
    seed = int(get(params, "seed"))
    cfg = get(params, "data.synthetic")
    size = int(get(params, "data.image_size"))

    for name in classes:
        target = raw_dir / name
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)

    n_images = int(cfg["n_images"])
    counts = dict.fromkeys(classes, 0)
    meta_rows: list[dict[str, Any]] = []

    log.info("Rendering %d synthetic %dx%d frames (difficulty=%.2f, seed=%d)",
             n_images, size, size, float(cfg["difficulty"]), seed)

    for img, meta in generate_dataset(
        n_images,
        seed=seed,
        size=size,
        difficulty=float(cfg["difficulty"]),
        defect_ratio=float(cfg["defect_ratio"]),
        n_blades=int(cfg["n_blades"]),
    ):
        name = classes[meta.label]
        path = raw_dir / name / f"{name}_{meta.index:06d}.png"
        Image.fromarray(img, mode="L").save(path, format="PNG", optimize=True)
        counts[name] += 1
        meta_rows.append({"relpath": f"{name}/{path.name}", **meta.as_dict()})

    # The render metadata is ground truth we would not have in production; it is
    # kept for the EDA notebook and for the drift report's "why did this shift"
    # narrative, and is never joined into the training features.
    (raw_dir / "_render_meta.json").write_text(json.dumps(meta_rows, indent=1), encoding="utf-8")

    record = {
        "source": "synthetic",
        "generator": "defectvision.data.synth",
        "seed": seed,
        "image_size": size,
        "params": cfg,
        "counts": counts,
    }
    _write_provenance(raw_dir, record)
    log.info("Synthetic corpus ready: %s", counts)
    return record


def _resolve_kaggle_root(params: dict[str, Any]) -> tuple[Path, str]:
    """Locate the Kaggle corpus, preferring an existing local copy over a download.

    Downloading 8.6k JPEGs on every clean checkout is slow and needs credentials,
    so an already-extracted copy under ``data.kaggle_local_dir`` is used when
    present. Returns ``(root, how_it_was_obtained)``.
    """
    local = get(params, "data.kaggle_local_dir", None)
    if local:
        candidate = resolve(local)
        if candidate.is_dir():
            log.info("Using local Kaggle copy: %s", candidate)
            return candidate, "local_copy"
        log.info("data.kaggle_local_dir=%s not present; falling back to download", candidate)

    if not kaggle_credentials_status()["authenticated"]:
        raise RuntimeError(_CREDENTIALS_HELP)

    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "kagglehub is not installed. Install it with:\n"
            "    pip install kagglehub\n"
            "or set data.source = synthetic in params.yaml."
        ) from exc

    slug = str(get(params, "data.kaggle_dataset"))
    log.info("Downloading Kaggle dataset %s (cached after the first run)", slug)
    return Path(kagglehub.dataset_download(slug)), "kagglehub_download"


def acquire_kaggle(params: dict[str, Any]) -> dict[str, Any]:
    """Normalise the Kaggle casting dataset into ``data/raw/<class>/``.

    The archive ships two independent captures:

    * ``casting_data/{train,test}/{ok_front,def_front}`` - 7348 images at
      300x300, the standard benchmark corpus.
    * ``casting_512x512/{ok_front,def_front}`` - 1300 images at 512x512.

    Only the subset named by ``data.kaggle_subset`` is ingested. Mixing the two
    would put the same physical parts at two resolutions into one corpus, and
    the higher-resolution capture is more valuable held back: it is a genuine
    camera-upgrade distribution shift for the M5 monitoring work, which is
    worth far more than 1300 extra training rows.

    The published train/test folders are flattened and re-split by our own
    stage, so that the split policy (stratification, near-duplicate grouping,
    seed) lives in ``params.yaml`` and is versioned with everything else.
    """
    raw_dir = ensure_dir(get(params, "data.raw_dir"))
    classes = class_names(params)
    subset = str(get(params, "data.kaggle_subset", "casting_data"))

    download_root, how = _resolve_kaggle_root(params)

    # Find the subset directory wherever it sits (the archive double-nests it).
    subset_roots = [p for p in download_root.rglob(subset) if p.is_dir()]
    if not subset_roots:
        raise RuntimeError(
            f"Subset {subset!r} not found under {download_root}. "
            f"Available directories: {sorted({p.name for p in download_root.iterdir() if p.is_dir()})}"
        )
    # Prefer the deepest match: the archive nests casting_data/casting_data/.
    subset_root = max(subset_roots, key=lambda p: len(p.parts))
    log.info("Ingesting subset %s from %s", subset, subset_root)

    for name in classes:
        target = raw_dir / name
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)

    counts = dict.fromkeys(classes, 0)
    collisions = 0
    for src in sorted(subset_root.rglob("*")):
        if not src.is_file() or src.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        canonical = KAGGLE_CLASS_MAP.get(src.parent.name.lower())
        if canonical is None or canonical not in classes:
            continue
        # train/ and test/ reuse filenames, so keep the origin folder as a prefix.
        origin = src.parent.parent.name.lower()
        dest = raw_dir / canonical / f"{origin}_{src.name}"
        if dest.exists():
            collisions += 1
            dest = dest.with_name(f"{dest.stem}__{collisions}{dest.suffix}")
        shutil.copy2(src, dest)
        counts[canonical] += 1

    if sum(counts.values()) == 0:
        raise RuntimeError(
            f"No images matched {sorted(KAGGLE_CLASS_MAP)} under {subset_root}. "
            "The dataset layout may have changed; inspect the directory."
        )

    record = {
        "source": "kaggle",
        "dataset": str(get(params, "data.kaggle_dataset")),
        "subset": subset,
        "obtained_via": how,
        "source_root": str(subset_root),
        "class_map": KAGGLE_CLASS_MAP,
        "counts": counts,
        "citation": (
            "Ravirajsinh Dabhi, 'Casting product image data for quality inspection', "
            "Kaggle, 2020. Licensed CC BY-NC-SA 4.0."
        ),
    }
    _write_provenance(raw_dir, record)
    log.info("Kaggle corpus ready: %s (%d images)", counts, sum(counts.values()))
    return record


def acquire_local(params: dict[str, Any]) -> dict[str, Any]:
    """Validate that hand-placed images already satisfy the raw-layout contract."""
    raw_dir = resolve(get(params, "data.raw_dir"))
    classes = class_names(params)

    missing = [c for c in classes if not (raw_dir / c).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"data.source='local' expects one directory per class under {raw_dir}. "
            f"Missing: {missing}"
        )

    counts = {
        c: sum(1 for p in (raw_dir / c).iterdir()
               if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        for c in classes
    }
    empty = [c for c, n in counts.items() if n == 0]
    if empty:
        raise ValueError(f"No images found for class(es): {empty}")

    record = {"source": "local", "root": str(raw_dir), "counts": counts}
    _write_provenance(raw_dir, record)
    log.info("Local corpus accepted: %s", counts)
    return record


def acquire(params: dict[str, Any], source: str | None = None) -> dict[str, Any]:
    """Dispatch to the configured source and return its provenance record."""
    src = (source or get(params, "data.source")).lower()
    handlers = {
        "synthetic": acquire_synthetic,
        "kaggle": acquire_kaggle,
        "local": acquire_local,
    }
    if src not in handlers:
        raise ValueError(f"Unknown data.source {src!r}; expected one of {sorted(handlers)}")
    return handlers[src](params)


def preview_grid(params: dict[str, Any], out_path: str, n_per_class: int = 6) -> Path:
    """Save a contact sheet of raw samples -- the first sanity check on any dataset."""
    raw_dir = resolve(get(params, "data.raw_dir"))
    classes = class_names(params)
    size = int(get(params, "data.image_size"))

    rows = []
    for name in classes:
        files = sorted((raw_dir / name).glob("*"))[:n_per_class]
        tiles = []
        for f in files:
            with Image.open(f) as im:
                tiles.append(np.asarray(im.convert("L").resize((size, size))))
        while len(tiles) < n_per_class:
            tiles.append(np.zeros((size, size), dtype=np.uint8))
        rows.append(np.hstack(tiles))

    sheet = np.vstack(rows)
    dest = resolve(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet, mode="L").save(dest)
    log.info("Wrote preview grid -> %s", dest)
    return dest
