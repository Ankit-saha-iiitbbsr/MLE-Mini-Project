"""Torch ``Dataset`` / ``DataLoader`` construction from the versioned manifest.

Datasets are always built *from the manifest*, never by globbing the filesystem.
That is what makes a training run reproducible: the manifest is a DVC-tracked
artifact, so re-running with the same manifest hash trains on exactly the same
images in the same folds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..config import class_names, get, resolve
from ..logging_utils import get_logger, seed_worker
from .split import load_manifest
from .transforms import PreprocessSpec, build_transforms

log = get_logger(__name__)


class DefectImageDataset(Dataset):
    """Grayscale casting images plus binary labels, driven by a manifest slice.

    Parameters
    ----------
    manifest:
        Rows with at least ``relpath`` and ``label``.
    root:
        Directory the ``relpath`` values are relative to (``data/raw``).
    transform:
        Callable applied to the PIL image.
    expand_to_rgb:
        Repeat the single channel three times, for ImageNet backbones whose
        first conv expects 3 channels.
    cache:
        Hold decoded images in memory. The corpus is a few thousand small
        grayscale frames (~50 MB), and on CPU the JPEG/PNG decode dominates
        epoch time, so caching is roughly a 3x speedup for a trivial cost.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        root: str | Path,
        transform: Any,
        *,
        expand_to_rgb: bool = False,
        cache: bool = True,
    ) -> None:
        if manifest.empty:
            raise ValueError("Cannot build a dataset from an empty manifest slice")
        missing = {"relpath", "label"} - set(manifest.columns)
        if missing:
            raise ValueError(f"Manifest is missing required column(s): {sorted(missing)}")

        self.root = Path(root)
        self.paths = [self.root / rp for rp in manifest["relpath"].tolist()]
        self.labels = manifest["label"].to_numpy(dtype=np.int64)
        self.relpaths = manifest["relpath"].tolist()
        self.transform = transform
        self.expand_to_rgb = expand_to_rgb
        self._cache: list[Image.Image] | None = [] if cache else None

        if self._cache is not None:
            for p in self.paths:
                with Image.open(p) as im:
                    self._cache.append(im.convert("L").copy())

    def __len__(self) -> int:
        return len(self.paths)

    def _read(self, idx: int) -> Image.Image:
        if self._cache is not None:
            return self._cache[idx]
        with Image.open(self.paths[idx]) as im:
            return im.convert("L").copy()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        tensor = self.transform(self._read(idx))
        if self.expand_to_rgb and tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor, int(self.labels[idx])

    # -- helpers ----------------------------------------------------------

    @property
    def class_distribution(self) -> dict[int, int]:
        values, counts = np.unique(self.labels, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts, strict=True)}


def compute_class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1.

    Used as ``pos_weight``/``weight`` in the loss so the minority class is not
    ignored. Normalising to mean 1 keeps the loss magnitude (and therefore a
    sensible learning rate) comparable to the unweighted case.
    """
    counts = np.bincount(labels.astype(np.int64), minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_datasets(
    params: dict[str, Any],
    *,
    expand_to_rgb: bool = False,
    cache: bool = True,
) -> tuple[dict[str, DefectImageDataset], PreprocessSpec]:
    """Build train/val/test datasets. Only ``train`` receives augmentation."""
    train_tf, eval_tf, spec = build_transforms(params)
    root = resolve(get(params, "data.raw_dir"))
    manifest = load_manifest(params)

    datasets: dict[str, DefectImageDataset] = {}
    for split in ("train", "val", "test"):
        sub = manifest[manifest["split"] == split].reset_index(drop=True)
        if sub.empty:
            log.warning("Split %r has no rows; skipping", split)
            continue
        datasets[split] = DefectImageDataset(
            sub, root,
            transform=train_tf if split == "train" else eval_tf,
            expand_to_rgb=expand_to_rgb,
            cache=cache,
        )
        log.info("Dataset %-5s n=%-5d distribution=%s",
                 split, len(datasets[split]), datasets[split].class_distribution)

    if "train" not in datasets:
        raise ValueError("Manifest contains no training rows")
    return datasets, spec


def build_dataloaders(
    params: dict[str, Any],
    *,
    batch_size: int | None = None,
    expand_to_rgb: bool = False,
    cache: bool = True,
) -> tuple[dict[str, DataLoader], PreprocessSpec, torch.Tensor]:
    """Build dataloaders plus the class-weight vector for the loss.

    Returns ``(loaders, preprocess_spec, class_weights)``.
    """
    datasets, spec = build_datasets(params, expand_to_rgb=expand_to_rgb, cache=cache)

    bs = int(batch_size or get(params, "train.batch_size"))
    workers = int(get(params, "train.num_workers", 0))
    seed = int(get(params, "seed"))

    # Seeded generator so shuffling is identical across reruns.
    generator = torch.Generator()
    generator.manual_seed(seed)

    loaders: dict[str, DataLoader] = {}
    for split, ds in datasets.items():
        loaders[split] = DataLoader(
            ds,
            batch_size=bs,
            shuffle=(split == "train"),
            num_workers=workers,
            pin_memory=False,
            drop_last=False,
            generator=generator if split == "train" else None,
            worker_init_fn=seed_worker if workers > 0 else None,
            persistent_workers=workers > 0,
        )

    n_classes = len(class_names(params))
    if str(get(params, "train.class_weighting", "balanced")).lower() == "balanced":
        weights = compute_class_weights(datasets["train"].labels, n_classes)
    else:
        weights = torch.ones(n_classes, dtype=torch.float32)
    log.info("Class weights: %s", weights.tolist())

    return loaders, spec, weights
