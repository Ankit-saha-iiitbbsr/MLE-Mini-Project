"""Stage 3 of M2 - turn the validated scan into a versioned train/val/test manifest.

The manifest (``data/processed/manifest.csv``) is the dataset artifact the rest
of the project consumes. Producing a *file* rather than splitting on the fly
matters: the split becomes a reviewable, hashable, DVC-tracked object, so
"which images did run #7 train on" has an exact answer months later.

Two properties the splitter guarantees:

**Stratification.** Class proportions are preserved across all three folds, so
validation metrics are comparable to test metrics.

**Group integrity.** Near-duplicate images (same perceptual hash) are assigned
to the same fold. Industrial datasets photograph the same part repeatedly; if
copies straddle the train/test boundary the test score measures memorisation.
This is the single most common reason an image classifier scores 0.99 offline
and disappoints in production.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..config import class_names, ensure_parent, get, resolve
from ..logging_utils import get_logger

log = get_logger(__name__)

SPLITS = ("train", "val", "test")


def _assign_groups(df: pd.DataFrame, group_by_hash: bool) -> pd.Series:
    """Group key per row: the perceptual hash, or a unique id when grouping is off."""
    if group_by_hash:
        return df["dhash"].fillna("").replace("", pd.NA).fillna(pd.Series(df.index.astype(str)))
    return pd.Series(df.index.astype(str), index=df.index)


def _greedy_group_split(
    groups: list[tuple[Any, int, int]],
    targets: dict[str, float],
    rng: np.random.Generator,
) -> dict[Any, str]:
    """Assign whole groups to folds, keeping each fold near its target size.

    Groups are placed largest-first into whichever fold is furthest below quota
    (a standard greedy bin-packing heuristic). Largest-first matters: placing a
    big group last can overshoot a small fold by a wide margin.

    ``groups`` is ``(group_key, size, label)``; ``targets`` maps fold -> share.
    """
    total = sum(size for _, size, _ in groups)
    quota = {name: share * total for name, share in targets.items()}
    filled = dict.fromkeys(targets, 0.0)
    assignment: dict[Any, str] = {}

    # Shuffle first so equal-sized groups do not always land in the same fold,
    # then sort by size so the packing stays balanced.
    order = list(groups)
    rng.shuffle(order)  # type: ignore[arg-type]
    order.sort(key=lambda g: g[1], reverse=True)

    for key, size, _label in order:
        deficits = {name: quota[name] - filled[name] for name in targets}
        # Prefer the fold with the largest remaining deficit.
        choice = max(deficits, key=lambda n: deficits[n])
        assignment[key] = choice
        filled[choice] += size

    return assignment


def build_manifest(params: dict[str, Any]) -> pd.DataFrame:
    """Read the validation scan, split it, and write the dataset manifest + card."""
    interim = resolve(get(params, "data.interim_dir"))
    scan_path = interim / "scan.csv"
    if not scan_path.exists():
        raise FileNotFoundError(f"{scan_path} not found. Run the `validate` stage first.")

    scan = pd.read_csv(scan_path)
    classes = class_names(params)
    seed = int(get(params, "seed"))
    split_cfg = get(params, "data.split")
    val_size = float(split_cfg["val_size"])
    test_size = float(split_cfg["test_size"])
    stratify = bool(split_cfg.get("stratify", True))
    group_by_hash = bool(split_cfg.get("group_by_hash", True))

    if not 0 < val_size + test_size < 1:
        raise ValueError(f"val_size + test_size must be in (0, 1); got {val_size + test_size}")

    df = scan[scan["is_valid"]].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid images available to split. Check the validation report.")

    # Drop exact duplicates outright -- keeping them inflates the effective
    # weight of whichever part happened to be photographed twice.
    before = len(df)
    df = df.drop_duplicates(subset="sha256", keep="first").reset_index(drop=True)
    n_exact_dropped = before - len(df)
    if n_exact_dropped:
        log.info("Dropped %d exact duplicate(s) before splitting", n_exact_dropped)

    df["group"] = _assign_groups(df, group_by_hash)

    targets = {"train": 1.0 - val_size - test_size, "val": val_size, "test": test_size}
    rng = np.random.default_rng(seed)

    if stratify:
        # Split each class independently so proportions carry into every fold.
        # A group spanning two classes (identical hash, different label) would
        # break this; those are resolved by majority label before splitting.
        group_label = df.groupby("group")["label"].agg(lambda s: int(s.mode().iloc[0]))
        group_size = df.groupby("group").size()
        assignment: dict[Any, str] = {}
        for label, name in enumerate(classes):
            keys = group_label[group_label == label].index
            triples = [(k, int(group_size[k]), label) for k in keys]
            if not triples:
                log.warning("No groups for class %s", name)
                continue
            assignment.update(_greedy_group_split(triples, targets, rng))
    else:
        triples = [(k, int(n), -1) for k, n in df.groupby("group").size().items()]
        assignment = _greedy_group_split(triples, targets, rng)

    df["split"] = df["group"].map(assignment)

    missing = df["split"].isna().sum()
    if missing:  # pragma: no cover - defensive
        raise RuntimeError(f"{missing} rows were not assigned to a split")

    # --- report -----------------------------------------------------------
    pivot = df.pivot_table(index="split", columns="class_name", values="relpath",
                           aggfunc="count", fill_value=0).reindex(SPLITS, fill_value=0)
    log.info("Split sizes:\n%s", pivot.to_string())

    for split in SPLITS:
        sub = df[df["split"] == split]
        if sub.empty:
            log.warning("Split %r is empty -- check val_size/test_size", split)
        else:
            shares = (sub["class_name"].value_counts(normalize=True) * 100).round(1).to_dict()
            log.info("  %-5s n=%-5d class mix: %s", split, len(sub), shares)

    # Leakage assertion: no group may appear in two folds.
    crossing = df.groupby("group")["split"].nunique()
    n_crossing = int((crossing > 1).sum())
    if n_crossing:  # pragma: no cover - defensive
        raise RuntimeError(f"{n_crossing} group(s) span multiple splits -- leakage guard failed")
    log.info("Leakage check passed: %d groups, none spanning folds", len(crossing))

    # --- persist ----------------------------------------------------------
    keep = ["relpath", "class_name", "label", "split", "group", "sha256", "dhash",
            "width", "height", "file_bytes"]
    manifest = df[keep].sort_values(["split", "class_name", "relpath"]).reset_index(drop=True)

    manifest_path = ensure_parent(resolve(get(params, "data.processed_dir")) / "manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    card = {
        "n_images": int(len(manifest)),
        "classes": classes,
        "class_counts": manifest["class_name"].value_counts().to_dict(),
        "split_counts": manifest["split"].value_counts().to_dict(),
        "split_class_matrix": pivot.to_dict(),
        "n_groups": int(manifest["group"].nunique()),
        "exact_duplicates_dropped": int(n_exact_dropped),
        "policy": {
            "seed": seed,
            "val_size": val_size,
            "test_size": test_size,
            "stratify": stratify,
            "group_by_hash": group_by_hash,
        },
        "source": _read_source_record(params),
    }
    card_path = ensure_parent(resolve(get(params, "data.processed_dir")) / "dataset_card.json")
    card_path.write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")

    log.info("Manifest     -> %s (%d rows)", manifest_path, len(manifest))
    log.info("Dataset card -> %s", card_path)
    return manifest


def _read_source_record(params: dict[str, Any]) -> dict[str, Any]:
    """Carry the acquisition provenance into the dataset card."""
    path = resolve(get(params, "data.raw_dir")) / "_source.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:  # pragma: no cover
            pass
    return {}


def load_manifest(params: dict[str, Any], split: str | None = None) -> pd.DataFrame:
    """Read the manifest, optionally filtered to one split."""
    path = resolve(get(params, "data.processed_dir")) / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the `split` stage first.")
    df = pd.read_csv(path)
    if split is not None:
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
        df = df[df["split"] == split].reset_index(drop=True)
    return df
