"""Build the drift reference baseline from the training split.

The reference *must* be the training distribution, not "the first day of
production traffic". The question monitoring answers is "does the model still
see the kind of data it learned from", so the baseline has to be the data it
learned from. Anchoring on early production traffic bakes in whatever was
already wrong on day one.

Stored per feature:

* the raw sample values, so a future window can be KS-tested against the real
  empirical distribution rather than a summary of it;
* **frozen quantile bin edges**, so PSI compares like with like on every window
  (see :mod:`defectvision.monitoring.drift`);
* descriptive statistics, for the human-readable report.

Also stored: the reference *model behaviour* -- score distribution and test
accuracy. Data drift and performance drift are different failures, and catching
the second before labels arrive needs a baseline for what the model's
confidence normally looks like.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from ..config import ensure_parent, get, resolve
from ..data.split import load_manifest
from ..features.image_stats import image_statistics
from ..logging_utils import get_logger
from .drift import quantile_bin_edges

log = get_logger(__name__)

#: Cap on reference rows retained. The full training split would make the JSON
#: unwieldy; a few thousand samples estimate these distributions to well within
#: the precision PSI needs.
MAX_REFERENCE_SAMPLES = 4000


def build_reference(params: dict[str, Any], *, split: str = "train") -> dict[str, Any]:
    """Compute and persist the drift baseline from a manifest split."""
    manifest = load_manifest(params, split)
    if manifest.empty:
        raise ValueError(f"Split {split!r} is empty; cannot build a reference")

    root = resolve(get(params, "data.raw_dir"))
    feature_names = list(get(params, "monitoring.drift_features"))
    n_bins = int(get(params, "monitoring.psi_bins", 10))
    seed = int(get(params, "seed"))

    # The scan table already holds statistics for every image, so reuse them
    # rather than decoding 4700 JPEGs a second time.
    scan_path = resolve(get(params, "data.interim_dir")) / "scan.csv"
    rows: pd.DataFrame | None = None
    if scan_path.exists():
        scan = pd.read_csv(scan_path)
        merged = manifest.merge(scan[["relpath", *feature_names]], on="relpath", how="left")
        if not merged[feature_names].isna().any().any():
            rows = merged
            log.info("Reusing image statistics from %s", scan_path)

    if rows is None:
        log.info("Computing image statistics for %d %s images...", len(manifest), split)
        records = []
        for relpath in manifest["relpath"]:
            with Image.open(root / relpath) as im:
                records.append(image_statistics(im.convert("L")))
        rows = pd.concat([manifest.reset_index(drop=True), pd.DataFrame(records)], axis=1)

    if len(rows) > MAX_REFERENCE_SAMPLES:
        rows = rows.sample(MAX_REFERENCE_SAMPLES, random_state=seed).reset_index(drop=True)
        log.info("Sub-sampled reference to %d rows", MAX_REFERENCE_SAMPLES)

    features: dict[str, Any] = {}
    for name in feature_names:
        values = rows[name].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        edges = quantile_bin_edges(values, n_bins)
        q = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
        features[name] = {
            "samples": [float(v) for v in values],
            # -inf/+inf are not valid JSON; sentinel them and restore on load.
            "bin_edges": [float(e) if np.isfinite(e) else (-1e308 if e < 0 else 1e308)
                          for e in edges],
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "p01": float(q[0]), "p05": float(q[1]), "p25": float(q[2]),
            "p50": float(q[3]), "p75": float(q[4]), "p95": float(q[5]), "p99": float(q[6]),
        }

    reference: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split": split,
        "n_samples": int(len(rows)),
        "psi_bins": n_bins,
        "feature_names": feature_names,
        "features": features,
        "class_distribution": rows["class_name"].value_counts().to_dict(),
    }

    model_baseline = _model_baseline(params)
    if model_baseline:
        reference["model_baseline"] = model_baseline

    path = ensure_parent(get(params, "monitoring.reference_stats"))
    path.write_text(json.dumps(reference, indent=1), encoding="utf-8")
    log.info("Drift reference (%d samples, %d features) -> %s",
             reference["n_samples"], len(features), path)
    return reference


def _model_baseline(params: dict[str, Any]) -> dict[str, Any]:
    """Reference model behaviour, read from the promoted bundle's own metrics."""
    bundle_path = resolve(get(params, "serving.model_bundle"))
    if not bundle_path.is_file():
        log.info("No production bundle yet; reference will carry data statistics only")
        return {}

    from ..bundle import load_bundle

    model = load_bundle(bundle_path)
    test_metrics = (model.metrics or {}).get("test", {})
    return {
        "model_name": model.model_name,
        "mlflow_run_id": model.mlflow_run_id,
        "threshold": model.threshold,
        # Baselines the online detectors compare against before labels arrive.
        "reference_accuracy": float(test_metrics.get("accuracy", float("nan"))),
        "reference_f1": float(test_metrics.get("f1", float("nan"))),
        "reference_recall": float(test_metrics.get("recall", float("nan"))),
        "reference_predicted_defect_rate": float(
            test_metrics.get("predicted_positive_rate", float("nan"))
        ),
    }


def load_reference(params: dict[str, Any]) -> dict[str, Any]:
    """Load the persisted baseline, restoring infinite bin edges."""
    path = resolve(get(params, "monitoring.reference_stats"))
    if not path.is_file():
        raise FileNotFoundError(
            f"Drift reference not found at {path}. Build it with:\n"
            "    defectvision reference-stats"
        )
    reference = json.loads(path.read_text(encoding="utf-8"))

    for entry in reference.get("features", {}).values():
        edges = entry.get("bin_edges")
        if edges:
            entry["bin_edges"] = [
                -np.inf if e <= -1e307 else (np.inf if e >= 1e307 else e) for e in edges
            ]
    return reference


def reference_model_baseline(reference: dict[str, Any]) -> dict[str, Any]:
    """Convenience accessor with sane defaults when no model was promoted yet."""
    return reference.get("model_baseline", {})
