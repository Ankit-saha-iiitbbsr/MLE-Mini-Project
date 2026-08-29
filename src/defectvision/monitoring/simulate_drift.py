"""Deliberately shift the input distribution and watch what the system does (M5).

A monitoring stack that has never seen drift is untested code on the most
important path. This module manufactures shift on purpose, pushes it through
the *same* prediction path the API uses, and records what the detectors and the
model actually did.

Scenarios come from two places:

* **Synthetic** - the operators in ``drift_simulation.scenarios``, each modelling
  a specific cell failure (lamp drift, camera remount, focus loss, sensor
  noise). Controlled and repeatable, so a detector's sensitivity can be
  measured against a known corruption magnitude.
* **Real** - the ``casting_512x512`` capture that :mod:`defectvision.data.acquire`
  deliberately holds back. It is the same production line photographed by a
  different camera at a different resolution: an authentic covariate shift that
  no corruption operator was tuned against. Detectors that only fire on the
  synthetic scenarios but not on this one are overfitted to the simulation.

A ``baseline`` control (uncorrupted test images) always runs first. Without it,
a non-zero PSI cannot be attributed to the corruption rather than to ordinary
sampling noise between the reference and any fresh sample.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from ..config import ensure_dir, ensure_parent, get, resolve
from ..data.corruptions import apply_scenario, describe_scenario
from ..data.split import load_manifest
from ..logging_utils import get_logger
from ..serving.predictor import Predictor
from ..training.evaluate import classification_metrics
from .store import PredictionRecord, PredictionStore

log = get_logger(__name__)

BASELINE = "baseline"
REAL_SHIFT = "real_camera_upgrade"

#: Images per scenario. Enough for stable PSI (which needs a few hundred),
#: small enough that the whole sweep runs in about a minute on CPU.
DEFAULT_SAMPLE_SIZE = 400


def _sample_test_images(params: dict[str, Any], n: int, seed: int) -> pd.DataFrame:
    """A stratified sample of the test split -- never of train.

    Using test images means the ground truth is known, so the simulation can
    report the *actual* accuracy drop each shift causes, not merely that the
    inputs looked different.
    """
    manifest = load_manifest(params, "test")
    if len(manifest) <= n:
        return manifest.reset_index(drop=True)

    # Sample each class in proportion to its share, so the shifted set has the
    # same class mix as the test split and accuracy stays comparable.
    parts = []
    for _, group in manifest.groupby("class_name", sort=True):
        take = max(1, int(round(n * len(group) / len(manifest))))
        parts.append(group.sample(min(take, len(group)), random_state=seed))
    return pd.concat(parts).sort_index().reset_index(drop=True)


def _real_shift_images(params: dict[str, Any], n: int, seed: int) -> list[tuple[Path, int]]:
    """Locate the held-back 512x512 capture, if it is available."""
    local = get(params, "data.kaggle_local_dir", None)
    if not local:
        return []
    roots = [p for p in resolve(local).rglob("casting_512x512") if p.is_dir()]
    if not roots:
        return []
    root = max(roots, key=lambda p: len(p.parts))

    class_map = {"ok_front": 0, "def_front": 1}
    files: list[tuple[Path, int]] = []
    for folder, label in class_map.items():
        directory = root / folder
        if directory.is_dir():
            files.extend((p, label) for p in sorted(directory.glob("*"))
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not files:
        return []

    rng = np.random.default_rng(seed)
    if len(files) > n:
        idx = rng.choice(len(files), size=n, replace=False)
        files = [files[i] for i in sorted(idx)]
    log.info("Real shift set: %d images from %s", len(files), root)
    return files


def _score_batch(
    predictor: Predictor,
    images: list[tuple[Image.Image, int, str]],
    scenario_name: str,
    store: PredictionStore,
    *,
    send_to_api: bool = False,
    base_url: str = "http://127.0.0.1:8000",
) -> tuple[np.ndarray, np.ndarray]:
    """Score images and log them under *scenario_name*. Returns ``(probs, labels)``."""
    probabilities: list[float] = []
    truths: list[int] = []
    records: list[PredictionRecord] = []

    if send_to_api:
        import httpx

        with httpx.Client(base_url=base_url, timeout=30.0) as client:
            for image, label, name in images:
                import io

                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                buffer.seek(0)
                response = client.post(
                    "/predict",
                    files={"file": (name, buffer.getvalue(), "image/png")},
                    data={"scenario": scenario_name},
                )
                response.raise_for_status()
                body = response.json()
                probabilities.append(float(body["probability_defect"]))
                truths.append(label)
                # The service already logged the row; attach the label we know.
                store.attach_ground_truth(body["request_id"], label, source="drift_sim")
        return np.array(probabilities), np.array(truths)

    for image, label, name in images:
        outcome = predictor.predict_image(
            image, filename=name, source="drift_sim", scenario=scenario_name,
            ground_truth=label, persist=False,
        )
        probabilities.append(outcome.probability_defect)
        truths.append(label)
        records.append(PredictionRecord(
            request_id=outcome.request_id,
            probability=outcome.probability_defect,
            predicted_label=outcome.predicted_label,
            predicted_class=outcome.predicted_class,
            decision=outcome.decision,
            latency_ms=outcome.latency_ms,
            model_name=predictor.model.model_name,  # type: ignore[union-attr]
            model_run_id=predictor.model.mlflow_run_id,  # type: ignore[union-attr]
            model_threshold=outcome.threshold,
            source="drift_sim",
            scenario=scenario_name,
            filename=name,
            width=outcome.width,
            height=outcome.height,
            file_bytes=outcome.file_bytes,
            image_stats=outcome.image_stats,
            ground_truth=label,
        ))

    # One bulk insert rather than a few hundred round-trips.
    store.log_many(records)
    return np.array(probabilities), np.array(truths)


def run_simulation(
    params: dict[str, Any],
    *,
    scenario: str | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    send_to_api: bool = False,
    base_url: str = "http://127.0.0.1:8000",
    include_real: bool = True,
) -> dict[str, Any]:
    """Run the drift sweep and return a structured report."""
    seed = int(get(params, "seed"))
    raw_root = resolve(get(params, "data.raw_dir"))
    scenarios: dict[str, dict[str, Any]] = dict(get(params, "drift_simulation.scenarios"))

    if scenario is not None and scenario not in scenarios and scenario not in (BASELINE, REAL_SHIFT):
        raise ValueError(
            f"Unknown scenario {scenario!r}. Available: "
            f"{[BASELINE, *sorted(scenarios), REAL_SHIFT]}"
        )

    predictor = Predictor(params)
    if not predictor.load():
        raise RuntimeError(
            f"No production model to evaluate ({predictor.load_error}). "
            "Run: defectvision train --all && defectvision package"
        )

    store = PredictionStore(get(params, "monitoring.db_path"))
    sample = _sample_test_images(params, sample_size, seed)
    log.info("Drift sweep over %d test images per scenario", len(sample))

    # Decode once; every scenario corrupts the same source images so the only
    # difference between scenarios is the corruption itself.
    originals: list[tuple[Image.Image, int, str]] = []
    for row in sample.itertuples():
        with Image.open(raw_root / row.relpath) as im:
            originals.append((im.convert("L").copy(), int(row.label), Path(row.relpath).name))

    to_run: list[tuple[str, dict[str, Any]]] = [(BASELINE, {})]
    if scenario is None:
        to_run += sorted(scenarios.items())
    elif scenario != BASELINE:
        to_run = [(scenario, scenarios.get(scenario, {}))]

    figures_dir = ensure_dir("reports/figures/drift")
    results: dict[str, Any] = {}
    baseline_metrics: dict[str, float] | None = None

    for name, spec in to_run:
        started = time.perf_counter()
        # Re-running a scenario replaces its rows rather than appending, so the
        # log always reflects one sweep and PSI is not diluted by stale windows.
        store.clear(scenario=name)

        corrupted = [
            (apply_scenario(image, spec, seed=seed + i) if spec else image, label, fname)
            for i, (image, label, fname) in enumerate(originals)
        ]
        probs, truths = _score_batch(predictor, corrupted, name, store,
                                     send_to_api=send_to_api, base_url=base_url)

        threshold = predictor.model.threshold  # type: ignore[union-attr]
        metrics = classification_metrics(truths, probs, threshold)
        if name == BASELINE:
            baseline_metrics = metrics

        _save_examples(originals, corrupted, name, figures_dir)

        results[name] = {
            "scenario": name,
            "description": describe_scenario(spec) if spec else "unmodified test images (control)",
            "parameters": spec,
            "n_images": len(corrupted),
            "metrics": metrics,
            "mean_confidence": float(np.mean(np.maximum(probs, 1.0 - probs))),
            "mean_probability": float(probs.mean()),
            "predicted_defect_rate": float((probs >= threshold).mean()),
            "seconds": time.perf_counter() - started,
        }
        if baseline_metrics is not None and name != BASELINE:
            results[name]["accuracy_drop"] = baseline_metrics["accuracy"] - metrics["accuracy"]
            results[name]["recall_drop"] = baseline_metrics["recall"] - metrics["recall"]
            results[name]["f1_drop"] = baseline_metrics["f1"] - metrics["f1"]

        log.info("%-20s acc=%.4f recall=%.4f f1=%.4f  (%s)",
                 name, metrics["accuracy"], metrics["recall"], metrics["f1"],
                 results[name]["description"])

    # --- real, un-simulated shift ----------------------------------------
    if include_real and scenario in (None, REAL_SHIFT):
        real_files = _real_shift_images(params, sample_size, seed)
        if real_files:
            store.clear(scenario=REAL_SHIFT)
            real_images = []
            for path, label in real_files:
                with Image.open(path) as im:
                    real_images.append((im.convert("L").copy(), label, path.name))
            probs, truths = _score_batch(predictor, real_images, REAL_SHIFT, store,
                                         send_to_api=send_to_api, base_url=base_url)
            threshold = predictor.model.threshold  # type: ignore[union-attr]
            metrics = classification_metrics(truths, probs, threshold)
            results[REAL_SHIFT] = {
                "scenario": REAL_SHIFT,
                "description": "REAL shift: casting_512x512 capture (different camera/resolution)",
                "parameters": {"source": "casting_512x512"},
                "n_images": len(real_images),
                "metrics": metrics,
                "mean_confidence": float(np.mean(np.maximum(probs, 1.0 - probs))),
                "mean_probability": float(probs.mean()),
                "predicted_defect_rate": float((probs >= threshold).mean()),
                "is_real_shift": True,
            }
            if baseline_metrics is not None:
                results[REAL_SHIFT]["accuracy_drop"] = (
                    baseline_metrics["accuracy"] - metrics["accuracy"])
                results[REAL_SHIFT]["recall_drop"] = baseline_metrics["recall"] - metrics["recall"]
                results[REAL_SHIFT]["f1_drop"] = baseline_metrics["f1"] - metrics["f1"]
            log.info("%-20s acc=%.4f recall=%.4f f1=%.4f  (REAL camera shift)",
                     REAL_SHIFT, metrics["accuracy"], metrics["recall"], metrics["f1"])
        else:
            log.info("casting_512x512 not available; skipping the real-shift scenario")

    summary = {
        name: {
            "accuracy": round(r["metrics"]["accuracy"], 4),
            "recall": round(r["metrics"]["recall"], 4),
            "f1": round(r["metrics"]["f1"], 4),
            "accuracy_drop": round(r.get("accuracy_drop", 0.0), 4),
            "mean_confidence": round(r["mean_confidence"], 4),
        }
        for name, r in results.items()
    }

    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": predictor.model.info(),  # type: ignore[union-attr]
        "sample_size": len(sample),
        "scenarios": results,
        "summary": summary,
    }
    path = ensure_parent("reports/drift_simulation.json")
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info("Drift simulation report -> %s", path)
    return report


def _save_examples(
    originals: list[tuple[Image.Image, int, str]],
    corrupted: list[tuple[Image.Image, int, str]],
    name: str,
    out_dir: Path,
    n: int = 4,
) -> Path:
    """Save a before/after strip so the report shows what each shift looks like."""
    top = np.hstack([np.asarray(img.resize((128, 128))) for img, _, _ in originals[:n]])
    bottom = np.hstack([np.asarray(img.resize((128, 128))) for img, _, _ in corrupted[:n]])
    path = out_dir / f"{name}.png"
    Image.fromarray(np.vstack([top, bottom])).save(path)
    return path
