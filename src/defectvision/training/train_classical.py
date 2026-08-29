"""Training path for the classical HOG + logistic-regression control arm.

Kept separate from :mod:`defectvision.training.train` because a scikit-learn
pipeline shares almost nothing with the torch loop -- no epochs, no batches, no
early stopping. Forcing both through one abstraction would obscure both.

What it *does* share is the parts that make the comparison fair and the run
citable: the same manifest and folds, the same preprocessing geometry, the same
validation-tuned threshold policy, the same metric set, and the same MLflow
schema. Only the estimator differs, which is the point of a control.

This arm is deliberately **not** deployable through the standard bundle: the
packaging step (M4) handles torch bundles only. Promoting it would mean
shipping a second serving path, and the measured gap does not justify that.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..config import class_names, ensure_dir, get, resolve
from ..data.split import load_manifest
from ..data.transforms import PreprocessSpec
from ..logging_utils import banner, get_logger, set_seed
from ..models.classical import build_classical_model, hog_feature_dim, hog_features
from . import reproducibility
from .evaluate import (
    bootstrap_ci,
    choose_threshold,
    classification_metrics,
    save_evaluation_plots,
)

log = get_logger(__name__)


def _load_split_features(
    params: dict[str, Any], split: str, spec: PreprocessSpec
) -> tuple[np.ndarray, np.ndarray]:
    """HOG descriptors + labels for one split, at the same geometry the CNNs see."""
    root = resolve(get(params, "data.raw_dir"))
    rows = load_manifest(params, split)
    if rows.empty:
        raise ValueError(f"Split {split!r} is empty")

    features: list[np.ndarray] = []
    for relpath in rows["relpath"]:
        with Image.open(root / relpath) as im:
            gray = im.convert("L").resize((spec.resize, spec.resize), Image.Resampling.BILINEAR)
            arr = np.asarray(gray, dtype=np.float64) / 255.0
        features.append(hog_features(arr))

    return np.vstack(features), rows["label"].to_numpy(dtype=np.int64)


def train_classical(params: dict[str, Any], model_name: str = "logreg_hog"):
    """Fit the classical baseline and log it as a comparable MLflow run."""
    from .train import TrainingResult, setup_mlflow  # local import avoids a cycle

    banner(log, f"M3 | Training '{model_name}' (classical control: HOG + logistic regression)")

    seed = int(get(params, "seed"))
    set_seed(seed)

    model_cfg = dict(get(params, f"train.models.{model_name}"))
    spec = PreprocessSpec.from_params(params)
    classes = class_names(params)
    provenance = reproducibility.capture(params)

    t_start = time.perf_counter()

    log.info("Extracting HOG descriptors (dim=%d per image)...",
             hog_feature_dim(spec.resize))
    x_train, y_train = _load_split_features(params, "train", spec)
    x_val, y_val = _load_split_features(params, "val", spec)
    x_test, y_test = _load_split_features(params, "test", spec)
    log.info("Feature matrices: train=%s val=%s test=%s",
             x_train.shape, x_val.shape, x_test.shape)

    pipeline = build_classical_model(model_cfg)

    mlflow = setup_mlflow(params)
    run_name = f"{model_name}-{time.strftime('%Y%m%d-%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        mlflow.log_params({
            "model_name": model_name,
            "arch": "logreg_hog",
            "seed": seed,
            "C": float(model_cfg.get("C", 1.0)),
            "solver": "liblinear",
            "class_weighting": "balanced",
            "hog_cell": 16, "hog_bins": 9, "hog_block": 2,
            "feature_dim": int(x_train.shape[1]),
            "input_size": spec.resize,
        })
        mlflow.set_tags({
            "module": "M3",
            "task": "defect-classification",
            "model_family": "classical",
            "deployable": "false",
            **{k: str(v) for k, v in provenance.items() if k.startswith("git")},
            "dataset_manifest_sha256": str(provenance.get("manifest_sha256")),
        })
        mlflow.log_dict(provenance, "reproducibility/environment.json")

        with_fit = time.perf_counter()
        pipeline.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - with_fit
        log.info("Fitted in %.1fs", fit_seconds)

        val_probs = pipeline.predict_proba(x_val)[:, 1]
        eval_cfg = get(params, "evaluate")
        choice = choose_threshold(
            y_val, val_probs,
            strategy=str(eval_cfg["threshold_strategy"]),
            fixed_threshold=float(eval_cfg["fixed_threshold"]),
            target_recall=float(eval_cfg["target_recall"]),
        )
        log.info("Operating threshold %.4f -- %s", choice.threshold, choice.rationale)

        test_probs = pipeline.predict_proba(x_test)[:, 1]
        test_metrics = classification_metrics(y_test, test_probs, choice.threshold)
        test_ci = {
            m: bootstrap_ci(y_test, test_probs, choice.threshold, m,
                            n_samples=int(eval_cfg.get("bootstrap_samples", 500)), seed=seed)
            for m in ("f1", "recall", "precision", "accuracy")
        }

        # Latency here is dominated by descriptor extraction, not by the linear
        # model, so it is measured over the full image -> decision path.
        sample = np.random.default_rng(seed).random((spec.resize, spec.resize))
        timings = []
        for _ in range(20):
            t0 = time.perf_counter()
            pipeline.predict_proba(hog_features(sample).reshape(1, -1))
            timings.append((time.perf_counter() - t0) * 1000.0)
        arr = np.array(timings)
        latency = {
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "throughput_single_ips": float(1000.0 / arr.mean()),
            "throughput_batched_ips": float(1000.0 / arr.mean()),
        }

        mlflow.log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items()
                            if isinstance(v, (int, float))})
        mlflow.log_metrics({f"val_final_{k}": float(v)
                            for k, v in choice.validation_metrics.items()
                            if isinstance(v, (int, float))})
        mlflow.log_metrics({f"latency_{k}": float(v) for k, v in latency.items()})
        # Logged so the MLflow-derived comparison table matches the in-memory
        # one for this same run. `best_epoch` has no meaning for a one-shot fit,
        # but the column exists for the deep arms and an absent metric comes
        # back as NaN from search_runs -- 0 is both truthful here and what
        # TrainingResult reports.
        mlflow.log_metrics({
            "train_seconds": float(fit_seconds),
            "total_params": float(x_train.shape[1] + 1),
            "best_epoch": 0.0,
        })
        mlflow.log_dict(test_ci, "metrics/test_bootstrap_ci.json")
        mlflow.log_dict(choice.to_dict(), "metrics/threshold_choice.json")

        fig_dir = ensure_dir(Path("reports/figures") / model_name)
        for p in save_evaluation_plots(y_test, test_probs, choice.threshold, fig_dir, "test"):
            mlflow.log_artifact(p, artifact_path=f"figures/{model_name}")

        # Persisted for inspection/reruns only -- not a deployment artifact.
        import joblib

        out_path = ensure_dir(resolve(f"models/candidates/{model_name}")) / "sklearn_pipeline.joblib"
        joblib.dump({"pipeline": pipeline, "threshold": choice.threshold,
                     "classes": classes, "preprocess": spec.to_dict()}, out_path)
        mlflow.log_artifact(str(out_path), artifact_path="model_bundle")

        log.info("TEST  f1=%.4f recall=%.4f precision=%.4f acc=%.4f auc=%.4f",
                 test_metrics["f1"], test_metrics["recall"], test_metrics["precision"],
                 test_metrics["accuracy"], test_metrics["roc_auc"])

    n_features = int(x_train.shape[1])
    return TrainingResult(
        model_name=model_name,
        arch="logreg_hog",
        run_id=run_id,
        best_epoch=0,
        threshold=choice.threshold,
        threshold_rationale=choice.rationale,
        val_metrics=choice.validation_metrics,
        test_metrics=test_metrics,
        test_ci=test_ci,
        latency=latency,
        model_info={"model_class": "Pipeline(StandardScaler, LogisticRegression)",
                    "arch": "logreg_hog",
                    "total_params": n_features + 1,
                    "trainable_params": n_features + 1,
                    "param_size_mb": round((n_features + 1) * 8 / (1024 ** 2), 4),
                    "pretrained_loaded": False,
                    "frozen_backbone": False},
        history=[],
        bundle_path=str(out_path),
        train_seconds=time.perf_counter() - t_start,
        provenance=provenance,
    )
