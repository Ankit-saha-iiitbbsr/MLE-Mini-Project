"""M3 - the training loop, wrapped in experiment tracking.

One call to :func:`train_model` produces a complete, self-describing experiment:
a tracked MLflow run carrying the config, the dataset fingerprint, the git
commit, per-epoch curves, evaluation plots, a latency profile, and a deployable
bundle. Comparing models then means comparing runs, not comparing notebook
cells someone re-executed in a different order.

Sequencing that matters:

* The model is selected on **validation** loss/metric via early stopping, the
  threshold is tuned on **validation**, and only then is the test set touched
  -- once. Test data influences no decision, so the reported number is an
  honest estimate of production performance.
* Latency is measured as part of every run, not as an afterthought. A model
  that is 0.4 points better and three times slower is often the wrong choice,
  and that trade-off is only visible if both numbers live on the same run.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..bundle import BundleMetadata, save_bundle
from ..config import class_names, ensure_dir, get, resolve
from ..data.dataset import build_dataloaders
from ..logging_utils import banner, get_logger, set_seed
from ..models.factory import build_model, describe_model
from . import reproducibility
from .evaluate import (
    bootstrap_ci,
    choose_threshold,
    classification_metrics,
    save_evaluation_plots,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Everything one experiment produced."""

    model_name: str
    arch: str
    run_id: str | None
    best_epoch: int
    threshold: float
    threshold_rationale: str
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    test_ci: dict[str, Any]
    latency: dict[str, float]
    model_info: dict[str, Any]
    history: list[dict[str, float]]
    bundle_path: str
    train_seconds: float
    provenance: dict[str, Any] = field(default_factory=dict)

    def summary_row(self) -> dict[str, Any]:
        """One row for the model-comparison table."""
        return {
            "model": self.model_name,
            "arch": self.arch,
            "run_id": (self.run_id or "")[:8],
            "params_M": round(self.model_info.get("total_params", 0) / 1e6, 3),
            "threshold": round(self.threshold, 4),
            "test_f1": round(self.test_metrics.get("f1", float("nan")), 4),
            "test_recall": round(self.test_metrics.get("recall", float("nan")), 4),
            "test_precision": round(self.test_metrics.get("precision", float("nan")), 4),
            "test_accuracy": round(self.test_metrics.get("accuracy", float("nan")), 4),
            "test_roc_auc": round(self.test_metrics.get("roc_auc", float("nan")), 4),
            "latency_p95_ms": round(self.latency.get("p95_ms", float("nan")), 2),
            "throughput_ips": round(self.latency.get("throughput_batched_ips", float("nan")), 1),
            "train_s": round(self.train_seconds, 1),
            "best_epoch": self.best_epoch,
        }


# ---------------------------------------------------------------------------
# Optimisation helpers
# ---------------------------------------------------------------------------


def _build_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float):
    params = [p for p in model.parameters() if p.requires_grad]
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay,
                               nesterov=True)
    raise ValueError(f"Unknown optimizer {name!r}; expected adamw, adam or sgd")


def _build_scheduler(optimizer, name: str, epochs: int):
    name = str(name).lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.1)
    if name in ("none", "", "null"):
        return None
    raise ValueError(f"Unknown scheduler {name!r}; expected cosine, step or none")


def _pos_weight(class_weights: torch.Tensor) -> torch.Tensor:
    """Convert a 2-class weight vector into BCE ``pos_weight``.

    ``BCEWithLogitsLoss`` takes a single scalar scaling the positive term, so the
    ratio of the two class weights is what carries the imbalance correction.
    """
    return torch.tensor([float(class_weights[1] / class_weights[0])], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Epoch loop
# ---------------------------------------------------------------------------


def _run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
    max_batches: int | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one pass. Training when *optimizer* is given, evaluation otherwise."""
    training = optimizer is not None
    model.train(training)

    total_loss, n_seen = 0.0, 0
    probs_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for i, (images, labels) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            targets = labels.to(device, non_blocking=True).float()

            logits = model(images)
            loss = criterion(logits, targets)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Clipping guards the from-scratch arm against the occasional
                # exploding step in the first epochs, when BatchNorm statistics
                # are still settling.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            batch_n = images.size(0)
            total_loss += float(loss.item()) * batch_n
            n_seen += batch_n
            probs_all.append(torch.sigmoid(logits.detach()).cpu().numpy())
            labels_all.append(labels.numpy())

    if n_seen == 0:  # pragma: no cover - empty loader
        return 0.0, np.empty(0), np.empty(0)
    return (total_loss / n_seen,
            np.concatenate(probs_all).astype(np.float64),
            np.concatenate(labels_all).astype(np.int64))


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


def measure_latency(
    model: nn.Module,
    input_size: int,
    in_channels: int,
    device: torch.device,
    *,
    n_warmup: int = 10,
    n_trials: int = 60,
    batch_size: int = 16,
) -> dict[str, float]:
    """Profile single-image latency and batched throughput.

    Both numbers matter and they answer different questions. Single-image
    latency is what an in-line trigger experiences per part; batched throughput
    is what an offline sweep of a shift's images achieves. Warm-up iterations
    are discarded because the first passes pay lazy allocation and kernel
    selection costs that do not recur.
    """
    model.eval()
    sample = torch.randn(1, in_channels, input_size, input_size, device=device)
    batch = torch.randn(batch_size, in_channels, input_size, input_size, device=device)

    with torch.inference_mode():
        for _ in range(n_warmup):
            model(sample)

        timings: list[float] = []
        for _ in range(n_trials):
            start = time.perf_counter()
            model(sample)
            timings.append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        for _ in range(5):
            model(batch)
        batched_s = (time.perf_counter() - start) / 5.0

    arr = np.array(timings)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "throughput_single_ips": float(1000.0 / arr.mean()) if arr.mean() > 0 else 0.0,
        "throughput_batched_ips": float(batch_size / batched_s) if batched_s > 0 else 0.0,
        "batch_size": float(batch_size),
    }


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------


def _flatten(prefix: str, node: Any, out: dict[str, Any]) -> None:
    """Flatten nested config into MLflow's flat param namespace."""
    if isinstance(node, dict):
        for k, v in node.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(node, (list, tuple)):
        out[prefix] = json.dumps(list(node))
    else:
        out[prefix] = node


def setup_mlflow(params: dict[str, Any]):
    """Point MLflow at the configured store and experiment; return the module.

    Store locations in ``params.yaml`` are written relative to the repository,
    then anchored to the project root here. Without that anchoring the tracking
    store would land wherever the caller happened to be, so a run launched by
    DVC and a run launched by hand would write to two different databases.
    """
    import mlflow

    uri = str(get(params, "mlflow.tracking_uri"))
    if uri.startswith("sqlite:///"):
        raw = uri[len("sqlite:///"):]
        if not Path(raw).is_absolute():
            db = resolve(raw)
            db.parent.mkdir(parents=True, exist_ok=True)
            uri = f"sqlite:///{db.as_posix()}"
    elif uri.startswith("file:./"):
        uri = resolve(uri[len("file:./"):]).as_uri()
    mlflow.set_tracking_uri(uri)

    experiment_name = str(get(params, "mlflow.experiment_name"))
    if mlflow.get_experiment_by_name(experiment_name) is None:
        artifact_root = ensure_dir(get(params, "mlflow.artifact_location", "mlartifacts"))
        mlflow.create_experiment(experiment_name, artifact_location=artifact_root.as_uri())
    mlflow.set_experiment(experiment_name)
    return mlflow


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def train_model(
    params: dict[str, Any],
    model_name: str,
    *,
    run_name: str | None = None,
    extra_tags: dict[str, str] | None = None,
) -> TrainingResult:
    """Train one configured model arm end-to-end and log it as an MLflow run."""
    models_cfg = get(params, "train.models")
    if model_name not in models_cfg:
        raise KeyError(
            f"Unknown model {model_name!r}. Configured arms: {sorted(models_cfg)}"
        )
    model_cfg = dict(models_cfg[model_name])
    arch = str(model_cfg.get("arch", model_name))

    if arch == "logreg_hog":
        from .train_classical import train_classical
        return train_classical(params, model_name)

    banner(log, f"M3 | Training '{model_name}' (arch={arch})")

    seed = int(get(params, "seed"))
    set_seed(seed)

    train_cfg = get(params, "train")
    epochs = int(model_cfg.get("epochs", train_cfg["epochs"]))
    lr = float(model_cfg.get("learning_rate", train_cfg["learning_rate"]))
    weight_decay = float(model_cfg.get("weight_decay", train_cfg["weight_decay"]))
    patience = int(train_cfg.get("early_stopping_patience", 5))
    monitor = str(train_cfg.get("monitor_metric", "f1"))
    max_batches = train_cfg.get("max_train_batches")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = class_names(params)
    provenance = reproducibility.capture(params)

    # --- data -------------------------------------------------------------
    loaders, spec, class_weights = build_dataloaders(params)
    if "val" not in loaders or "test" not in loaders:
        raise ValueError("Training requires both a val and a test split in the manifest")

    # --- model ------------------------------------------------------------
    model = build_model(model_cfg, in_channels=spec.channels).to(device)
    model_info = describe_model(model)
    log.info("Model: %(model_class)s  params=%(total_params)s  trainable=%(trainable_params)s",
             model_info)

    criterion = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(class_weights).to(device))
    optimizer = _build_optimizer(model, str(train_cfg["optimizer"]), lr, weight_decay)
    scheduler = _build_scheduler(optimizer, train_cfg.get("scheduler", "cosine"), epochs)

    mlflow = setup_mlflow(params)
    run_name = run_name or f"{model_name}-{time.strftime('%Y%m%d-%H%M%S')}"

    history: list[dict[str, float]] = []
    best_score = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    t_start = time.perf_counter()

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # --- log configuration -------------------------------------------
        flat: dict[str, Any] = {}
        _flatten("", {"data": get(params, "data"), "preprocess": get(params, "preprocess"),
                      "augment": get(params, "augment"), "evaluate": get(params, "evaluate")}, flat)
        _flatten("model", model_cfg, flat)
        mlflow.log_params({
            **{k: v for k, v in flat.items() if v is not None},
            "model_name": model_name,
            "arch": arch,
            "seed": seed,
            "epochs": epochs,
            "batch_size": int(train_cfg["batch_size"]),
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "optimizer": train_cfg["optimizer"],
            "scheduler": train_cfg.get("scheduler", "cosine"),
            "class_weighting": train_cfg.get("class_weighting", "balanced"),
            "device": str(device),
        })
        mlflow.set_tags({
            "module": "M3",
            "task": "defect-classification",
            "model_family": "transfer" if arch != "baseline_cnn" else "from_scratch",
            **{k: str(v) for k, v in provenance.items() if k.startswith("git")},
            "dataset_manifest_sha256": str(provenance.get("manifest_sha256")),
            "params_config_hash": str(provenance.get("params_config_hash")),
            **(extra_tags or {}),
        })
        mlflow.log_dict(provenance, "reproducibility/environment.json")
        mlflow.log_dict(params, "reproducibility/params_snapshot.json")
        mlflow.log_dict(model_info, "model/architecture.json")

        # --- training loop ------------------------------------------------
        for epoch in range(1, epochs + 1):
            t_epoch = time.perf_counter()
            train_loss, train_probs, train_labels = _run_epoch(
                model, loaders["train"], criterion, device, optimizer, max_batches
            )
            val_loss, val_probs, val_labels = _run_epoch(
                model, loaders["val"], criterion, device
            )
            if scheduler is not None:
                scheduler.step()

            train_m = classification_metrics(train_labels, train_probs, 0.5)
            val_m = classification_metrics(val_labels, val_probs, 0.5)
            score = val_m[monitor]

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_f1": train_m["f1"],
                "val_f1": val_m["f1"],
                "val_recall": val_m["recall"],
                "val_precision": val_m["precision"],
                "val_accuracy": val_m["accuracy"],
                "val_roc_auc": val_m["roc_auc"],
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.perf_counter() - t_epoch,
            }
            history.append(row)
            mlflow.log_metrics({k: float(v) for k, v in row.items() if k != "epoch"}, step=epoch)

            log.info(
                "epoch %2d/%d | train_loss=%.4f val_loss=%.4f | val_f1=%.4f "
                "val_recall=%.4f val_auc=%.4f | %.1fs",
                epoch, epochs, train_loss, val_loss, val_m["f1"], val_m["recall"],
                val_m["roc_auc"], row["epoch_seconds"],
            )

            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    log.info("Early stopping at epoch %d (no val %s gain for %d epochs)",
                             epoch, monitor, patience)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        log.info("Best epoch: %d (val %s = %.4f)", best_epoch, monitor, best_score)

        # --- threshold selection on VALIDATION ----------------------------
        _, val_probs, val_labels = _run_epoch(model, loaders["val"], criterion, device)
        eval_cfg = get(params, "evaluate")
        choice = choose_threshold(
            val_labels, val_probs,
            strategy=str(eval_cfg["threshold_strategy"]),
            fixed_threshold=float(eval_cfg["fixed_threshold"]),
            target_recall=float(eval_cfg["target_recall"]),
        )
        log.info("Operating threshold %.4f -- %s", choice.threshold, choice.rationale)

        # --- final evaluation on TEST (single use) ------------------------
        _, test_probs, test_labels = _run_epoch(model, loaders["test"], criterion, device)
        test_metrics = classification_metrics(test_labels, test_probs, choice.threshold)
        test_ci = {
            m: bootstrap_ci(test_labels, test_probs, choice.threshold, m,
                            n_samples=int(eval_cfg.get("bootstrap_samples", 500)), seed=seed)
            for m in ("f1", "recall", "precision", "accuracy")
        }

        latency = measure_latency(model, spec.resize, spec.channels, device)
        train_seconds = time.perf_counter() - t_start

        mlflow.log_metrics({f"val_final_{k}": float(v)
                            for k, v in choice.validation_metrics.items()
                            if isinstance(v, (int, float))})
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items()
                            if isinstance(v, (int, float))})
        mlflow.log_metrics({f"latency_{k}": float(v) for k, v in latency.items()})
        mlflow.log_metrics({
            "best_epoch": float(best_epoch),
            "train_seconds": float(train_seconds),
            "total_params": float(model_info["total_params"]),
        })
        mlflow.log_dict(dict(test_ci), "metrics/test_bootstrap_ci.json")
        mlflow.log_dict(choice.to_dict(), "metrics/threshold_choice.json")
        mlflow.log_dict({"history": history}, "metrics/history.json")

        # --- artifacts ----------------------------------------------------
        fig_dir = ensure_dir(Path("reports/figures") / model_name)
        plots = save_evaluation_plots(test_labels, test_probs, choice.threshold, fig_dir, "test")
        plots.append(str(_plot_history(history, fig_dir)))
        for p in plots:
            mlflow.log_artifact(p, artifact_path=f"figures/{model_name}")

        # --- deployable bundle --------------------------------------------
        meta = BundleMetadata(
            model_name=model_name,
            arch=arch,
            model_config=model_cfg,
            preprocess=spec.to_dict(),
            classes=classes,
            threshold=choice.threshold,
            threshold_strategy=choice.strategy,
            in_channels=spec.channels,
            metrics={"val": choice.validation_metrics, "test": test_metrics,
                     "test_ci": test_ci, "latency": latency},
            provenance={**provenance, "best_epoch": best_epoch,
                        "threshold_rationale": choice.rationale},
            mlflow_run_id=run_id,
        )
        bundle_path = save_bundle(
            resolve(f"models/candidates/{model_name}/model_bundle.pt"), model, meta
        )
        mlflow.log_artifact(str(bundle_path), artifact_path="model_bundle")

        repro_cmd = reproducibility.reproduction_command(provenance, model_name)
        mlflow.log_text(repro_cmd, "reproducibility/how_to_reproduce.txt")

        log.info("TEST  f1=%.4f recall=%.4f precision=%.4f acc=%.4f auc=%.4f",
                 test_metrics["f1"], test_metrics["recall"], test_metrics["precision"],
                 test_metrics["accuracy"], test_metrics["roc_auc"])
        log.info("LATENCY p50=%.1fms p95=%.1fms | batched %.0f img/s",
                 latency["p50_ms"], latency["p95_ms"], latency["throughput_batched_ips"])

    return TrainingResult(
        model_name=model_name,
        arch=arch,
        run_id=run_id,
        best_epoch=best_epoch,
        threshold=choice.threshold,
        threshold_rationale=choice.rationale,
        val_metrics=choice.validation_metrics,
        test_metrics=test_metrics,
        test_ci=test_ci,
        latency=latency,
        model_info=model_info,
        history=history,
        bundle_path=str(bundle_path),
        train_seconds=train_seconds,
        provenance=provenance,
    )


def _plot_history(history: list[dict[str, float]], out_dir: Path) -> Path:
    """Loss and validation-F1 curves -- the first thing to check when a run looks wrong."""
    from ..plotting import save_figure, use_headless_backend

    plt = use_headless_backend()

    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax1.plot(epochs, [h["train_loss"] for h in history], label="train", lw=2)
    ax1.plot(epochs, [h["val_loss"] for h in history], label="val", lw=2)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("BCE loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, [h["train_f1"] for h in history], label="train F1", lw=2)
    ax2.plot(epochs, [h["val_f1"] for h in history], label="val F1", lw=2)
    ax2.plot(epochs, [h["val_recall"] for h in history], label="val recall", lw=1.5, ls="--")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("score")
    ax2.set_title("Metrics @ t=0.5")
    ax2.set_ylim(0, 1.02)
    ax2.legend()
    ax2.grid(alpha=0.3)

    return Path(save_figure(fig, out_dir / "training_history.png"))
