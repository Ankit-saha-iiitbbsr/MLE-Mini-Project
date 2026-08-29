"""The deployment artifact: a self-contained model bundle (M4).

A bare ``state_dict`` is not a deployable model. To turn an image into a
decision the service also needs the architecture that built the weights, the
*exact* preprocessing used in training, the class ordering, and the tuned
operating threshold. Shipping those separately is how a model ends up served
with the wrong resize or a stale threshold.

This module defines one file that carries all of it, plus the provenance needed
to trace a running container back to the run that produced it. Two consequences
worth calling out:

* **The serving container does not need MLflow.** The bundle is plain
  ``torch.save`` output; MLflow stays an experiment-tracking concern and never
  becomes a runtime dependency. That keeps the image small and removes a
  failure mode from the request path.
* **Preprocessing travels with the weights.** :class:`LoadedModel` builds its
  transform from the bundle, not from ``params.yaml``. Editing config after a
  model ships cannot silently change how that model sees an image.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .data.transforms import PreprocessSpec
from .logging_utils import get_logger

log = get_logger(__name__)

#: Bumped when the on-disk layout changes incompatibly.
BUNDLE_FORMAT_VERSION = 1


@dataclass
class BundleMetadata:
    """Everything about a bundle except the weights themselves."""

    model_name: str
    arch: str
    model_config: dict[str, Any]
    preprocess: dict[str, Any]
    classes: list[str]
    threshold: float
    threshold_strategy: str = "max_f1"
    in_channels: int = 1
    metrics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    mlflow_run_id: str | None = None
    created_at: str = ""
    format_version: int = BUNDLE_FORMAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        d = {
            "model_name": self.model_name,
            "arch": self.arch,
            "model_config": self.model_config,
            "preprocess": self.preprocess,
            "classes": list(self.classes),
            "threshold": float(self.threshold),
            "threshold_strategy": self.threshold_strategy,
            "in_channels": int(self.in_channels),
            "metrics": self.metrics,
            "provenance": self.provenance,
            "mlflow_run_id": self.mlflow_run_id,
            "created_at": self.created_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "format_version": self.format_version,
        }
        return d


def save_bundle(path: str | Path, model: torch.nn.Module, meta: BundleMetadata) -> Path:
    """Serialise weights + metadata to a single ``.pt`` file."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        **meta.to_dict(),
        # Always store CPU tensors: a bundle trained on GPU must load on a
        # CPU-only inference node without a map_location dance.
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    torch.save(payload, dest)
    size_mb = dest.stat().st_size / (1024 ** 2)
    log.info("Saved model bundle -> %s (%.2f MB)", dest, size_mb)
    return dest


class LoadedModel:
    """A bundle loaded and ready to serve.

    Exposes exactly what the API layer needs and nothing more: turn images into
    P(defect), then apply the frozen threshold.
    """

    def __init__(self, payload: dict[str, Any], device: str = "cpu") -> None:
        fmt = int(payload.get("format_version", 0))
        if fmt != BUNDLE_FORMAT_VERSION:
            raise ValueError(
                f"Bundle format version {fmt} is not supported by this build "
                f"(expected {BUNDLE_FORMAT_VERSION}). Re-export the model."
            )

        self.device = torch.device(device)
        self.model_name: str = payload["model_name"]
        self.arch: str = payload["arch"]
        self.model_config: dict[str, Any] = payload.get("model_config", {})
        self.classes: list[str] = list(payload["classes"])
        self.threshold: float = float(payload["threshold"])
        self.threshold_strategy: str = payload.get("threshold_strategy", "max_f1")
        self.in_channels: int = int(payload.get("in_channels", 1))
        self.metrics: dict[str, Any] = payload.get("metrics", {})
        self.provenance: dict[str, Any] = payload.get("provenance", {})
        self.mlflow_run_id: str | None = payload.get("mlflow_run_id")
        self.created_at: str = payload.get("created_at", "")

        self.preprocess_spec = PreprocessSpec.from_dict(payload["preprocess"])
        self._transform = self.preprocess_spec.build()

        # Rebuild the architecture from the recorded config, then load weights.
        from .models.factory import build_model

        self.model = build_model(self.model_config, in_channels=self.in_channels)
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    # -- inference --------------------------------------------------------

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """PIL image -> normalised CHW tensor, using the bundled transform."""
        return self._transform(image)

    @torch.inference_mode()
    def predict_proba(self, images: list[Image.Image] | Image.Image) -> np.ndarray:
        """Return P(defect) for one image or a list of images."""
        if isinstance(images, Image.Image):
            images = [images]
        if not images:
            return np.empty((0,), dtype=np.float64)

        batch = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        logits = self.model(batch)
        # Models emit a single logit per image; guard against a stray extra dim.
        if logits.ndim > 1:
            logits = logits.squeeze(-1)
        return torch.sigmoid(logits).cpu().numpy().astype(np.float64)

    def predict(self, images: list[Image.Image] | Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(probabilities, predicted_label_indices)`` at the frozen threshold."""
        probs = self.predict_proba(images)
        return probs, (probs >= self.threshold).astype(np.int64)

    def label_for(self, prob: float) -> str:
        """Class name for a probability, using the bundle's threshold."""
        return self.classes[1] if prob >= self.threshold else self.classes[0]

    # -- introspection ----------------------------------------------------

    def info(self) -> dict[str, Any]:
        """Serving-safe summary for the ``/model`` endpoint."""
        return {
            "model_name": self.model_name,
            "arch": self.arch,
            "classes": self.classes,
            "threshold": self.threshold,
            "threshold_strategy": self.threshold_strategy,
            "input_size": self.preprocess_spec.resize,
            "in_channels": self.in_channels,
            "created_at": self.created_at,
            "mlflow_run_id": self.mlflow_run_id,
            "git_commit": self.provenance.get("git_commit_short"),
            "dataset_manifest_sha256": self.provenance.get("manifest_sha256"),
            "test_metrics": {
                k: round(float(v), 5)
                for k, v in (self.metrics.get("test") or {}).items()
                if isinstance(v, (int, float))
            },
        }

    def __repr__(self) -> str:
        return (f"LoadedModel(name={self.model_name!r}, arch={self.arch!r}, "
                f"threshold={self.threshold:.3f}, device={self.device})")


def load_bundle(path: str | Path, device: str = "cpu") -> LoadedModel:
    """Load a bundle from disk.

    ``weights_only=False`` is required because the payload carries metadata
    dicts alongside tensors. That makes the file a trusted input: load bundles
    this project produced, not bundles from untrusted sources.
    """
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(
            f"Model bundle not found: {src}\n"
            "Train and package a model first:  defectvision train --all && defectvision package"
        )
    payload = torch.load(src, map_location="cpu", weights_only=False)
    model = LoadedModel(payload, device=device)
    log.info("Loaded %s (threshold=%.4f, input=%dpx)",
             model.model_name, model.threshold, model.preprocess_spec.resize)
    return model
