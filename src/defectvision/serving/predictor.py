"""Model holder and the image -> decision path used by the API.

Kept separate from :mod:`defectvision.serving.app` so the inference logic can be
unit-tested without an HTTP layer, and so the drift simulator can reuse the
exact same code path the service uses. Reusing it is not a convenience: if the
simulator scored images through a different function, it would be measuring a
model the service does not run.

Two behaviours worth noting:

**Decoding is defensive.** Upload bytes are untrusted. Pillow will happily
allocate gigabytes for a crafted image header, so a decompression-bomb guard is
set and every decode failure is converted into a typed error the API can turn
into a clean 4xx instead of a 500.

**A confidence band routes to a human.** A probability just past the threshold
is a coin-flip dressed as a decision. Those parts go to ``human_review`` rather
than being auto-actioned -- the human-in-the-loop pattern from M5/CS10 -- and
the share of traffic landing in that band is itself a monitored signal.
"""

from __future__ import annotations

import io
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from ..bundle import LoadedModel, load_bundle
from ..config import get, resolve
from ..features.image_stats import image_statistics
from ..logging_utils import get_logger
from ..monitoring.store import PredictionRecord, PredictionStore

log = get_logger(__name__)

#: Pillow's own bomb guard. 64 MP is far above any inspection camera and far
#: below what would exhaust a serving container.
Image.MAX_IMAGE_PIXELS = 64_000_000

ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/bmp", "image/x-ms-bmp",
    "application/octet-stream",  # some clients omit a real type
}
ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "MPO"}


class ImageDecodeError(ValueError):
    """Raised when upload bytes cannot be turned into a usable image."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


class PayloadTooLarge(ValueError):
    """Raised when an upload exceeds the configured size limit."""


@dataclass
class PredictionOutcome:
    """Everything one inference produced, before it becomes JSON."""

    request_id: str
    probability_defect: float
    predicted_label: int
    predicted_class: str
    confidence: float
    threshold: float
    decision: str
    latency_ms: float
    image_stats: dict[str, float]
    width: int
    height: int
    file_bytes: int
    filename: str | None = None


def decode_image(
    data: bytes,
    *,
    filename: str | None = None,
    content_type: str | None = None,
    max_bytes: int | None = None,
) -> Image.Image:
    """Turn upload bytes into a grayscale PIL image, or raise a typed error."""
    if not data:
        raise ImageDecodeError(
            "Uploaded file is empty (0 bytes)",
            hint="Attach a non-empty image file in the 'file' form field.",
        )

    if max_bytes is not None and len(data) > max_bytes:
        raise PayloadTooLarge(
            f"Image is {len(data) / 1e6:.1f} MB; the limit is {max_bytes / 1e6:.1f} MB"
        )

    if content_type and content_type.split(";")[0].strip().lower() not in ALLOWED_CONTENT_TYPES:
        raise ImageDecodeError(
            f"Unsupported content type {content_type!r}",
            hint=f"Send one of: {', '.join(sorted(ALLOWED_CONTENT_TYPES - {'application/octet-stream'}))}",
        )

    try:
        # Verify on a throwaway handle first: verify() invalidates the object,
        # so the real decode needs a second one.
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(data)) as im:
            fmt = (im.format or "").upper()
            if fmt and fmt not in ALLOWED_FORMATS:
                raise ImageDecodeError(
                    f"Unsupported image format {fmt!r}",
                    hint=f"Convert to one of: {', '.join(sorted(ALLOWED_FORMATS - {'MPO'}))}",
                )
            gray = im.convert("L")
            gray.load()
            return gray
    except ImageDecodeError:
        raise
    except UnidentifiedImageError as exc:
        raise ImageDecodeError(
            f"File{f' {filename!r}' if filename else ''} is not a recognisable image",
            hint="The bytes are not a valid JPEG/PNG/BMP. Check the file is not truncated.",
        ) from exc
    except Image.DecompressionBombError as exc:
        raise ImageDecodeError(
            "Image dimensions exceed the safety limit",
            hint="Downscale the image before uploading.",
        ) from exc
    except (OSError, ValueError) as exc:
        raise ImageDecodeError(
            f"Failed to decode image: {exc}",
            hint="The file may be corrupt or truncated.",
        ) from exc


class Predictor:
    """Wraps a loaded bundle with request handling and prediction logging."""

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.model: LoadedModel | None = None
        self.load_error: str | None = None

        serving = get(params, "serving")
        self.max_upload_bytes = int(serving["max_upload_bytes"])
        self.max_batch_size = int(serving["max_batch_size"])
        self.log_predictions = bool(serving.get("log_predictions", True))
        low, high = serving.get("review_band", [0.35, 0.65])
        self.review_low, self.review_high = float(low), float(high)

        self.store: PredictionStore | None = None
        if self.log_predictions:
            try:
                self.store = PredictionStore(get(params, "monitoring.db_path"))
            except Exception as exc:  # pragma: no cover - disk/permission issues
                log.warning("Prediction logging disabled (%s: %s)", type(exc).__name__, exc)

    # -- lifecycle --------------------------------------------------------

    def load(self, bundle_path: str | Path | None = None) -> bool:
        """Load the production bundle. Returns success; never raises.

        Failure is recorded rather than raised so the process can still start
        and report *why* it is not ready on ``/readyz``. A container that exits
        on a missing model gives an orchestrator nothing to display.
        """
        path = resolve(bundle_path or get(self.params, "serving.model_bundle"))
        try:
            self.model = load_bundle(path)
            self.load_error = None
            log.info("Predictor ready: %r", self.model)
            return True
        except Exception as exc:
            self.model = None
            self.load_error = f"{type(exc).__name__}: {exc}"
            log.error("Failed to load model bundle from %s -- %s", path, self.load_error)
            return False

    @property
    def ready(self) -> bool:
        return self.model is not None

    # -- inference --------------------------------------------------------

    def _decide(self, probability: float, threshold: float) -> str:
        """Auto-action, or route to a human when confidence is borderline."""
        if self.review_low <= probability <= self.review_high:
            return "human_review"
        return "auto_reject" if probability >= threshold else "auto_accept"

    def predict_image(
        self,
        image: Image.Image,
        *,
        filename: str | None = None,
        file_bytes: int = 0,
        request_id: str | None = None,
        source: str = "api",
        scenario: str | None = None,
        ground_truth: int | None = None,
        persist: bool = True,
    ) -> PredictionOutcome:
        """Score one already-decoded image and (optionally) log the result."""
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        request_id = request_id or uuid.uuid4().hex
        started = time.perf_counter()

        probability = float(self.model.predict_proba([image])[0])
        stats = image_statistics(image)

        latency_ms = (time.perf_counter() - started) * 1000.0
        threshold = self.model.threshold
        label = int(probability >= threshold)
        predicted_class = self.model.classes[label]

        outcome = PredictionOutcome(
            request_id=request_id,
            probability_defect=probability,
            predicted_label=label,
            predicted_class=predicted_class,
            # Confidence is in the *predicted* class, so an "ok" call at
            # p(defect)=0.02 reports 0.98, not 0.02.
            confidence=probability if label == 1 else 1.0 - probability,
            threshold=threshold,
            decision=self._decide(probability, threshold),
            latency_ms=latency_ms,
            image_stats=stats,
            width=image.width,
            height=image.height,
            file_bytes=file_bytes,
            filename=filename,
        )

        if persist and self.store is not None:
            try:
                self.store.log(PredictionRecord(
                    request_id=request_id,
                    probability=probability,
                    predicted_label=label,
                    predicted_class=predicted_class,
                    decision=outcome.decision,
                    latency_ms=latency_ms,
                    model_name=self.model.model_name,
                    model_run_id=self.model.mlflow_run_id,
                    model_threshold=threshold,
                    source=source,
                    scenario=scenario,
                    filename=filename,
                    width=image.width,
                    height=image.height,
                    file_bytes=file_bytes,
                    image_stats=stats,
                    ground_truth=ground_truth,
                ))
            except Exception as exc:  # pragma: no cover
                # Monitoring must never take the serving path down with it.
                log.warning("Failed to log prediction %s (%s: %s)",
                            request_id, type(exc).__name__, exc)

        return outcome

    def predict_bytes(
        self,
        data: bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
        request_id: str | None = None,
        **kwargs: Any,
    ) -> PredictionOutcome:
        """Decode then score. Raises :class:`ImageDecodeError` on bad input."""
        image = decode_image(
            data, filename=filename, content_type=content_type,
            max_bytes=self.max_upload_bytes,
        )
        return self.predict_image(
            image, filename=filename, file_bytes=len(data),
            request_id=request_id, **kwargs,
        )

    def model_ref(self) -> dict[str, Any]:
        """Identity block embedded in every prediction response."""
        if self.model is None:
            return {}
        return {
            "name": self.model.model_name,
            "arch": self.model.arch,
            "threshold": self.model.threshold,
            "mlflow_run_id": self.model.mlflow_run_id,
            "git_commit": self.model.provenance.get("git_commit_short"),
        }
