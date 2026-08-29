"""FastAPI inference service (M4).

Endpoint design notes:

**``/healthz`` and ``/readyz`` are different things.** Liveness says the process
is alive; readiness says it can actually serve. A container whose model failed
to load is alive but must not receive traffic -- collapsing the two into one
endpoint either restart-loops a healthy process or routes requests to a broken
one.

**Errors are typed, never 500.** Every foreseeable bad input -- empty file,
wrong content type, corrupt bytes, oversized upload, too many files -- maps to a
specific 4xx with a ``hint`` telling the caller how to fix it. A 500 in the logs
should mean "we have a bug", and that signal is worthless if malformed uploads
also produce 500s.

**Batch is partially fault-tolerant.** A well-formed batch returns 200 with
per-item success flags; one corrupt file does not discard fifteen good
predictions.

**Every response carries a ``request_id``** that matches the prediction-log row,
so a customer complaint can be traced to the exact inference and its inputs.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile, status
from fastapi.responses import JSONResponse, RedirectResponse

from .. import __version__
from ..config import get, load_params
from ..logging_utils import configure_logging, get_logger
from .predictor import ImageDecodeError, PayloadTooLarge, Predictor
from .schemas import (
    BatchItem,
    BatchPredictionResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReadinessResponse,
)

log = get_logger(__name__)


class ServiceState:
    """Process-local state: the model, and in-memory operational counters."""

    def __init__(self) -> None:
        self.started_at = time.time()
        self.params: dict[str, Any] = {}
        self.predictor: Predictor | None = None
        self.requests_total = 0
        self.requests_failed = 0
        self.predictions_total = 0
        # Bounded: an unbounded latency list is a slow memory leak in a
        # long-running service.
        self.latencies: deque[float] = deque(maxlen=2000)

    @property
    def uptime(self) -> float:
        return time.time() - self.started_at


state = ServiceState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load configuration and the model once, at startup."""
    configure_logging()
    state.params = load_params()
    state.predictor = Predictor(state.params)
    if not state.predictor.load():
        log.warning(
            "Service started WITHOUT a model. /readyz will report not-ready until a "
            "bundle exists at %s. Train and promote one:  "
            "defectvision train --all && defectvision package",
            get(state.params, "serving.model_bundle"),
        )
    yield
    log.info("Shutting down after %.0fs, %d requests", state.uptime, state.requests_total)


app = FastAPI(
    title="DefectVision Inference API",
    description=(
        "Image-based casting defect / quality classifier.\n\n"
        "Upload a product image to `/predict` and receive a defect probability, a "
        "thresholded decision, and the image statistics used for drift monitoring.\n\n"
        "*BITS PCAM\\* ZC412 Machine Learning Engineering mini-project, Flavor B.*"
    ),
    version=__version__,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a request id, time the request, and record counters."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()

    state.requests_total += 1
    try:
        response = await call_next(request)
    except Exception:
        state.requests_failed += 1
        log.exception("Unhandled error on %s %s", request.method, request.url.path,
                      extra={"request_id": request_id})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="internal_error",
                detail="An unexpected error occurred while handling the request.",
                request_id=request_id,
            ).model_dump(),
        )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if request.url.path.startswith("/predict"):
        state.latencies.append(elapsed_ms)
    if response.status_code >= 400:
        state.requests_failed += 1

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.2f}"
    return response


def _error(status_code: int, code: str, detail: str,
           request_id: str | None = None, hint: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=code, detail=detail,
                              request_id=request_id, hint=hint).model_dump(),
    )


def _require_model(request_id: str) -> JSONResponse | None:
    """503 when no model is loaded -- a retryable condition, not a client error."""
    if state.predictor is None or not state.predictor.ready:
        reason = state.predictor.load_error if state.predictor else "predictor not initialised"
        return _error(
            status.HTTP_503_SERVICE_UNAVAILABLE, "model_not_loaded",
            f"No model is available to serve predictions ({reason}).",
            request_id,
            hint="Train and promote a model: defectvision train --all && defectvision package",
        )
    return None


# ---------------------------------------------------------------------------
# Operational endpoints
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/healthz", response_model=HealthResponse, tags=["ops"])
async def healthz() -> HealthResponse:
    """Liveness: the process is running. Always 200 while the server responds."""
    return HealthResponse(status="ok", version=__version__, uptime_seconds=state.uptime)


@app.get("/readyz", response_model=ReadinessResponse, tags=["ops"])
async def readyz() -> JSONResponse:
    """Readiness: a model is loaded and the service can serve traffic."""
    ready = state.predictor is not None and state.predictor.ready
    payload = ReadinessResponse(
        status="ready" if ready else "not_ready",
        model_loaded=ready,
        model_name=state.predictor.model.model_name if ready else None,  # type: ignore[union-attr]
        detail=None if ready else (
            state.predictor.load_error if state.predictor else "predictor not initialised"
        ),
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
        content=payload.model_dump(),
    )


@app.get("/model", response_model=ModelInfoResponse, tags=["ops"])
async def model_info(request: Request) -> Any:
    """Model card: identity, threshold, training provenance and test metrics."""
    if (err := _require_model(request.state.request_id)) is not None:
        return err
    return state.predictor.model.info()  # type: ignore[union-attr]


@app.get("/metrics", response_model=MetricsResponse, tags=["ops"])
async def metrics() -> MetricsResponse:
    """Operational counters, including latency percentiles over recent requests."""
    lat = np.array(state.latencies, dtype=np.float64) if state.latencies else None
    log_summary: dict[str, Any] = {}
    if state.predictor is not None and state.predictor.store is not None:
        try:
            log_summary = state.predictor.store.summary()
        except Exception as exc:  # pragma: no cover
            log_summary = {"error": f"{type(exc).__name__}: {exc}"}

    return MetricsResponse(
        uptime_seconds=state.uptime,
        requests_total=state.requests_total,
        requests_failed=state.requests_failed,
        predictions_total=state.predictions_total,
        latency_p50_ms=float(np.percentile(lat, 50)) if lat is not None else None,
        latency_p95_ms=float(np.percentile(lat, 95)) if lat is not None else None,
        latency_p99_ms=float(np.percentile(lat, 99)) if lat is not None else None,
        prediction_log=log_summary,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Malformed or undecodable image"},
        413: {"model": ErrorResponse, "description": "Upload exceeds the size limit"},
        415: {"model": ErrorResponse, "description": "Unsupported media type"},
        503: {"model": ErrorResponse, "description": "No model loaded"},
    },
    tags=["inference"],
)
async def predict(
    request: Request,
    file: UploadFile = File(..., description="Product image (JPEG/PNG/BMP)"),
    scenario: str | None = Form(None, description="Optional tag for drift experiments"),
) -> Any:
    """Classify a single product image as `ok` or `defect`."""
    request_id = request.state.request_id
    if (err := _require_model(request_id)) is not None:
        return err

    try:
        data = await file.read()
    except Exception as exc:
        return _error(status.HTTP_400_BAD_REQUEST, "upload_read_failed",
                      f"Could not read the uploaded file: {exc}", request_id)

    try:
        outcome = state.predictor.predict_bytes(  # type: ignore[union-attr]
            data,
            filename=file.filename,
            content_type=file.content_type,
            request_id=request_id,
            scenario=scenario,
        )
    except PayloadTooLarge as exc:
        return _error(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "payload_too_large",
                      str(exc), request_id,
                      hint="Downscale the image or raise serving.max_upload_bytes.")
    except ImageDecodeError as exc:
        code = (status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
                if "Unsupported" in str(exc) else status.HTTP_400_BAD_REQUEST)
        return _error(code, "invalid_image", str(exc), request_id, hint=exc.hint)

    state.predictions_total += 1
    return PredictionResponse(
        request_id=outcome.request_id,
        filename=outcome.filename,
        predicted_class=outcome.predicted_class,
        predicted_label=outcome.predicted_label,
        probability_defect=outcome.probability_defect,
        confidence=outcome.confidence,
        threshold=outcome.threshold,
        decision=outcome.decision,
        latency_ms=outcome.latency_ms,
        image_stats=outcome.image_stats,  # type: ignore[arg-type]
        model=state.predictor.model_ref(),  # type: ignore[union-attr,arg-type]
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse, "description": "Too many files in one request"},
        503: {"model": ErrorResponse},
    },
    tags=["inference"],
)
async def predict_batch(
    request: Request,
    files: list[UploadFile] = File(..., description="Up to serving.max_batch_size images"),
    scenario: str | None = Form(None),
) -> Any:
    """Classify several images in one request, with per-item error reporting."""
    request_id = request.state.request_id
    if (err := _require_model(request_id)) is not None:
        return err

    predictor = state.predictor
    assert predictor is not None

    if not files:
        return _error(status.HTTP_400_BAD_REQUEST, "empty_batch",
                      "No files were supplied.", request_id,
                      hint="Attach at least one file in the 'files' field.")
    if len(files) > predictor.max_batch_size:
        return _error(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "batch_too_large",
            f"{len(files)} files exceeds the limit of {predictor.max_batch_size}.",
            request_id, hint="Split the request or raise serving.max_batch_size.",
        )

    started = time.perf_counter()
    items: list[BatchItem] = []

    for index, upload in enumerate(files):
        item_id = f"{request_id}-{index:03d}"
        try:
            data = await upload.read()
            outcome = predictor.predict_bytes(
                data, filename=upload.filename, content_type=upload.content_type,
                request_id=item_id, scenario=scenario,
            )
        except (ImageDecodeError, PayloadTooLarge) as exc:
            items.append(BatchItem(index=index, filename=upload.filename,
                                   success=False, error=str(exc)))
            continue
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("Batch item %d failed", index)
            items.append(BatchItem(index=index, filename=upload.filename, success=False,
                                   error=f"{type(exc).__name__}: {exc}"))
            continue

        state.predictions_total += 1
        items.append(BatchItem(
            index=index, filename=upload.filename, success=True,
            result=PredictionResponse(
                request_id=outcome.request_id,
                filename=outcome.filename,
                predicted_class=outcome.predicted_class,
                predicted_label=outcome.predicted_label,
                probability_defect=outcome.probability_defect,
                confidence=outcome.confidence,
                threshold=outcome.threshold,
                decision=outcome.decision,
                latency_ms=outcome.latency_ms,
                image_stats=outcome.image_stats,  # type: ignore[arg-type]
                model=predictor.model_ref(),  # type: ignore[arg-type]
            ),
        ))

    n_ok = sum(1 for i in items if i.success)
    return BatchPredictionResponse(
        request_id=request_id,
        n_requested=len(files),
        n_succeeded=n_ok,
        n_failed=len(files) - n_ok,
        total_latency_ms=(time.perf_counter() - started) * 1000.0,
        items=items,
    )


@app.post("/feedback", response_model=FeedbackResponse, tags=["monitoring"])
async def feedback(request: Request, payload: FeedbackRequest) -> Any:
    """Attach a ground-truth label to an earlier prediction.

    Labels on a real line arrive later than predictions -- from teardown,
    rework, or a customer return. This endpoint closes that loop so accuracy
    monitoring and the retraining trigger have something to work with.
    """
    request_id = request.state.request_id
    if state.predictor is None or state.predictor.store is None:
        return _error(status.HTTP_503_SERVICE_UNAVAILABLE, "logging_disabled",
                      "Prediction logging is disabled; feedback cannot be recorded.",
                      request_id)

    recorded = state.predictor.store.attach_ground_truth(
        payload.request_id, payload.ground_truth, payload.source
    )
    if not recorded:
        return _error(status.HTTP_404_NOT_FOUND, "unknown_request_id",
                      f"No logged prediction with request_id {payload.request_id!r}.",
                      request_id,
                      hint="Use the request_id returned by /predict.")

    return FeedbackResponse(
        request_id=payload.request_id, recorded=True,
        detail=f"Ground truth {payload.ground_truth} recorded from {payload.source!r}.",
    )
