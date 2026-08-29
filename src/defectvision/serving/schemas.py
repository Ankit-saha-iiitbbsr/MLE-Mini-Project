"""Pydantic request/response contracts for the inference API.

The response is wider than "class + confidence" on purpose. Each extra field
exists because something downstream needs it:

``decision``       the caller must know whether to act automatically or route
                   the part to a human -- a 0.51 probability is not the same
                   instruction as a 0.99 one, even though both cross the
                   threshold
``threshold``      makes the decision auditable: the same probability yields a
                   different label under a different threshold, so the
                   threshold has to travel with the answer
``model``          which model version produced this, for incident triage
``image_stats``    the drift features, returned so a caller can monitor without
                   a second pass over the image
``request_id``     ties the API response to the row in the prediction log
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ImageStats(BaseModel):
    """Image statistics computed at request time; the drift feature vector."""

    mean_intensity: float
    std_intensity: float
    p05_intensity: float
    p95_intensity: float
    edge_density: float
    laplacian_var: float
    entropy: float


class ModelRef(BaseModel):
    """Identity of the model that produced a prediction."""

    name: str
    arch: str
    threshold: float
    mlflow_run_id: str | None = None
    git_commit: str | None = None


class PredictionResponse(BaseModel):
    """Result for a single image."""

    # protected_namespaces=() lets the response carry a field literally named
    # `model` (the model that produced the prediction) without Pydantic warning
    # about a clash with its own `model_` attributes.
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "request_id": "9f2b1c8a4e7d4f1e",
                "filename": "cast_def_0_1042.jpeg",
                "predicted_class": "defect",
                "predicted_label": 1,
                "probability_defect": 0.9963,
                "confidence": 0.9963,
                "threshold": 0.42,
                "decision": "auto_reject",
                "latency_ms": 11.7,
            }
        },
    )

    request_id: str = Field(..., description="Correlates with the prediction log row")
    filename: str | None = Field(None, description="Original upload filename")

    predicted_class: str = Field(..., description="'ok' or 'defect'")
    predicted_label: int = Field(..., description="0 = ok, 1 = defect")
    probability_defect: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Probability of the predicted class (not always of 'defect')",
    )
    threshold: float = Field(..., description="Operating threshold applied")
    decision: str = Field(
        ...,
        description="auto_accept | auto_reject | human_review (low-confidence band)",
    )

    latency_ms: float
    image_stats: ImageStats
    model: ModelRef


class BatchItem(BaseModel):
    """One element of a batch response; carries an error instead of a result on failure."""

    model_config = ConfigDict(protected_namespaces=())

    index: int
    filename: str | None = None
    success: bool
    result: PredictionResponse | None = None
    error: str | None = None


class BatchPredictionResponse(BaseModel):
    """Result for a multi-image request.

    A batch returns HTTP 200 whenever the *request* was well-formed, with
    per-item success flags inside. One corrupt file in a batch of sixteen should
    not discard the other fifteen valid predictions.
    """

    request_id: str
    n_requested: int
    n_succeeded: int
    n_failed: int
    total_latency_ms: float
    items: list[BatchItem]


class ErrorResponse(BaseModel):
    """Uniform error envelope for every 4xx/5xx response."""

    error: str = Field(..., description="Short machine-readable error code")
    detail: str = Field(..., description="Human-readable explanation")
    request_id: str | None = None
    hint: str | None = Field(None, description="How to fix the request")


class HealthResponse(BaseModel):
    """Liveness payload."""

    status: str
    version: str
    uptime_seconds: float


class ReadinessResponse(BaseModel):
    """Readiness payload -- distinct from liveness (see app.py)."""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_name: str | None = None
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    """Model card served at ``/model``."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    arch: str
    classes: list[str]
    threshold: float
    threshold_strategy: str
    input_size: int
    in_channels: int
    created_at: str
    mlflow_run_id: str | None = None
    git_commit: str | None = None
    dataset_manifest_sha256: str | None = None
    test_metrics: dict[str, float] = Field(default_factory=dict)


class MetricsResponse(BaseModel):
    """Operational counters served at ``/metrics``."""

    uptime_seconds: float
    requests_total: int
    requests_failed: int
    predictions_total: int
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_p99_ms: float | None = None
    prediction_log: dict[str, Any] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    """Delayed ground-truth label for a previous prediction."""

    request_id: str = Field(..., description="request_id returned by /predict")
    ground_truth: int = Field(..., ge=0, le=1, description="0 = ok, 1 = defect")
    source: str = Field("manual", description="Where the label came from")


class FeedbackResponse(BaseModel):
    """Acknowledgement for a feedback submission."""

    request_id: str
    recorded: bool
    detail: str
