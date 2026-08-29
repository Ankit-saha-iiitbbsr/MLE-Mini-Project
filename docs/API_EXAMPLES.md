# API Test Calls

**Deliverable 3 · Module M4 · PCAM\* ZC412 Mini-Project (Flavor B)**

Sample requests and responses for the deployed inference service. A ready-to-import
Postman collection is at [`postman_collection.json`](postman_collection.json).

Start the service first:

```bash
python -m defectvision.cli serve
```

Interactive OpenAPI docs: <http://localhost:8000/docs>

---

## 1. Health and readiness

Liveness and readiness are **separate endpoints** and answer different
questions. A container whose model failed to load is alive but must not receive
traffic.

```bash
curl -s http://localhost:8000/healthz
```

```json
{ "status": "ok", "version": "1.0.0", "uptime_seconds": 42.7 }
```

```bash
curl -s http://localhost:8000/readyz
```

```json
{ "status": "ready", "model_loaded": true, "model_name": "resnet18", "detail": null }
```

With **no model bundle present**, `/healthz` still returns 200 while `/readyz`
returns **503** — which is what an orchestrator needs to stop routing traffic
without restart-looping a healthy process:

```json
{
  "status": "not_ready",
  "model_loaded": false,
  "model_name": null,
  "detail": "FileNotFoundError: Model bundle not found: models/production/model_bundle.pt"
}
```

---

## 2. Model card

```bash
curl -s http://localhost:8000/model
```

```json
{
  "model_name": "resnet18",
  "arch": "resnet18",
  "classes": ["ok", "defect"],
  "threshold": 0.16,
  "threshold_strategy": "max_f1",
  "input_size": 128,
  "in_channels": 1,
  "created_at": "2026-08-15T12:25:04Z",
  "mlflow_run_id": "b3c9e1a74f2d4e88a1c05f6d2e7b9a13",
  "git_commit": "df4878a",
  "dataset_manifest_sha256": "8f21c4a9e37b6d05...",
  "test_metrics": { "f1": 0.999, "recall": 0.9988, "precision": 0.9993, "accuracy": 0.9986 }
}
```

The `git_commit` and `dataset_manifest_sha256` fields are what let you trace a
running container back to the exact run, commit and dataset version that
produced it.

---

## 3. Single prediction

```bash
curl -s -F "file=@data/raw/defect/train_cast_def_0_1042.jpeg" http://localhost:8000/predict
```

```json
{
  "request_id": "9f2b1c8a4e7d4f1e8b03a6c5d9e2f741",
  "filename": "train_cast_def_0_1042.jpeg",
  "predicted_class": "defect",
  "predicted_label": 1,
  "probability_defect": 0.9963,
  "confidence": 0.9963,
  "threshold": 0.16,
  "decision": "auto_reject",
  "latency_ms": 11.74,
  "image_stats": {
    "mean_intensity": 0.5312,
    "std_intensity": 0.2094,
    "p05_intensity": 0.1725,
    "p95_intensity": 0.8353,
    "edge_density": 0.1417,
    "laplacian_var": 0.00382,
    "entropy": 4.9216
  },
  "model": {
    "name": "resnet18",
    "arch": "resnet18",
    "threshold": 0.16,
    "mlflow_run_id": "b3c9e1a74f2d4e88a1c05f6d2e7b9a13",
    "git_commit": "df4878a"
  }
}
```

### Fields worth understanding

| field | why it is in the response |
| --- | --- |
| `decision` | `auto_accept` / `auto_reject` / `human_review`. A probability just past the threshold is a coin-flip dressed as a decision, so borderline scores are routed to a person instead of auto-actioned. |
| `confidence` | Probability of the **predicted** class, not of `defect`. An `ok` call at `probability_defect = 0.02` reports `confidence = 0.98`. |
| `threshold` | The same probability yields a different label under a different threshold, so the threshold must travel with the answer for the decision to be auditable. |
| `image_stats` | The drift feature vector, returned so a caller can monitor without a second pass over the image. |
| `request_id` | Matches the row in the prediction log — a complaint traces to the exact inference. Also returned as the `X-Request-ID` header. |

A **good** part:

```json
{
  "predicted_class": "ok",
  "predicted_label": 0,
  "probability_defect": 0.0041,
  "confidence": 0.9959,
  "decision": "auto_accept"
}
```

---

## 4. Batch prediction

```bash
curl -s -F "files=@img1.jpeg" -F "files=@img2.jpeg" -F "files=@img3.jpeg" http://localhost:8000/predict/batch
```

```json
{
  "request_id": "4d1e7a93c8b25f60",
  "n_requested": 3,
  "n_succeeded": 3,
  "n_failed": 0,
  "total_latency_ms": 34.18,
  "items": [
    { "index": 0, "filename": "img1.jpeg", "success": true, "result": { "predicted_class": "defect", "...": "..." } },
    { "index": 1, "filename": "img2.jpeg", "success": true, "result": { "predicted_class": "ok", "...": "..." } },
    { "index": 2, "filename": "img3.jpeg", "success": true, "result": { "predicted_class": "ok", "...": "..." } }
  ]
}
```

### Partial failure returns 200, not an error

One corrupt file must not discard the valid predictions beside it:

```json
{
  "n_requested": 3,
  "n_succeeded": 2,
  "n_failed": 1,
  "items": [
    { "index": 0, "success": true,  "result": { "predicted_class": "defect", "...": "..." } },
    { "index": 1, "success": false, "error": "File 'bad.png' is not a recognisable image" },
    { "index": 2, "success": true,  "result": { "predicted_class": "ok", "...": "..." } }
  ]
}
```

---

## 5. Error handling

Every foreseeable bad input maps to a specific 4xx carrying a `hint`. A 500
should mean "the service has a bug" — a signal that is worthless if malformed
uploads also produce 500s.

### Empty file → 400

```bash
curl -s -w "\nHTTP %{http_code}\n" -F "file=@/dev/null;filename=empty.png" http://localhost:8000/predict
```

```json
{
  "error": "invalid_image",
  "detail": "Uploaded file is empty (0 bytes)",
  "request_id": "1a2b3c4d",
  "hint": "Attach a non-empty image file in the 'file' form field."
}
```

### Not an image → 400

```bash
echo "definitely not an image" > bad.png && curl -s -w "\nHTTP %{http_code}\n" -F "file=@bad.png" http://localhost:8000/predict
```

```json
{
  "error": "invalid_image",
  "detail": "File 'bad.png' is not a recognisable image",
  "hint": "The bytes are not a valid JPEG/PNG/BMP. Check the file is not truncated."
}
```

### Wrong content type → 415

```bash
curl -s -w "\nHTTP %{http_code}\n" -F "file=@report.pdf;type=application/pdf" http://localhost:8000/predict
```

### Oversized upload → 413

```json
{
  "error": "payload_too_large",
  "detail": "Image is 12.4 MB; the limit is 5.2 MB",
  "hint": "Downscale the image or raise serving.max_upload_bytes."
}
```

### Batch over the limit → 413

```json
{
  "error": "batch_too_large",
  "detail": "40 files exceeds the limit of 16.",
  "hint": "Split the request or raise serving.max_batch_size."
}
```

### Missing file field → 422

FastAPI's own request validation, before any handler code runs.

### Summary

| condition | status | `error` code |
| --- | --- | --- |
| Missing `file` field | `422` | *(FastAPI validation)* |
| Empty file | `400` | `invalid_image` |
| Non-image or truncated bytes | `400` | `invalid_image` |
| Unsupported content type | `415` | `invalid_image` |
| Upload over size limit | `413` | `payload_too_large` |
| Batch over item limit | `413` | `batch_too_large` |
| Empty batch | `400` | `empty_batch` |
| No model loaded | `503` | `model_not_loaded` |

---

## 6. Feedback (delayed ground truth)

Labels on a real line arrive hours or days after the prediction — from teardown,
rework, or a customer return. This closes that loop so accuracy monitoring and
the retraining trigger have something to work with.

```bash
curl -s -X POST http://localhost:8000/feedback -H "Content-Type: application/json" -d '{"request_id":"9f2b1c8a4e7d4f1e8b03a6c5d9e2f741","ground_truth":1,"source":"teardown"}'
```

```json
{
  "request_id": "9f2b1c8a4e7d4f1e8b03a6c5d9e2f741",
  "recorded": true,
  "detail": "Ground truth 1 recorded from 'teardown'."
}
```

An unknown `request_id` returns **404** rather than silently succeeding.

---

## 7. Operational metrics

```bash
curl -s http://localhost:8000/metrics
```

```json
{
  "uptime_seconds": 903.4,
  "requests_total": 1274,
  "requests_failed": 6,
  "predictions_total": 1265,
  "latency_p50_ms": 12.4,
  "latency_p95_ms": 24.8,
  "latency_p99_ms": 41.2,
  "prediction_log": {
    "n_predictions": 1265,
    "n_predicted_defect": 731,
    "n_human_review": 38,
    "n_labeled": 402,
    "review_rate": 0.0300,
    "predicted_defect_rate": 0.5779,
    "mean_probability": 0.5612,
    "mean_latency_ms": 11.9
  }
}
```

Percentiles rather than a mean: latency distributions are right-skewed, so a
mean hides exactly the tail an SLA is written against.

---

## 8. Load benchmark

```bash
python -m defectvision.cli benchmark --n 200 --concurrency 4
```

Measures **end-to-end HTTP latency** against a running service — multipart
parsing, JPEG decode, preprocessing, inference and serialisation. Results are
written to `reports/api_benchmark.json`.

```json
{
  "n_requests": 200,
  "concurrency": 4,
  "n_succeeded": 200,
  "throughput_rps": 82.4,
  "latency_ms": { "p50": 44.1, "p95": 68.3, "p99": 91.7, "max": 118.2 },
  "status_codes": { "200": 200 }
}
```

---

## 9. Docker

```bash
docker build -t defectvision:1.0.0 .
```

```bash
docker run --rm -p 8000:8000 -v "$(pwd)/models/production:/app/models/production:ro" -v "$(pwd)/monitoring:/app/monitoring" defectvision:1.0.0
```

The model is **mounted read-only**, not baked into the image: a newly promoted
model is picked up by restarting the container rather than rebuilding it, so the
image stays a pure runtime and the model stays a versioned artifact.
