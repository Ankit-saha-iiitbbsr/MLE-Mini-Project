"""End-to-end API load benchmark (M4: "basic latency/throughput awareness").

Deliberately measured over HTTP against a *running* service, not by timing
``model.forward()`` in-process. The two differ by more than people expect:
multipart parsing, JPEG decode, preprocessing, Pydantic serialisation and the
prediction-log write all sit on the request path and are frequently a larger
share of wall-clock time than the convolutions. Optimising the model while the
decode dominates is wasted effort, and only the end-to-end number reveals that.

Reported percentiles rather than a mean: latency distributions are right-skewed,
so a mean hides exactly the tail an SLA is written against.
"""

from __future__ import annotations

import io
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from PIL import Image

from ..config import get, resolve
from ..data.split import load_manifest
from ..logging_utils import get_logger

log = get_logger(__name__)


def _load_sample_images(params: dict[str, Any], n: int) -> list[tuple[str, bytes]]:
    """Real test images as encoded bytes, so decode cost is measured honestly."""
    raw_root = resolve(get(params, "data.raw_dir"))
    try:
        manifest = load_manifest(params, "test")
    except FileNotFoundError:
        manifest = None

    samples: list[tuple[str, bytes]] = []
    if manifest is not None and not manifest.empty:
        for relpath in manifest["relpath"].head(max(n, 1)):
            path = raw_root / relpath
            if path.is_file():
                samples.append((path.name, path.read_bytes()))
    if samples:
        return samples

    # Fall back to a synthetic frame so the benchmark still runs on a machine
    # with no dataset present.
    log.warning("No test images found; benchmarking with a generated image")
    buffer = io.BytesIO()
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (300, 300), dtype=np.uint8), mode="L").save(
        buffer, format="JPEG", quality=90
    )
    return [("synthetic.jpg", buffer.getvalue())]


def run_benchmark(
    params: dict[str, Any],
    *,
    base_url: str = "http://127.0.0.1:8000",
    n_requests: int = 200,
    concurrency: int = 4,
    warmup: int = 10,
) -> dict[str, Any]:
    """Fire *n_requests* at ``/predict`` and summarise the latency distribution."""
    import httpx

    samples = _load_sample_images(params, min(n_requests, 64))
    log.info("Benchmarking %s: %d requests, concurrency %d, %d distinct images",
             base_url, n_requests, concurrency, len(samples))

    # One pooled client shared by every worker. Opening a fresh connection per
    # request would measure TCP handshake cost rather than service cost, and on
    # a local service that overhead dominates -- it understates throughput by
    # roughly an order of magnitude. httpx.Client is thread-safe and keeps a
    # connection pool, which is also how a real caller would behave.
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    with httpx.Client(base_url=base_url, timeout=30.0, limits=limits) as client:
        try:
            ready = client.get("/readyz")
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Cannot reach the service at {base_url}. Start it with:\n"
                "    defectvision serve"
            ) from exc
        if ready.status_code != 200:
            raise RuntimeError(f"Service is not ready: {ready.status_code} {ready.text}")

        # Warm-up requests are discarded: the first few pay lazy-import and
        # allocator costs that never recur.
        for i in range(warmup):
            name, blob = samples[i % len(samples)]
            client.post("/predict", files={"file": (name, blob, "image/jpeg")})

        def one_request(index: int) -> tuple[float, int]:
            name, blob = samples[index % len(samples)]
            started = time.perf_counter()
            response = client.post("/predict", files={"file": (name, blob, "image/jpeg")})
            return (time.perf_counter() - started) * 1000.0, response.status_code

        wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(one_request, range(n_requests)))
        wall_elapsed = time.perf_counter() - wall_start

    latencies = np.array([r[0] for r in results])
    statuses = [r[1] for r in results]
    n_ok = sum(1 for s in statuses if s == 200)

    report = {
        "base_url": base_url,
        "n_requests": n_requests,
        "concurrency": concurrency,
        "n_succeeded": n_ok,
        "n_failed": n_requests - n_ok,
        "wall_seconds": round(wall_elapsed, 3),
        "throughput_rps": round(n_requests / wall_elapsed, 2) if wall_elapsed > 0 else 0.0,
        "latency_ms": {
            "mean": round(float(latencies.mean()), 2),
            "stdev": round(float(statistics.pstdev(latencies.tolist())), 2),
            "min": round(float(latencies.min()), 2),
            "p50": round(float(np.percentile(latencies, 50)), 2),
            "p90": round(float(np.percentile(latencies, 90)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
            "p99": round(float(np.percentile(latencies, 99)), 2),
            "max": round(float(latencies.max()), 2),
        },
        "status_codes": {str(code): statuses.count(code) for code in sorted(set(statuses))},
    }

    log.info("Throughput %.1f req/s | p50=%.1fms p95=%.1fms p99=%.1fms | %d/%d OK",
             report["throughput_rps"], report["latency_ms"]["p50"],
             report["latency_ms"]["p95"], report["latency_ms"]["p99"], n_ok, n_requests)

    import json

    from ..config import ensure_parent

    path = ensure_parent("reports/api_benchmark.json")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Benchmark report -> %s", path)
    return report
