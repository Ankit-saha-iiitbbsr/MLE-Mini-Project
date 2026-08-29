"""M4 tests: the API contract, and above all the error paths.

The happy path is the easy half. Most of this file is malformed input, because
"handle malformed/edge-case inputs" is an explicit requirement and because a
service that returns 500 for a corrupt upload makes real 500s undiagnosable.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from defectvision.serving.predictor import (
    ImageDecodeError,
    PayloadTooLarge,
    decode_image,
)

# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


class TestDecodeImage:
    def test_decodes_a_valid_png(self, image_bytes):
        image = decode_image(image_bytes, filename="a.png", content_type="image/png")
        assert image.mode == "L"
        assert image.size == (64, 64)

    def test_decodes_jpeg(self, sample_image):
        buffer = io.BytesIO()
        sample_image.save(buffer, format="JPEG")
        image = decode_image(buffer.getvalue(), content_type="image/jpeg")
        assert image.mode == "L"

    def test_rejects_empty_upload(self):
        with pytest.raises(ImageDecodeError, match="empty"):
            decode_image(b"")

    def test_rejects_non_image_bytes(self):
        with pytest.raises(ImageDecodeError, match="not a recognisable image"):
            decode_image(b"#!/bin/sh\necho hello\n", filename="script.sh")

    def test_rejects_truncated_image(self, image_bytes):
        with pytest.raises(ImageDecodeError):
            decode_image(image_bytes[: len(image_bytes) // 3], filename="cut.png")

    def test_rejects_unsupported_content_type(self, image_bytes):
        with pytest.raises(ImageDecodeError, match="Unsupported content type"):
            decode_image(image_bytes, content_type="application/pdf")

    def test_rejects_oversized_upload(self, image_bytes):
        with pytest.raises(PayloadTooLarge, match="limit"):
            decode_image(image_bytes, max_bytes=10)

    def test_errors_carry_an_actionable_hint(self):
        with pytest.raises(ImageDecodeError) as excinfo:
            decode_image(b"")
        assert excinfo.value.hint


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class TestPredictor:
    def test_reports_not_ready_without_a_bundle(self, tiny_params, tmp_path):
        from defectvision.config import apply_overrides
        from defectvision.serving.predictor import Predictor

        params = apply_overrides(tiny_params, {
            "serving.model_bundle": str(tmp_path / "does_not_exist.pt"),
        })
        predictor = Predictor(params)
        assert predictor.load() is False
        assert predictor.ready is False
        assert predictor.load_error  # the reason is retained for /readyz

    def test_review_band_routes_borderline_scores_to_a_human(self, tiny_params):
        from defectvision.serving.predictor import Predictor

        predictor = Predictor(tiny_params)
        low, high = predictor.review_low, predictor.review_high
        midpoint = (low + high) / 2

        assert predictor._decide(midpoint, 0.5) == "human_review"
        assert predictor._decide(high + 0.2, 0.5) == "auto_reject"
        assert predictor._decide(max(low - 0.2, 0.0), 0.5) == "auto_accept"


# ---------------------------------------------------------------------------
# Bundle round-trip
# ---------------------------------------------------------------------------


class TestBundle:
    def test_round_trip_preserves_behaviour(self, tmp_path, sample_image):
        """A bundle must reproduce the in-memory model's outputs exactly.

        This is the packaging contract: if the reloaded model disagreed with the
        trained one, every offline metric would be a lie about production.
        """
        import torch

        from defectvision.bundle import BundleMetadata, load_bundle, save_bundle
        from defectvision.data.transforms import PreprocessSpec
        from defectvision.models.factory import build_model

        spec = PreprocessSpec(resize=64, grayscale=True, mean=(0.485,), std=(0.229,))
        cfg = {"arch": "baseline_cnn", "channels": [8, 16], "dropout": 0.1}
        model = build_model(cfg, in_channels=1)
        model.eval()

        batch = spec.build()(sample_image).unsqueeze(0)
        with torch.inference_mode():
            expected = torch.sigmoid(model(batch)).item()

        path = tmp_path / "bundle.pt"
        save_bundle(path, model, BundleMetadata(
            model_name="test_cnn", arch="baseline_cnn", model_config=cfg,
            preprocess=spec.to_dict(), classes=["ok", "defect"], threshold=0.37,
        ))

        loaded = load_bundle(path)
        assert loaded.threshold == pytest.approx(0.37)
        assert loaded.classes == ["ok", "defect"]
        assert loaded.predict_proba(sample_image)[0] == pytest.approx(expected, abs=1e-6)

    def test_preprocessing_travels_with_the_weights(self, tmp_path):
        """Config drift after shipping must not change how a model sees an image."""
        from defectvision.bundle import BundleMetadata, load_bundle, save_bundle
        from defectvision.data.transforms import PreprocessSpec
        from defectvision.models.factory import build_model

        spec = PreprocessSpec(resize=48, grayscale=True, mean=(0.5,), std=(0.25,))
        cfg = {"arch": "baseline_cnn", "channels": [8], "dropout": 0.0}
        path = tmp_path / "b.pt"
        save_bundle(path, build_model(cfg, in_channels=1), BundleMetadata(
            model_name="m", arch="baseline_cnn", model_config=cfg,
            preprocess=spec.to_dict(), classes=["ok", "defect"], threshold=0.5,
        ))

        loaded = load_bundle(path)
        assert loaded.preprocess_spec.resize == 48
        assert loaded.preprocess_spec.mean == (0.5,)

    def test_rejects_an_unsupported_format_version(self, tmp_path):
        import torch

        from defectvision.bundle import load_bundle

        torch.save({"format_version": 999}, tmp_path / "future.pt")
        with pytest.raises(ValueError, match="format version"):
            load_bundle(tmp_path / "future.pt")

    def test_missing_bundle_gives_an_actionable_error(self, tmp_path):
        from defectvision.bundle import load_bundle

        with pytest.raises(FileNotFoundError, match="defectvision train"):
            load_bundle(tmp_path / "nope.pt")


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


@pytest.fixture
def api_client(tmp_path, tiny_params, monkeypatch):
    """A TestClient backed by a real (tiny, untrained) bundle."""
    from fastapi.testclient import TestClient

    from defectvision.bundle import BundleMetadata, save_bundle
    from defectvision.config import apply_overrides
    from defectvision.data.transforms import PreprocessSpec
    from defectvision.models.factory import build_model

    bundle_path = tmp_path / "model_bundle.pt"
    spec = PreprocessSpec(resize=64, grayscale=True, mean=(0.485,), std=(0.229,))
    cfg = {"arch": "baseline_cnn", "channels": [8, 16], "dropout": 0.0}
    save_bundle(bundle_path, build_model(cfg, in_channels=1), BundleMetadata(
        model_name="test_cnn", arch="baseline_cnn", model_config=cfg,
        preprocess=spec.to_dict(), classes=["ok", "defect"], threshold=0.5,
        metrics={"test": {"f1": 0.99, "accuracy": 0.99}},
    ))

    params = apply_overrides(tiny_params, {"serving.model_bundle": str(bundle_path)})
    monkeypatch.setattr("defectvision.serving.app.load_params", lambda: params)

    from defectvision.serving.app import app

    with TestClient(app) as client:
        yield client


class TestAPI:
    def test_healthz_is_always_ok(self, api_client):
        response = api_client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_readyz_reports_the_loaded_model(self, api_client):
        response = api_client.get("/readyz")
        assert response.status_code == 200
        body = response.json()
        assert body["model_loaded"] is True
        assert body["model_name"] == "test_cnn"

    def test_model_endpoint_returns_the_model_card(self, api_client):
        body = api_client.get("/model").json()
        assert body["model_name"] == "test_cnn"
        assert body["classes"] == ["ok", "defect"]
        assert body["input_size"] == 64

    def test_predict_returns_a_complete_response(self, api_client, image_bytes):
        response = api_client.post(
            "/predict", files={"file": ("part.png", image_bytes, "image/png")}
        )
        assert response.status_code == 200
        body = response.json()

        assert body["predicted_class"] in ("ok", "defect")
        assert 0.0 <= body["probability_defect"] <= 1.0
        assert body["decision"] in ("auto_accept", "auto_reject", "human_review")
        assert body["threshold"] == pytest.approx(0.5)
        assert set(body["image_stats"]) >= {"mean_intensity", "edge_density", "entropy"}
        assert body["model"]["name"] == "test_cnn"
        assert response.headers["X-Request-ID"] == body["request_id"]

    def test_confidence_refers_to_the_predicted_class(self, api_client, image_bytes):
        body = api_client.post(
            "/predict", files={"file": ("p.png", image_bytes, "image/png")}
        ).json()
        expected = (body["probability_defect"] if body["predicted_label"] == 1
                    else 1.0 - body["probability_defect"])
        assert body["confidence"] == pytest.approx(expected)

    # -- error paths ------------------------------------------------------

    def test_missing_file_field_is_422(self, api_client):
        assert api_client.post("/predict").status_code == 422

    def test_empty_file_is_400_with_a_hint(self, api_client):
        response = api_client.post("/predict", files={"file": ("e.png", b"", "image/png")})
        assert response.status_code == 400
        body = response.json()
        assert body["error"] == "invalid_image"
        assert body["hint"]

    def test_garbage_bytes_are_400_not_500(self, api_client):
        response = api_client.post(
            "/predict", files={"file": ("x.png", b"not an image at all", "image/png")}
        )
        assert response.status_code == 400
        assert response.json()["error"] == "invalid_image"

    def test_unsupported_content_type_is_415(self, api_client, image_bytes):
        response = api_client.post(
            "/predict", files={"file": ("doc.pdf", image_bytes, "application/pdf")}
        )
        assert response.status_code == 415

    def test_oversized_upload_is_413(self, api_client):
        big = io.BytesIO()
        rng = np.random.default_rng(0)
        Image.fromarray(rng.integers(0, 255, (3000, 3000), dtype=np.uint8), mode="L").save(
            big, format="PNG"
        )
        response = api_client.post(
            "/predict", files={"file": ("big.png", big.getvalue(), "image/png")}
        )
        assert response.status_code == 413

    # -- batch ------------------------------------------------------------

    def test_batch_returns_one_item_per_file(self, api_client, image_bytes):
        files = [("files", (f"p{i}.png", image_bytes, "image/png")) for i in range(3)]
        body = api_client.post("/predict/batch", files=files).json()
        assert body["n_requested"] == 3
        assert body["n_succeeded"] == 3
        assert len(body["items"]) == 3

    def test_batch_isolates_a_bad_file(self, api_client, image_bytes):
        """One corrupt file must not discard the good predictions beside it."""
        files = [
            ("files", ("good1.png", image_bytes, "image/png")),
            ("files", ("bad.png", b"garbage", "image/png")),
            ("files", ("good2.png", image_bytes, "image/png")),
        ]
        response = api_client.post("/predict/batch", files=files)
        assert response.status_code == 200

        body = response.json()
        assert body["n_succeeded"] == 2
        assert body["n_failed"] == 1
        assert body["items"][1]["success"] is False
        assert body["items"][0]["result"]["predicted_class"] in ("ok", "defect")

    def test_batch_over_the_limit_is_413(self, api_client, image_bytes):
        files = [("files", (f"p{i}.png", image_bytes, "image/png")) for i in range(40)]
        response = api_client.post("/predict/batch", files=files)
        assert response.status_code == 413
        assert response.json()["error"] == "batch_too_large"

    # -- monitoring integration -------------------------------------------

    def test_feedback_attaches_ground_truth(self, api_client, image_bytes):
        predicted = api_client.post(
            "/predict", files={"file": ("p.png", image_bytes, "image/png")}
        ).json()

        response = api_client.post("/feedback", json={
            "request_id": predicted["request_id"], "ground_truth": 1, "source": "teardown",
        })
        assert response.status_code == 200
        assert response.json()["recorded"] is True

    def test_feedback_for_an_unknown_request_is_404(self, api_client):
        response = api_client.post("/feedback", json={
            "request_id": "does-not-exist", "ground_truth": 0,
        })
        assert response.status_code == 404

    def test_metrics_counts_predictions(self, api_client, image_bytes):
        for _ in range(3):
            api_client.post("/predict", files={"file": ("p.png", image_bytes, "image/png")})
        body = api_client.get("/metrics").json()
        assert body["predictions_total"] >= 3
        assert body["latency_p50_ms"] is not None
