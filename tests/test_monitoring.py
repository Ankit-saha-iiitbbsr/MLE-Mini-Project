"""M5 tests: drift maths, the prediction log, corruptions, and trigger policy."""

from __future__ import annotations

import numpy as np
import pytest

from defectvision.data.corruptions import apply_scenario, describe_scenario
from defectvision.features.image_stats import image_statistics
from defectvision.monitoring.drift import (
    classify_severity,
    ks_test,
    population_stability_index,
    quantile_bin_edges,
    summarise_drift,
)
from defectvision.monitoring.retrain_trigger import evaluate_rule
from defectvision.monitoring.store import PredictionRecord, PredictionStore

# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------


class TestPSI:
    def test_identical_distributions_score_near_zero(self, rng):
        sample = rng.normal(0, 1, 5000)
        psi, _ = population_stability_index(sample, sample)
        assert psi < 0.01

    def test_a_shift_produces_a_large_psi(self, rng):
        reference = rng.normal(0, 1, 5000)
        shifted = rng.normal(1.5, 1, 5000)
        psi, _ = population_stability_index(reference, shifted)
        assert psi > 0.25

    def test_psi_grows_monotonically_with_the_shift(self, rng):
        reference = rng.normal(0, 1, 5000)
        values = [
            population_stability_index(reference, rng.normal(delta, 1, 5000))[0]
            for delta in (0.0, 0.25, 0.5, 1.0, 2.0)
        ]
        assert values == sorted(values)

    def test_psi_is_insensitive_to_sample_size(self, rng):
        """The property that makes PSI safe to threshold on.

        A hypothesis test would grow more confident (smaller p) with more data
        for the same effect; PSI must report roughly the same magnitude.
        """
        reference = rng.normal(0, 1, 8000)
        edges = quantile_bin_edges(reference, 10)
        small, _ = population_stability_index(
            reference, rng.normal(0.6, 1, 400), edges=edges)
        large, _ = population_stability_index(
            reference, rng.normal(0.6, 1, 8000), edges=edges)
        assert small == pytest.approx(large, rel=0.45)

    def test_frozen_edges_are_reused(self, rng):
        """Recomputing bins per window would make PSI blind to shift."""
        reference = rng.normal(0, 1, 2000)
        edges = quantile_bin_edges(reference, 10)
        shifted = rng.normal(2.0, 1, 2000)

        with_frozen, _ = population_stability_index(reference, shifted, edges=edges)
        recomputed, _ = population_stability_index(shifted, shifted)
        assert with_frozen > 0.5
        assert recomputed < 0.01  # self-comparison always looks stable

    def test_handles_a_constant_feature(self):
        constant = np.full(500, 3.0)
        psi, _ = population_stability_index(constant, constant)
        assert np.isfinite(psi)

    def test_empty_bins_do_not_produce_infinity(self, rng):
        reference = rng.normal(0, 1, 1000)
        far_away = np.full(500, 50.0)
        psi, _ = population_stability_index(reference, far_away)
        assert np.isfinite(psi)
        assert psi > 0.25

    def test_severity_bands(self):
        assert classify_severity(0.05) == "none"
        assert classify_severity(0.15) == "moderate"
        assert classify_severity(0.40) == "significant"


class TestKS:
    def test_same_distribution_is_not_significant(self, rng):
        _, p = ks_test(rng.normal(0, 1, 1000), rng.normal(0, 1, 1000))
        assert p > 0.01

    def test_shifted_distribution_is_significant(self, rng):
        _, p = ks_test(rng.normal(0, 1, 1000), rng.normal(1.0, 1, 1000))
        assert p < 0.01

    def test_degenerate_input_returns_nan(self):
        stat, p = ks_test(np.array([1.0]), np.array([2.0]))
        assert np.isnan(stat) and np.isnan(p)


class TestSummarise:
    def test_empty_input_is_handled(self):
        summary = summarise_drift({})
        assert summary["psi_max"] == 0.0
        assert summary["drifted_features"] == []


# ---------------------------------------------------------------------------
# Corruptions
# ---------------------------------------------------------------------------


class TestCorruptions:
    def test_empty_scenario_is_a_no_op(self, sample_image):
        result = apply_scenario(sample_image, {})
        np.testing.assert_array_equal(np.asarray(result), np.asarray(sample_image))

    def test_brightness_moves_mean_intensity_the_right_way(self, sample_image):
        base = image_statistics(sample_image)["mean_intensity"]
        brighter = image_statistics(apply_scenario(sample_image, {"brightness": 0.25}))
        dimmer = image_statistics(apply_scenario(sample_image, {"brightness": -0.25}))
        assert brighter["mean_intensity"] > base > dimmer["mean_intensity"]

    def test_blur_reduces_the_focus_measure(self, sample_image):
        base = image_statistics(sample_image)["laplacian_var"]
        blurred = image_statistics(apply_scenario(sample_image, {"blur_sigma": 2.5}))
        assert blurred["laplacian_var"] < base

    def test_noise_raises_the_focus_measure(self, sample_image):
        """Noise and blur move laplacian_var in opposite directions.

        Worth pinning: it means the statistic alone cannot distinguish the two
        failures, which is exactly why several features are monitored together.
        """
        base = image_statistics(sample_image)["laplacian_var"]
        noisy = image_statistics(apply_scenario(sample_image, {"noise_std": 0.08}, seed=0))
        assert noisy["laplacian_var"] > base

    def test_geometry_preserves_image_size(self, sample_image):
        result = apply_scenario(sample_image, {"rotation_deg": 25, "scale": 0.8})
        assert result.size == sample_image.size

    def test_corruption_is_deterministic_for_a_seed(self, sample_image):
        a = apply_scenario(sample_image, {"noise_std": 0.05}, seed=3)
        b = apply_scenario(sample_image, {"noise_std": 0.05}, seed=3)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_unknown_parameter_raises(self, sample_image):
        """A typo must fail loudly, not silently return an uncorrupted image."""
        with pytest.raises(ValueError, match="Unknown drift scenario parameter"):
            apply_scenario(sample_image, {"brihgtness": 0.2})

    def test_describe_scenario_is_readable(self):
        text = describe_scenario({"brightness": -0.35, "contrast": 0.85})
        assert "brightness" in text and "contrast" in text


# ---------------------------------------------------------------------------
# Prediction store
# ---------------------------------------------------------------------------


class TestPredictionStore:
    @pytest.fixture
    def store(self, tmp_path):
        return PredictionStore(tmp_path / "predictions.db")

    def _record(self, i: int, **kwargs) -> PredictionRecord:
        return PredictionRecord(
            probability=0.1 * (i % 10),
            predicted_label=i % 2,
            predicted_class="defect" if i % 2 else "ok",
            decision="auto_reject" if i % 2 else "auto_accept",
            model_name="test",
            latency_ms=5.0,
            image_stats={
                "mean_intensity": 0.5, "std_intensity": 0.2, "p05_intensity": 0.1,
                "p95_intensity": 0.9, "edge_density": 0.15, "laplacian_var": 0.01,
                "entropy": 4.2,
            },
            **kwargs,
        )

    def test_log_and_count(self, store):
        store.log(self._record(1))
        assert store.count() == 1

    def test_bulk_insert(self, store):
        assert store.log_many([self._record(i) for i in range(50)]) == 50
        assert store.count() == 50

    def test_fetch_returns_the_logged_columns(self, store):
        store.log(self._record(1, filename="a.png"))
        df = store.fetch()
        assert df.loc[0, "filename"] == "a.png"
        assert df.loc[0, "mean_intensity"] == pytest.approx(0.5)

    def test_scenario_filtering(self, store):
        store.log_many([self._record(i, scenario="dim") for i in range(5)])
        store.log_many([self._record(i, scenario="bright") for i in range(3)])
        assert len(store.fetch(scenario="dim")) == 5
        assert set(store.scenarios()) == {"dim", "bright"}

    def test_ground_truth_can_arrive_later(self, store):
        """Labels on a real line arrive after the prediction, not with it."""
        record = self._record(1)
        store.log(record)
        assert store.attach_ground_truth(record.request_id, 1, "teardown") is True

        row = store.fetch().iloc[0]
        assert row["ground_truth"] == 1
        assert row["feedback_source"] == "teardown"

    def test_ground_truth_for_unknown_id_returns_false(self, store):
        assert store.attach_ground_truth("nope", 1) is False

    def test_latest_window_is_ordered_oldest_first(self, store):
        store.log_many([self._record(i) for i in range(30)])
        window = store.latest_window(10)
        assert len(window) == 10
        assert window["ts"].is_monotonic_increasing

    def test_summary_counts(self, store):
        store.log_many([self._record(i) for i in range(20)])
        summary = store.summary()
        assert summary["n_predictions"] == 20
        assert 0.0 <= summary["review_rate"] <= 1.0

    def test_export_jsonl(self, store, tmp_path):
        store.log_many([self._record(i) for i in range(5)])
        path = store.export_jsonl(tmp_path / "log.jsonl")
        assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 5


# ---------------------------------------------------------------------------
# Trigger rules
# ---------------------------------------------------------------------------


class TestTriggerRules:
    def test_breach_is_detected(self):
        rule = {"name": "data_drift_psi", "metric": "psi_max", "op": ">=",
                "threshold": 0.25, "severity": "high"}
        assert evaluate_rule(rule, {"psi_max": 0.4})["breached"] is True
        assert evaluate_rule(rule, {"psi_max": 0.1})["breached"] is False

    def test_missing_metric_does_not_breach(self):
        """An unavailable metric (no labels yet) must never fire a trigger."""
        rule = {"name": "accuracy_degradation", "metric": "accuracy_drop",
                "op": ">=", "threshold": 0.05}
        result = evaluate_rule(rule, {})
        assert result["breached"] is False
        assert "unavailable" in result["detail"]

    def test_nan_metric_does_not_breach(self):
        rule = {"name": "accuracy_degradation", "metric": "accuracy_drop",
                "op": ">=", "threshold": 0.05}
        assert evaluate_rule(rule, {"accuracy_drop": float("nan")})["breached"] is False
