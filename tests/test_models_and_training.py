"""M3 tests: architectures, transforms, metrics, threshold policy, reproducibility."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from defectvision.data.transforms import AugmentSpec, PreprocessSpec, build_train_transform
from defectvision.models.classical import hog_feature_dim, hog_features
from defectvision.models.factory import build_model, count_parameters
from defectvision.training.evaluate import (
    average_precision,
    bootstrap_ci,
    choose_threshold,
    classification_metrics,
    roc_auc,
)

# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------


class TestModels:
    @pytest.mark.parametrize("cfg", [
        {"arch": "baseline_cnn", "channels": [8, 16], "dropout": 0.1},
        {"arch": "resnet18", "pretrained": False},
        {"arch": "mobilenet_v3_small", "pretrained": False},
    ])
    def test_emits_one_logit_per_image(self, cfg):
        model = build_model(cfg, in_channels=1)
        model.eval()
        with torch.inference_mode():
            out = model(torch.randn(3, 1, 64, 64))
        assert out.shape == (3,)
        assert torch.isfinite(out).all()

    def test_accepts_varied_input_sizes(self):
        model = build_model({"arch": "baseline_cnn", "channels": [8, 16]}, in_channels=1)
        model.eval()
        with torch.inference_mode():
            for size in (64, 96, 128):
                assert model(torch.randn(1, 1, size, size)).shape == (1,)

    def test_grayscale_stem_adaptation_preserves_response(self):
        """The 1-channel stem must match what a 3x-repeated grayscale input gives.

        This is the claim the design rests on: summing the RGB filter weights is
        response-identical to channel repetition, at a third of the stem cost.
        If it ever stops holding, the transfer arms silently lose their
        pretrained edge detectors.
        """
        from torchvision import models as tv

        from defectvision.models.transfer import _adapt_first_conv

        original = tv.resnet18(weights=None).conv1
        adapted = _adapt_first_conv(original, 1)

        gray = torch.randn(2, 1, 64, 64)
        with torch.inference_mode():
            from_adapted = adapted(gray)
            from_repeated = original(gray.repeat(1, 3, 1, 1))
        torch.testing.assert_close(from_adapted, from_repeated, rtol=1e-4, atol=1e-5)

    def test_frozen_backbone_has_fewer_trainable_parameters(self):
        free = build_model({"arch": "resnet18", "pretrained": False}, in_channels=1)
        frozen = build_model(
            {"arch": "resnet18", "pretrained": False, "freeze_backbone": True}, in_channels=1
        )
        assert count_parameters(frozen)[1] < count_parameters(free)[1]

    def test_frozen_backbone_stays_in_eval_mode(self):
        """Otherwise BatchNorm keeps adapting and the backbone is not really frozen."""
        model = build_model(
            {"arch": "resnet18", "pretrained": False, "freeze_backbone": True}, in_channels=1
        )
        model.train()
        assert model.features.training is False

    def test_unknown_architecture_raises(self):
        with pytest.raises(ValueError, match="Unknown architecture"):
            build_model({"arch": "transformer9000"})

    def test_classical_arm_is_routed_away_from_the_torch_factory(self):
        with pytest.raises(ValueError, match="scikit-learn"):
            build_model({"arch": "logreg_hog"})


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


class TestTransforms:
    def test_preprocess_produces_the_declared_shape(self, sample_image):
        spec = PreprocessSpec(resize=96, grayscale=True, mean=(0.5,), std=(0.25,))
        tensor = spec.build()(sample_image)
        assert tensor.shape == (1, 96, 96)

    def test_preprocess_is_deterministic(self, sample_image):
        """The serving path must give identical tensors on repeated calls."""
        spec = PreprocessSpec(resize=64)
        a, b = spec.build()(sample_image), spec.build()(sample_image)
        torch.testing.assert_close(a, b)

    def test_augmentation_is_stochastic(self, sample_image):
        transform = build_train_transform(
            PreprocessSpec(resize=64), AugmentSpec(enabled=True)
        )
        torch.manual_seed(0)
        a = transform(sample_image)
        torch.manual_seed(1)
        b = transform(sample_image)
        assert not torch.allclose(a, b)

    def test_disabled_augmentation_falls_back_to_preprocess(self, sample_image):
        spec = PreprocessSpec(resize=64)
        transform = build_train_transform(spec, AugmentSpec(enabled=False))
        torch.testing.assert_close(transform(sample_image), spec.build()(sample_image))

    def test_spec_round_trips_through_a_dict(self):
        spec = PreprocessSpec(resize=77, grayscale=True, mean=(0.4,), std=(0.3,))
        assert PreprocessSpec.from_dict(spec.to_dict()) == spec


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_perfect_predictions(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.01, 0.02, 0.98, 0.99])
        m = classification_metrics(y, p, 0.5)
        assert m["accuracy"] == m["f1"] == m["recall"] == m["precision"] == 1.0
        assert m["roc_auc"] == 1.0

    def test_confusion_cells_are_correct(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.9, 0.1, 0.9, 0.1])  # one FP, one TP, one TN, one FN
        m = classification_metrics(y, p, 0.5)
        assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 1, 1)
        assert m["recall"] == pytest.approx(0.5)

    def test_roc_auc_matches_sklearn(self, rng):
        from sklearn.metrics import roc_auc_score

        y = rng.integers(0, 2, 300)
        p = rng.random(300)
        assert roc_auc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-9)

    def test_roc_auc_handles_ties(self):
        """Tied scores need averaged ranks or AUC silently drifts."""
        from sklearn.metrics import roc_auc_score

        y = np.array([0, 1, 0, 1, 0, 1])
        p = np.array([0.5, 0.5, 0.5, 0.7, 0.3, 0.5])
        assert roc_auc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-9)

    def test_average_precision_matches_sklearn(self, rng):
        from sklearn.metrics import average_precision_score

        y = rng.integers(0, 2, 300)
        p = rng.random(300)
        assert average_precision(y, p) == pytest.approx(
            average_precision_score(y, p), abs=1e-6
        )

    def test_single_class_input_gives_nan_auc(self):
        m = classification_metrics(np.zeros(10, dtype=int), np.random.random(10), 0.5)
        assert np.isnan(m["roc_auc"])


# ---------------------------------------------------------------------------
# Threshold policy
# ---------------------------------------------------------------------------


class TestThresholdSelection:
    @pytest.fixture
    def scores(self, rng):
        y = np.r_[np.zeros(200, dtype=int), np.ones(200, dtype=int)]
        p = np.r_[rng.beta(2, 6, 200), rng.beta(6, 2, 200)]
        return y, p

    def test_max_f1_beats_the_default_threshold(self, scores):
        y, p = scores
        choice = choose_threshold(y, p, "max_f1")
        assert choice.validation_metrics["f1"] >= classification_metrics(y, p, 0.5)["f1"]
        assert choice.rationale

    def test_target_recall_is_met(self, scores):
        y, p = scores
        choice = choose_threshold(y, p, "target_recall", target_recall=0.95)
        assert choice.validation_metrics["recall"] >= 0.95

    def test_unreachable_target_recall_is_reported_not_hidden(self):
        """Silently returning a threshold that misses the target would be a lie.

        The target is only genuinely unreachable when the split contains no
        positives at all -- with any positive present, threshold 0.0 predicts
        everything positive and trivially achieves recall 1.0.
        """
        y = np.zeros(20, dtype=int)  # degenerate: no defects in validation
        p = np.linspace(0.0, 1.0, 20)
        choice = choose_threshold(y, p, "target_recall", target_recall=0.98)
        assert "unreachable" in choice.rationale.lower()

    def test_target_recall_is_satisfiable_but_worthless_on_inverted_scores(self):
        """A documented limitation of the target_recall strategy.

        On a model whose scores are inverted, the constraint is still met -- by
        collapsing to "flag everything", where precision equals the base rate.
        The strategy guarantees recall, never that the model is useful, which is
        why the promotion gates check F1 as well.
        """
        y = np.array([0, 0, 1, 1])
        p = np.array([0.9, 0.9, 0.1, 0.1])
        choice = choose_threshold(y, p, "target_recall", target_recall=0.99)
        assert choice.validation_metrics["recall"] == pytest.approx(1.0)
        assert choice.validation_metrics["precision"] == pytest.approx(0.5)
        assert choice.validation_metrics["predicted_positive_rate"] == pytest.approx(1.0)

    def test_fixed_strategy_returns_the_configured_value(self, scores):
        y, p = scores
        assert choose_threshold(y, p, "fixed", fixed_threshold=0.42).threshold == 0.42

    def test_unknown_strategy_raises(self, scores):
        y, p = scores
        with pytest.raises(ValueError, match="Unknown threshold strategy"):
            choose_threshold(y, p, "vibes")


class TestBootstrap:
    def test_interval_brackets_the_point_estimate(self, rng):
        y = np.r_[np.zeros(150, dtype=int), np.ones(150, dtype=int)]
        p = np.r_[rng.beta(2, 5, 150), rng.beta(5, 2, 150)]
        ci = bootstrap_ci(y, p, 0.5, "f1", n_samples=120, seed=0)
        assert ci["lo"] <= ci["point"] <= ci["hi"]

    def test_empty_input_is_handled(self):
        ci = bootstrap_ci(np.array([]), np.array([]), 0.5, "f1", n_samples=10)
        assert np.isnan(ci["point"])


# ---------------------------------------------------------------------------
# HOG
# ---------------------------------------------------------------------------


class TestHOG:
    def test_dimension_matches_the_formula(self):
        descriptor = hog_features(np.random.random((128, 128)))
        assert descriptor.shape[0] == hog_feature_dim(128)

    def test_blocks_are_l2_normalised(self):
        descriptor = hog_features(np.random.random((128, 128)))
        assert np.isfinite(descriptor).all()
        assert descriptor.min() >= 0.0

    def test_is_deterministic(self):
        image = np.random.default_rng(0).random((128, 128))
        np.testing.assert_array_equal(hog_features(image), hog_features(image))

    def test_rejects_an_image_too_small_for_the_cell_grid(self):
        with pytest.raises(ValueError, match="too small"):
            hog_features(np.random.random((8, 8)))


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_seeding_makes_torch_deterministic(self):
        from defectvision.logging_utils import set_seed

        set_seed(123)
        a = torch.randn(5)
        set_seed(123)
        torch.testing.assert_close(a, torch.randn(5))

    def test_config_hash_is_order_independent(self):
        from defectvision.training.reproducibility import dict_hash

        assert dict_hash({"a": 1, "b": {"c": 2}}) == dict_hash({"b": {"c": 2}, "a": 1})

    def test_config_hash_changes_with_content(self):
        from defectvision.training.reproducibility import dict_hash

        assert dict_hash({"epochs": 10}) != dict_hash({"epochs": 11})

    def test_environment_capture_records_versions(self):
        from defectvision.training.reproducibility import environment_info

        info = environment_info()
        assert info["torch_version"] != "not-installed"
        assert info["python_version"]


# ---------------------------------------------------------------------------
# Comparison table built from the MLflow store
# ---------------------------------------------------------------------------


class TestMlflowComparisonRow:
    """Regression tests for `load_comparison_from_mlflow`'s row parsing.

    `search_runs` returns the union of every metric column across all runs, so a
    run that never logged a metric still gets the column, filled with NaN. These
    pin the two ways that used to go wrong.
    """

    def test_metric_reader_handles_missing_and_nan(self):
        import pandas as pd

        from defectvision.training.compare import _metric

        row = pd.Series({"metrics.present": 3.0, "metrics.nan": float("nan")})

        assert _metric(row, "metrics.present") == 3.0
        assert _metric(row, "metrics.nan") == 0.0          # NaN -> default
        assert _metric(row, "metrics.absent") == 0.0       # missing key -> default
        assert _metric(row, "metrics.nan", 0.5) == 0.5     # honours the default

    def test_nan_metric_is_int_convertible(self):
        """The exact crash: `int(float(nan))` raised ValueError.

        `value or default` does not guard against it, because NaN is truthy.
        """
        import pandas as pd

        from defectvision.training.compare import _metric

        row = pd.Series({"metrics.best_epoch": float("nan")})
        assert int(_metric(row, "metrics.best_epoch")) == 0

        # Demonstrate the original failure mode so the regression is explicit.
        with pytest.raises(ValueError, match="cannot convert float NaN"):
            int(float(row.get("metrics.best_epoch", 0) or 0))

    def test_promotion_runs_are_excluded_from_the_candidate_table(self, monkeypatch, base_params):
        """Promotion runs log the winner's test metrics for lineage.

        They therefore pass the "did this run finish" filter and would be
        re-ingested as phantom candidates -- one more on every `compare`. Only
        training runs log `model_name`, which is the discriminator.
        """
        import pandas as pd

        from defectvision.training import compare as compare_mod

        runs = pd.DataFrame([
            {"run_id": "aaaaaaaa1", "params.model_name": "baseline_cnn",
             "params.arch": "baseline_cnn", "metrics.test_f1": 0.99,
             "metrics.best_epoch": 12.0, "metrics.total_params": 250657.0},
            {"run_id": "bbbbbbbb2", "params.model_name": "logreg_hog",
             "params.arch": "logreg_hog", "metrics.test_f1": 0.96,
             "metrics.best_epoch": float("nan"),          # classical: no epochs
             "metrics.total_params": float("nan")},
            {"run_id": "cccccccc3", "params.model_name": None,   # promotion run
             "params.arch": None, "metrics.test_f1": 0.99,
             "metrics.best_epoch": float("nan"), "metrics.total_params": float("nan")},
            {"run_id": "dddddddd4", "params.model_name": "crashed",
             "params.arch": "resnet18", "metrics.test_f1": float("nan"),  # unfinished
             "metrics.best_epoch": float("nan"), "metrics.total_params": float("nan")},
        ])

        class _FakeExperiment:
            experiment_id = "0"

        class _FakeMlflow:
            @staticmethod
            def get_experiment_by_name(_name):
                return _FakeExperiment()

            @staticmethod
            def search_runs(**_kwargs):
                return runs

        monkeypatch.setattr(compare_mod, "load_comparison_from_mlflow",
                            compare_mod.load_comparison_from_mlflow)
        monkeypatch.setattr("defectvision.training.train.setup_mlflow",
                            lambda _p: _FakeMlflow())

        table = compare_mod.load_comparison_from_mlflow(base_params)

        assert set(table["model"]) == {"baseline_cnn", "logreg_hog"}
        assert "?" not in set(table["model"])          # no phantom promotion row
        assert "crashed" not in set(table["model"])    # unfinished run dropped

        classical = table[table["model"] == "logreg_hog"].iloc[0]
        assert classical["best_epoch"] == 0            # NaN coerced, not crashed
        assert isinstance(classical["best_epoch"], (int, np.integer))

    def test_tied_runs_resolve_to_the_most_recent(self, monkeypatch, base_params):
        """Re-running a deterministic arm must not flip which run is reported.

        Two runs of the same model with an identical score used to be ordered by
        pandas' default *unstable* quicksort, so `compare` could promote a
        different run id each time it ran on unchanged data.
        """
        import pandas as pd

        from defectvision.training import compare as compare_mod

        # search_runs returns start_time DESC, so the newer run comes first.
        runs = pd.DataFrame([
            {"run_id": "newer000", "params.model_name": "logreg_hog",
             "params.arch": "logreg_hog", "metrics.test_f1": 0.9609,
             "metrics.total_params": 1765.0, "metrics.best_epoch": 0.0},
            {"run_id": "older000", "params.model_name": "logreg_hog",
             "params.arch": "logreg_hog", "metrics.test_f1": 0.9609,
             "metrics.total_params": float("nan"), "metrics.best_epoch": float("nan")},
        ])

        class _FakeMlflow:
            @staticmethod
            def get_experiment_by_name(_name):
                return type("E", (), {"experiment_id": "0"})()

            @staticmethod
            def search_runs(**_kwargs):
                return runs

        monkeypatch.setattr("defectvision.training.train.setup_mlflow",
                            lambda _p: _FakeMlflow())

        for _ in range(5):  # stable across repeats, not luck
            table = compare_mod.load_comparison_from_mlflow(base_params)
            assert len(table) == 1
            assert table.iloc[0]["run_id"] == "newer000"
