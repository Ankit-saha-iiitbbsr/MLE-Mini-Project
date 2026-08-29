"""M2 tests: generation, statistics, validation gates, and split integrity."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from defectvision.data.acquire import acquire_synthetic
from defectvision.data.split import build_manifest, load_manifest
from defectvision.data.synth import DEFECT_KINDS, render_casting
from defectvision.data.validate import DataValidationError, dhash, run_validation, sha256_file
from defectvision.features.image_stats import FEATURE_NAMES, image_statistics, to_vector

# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------


class TestSynthGenerator:
    def test_is_deterministic_in_seed_and_index(self):
        a, meta_a = render_casting(5, seed=42, size=64, defective=True)
        b, meta_b = render_casting(5, seed=42, size=64, defective=True)
        np.testing.assert_array_equal(a, b)
        assert meta_a.rotation_deg == meta_b.rotation_deg

    def test_different_seeds_give_different_images(self):
        a, _ = render_casting(5, seed=1, size=64, defective=True)
        b, _ = render_casting(5, seed=2, size=64, defective=True)
        assert not np.array_equal(a, b)

    def test_shape_and_dtype(self):
        img, _ = render_casting(0, size=96)
        assert img.shape == (96, 96)
        assert img.dtype == np.uint8

    def test_defective_images_record_their_defects(self):
        _, meta = render_casting(3, size=64, defective=True)
        assert meta.label == 1
        assert 1 <= meta.n_defects <= 3
        assert all(kind in DEFECT_KINDS for kind in meta.defect_kinds)

    def test_clean_images_have_no_defects(self):
        _, meta = render_casting(3, size=64, defective=False)
        assert meta.label == 0
        assert meta.n_defects == 0

    def test_no_global_statistic_separates_the_classes(self):
        """The generator must not leave a shortcut a CNN could exploit.

        If any single global statistic separated the classes, the task would be
        solvable without localising a defect and the whole benchmark would be
        meaningless. Guarding it here stops a future tweak from silently
        reintroducing one.
        """
        rows, labels = [], []
        for i in range(120):
            label = i % 2
            img, _ = render_casting(i, size=64, difficulty=0.45, defective=bool(label))
            rows.append(to_vector(image_statistics(img)))
            labels.append(label)

        matrix, labels = np.array(rows), np.array(labels)
        group_a, group_b = matrix[labels == 0], matrix[labels == 1]
        for j, name in enumerate(FEATURE_NAMES):
            pooled = np.sqrt((group_a[:, j].var() + group_b[:, j].var()) / 2)
            if pooled < 1e-9:
                continue
            cohens_d = abs(group_a[:, j].mean() - group_b[:, j].mean()) / pooled
            assert cohens_d < 1.5, f"{name} separates the classes too well (d={cohens_d:.2f})"


# ---------------------------------------------------------------------------
# Image statistics
# ---------------------------------------------------------------------------


class TestImageStatistics:
    def test_returns_every_declared_feature(self, sample_image):
        stats = image_statistics(sample_image)
        assert set(stats) == set(FEATURE_NAMES)
        assert all(np.isfinite(v) for v in stats.values())

    def test_uint8_and_float_inputs_agree(self, sample_image):
        as_uint8 = np.asarray(sample_image)
        from_uint8 = image_statistics(as_uint8)
        from_float = image_statistics(as_uint8.astype(np.float64) / 255.0)
        for name in FEATURE_NAMES:
            assert from_uint8[name] == pytest.approx(from_float[name], abs=1e-9)

    def test_brightening_raises_mean_intensity(self, sample_image):
        arr = np.asarray(sample_image, dtype=np.float64)
        base = image_statistics(arr)
        brighter = image_statistics(np.clip(arr + 30, 0, 255))
        assert brighter["mean_intensity"] > base["mean_intensity"]

    def test_blurring_lowers_focus_and_edge_measures(self, sample_image):
        from PIL import ImageFilter

        sharp = image_statistics(sample_image)
        blurred = image_statistics(sample_image.filter(ImageFilter.GaussianBlur(3)))
        assert blurred["laplacian_var"] < sharp["laplacian_var"]
        assert blurred["edge_density"] < sharp["edge_density"]

    def test_uniform_image_has_zero_variation(self):
        stats = image_statistics(np.full((32, 32), 128, dtype=np.uint8))
        assert stats["std_intensity"] == pytest.approx(0.0, abs=1e-9)
        assert stats["entropy"] == pytest.approx(0.0, abs=1e-9)

    def test_rejects_empty_image(self):
        with pytest.raises(ValueError):
            image_statistics(np.empty((0, 0)))


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


class TestHashing:
    def test_dhash_is_stable_under_reencoding(self, sample_image, tmp_path):
        """A perceptual hash must survive a lossy round-trip; a byte hash must not."""
        png = tmp_path / "a.png"
        jpg = tmp_path / "a.jpg"
        sample_image.save(png)
        sample_image.save(jpg, quality=92)

        with Image.open(png) as a, Image.open(jpg) as b:
            assert dhash(a) == dhash(b)
        assert sha256_file(png) != sha256_file(jpg)

    def test_dhash_differs_for_different_images(self, sample_image, ok_image):
        assert dhash(sample_image) != dhash(ok_image)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_clean_corpus_passes_every_gate(self, tiny_params):
        acquire_synthetic(tiny_params)
        df, report = run_validation(tiny_params)

        assert report.passed
        assert report.n_valid == report.n_files == 80
        assert report.n_quarantined == 0
        assert all(check.status in ("PASS", "WARN") for check in report.checks)
        assert set(FEATURE_NAMES).issubset(df.columns)

    def test_quarantines_corrupt_files_without_failing(self, tiny_params):
        """A handful of bad files is normal on a line and must not abort the run."""
        acquire_synthetic(tiny_params)
        raw = __import__("pathlib").Path(tiny_params["data"]["raw_dir"])
        (raw / "ok" / "corrupt.png").write_bytes(b"this is definitely not a png")

        _, report = run_validation(tiny_params)
        assert report.n_quarantined == 1
        assert report.passed  # 1/81 is below max_corrupt_ratio
        assert any("unreadable" in q["issues"] for q in report.quarantine)

    def test_blocks_when_corrupt_ratio_exceeds_the_limit(self, tiny_params):
        acquire_synthetic(tiny_params)
        raw = __import__("pathlib").Path(tiny_params["data"]["raw_dir"])
        for i in range(30):  # 30/110 >> 2% limit
            (raw / "ok" / f"bad_{i}.png").write_bytes(b"nope")

        with pytest.raises(DataValidationError, match="corrupt_file_ratio"):
            run_validation(tiny_params)

    def test_report_is_written_under_the_configured_reports_dir(self, tiny_params, tmp_path):
        """A test run must not overwrite the real, submitted validation report.

        The path used to be a literal `reports/validation_report.json`, so
        running pytest replaced the report for the 7,348-image corpus with this
        fixture's 80-image output -- silently corrupting a graded deliverable.
        """
        from pathlib import Path

        acquire_synthetic(tiny_params)
        run_validation(tiny_params)

        written = Path(tiny_params["reports_dir"]) / "validation_report.json"
        assert written.is_file(), "report was not written to the configured directory"
        assert str(written).startswith(str(tmp_path)), "report escaped the temp directory"

    def test_flags_an_all_black_frame(self, tiny_params):
        """A covered lens produces a technically valid but useless image."""
        acquire_synthetic(tiny_params)
        raw = __import__("pathlib").Path(tiny_params["data"]["raw_dir"])
        Image.fromarray(np.zeros((64, 64), dtype=np.uint8), mode="L").save(raw / "ok" / "black.png")

        _, report = run_validation(tiny_params)
        issues = " ".join(q["issues"] for q in report.quarantine)
        assert "too_dark" in issues or "no_variation" in issues


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


class TestSplit:
    @pytest.fixture
    def manifest(self, tiny_params):
        acquire_synthetic(tiny_params)
        run_validation(tiny_params)
        return build_manifest(tiny_params)

    def test_produces_all_three_splits(self, manifest):
        assert set(manifest["split"].unique()) == {"train", "val", "test"}

    def test_no_group_spans_two_splits(self, manifest):
        """The leakage guard: near-duplicates must stay in one fold."""
        crossing = manifest.groupby("group")["split"].nunique()
        assert (crossing == 1).all()

    def test_class_proportions_are_preserved(self, manifest):
        overall = manifest["label"].mean()
        for split in ("train", "val", "test"):
            subset = manifest[manifest["split"] == split]
            assert subset["label"].mean() == pytest.approx(overall, abs=0.20)

    def test_is_reproducible_for_a_fixed_seed(self, tiny_params, manifest):
        again = build_manifest(tiny_params)
        assert manifest["split"].tolist() == again["split"].tolist()

    def test_load_manifest_filters_by_split(self, tiny_params, manifest):
        train = load_manifest(tiny_params, "train")
        assert (train["split"] == "train").all()
        assert len(train) == (manifest["split"] == "train").sum()

    def test_rejects_an_unknown_split_name(self, tiny_params, manifest):
        with pytest.raises(ValueError, match="Unknown split"):
            load_manifest(tiny_params, "holdout")
