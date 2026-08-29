"""Stage 2 of M2 - scan and validate the raw corpus before anything trains on it.

The stage answers three questions and refuses to let a bad answer through
silently:

1. **Is each file usable?** Readable, a permitted format, sane dimensions and
   size, and carrying actual signal (not an all-black frame from a covered lens
   or a blown-out frame from a misfired flash). Individually bad files are
   *quarantined*, not fatal -- a production line will always produce a few.
2. **Is the corpus as a whole usable?** Enough images per class, tolerable
   class imbalance, and few enough corrupt/duplicate files that the quarantine
   is not hiding a systemic capture problem. These are the checks that abort
   the pipeline.
3. **What does 'normal' look like?** Per-image statistics are computed here and
   reused as the drift reference in M5, so the monitoring baseline is by
   construction the same distribution the model trained on.

Design note: severity is split into ``error`` (fails the run when
``validation.fail_on_error``) and ``warning`` (recorded, run continues). Making
every check fatal trains people to disable validation; making none fatal means
nobody reads it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from ..config import class_names, ensure_parent, get, resolve
from ..features.image_stats import FEATURE_NAMES, image_statistics
from ..logging_utils import get_logger

log = get_logger(__name__)

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class CheckResult:
    """Outcome of one corpus-level validation rule."""

    name: str
    status: str
    severity: str  # "error" | "warning"
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def blocking(self) -> bool:
        return self.status == FAIL and self.severity == "error"


@dataclass
class ValidationReport:
    """Aggregate result written to ``reports/validation_report.json``."""

    n_files: int
    n_valid: int
    n_quarantined: int
    class_counts: dict[str, int]
    checks: list[CheckResult]
    statistics: dict[str, dict[str, float]]
    quarantine: list[dict[str, str]]

    @property
    def passed(self) -> bool:
        return not any(c.blocking for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "n_files": self.n_files,
            "n_valid": self.n_valid,
            "n_quarantined": self.n_quarantined,
            "class_counts": self.class_counts,
            "checks": [asdict(c) for c in self.checks],
            "statistics": self.statistics,
            "quarantine": self.quarantine[:200],  # cap: the CSV holds the full list
        }


class DataValidationError(RuntimeError):
    """Raised when a blocking corpus-level check fails."""


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Content hash of the file bytes -- detects exact duplicates."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Difference hash -- detects *near*-duplicates (re-encodes, minor crops).

    Near-duplicates are the usual source of optimistic validation scores on
    industrial datasets, where the same part is photographed repeatedly. The
    split stage groups on this hash so copies of one part cannot straddle the
    train/test boundary.
    """
    small = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    return f"{int(''.join('1' if b else '0' for b in bits.flatten()), 2):016x}"


# ---------------------------------------------------------------------------
# Per-file validation
# ---------------------------------------------------------------------------


def _validate_file(path: Path, class_name: str, label: int,
                   rules: dict[str, Any], root: Path) -> dict[str, Any]:
    """Inspect one file. Returns a record whose ``issues`` list is empty when clean."""
    rec: dict[str, Any] = {
        "relpath": path.relative_to(root).as_posix(),
        "class_name": class_name,
        "label": label,
        "file_bytes": 0,
        "width": 0,
        "height": 0,
        "format": "",
        "mode": "",
        "sha256": "",
        "dhash": "",
        "issues": [],
    }
    rec.update(dict.fromkeys(FEATURE_NAMES, np.nan))

    suffix = path.suffix.lower().lstrip(".")
    if suffix not in {f.lower() for f in rules["allowed_formats"]}:
        rec["issues"].append(f"format_not_allowed:{suffix}")
        return rec

    try:
        rec["file_bytes"] = path.stat().st_size
    except OSError as exc:
        rec["issues"].append(f"stat_failed:{type(exc).__name__}")
        return rec

    if rec["file_bytes"] == 0:
        rec["issues"].append("empty_file")
        return rec
    if rec["file_bytes"] > int(rules["max_file_bytes"]):
        rec["issues"].append(f"file_too_large:{rec['file_bytes']}")

    try:
        with Image.open(path) as im:
            im.verify()  # cheap structural check; consumes the file handle
        with Image.open(path) as im:
            rec["format"] = (im.format or "").lower()
            rec["mode"] = im.mode
            rec["width"], rec["height"] = im.size
            gray = im.convert("L")
            gray.load()
            rec["dhash"] = dhash(gray)
            stats = image_statistics(gray)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        rec["issues"].append(f"unreadable:{type(exc).__name__}")
        return rec

    rec.update({k: float(v) for k, v in stats.items()})
    rec["sha256"] = sha256_file(path)

    if rec["width"] < int(rules["min_width"]) or rec["height"] < int(rules["min_height"]):
        rec["issues"].append(f"too_small:{rec['width']}x{rec['height']}")
    if rec["width"] > int(rules["max_width"]) or rec["height"] > int(rules["max_height"]):
        rec["issues"].append(f"too_large:{rec['width']}x{rec['height']}")

    # Statistics are on a [0, 1] scale; the configured gates are 0-255.
    mean255 = stats["mean_intensity"] * 255.0
    std255 = stats["std_intensity"] * 255.0
    if mean255 < float(rules["min_mean_intensity"]):
        rec["issues"].append(f"too_dark:mean={mean255:.1f}")
    if mean255 > float(rules["max_mean_intensity"]):
        rec["issues"].append(f"too_bright:mean={mean255:.1f}")
    if std255 < float(rules["min_std_intensity"]):
        rec["issues"].append(f"no_variation:std={std255:.1f}")

    return rec


# ---------------------------------------------------------------------------
# Corpus-level checks
# ---------------------------------------------------------------------------


def _corpus_checks(df: pd.DataFrame, rules: dict[str, Any],
                   classes: list[str]) -> list[CheckResult]:
    checks: list[CheckResult] = []
    n_total = len(df)
    valid = df[df["is_valid"]]

    # --- corrupt / unusable ratio ----------------------------------------
    n_bad = n_total - len(valid)
    bad_ratio = n_bad / n_total if n_total else 0.0
    limit = float(rules["max_corrupt_ratio"])
    checks.append(CheckResult(
        name="corrupt_file_ratio",
        status=PASS if bad_ratio <= limit else FAIL,
        severity="error",
        message=f"{n_bad}/{n_total} files quarantined ({bad_ratio:.2%}); limit {limit:.2%}",
        details={"quarantined": n_bad, "ratio": round(bad_ratio, 5), "limit": limit},
    ))

    # --- per-class volume -------------------------------------------------
    counts = valid["class_name"].value_counts().to_dict()
    min_per_class = int(rules["min_images_per_class"])
    short = {c: int(counts.get(c, 0)) for c in classes if counts.get(c, 0) < min_per_class}
    checks.append(CheckResult(
        name="min_images_per_class",
        status=PASS if not short else FAIL,
        severity="error",
        message=(f"all classes have >= {min_per_class} valid images"
                 if not short else f"under-populated classes: {short}"),
        details={"counts": {c: int(counts.get(c, 0)) for c in classes},
                 "minimum": min_per_class},
    ))

    # --- class balance ----------------------------------------------------
    present = [counts.get(c, 0) for c in classes]
    # An entirely absent class is infinitely imbalanced, not a division error.
    imbalance = max(present) / min(present) if min(present) > 0 else float("inf")
    max_imbalance = float(rules["max_class_imbalance_ratio"])
    checks.append(CheckResult(
        name="class_imbalance",
        status=PASS if imbalance <= max_imbalance else WARN,
        severity="warning",
        message=f"majority/minority = {imbalance:.2f} (limit {max_imbalance:.2f})",
        details={"ratio": None if imbalance == float("inf") else round(imbalance, 4),
                 "limit": max_imbalance,
                 "mitigation": "train.class_weighting=balanced re-weights the loss"},
    ))

    # --- exact duplicates -------------------------------------------------
    dup_mask = valid["sha256"].duplicated(keep="first")
    dup_ratio = float(dup_mask.mean()) if len(valid) else 0.0
    dup_limit = float(rules["max_duplicate_ratio"])
    checks.append(CheckResult(
        name="exact_duplicate_ratio",
        status=PASS if dup_ratio <= dup_limit else FAIL,
        severity="error",
        message=f"{int(dup_mask.sum())} exact duplicates ({dup_ratio:.2%}); limit {dup_limit:.2%}",
        details={"duplicates": int(dup_mask.sum()), "ratio": round(dup_ratio, 5),
                 "limit": dup_limit},
    ))

    # --- near duplicates --------------------------------------------------
    near_mask = valid["dhash"].duplicated(keep="first") & ~dup_mask
    near_ratio = float(near_mask.mean()) if len(valid) else 0.0
    checks.append(CheckResult(
        name="near_duplicate_ratio",
        status=PASS if near_ratio <= dup_limit else WARN,
        severity="warning",
        message=f"{int(near_mask.sum())} near-duplicates ({near_ratio:.2%}) by perceptual hash",
        details={"near_duplicates": int(near_mask.sum()), "ratio": round(near_ratio, 5),
                 "mitigation": "data.split.group_by_hash keeps them in one fold"},
    ))

    # --- label leakage through global statistics --------------------------
    # If a single global statistic separates the classes almost perfectly, the
    # dataset has a shortcut and reported accuracy will not survive contact
    # with production. Worth knowing BEFORE spending an afternoon training.
    leak: dict[str, float] = {}
    if len(classes) == 2 and len(valid) > 20:
        a = valid[valid["label"] == 0]
        b = valid[valid["label"] == 1]
        for feat in FEATURE_NAMES:
            xa, xb = a[feat].to_numpy(), b[feat].to_numpy()
            pooled = np.sqrt((xa.var() + xb.var()) / 2.0)
            if pooled > 1e-9:
                d = abs(xa.mean() - xb.mean()) / pooled  # Cohen's d
                if d > 2.0:
                    leak[feat] = round(float(d), 3)
    checks.append(CheckResult(
        name="global_statistic_leakage",
        status=PASS if not leak else WARN,
        severity="warning",
        message=("no single global statistic separates the classes"
                 if not leak else f"suspiciously separable statistics (Cohen's d > 2): {leak}"),
        details={"effect_sizes": leak},
    ))

    # --- resolution consistency -------------------------------------------
    shapes = valid.groupby(["width", "height"]).size().sort_values(ascending=False)
    n_shapes = len(shapes)
    checks.append(CheckResult(
        name="resolution_consistency",
        status=PASS if n_shapes <= 1 else WARN,
        severity="warning",
        message=f"{n_shapes} distinct resolution(s) present",
        details={"top_shapes": {f"{w}x{h}": int(n) for (w, h), n in shapes.head(5).items()},
                 "mitigation": "preprocess.resize normalises every input"},
    ))

    return checks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_validation(params: dict[str, Any]) -> tuple[pd.DataFrame, ValidationReport]:
    """Scan ``data/raw``, apply all gates, and persist the scan table + report."""
    raw_dir = resolve(get(params, "data.raw_dir"))
    classes = class_names(params)
    rules = get(params, "validation")

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}. Run `acquire` first.")

    records: list[dict[str, Any]] = []
    for label, name in enumerate(classes):
        class_dir = raw_dir / name
        if not class_dir.is_dir():
            log.warning("Class directory missing: %s", class_dir)
            continue
        files = sorted(p for p in class_dir.iterdir() if p.is_file())
        log.info("Validating %-8s %5d files", name, len(files))
        for path in files:
            records.append(_validate_file(path, name, label, rules, raw_dir))

    if not records:
        raise DataValidationError(f"No files found under {raw_dir}")

    df = pd.DataFrame.from_records(records)
    df["issues"] = df["issues"].apply(lambda xs: "|".join(xs))
    df["is_valid"] = df["issues"] == ""

    checks = _corpus_checks(df, rules, classes)

    valid = df[df["is_valid"]]
    stats_summary = {
        feat: {
            "mean": float(valid[feat].mean()),
            "std": float(valid[feat].std()),
            "min": float(valid[feat].min()),
            "max": float(valid[feat].max()),
        }
        for feat in FEATURE_NAMES
    } if len(valid) else {}

    report = ValidationReport(
        n_files=len(df),
        n_valid=int(df["is_valid"].sum()),
        n_quarantined=int((~df["is_valid"]).sum()),
        class_counts=valid["class_name"].value_counts().to_dict(),
        checks=checks,
        statistics=stats_summary,
        quarantine=df.loc[~df["is_valid"], ["relpath", "issues"]].to_dict("records"),
    )

    # --- persist ----------------------------------------------------------
    scan_path = ensure_parent(Path(get(params, "data.interim_dir")) / "scan.csv")
    df.to_csv(scan_path, index=False)

    # Report location comes from config, not a literal. The test suite runs the
    # real validation over an 80-image synthetic fixture; with a hard-coded path
    # that run overwrites the submitted report for the 7,348-image corpus.
    reports_dir = get(params, "reports_dir", "reports")
    report_path = ensure_parent(Path(reports_dir) / "validation_report.json")
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    # --- log outcome ------------------------------------------------------
    for c in checks:
        emit = log.error if c.blocking else (log.warning if c.status != PASS else log.info)
        emit("[%-4s] %-28s %s", c.status, c.name, c.message)

    log.info("Scan table  -> %s", scan_path)
    log.info("Report      -> %s", report_path)
    log.info("Valid: %d/%d  quarantined: %d", report.n_valid, report.n_files, report.n_quarantined)

    if bool(rules.get("fail_on_error", True)) and not report.passed:
        blocking = [c.name for c in checks if c.blocking]
        raise DataValidationError(
            f"Data validation failed on blocking check(s): {blocking}. "
            f"See {report_path} for details."
        )

    return df, report
