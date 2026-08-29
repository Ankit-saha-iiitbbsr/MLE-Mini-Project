"""Metrics, operating-threshold selection, and evaluation plots.

Three ideas drive this module:

**Accuracy is the wrong headline metric.** On an inspection line the two errors
have very different costs: a false alarm sends a good part to manual review
(cheap), a missed defect ships (expensive, possibly a recall). Reporting
centres on recall for the defect class and on F1, with accuracy present only
for completeness.

**0.5 is an arbitrary threshold.** It is where a sigmoid happens to cross, not
where the business optimum sits. :func:`choose_threshold` tunes the operating
point on the **validation** split and that value is then frozen and applied to
test. Tuning on test would leak the test set into the model's configuration and
inflate the reported score.

**A point estimate hides sampling noise.** With a few hundred test images, an
F1 of 0.94 and one of 0.96 may be indistinguishable. Bootstrap confidence
intervals accompany the headline metrics so model comparison is honest about
what the data can actually resolve.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Binary confusion matrix entries with the defect class as positive."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Full metric set at a given operating threshold.

    ``y_prob`` is P(defect). Threshold-free metrics (ROC-AUC, average precision)
    are included so models can be compared independently of the chosen
    operating point.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(int)

    c = confusion_counts(y_true, y_pred)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    n = max(len(y_true), 1)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    balanced_acc = (recall + specificity) / 2.0

    # Matthews correlation: the one number that stays honest under class
    # imbalance, because it uses all four confusion cells.
    mcc_den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn) - (fp * fn)) / mcc_den if mcc_den > 0 else 0.0

    metrics = {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / n,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,          # defect recall -- the headline metric
        "specificity": specificity,
        "f1": f1,
        "mcc": float(mcc),
        "false_negative_rate": fn / (tp + fn) if (tp + fn) else 0.0,
        "false_positive_rate": fp / (tn + fp) if (tn + fp) else 0.0,
        "roc_auc": roc_auc(y_true, y_prob),
        "average_precision": average_precision(y_true, y_prob),
        "brier": float(np.mean((y_prob - y_true) ** 2)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_samples": int(n),
        "positive_rate": float(y_true.mean()),
        "predicted_positive_rate": float(y_pred.mean()),
    }
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in metrics.items()}


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC via the rank (Mann-Whitney U) identity, with tie correction."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    # Average the ranks within each group of tied scores.
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1

    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray):
    """Return ``(precision, recall, thresholds)`` at every distinct score."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    scores_sorted = y_score[order]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    total_pos = max(int(y_true.sum()), 1)

    # Keep only the last index of each run of equal scores: a threshold cannot
    # split tied predictions.
    distinct = np.where(np.diff(scores_sorted))[0]
    idx = np.r_[distinct, len(y_sorted) - 1]

    precision = tp[idx] / np.maximum(tp[idx] + fp[idx], EPS)
    recall = tp[idx] / total_pos
    return precision, recall, scores_sorted[idx]


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve (step interpolation)."""
    y_true = np.asarray(y_true).astype(int)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_curve(y_true: np.ndarray, y_score: np.ndarray):
    """Return ``(fpr, tpr, thresholds)``."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    scores_sorted = y_score[order]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    n_pos = max(int(y_true.sum()), 1)
    n_neg = max(int((1 - y_true).sum()), 1)

    distinct = np.where(np.diff(scores_sorted))[0]
    idx = np.r_[distinct, len(y_sorted) - 1]
    return np.r_[0.0, fp[idx] / n_neg], np.r_[0.0, tp[idx] / n_pos], np.r_[np.inf, scores_sorted[idx]]


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------


@dataclass
class ThresholdChoice:
    """The chosen operating point and why it was chosen."""

    threshold: float
    strategy: str
    rationale: str
    validation_metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def choose_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: str = "max_f1",
    *,
    fixed_threshold: float = 0.5,
    target_recall: float = 0.98,
    n_grid: int = 501,
) -> ThresholdChoice:
    """Pick the decision threshold on validation data.

    Strategies
    ----------
    ``max_f1``
        Best balance of catching defects and not drowning the review queue.
        Sensible default when the two error costs are within an order of
        magnitude of each other.
    ``target_recall``
        Cheapest threshold (highest precision) that still catches at least
        ``target_recall`` of defects. This is the one to use when a missed
        defect carries a contractual or safety penalty -- recall becomes a
        constraint and precision is what gets optimised subject to it.
    ``fixed``
        Use the configured constant. Escape hatch for a threshold mandated
        elsewhere.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    grid = np.linspace(0.0, 1.0, n_grid)

    if strategy == "fixed":
        t = float(fixed_threshold)
        return ThresholdChoice(
            threshold=t,
            strategy=strategy,
            rationale=f"Configured fixed threshold {t:.3f}",
            validation_metrics=classification_metrics(y_true, y_prob, t),
        )

    if strategy == "max_f1":
        scores = [classification_metrics(y_true, y_prob, t)["f1"] for t in grid]
        best = int(np.argmax(scores))
        t = float(grid[best])
        return ThresholdChoice(
            threshold=t,
            strategy=strategy,
            rationale=f"Maximises validation F1 ({scores[best]:.4f}) over {n_grid} candidates",
            validation_metrics=classification_metrics(y_true, y_prob, t),
        )

    if strategy == "target_recall":
        feasible = []
        for t in grid:
            m = classification_metrics(y_true, y_prob, t)
            if m["recall"] >= target_recall:
                feasible.append((m["precision"], t, m))
        if feasible:
            precision, t, m = max(feasible, key=lambda x: x[0])
            rationale = (
                f"Highest precision ({precision:.4f}) among thresholds achieving "
                f"recall >= {target_recall:.2f}"
            )
        else:
            # No threshold reaches the target; fall back to maximum recall and
            # say so, rather than silently returning something that misses it.
            recalls = [classification_metrics(y_true, y_prob, t)["recall"] for t in grid]
            best = int(np.argmax(recalls))
            t = float(grid[best])
            m = classification_metrics(y_true, y_prob, t)
            rationale = (
                f"Target recall {target_recall:.2f} unreachable on validation "
                f"(best {recalls[best]:.4f}); using the max-recall threshold"
            )
        return ThresholdChoice(
            threshold=float(t), strategy=strategy, rationale=rationale, validation_metrics=m
        )

    raise ValueError(
        f"Unknown threshold strategy {strategy!r}; expected 'max_f1', 'target_recall' or 'fixed'"
    )


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    metric: str = "f1",
    n_samples: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile bootstrap CI for one metric, resampling test images with replacement."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = len(y_true)
    if n == 0 or n_samples <= 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}

    rng = np.random.default_rng(seed)
    point = classification_metrics(y_true, y_prob, threshold)[metric]

    values = []
    for _ in range(n_samples):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if yt.min() == yt.max():
            continue  # degenerate resample: metric undefined
        values.append(classification_metrics(yt, y_prob[idx], threshold)[metric])

    if not values:
        return {"point": float(point), "lo": float("nan"), "hi": float("nan")}

    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(point), "lo": float(lo), "hi": float(hi),
            "n_bootstrap": len(values), "alpha": alpha}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def save_evaluation_plots(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    out_dir: Any,
    prefix: str = "",
) -> list[str]:
    """Write confusion matrix, ROC, PR and score-distribution figures."""
    from pathlib import Path

    from ..plotting import COLOR_DEFECT, COLOR_OK, save_figure, use_headless_backend

    plt = use_headless_backend()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_" if prefix else ""
    written: list[str] = []

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(int)

    # --- confusion matrix -------------------------------------------------
    c = confusion_counts(y_true, y_pred)
    mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(mat, cmap="Blues")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v}", ha="center", va="center",
                color="white" if v > mat.max() / 2 else "black", fontsize=15)
    ax.set_xticks([0, 1], ["pred ok", "pred defect"])
    ax.set_yticks([0, 1], ["true ok", "true defect"])
    ax.set_title(f"Confusion matrix @ t={threshold:.3f}")
    written.append(save_figure(fig, out / f"{stem}confusion_matrix.png"))

    # --- ROC --------------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc(y_true, y_prob):.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    written.append(save_figure(fig, out / f"{stem}roc_curve.png"))

    # --- precision-recall -------------------------------------------------
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(recall, precision, lw=2, label=f"AP = {average_precision(y_true, y_prob):.4f}")
    ax.axhline(float(y_true.mean()), ls="--", c="grey", lw=1, label="no-skill")
    ax.set_xlabel("recall (defect)")
    ax.set_ylabel("precision (defect)")
    ax.set_title("Precision-Recall curve")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    written.append(save_figure(fig, out / f"{stem}pr_curve.png"))

    # --- score distribution ----------------------------------------------
    # The most diagnostic plot of the four: it shows whether the two classes
    # are actually separated, or merely separable at one lucky threshold.
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    bins = np.linspace(0, 1, 41)
    ax.hist(y_prob[y_true == 0], bins=bins, alpha=0.65, label="true ok", color=COLOR_OK)
    ax.hist(y_prob[y_true == 1], bins=bins, alpha=0.65, label="true defect", color=COLOR_DEFECT)
    ax.axvline(threshold, color="k", ls="--", lw=1.5, label=f"threshold {threshold:.3f}")
    ax.set_xlabel("P(defect)")
    ax.set_ylabel("count")
    ax.set_title("Predicted score distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    written.append(save_figure(fig, out / f"{stem}score_distribution.png"))

    return written
