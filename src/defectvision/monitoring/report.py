"""Turn the prediction log into monitoring signals, plots and a report (M5).

The report answers, per traffic window, the three questions an on-call engineer
actually asks:

1. **Has the input changed?** Per-feature PSI and KS against the frozen training
   reference. Available immediately -- no labels required.
2. **Has the model's behaviour changed?** Mean confidence, predicted-defect
   rate, and the share of traffic falling into the human-review band. These are
   leading indicators: a model losing grip on a shifted distribution gets less
   confident *before* anyone measures its accuracy.
3. **Has accuracy actually degraded?** Computed only over rows where ground
   truth has arrived. This is the ground truth of monitoring, and it is also
   the signal that arrives last -- which is exactly why the first two exist.

Windowing matters: a single window can move on chance alone. Signals are
computed per window so the trigger logic can require a breach to persist across
consecutive windows before acting.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd

from ..config import ensure_dir, ensure_parent, get
from ..logging_utils import get_logger
from ..training.evaluate import classification_metrics
from .drift import assess_feature_drift, chi_square_test, summarise_drift
from .reference import load_reference, reference_model_baseline
from .store import PredictionStore, get_store

log = get_logger(__name__)


def compute_window_signals(
    window: pd.DataFrame,
    reference: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """All monitoring signals for one window of logged predictions."""
    thresholds = get(params, "monitoring.thresholds")
    feature_names = [f for f in get(params, "monitoring.drift_features") if f in window.columns]
    n_bins = int(get(params, "monitoring.psi_bins", 10))
    baseline = reference_model_baseline(reference)

    # --- 1. data drift ----------------------------------------------------
    current = {name: window[name].to_numpy(dtype=np.float64) for name in feature_names}
    drift_results = assess_feature_drift(
        reference, current,
        n_bins=n_bins,
        psi_warn=float(thresholds["psi_warn"]),
        psi_alert=float(thresholds["psi_alert"]),
    )
    drift_summary = summarise_drift(drift_results)

    # --- 2. model behaviour ----------------------------------------------
    probabilities = window["probability"].to_numpy(dtype=np.float64)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    mean_confidence = float(confidence.mean()) if confidence.size else float("nan")

    reference_confidence = baseline.get("reference_mean_confidence")
    if reference_confidence is None:
        # Not snapshotted (older reference): fall back to the reference
        # accuracy, which a well-calibrated model's mean confidence tracks.
        reference_confidence = baseline.get("reference_accuracy", float("nan"))
    confidence_drop = (float(reference_confidence) - mean_confidence
                       if np.isfinite(reference_confidence) else float("nan"))

    predicted_defect_rate = float((window["predicted_label"] == 1).mean())
    review_rate = float((window["decision"] == "human_review").mean()) \
        if "decision" in window.columns else 0.0

    # Predicted-class mix vs. the reference class mix.
    ref_class_counts = reference.get("class_distribution", {})
    current_counts = {
        "ok": int((window["predicted_label"] == 0).sum()),
        "defect": int((window["predicted_label"] == 1).sum()),
    }
    chi2_stat, chi2_p = chi_square_test(
        {k: int(v) for k, v in ref_class_counts.items()}, current_counts
    )

    # --- 3. accuracy, where labels exist ----------------------------------
    performance: dict[str, Any] = {"n_labeled": 0}
    labeled = window[window["ground_truth"].notna()] if "ground_truth" in window.columns \
        else window.iloc[0:0]
    if len(labeled) >= 20:
        threshold = float(labeled["model_threshold"].iloc[0]) \
            if labeled["model_threshold"].notna().any() else 0.5
        metrics = classification_metrics(
            labeled["ground_truth"].to_numpy(dtype=int),
            labeled["probability"].to_numpy(dtype=np.float64),
            threshold,
        )
        reference_accuracy = baseline.get("reference_accuracy", float("nan"))
        performance = {
            "n_labeled": int(len(labeled)),
            **{k: metrics[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")},
            "reference_accuracy": reference_accuracy,
            "accuracy_drop": (float(reference_accuracy) - metrics["accuracy"]
                              if np.isfinite(reference_accuracy) else float("nan")),
            "reference_recall": baseline.get("reference_recall", float("nan")),
            "recall_drop": (float(baseline.get("reference_recall", np.nan)) - metrics["recall"]
                            if np.isfinite(baseline.get("reference_recall", np.nan))
                            else float("nan")),
        }

    # --- alerts -----------------------------------------------------------
    alerts: list[dict[str, Any]] = []
    if drift_summary["psi_max"] >= float(thresholds["psi_alert"]):
        alerts.append({
            "signal": "data_drift", "severity": "high",
            "detail": (f"PSI {drift_summary['psi_max']:.3f} on "
                       f"{drift_summary['worst_feature']} exceeds "
                       f"{thresholds['psi_alert']}"),
        })
    elif drift_summary["psi_max"] >= float(thresholds["psi_warn"]):
        alerts.append({
            "signal": "data_drift", "severity": "warning",
            "detail": (f"PSI {drift_summary['psi_max']:.3f} on "
                       f"{drift_summary['worst_feature']} exceeds the warning band"),
        })
    if np.isfinite(confidence_drop) and confidence_drop >= float(thresholds["confidence_drop_alert"]):
        alerts.append({
            "signal": "confidence_collapse", "severity": "medium",
            "detail": f"Mean confidence fell {confidence_drop:.3f} below baseline",
        })
    if review_rate >= float(thresholds["review_rate_alert"]):
        alerts.append({
            "signal": "review_queue_overflow", "severity": "medium",
            "detail": f"{review_rate:.1%} of traffic routed to human review",
        })
    accuracy_drop = performance.get("accuracy_drop", float("nan"))
    if np.isfinite(accuracy_drop) and accuracy_drop >= float(thresholds["accuracy_drop_alert"]):
        alerts.append({
            "signal": "accuracy_degradation", "severity": "high",
            "detail": f"Accuracy fell {accuracy_drop:.3f} below the reference",
        })

    return {
        "n_samples": int(len(window)),
        "scenario": (window["scenario"].dropna().iloc[0]
                     if "scenario" in window.columns and window["scenario"].notna().any()
                     else None),
        "drift": {
            "summary": drift_summary,
            "features": {k: v.to_dict() for k, v in drift_results.items()},
        },
        "behaviour": {
            "mean_confidence": mean_confidence,
            "reference_confidence": reference_confidence,
            "mean_confidence_drop": confidence_drop,
            "mean_probability": float(probabilities.mean()) if probabilities.size else float("nan"),
            "predicted_defect_rate": predicted_defect_rate,
            "review_rate": review_rate,
            "class_mix_chi2": chi2_stat,
            "class_mix_pvalue": chi2_p,
        },
        "performance": performance,
        "alerts": alerts,
        # Flat view consumed by the retraining trigger rules.
        "metrics": {
            "psi_max": drift_summary["psi_max"],
            "psi_mean": drift_summary["psi_mean"],
            "mean_confidence_drop": confidence_drop,
            "accuracy_drop": accuracy_drop,
            "review_rate": review_rate,
        },
    }


def build_monitoring_report(
    params: dict[str, Any],
    *,
    window: int | None = None,
    store: PredictionStore | None = None,
) -> dict[str, Any]:
    """Compute signals per scenario and per rolling window; write report + plots."""
    store = store or get_store(params)
    reference = load_reference(params)

    total = store.count()
    min_samples = int(get(params, "monitoring.min_samples_for_drift", 100))
    if total == 0:
        raise RuntimeError(
            "The prediction log is empty. Generate traffic first:\n"
            "    defectvision simulate-drift"
        )

    window_size = int(window or get(params, "retraining.window_size", 200))
    log.info("Analysing %d logged predictions (window=%d)", total, window_size)

    # --- per scenario -----------------------------------------------------
    per_scenario: dict[str, Any] = {}
    scenarios = store.scenarios()
    for name in scenarios:
        rows = store.fetch(scenario=name)
        if len(rows) < min_samples:
            log.info("Scenario %-20s skipped (%d rows < %d)", name, len(rows), min_samples)
            continue
        per_scenario[name] = compute_window_signals(rows, reference, params)
        signals = per_scenario[name]
        log.info("%-20s PSI_max=%.4f  conf_drop=%+.4f  acc_drop=%+.4f  alerts=%d",
                 name, signals["metrics"]["psi_max"],
                 signals["metrics"]["mean_confidence_drop"] or 0.0,
                 signals["metrics"]["accuracy_drop"] if
                 np.isfinite(signals["metrics"]["accuracy_drop"]) else 0.0,
                 len(signals["alerts"]))

    # --- rolling windows over the whole log -------------------------------
    all_rows = store.fetch()
    windows: list[dict[str, Any]] = []
    for start in range(0, len(all_rows), window_size):
        chunk = all_rows.iloc[start:start + window_size]
        if len(chunk) < min_samples:
            continue
        signals = compute_window_signals(chunk, reference, params)
        signals["window_index"] = len(windows)
        signals["row_range"] = [int(start), int(start + len(chunk))]
        windows.append(signals)

    figures = _plot_monitoring(all_rows, per_scenario, windows, reference, params)

    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_predictions": total,
        "window_size": window_size,
        "reference": {
            "created_at": reference.get("created_at"),
            "n_samples": reference.get("n_samples"),
            "model_baseline": reference_model_baseline(reference),
        },
        "per_scenario": per_scenario,
        "windows": windows,
        "figures": figures,
        "summary": {
            name: {
                "psi_max": round(s["metrics"]["psi_max"], 4),
                "drifted_features": s["drift"]["summary"]["drifted_features"],
                "accuracy": round(s["performance"].get("accuracy", float("nan")), 4)
                if s["performance"].get("n_labeled") else None,
                "accuracy_drop": (round(s["metrics"]["accuracy_drop"], 4)
                                  if np.isfinite(s["metrics"]["accuracy_drop"]) else None),
                "mean_confidence": round(s["behaviour"]["mean_confidence"], 4),
                "review_rate": round(s["behaviour"]["review_rate"], 4),
                "alerts": [a["signal"] for a in s["alerts"]],
            }
            for name, s in per_scenario.items()
        },
    }

    path = ensure_parent("reports/monitoring_report.json")
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info("Monitoring report -> %s", path)

    store.export_jsonl("reports/prediction_log.jsonl")
    _write_markdown(report, params)
    return report


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_monitoring(
    all_rows: pd.DataFrame,
    per_scenario: dict[str, Any],
    windows: list[dict[str, Any]],
    reference: dict[str, Any],
    params: dict[str, Any],
) -> list[str]:
    from ..plotting import COLOR_ACCENT, COLOR_DEFECT, COLOR_OK, save_figure, use_headless_backend

    plt = use_headless_backend()

    out = ensure_dir("reports/figures/monitoring")
    thresholds = get(params, "monitoring.thresholds")
    written: list[str] = []

    # --- PSI heatmap across scenarios ------------------------------------
    if per_scenario:
        names = list(per_scenario)
        features = list(next(iter(per_scenario.values()))["drift"]["features"])
        matrix = np.array([
            [per_scenario[n]["drift"]["features"].get(f, {}).get("psi", np.nan)
             for f in features] for n in names
        ])
        fig, ax = plt.subplots(figsize=(1.35 * len(features) + 3, 0.55 * len(names) + 2.2))
        im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=max(0.5, float(np.nanmax(matrix))))
        ax.set_xticks(range(len(features)), features, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        for (i, j), v in np.ndenumerate(matrix):
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.35 else "black")
        ax.set_title(f"PSI per feature per scenario (alert >= {thresholds['psi_alert']})")
        fig.colorbar(im, ax=ax, shrink=0.8, label="PSI")
        written.append(save_figure(fig, out / "psi_heatmap.png"))

    # --- accuracy and confidence per scenario -----------------------------
    if per_scenario:
        names = list(per_scenario)
        acc = [per_scenario[n]["performance"].get("accuracy", np.nan) for n in names]
        conf = [per_scenario[n]["behaviour"]["mean_confidence"] for n in names]
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(1.25 * len(names) + 3, 4.2))
        ax.bar(x - 0.2, acc, width=0.4, label="accuracy", color=COLOR_OK)
        ax.bar(x + 0.2, conf, width=0.4, label="mean confidence", color=COLOR_ACCENT)
        baseline_acc = reference_model_baseline(reference).get("reference_accuracy")
        if baseline_acc and np.isfinite(baseline_acc):
            ax.axhline(baseline_acc, ls="--", c="k", lw=1.2,
                       label=f"reference accuracy ({baseline_acc:.3f})")
        ax.set_xticks(x, names, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("score")
        ax.set_title("Model performance under simulated drift")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        written.append(save_figure(fig, out / "performance_by_scenario.png"))

    # --- rolling window trend --------------------------------------------
    if len(windows) > 1:
        idx = [w["window_index"] for w in windows]
        psi = [w["metrics"]["psi_max"] for w in windows]
        conf = [w["behaviour"]["mean_confidence"] for w in windows]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
        ax1.plot(idx, psi, marker="o", lw=2, color=COLOR_DEFECT)
        ax1.axhline(float(thresholds["psi_alert"]), ls="--", c="k", lw=1, label="alert")
        ax1.axhline(float(thresholds["psi_warn"]), ls=":", c="grey", lw=1, label="warn")
        ax1.set_ylabel("max PSI")
        ax1.set_title("Drift signals over consecutive windows")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
        ax2.plot(idx, conf, marker="s", lw=2, color=COLOR_ACCENT)
        ax2.set_ylabel("mean confidence")
        ax2.set_xlabel("window index")
        ax2.set_ylim(0.4, 1.02)
        ax2.grid(alpha=0.3)
        written.append(save_figure(fig, out / "window_trend.png"))

    # --- score distribution: reference vs drifted -------------------------
    if "scenario" in all_rows.columns and per_scenario:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        bins = np.linspace(0, 1, 41)
        for name in list(per_scenario)[:4]:
            subset = all_rows[all_rows["scenario"] == name]["probability"]
            ax.hist(subset, bins=bins, histtype="step", lw=2, label=name, density=True)
        ax.set_xlabel("P(defect)")
        ax.set_ylabel("density")
        ax.set_title("Predicted score distribution by scenario")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        written.append(save_figure(fig, out / "score_distributions.png"))

    return written


def _write_markdown(report: dict[str, Any], params: dict[str, Any]) -> None:
    """Render the human-readable drift/monitoring report."""
    thresholds = get(params, "monitoring.thresholds")
    lines = [
        "# Monitoring & Drift Report",
        "",
        f"*Generated {report['created_at']} from {report['n_predictions']} logged predictions.*",
        "",
        "## 1. What is monitored",
        "",
        "Every served prediction logs seven image statistics alongside the decision. "
        "Those statistics are compared against a baseline frozen from the **training "
        "split**, using PSI (with reference-derived, frozen bin edges) and a two-sample "
        "KS test. Model behaviour (confidence, predicted-defect rate, human-review rate) "
        "is tracked in parallel because it degrades *before* labelled accuracy does.",
        "",
        "## 2. Results by scenario",
        "",
        "| scenario | max PSI | drifted features | accuracy | acc. drop | mean conf. | alerts |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, s in report["summary"].items():
        acc = f"{s['accuracy']:.4f}" if s["accuracy"] is not None else "-"
        drop = f"{s['accuracy_drop']:+.4f}" if s["accuracy_drop"] is not None else "-"
        lines.append(
            f"| `{name}` | {s['psi_max']:.4f} | {len(s['drifted_features'])} "
            f"({', '.join(s['drifted_features'][:3]) or 'none'}) | {acc} | {drop} | "
            f"{s['mean_confidence']:.4f} | {', '.join(s['alerts']) or 'none'} |"
        )

    lines += [
        "",
        f"Alert thresholds: PSI warn `{thresholds['psi_warn']}`, "
        f"PSI alert `{thresholds['psi_alert']}`, "
        f"confidence drop `{thresholds['confidence_drop_alert']}`, "
        f"accuracy drop `{thresholds['accuracy_drop_alert']}`, "
        f"review rate `{thresholds['review_rate_alert']}`.",
        "",
        "## 3. Figures",
        "",
    ]
    for path in report.get("figures", []):
        rel = str(path).replace("\\", "/").split("reports/")[-1]
        lines.append(f"- `{rel}`")

    lines += [
        "",
        "## 4. Reading the results",
        "",
        "- The `baseline` row is the control: uncorrupted test images. Its PSI shows the "
        "noise floor, i.e. how much apparent drift arises from sampling alone. Any "
        "scenario must be read against that floor, not against zero.",
        "- `real_camera_upgrade` is not simulated. It is a genuine second capture of the "
        "same production line with a different camera, held back from training "
        "specifically so the detectors face a shift no corruption operator was tuned on.",
        "",
        "### On the magnitude of PSI",
        "",
        "The conventional bands (0.10 warn, 0.25 significant) are calibrated for the "
        "*subtle* shifts typical of tabular credit scoring. Values in double digits are "
        "not an error: PSI is unbounded, and a value near 12 is the arithmetic signature "
        "of a distribution that has moved **entirely outside the reference support** -- "
        "every sample landing in one reference bin, with the other nine effectively "
        "empty. Anything above roughly 1.0 should be read as \"a different distribution\", "
        "not as \"1.0/0.25 = 4x worse than the alert threshold\".",
        "",
        "This is also why the retraining policy routes very large PSI to "
        "`investigate_capture` rather than `retrain`: a shift that large is a hardware or "
        "configuration change, not a change in the parts being inspected.",
        "",
        "### The *pattern* of drifted features identifies the fault",
        "",
        "See `figures/monitoring/psi_heatmap.png`. The features were chosen so that each "
        "maps to a physical failure, and the heatmap shows that holding: a defocus "
        "scenario lights up `laplacian_var` while leaving `mean_intensity` at the noise "
        "floor; a lighting shift does the reverse; sensor noise fires `edge_density` and "
        "`laplacian_var` together. So the monitoring output is not merely \"something "
        "changed\" -- the signature says *which* piece of the capture rig to go and look "
        "at. An embedding-based detector would fire just as reliably and tell nobody "
        "where to start.",
        "",
        "### Confidence is not a sufficient monitor",
        "",
        "Compare the accuracy and mean-confidence columns above. A model can lose most of "
        "its accuracy while becoming *more* certain -- the classic silent failure. Any "
        "scenario in this table where accuracy fell sharply but confidence did not is a "
        "case that confidence-based monitoring alone would have missed entirely, and that "
        "input-distribution monitoring caught. That asymmetry is the reason both tiers "
        "exist.",
        "",
    ]
    ensure_parent("reports/drift_report.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("Drift report -> reports/drift_report.md")
