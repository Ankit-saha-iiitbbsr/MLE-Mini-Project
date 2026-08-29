"""When should the model be retrained? (M5)

Retraining is expensive and not always the right fix, so the trigger is a
policy rather than a threshold. Four safeguards separate this from
"if PSI > 0.25: retrain":

**Persistence.** A rule must be breached on ``consecutive_windows`` consecutive
windows. One bad window is usually a batch of odd parts or a shift change, and
retraining on it would chase noise.

**Data sufficiency.** Retraining needs enough *newly labelled* examples to
actually learn the new regime. Firing without labels produces a model retrained
on the old data plus noise -- strictly worse than doing nothing.

**Cooldown.** After a retrain, the trigger is muted for ``cooldown_hours``.
Without it, a drift that retraining cannot fix (a genuinely harder distribution)
causes an infinite retrain loop that burns compute and churns production models.

**Severity routing.** Not every breach means "retrain". Drift with intact
accuracy may only need a recalibrated threshold; a broken camera needs a
maintenance ticket, not a model. The decision therefore carries a recommended
*action*, and retraining is only one of the options.

The output is a structured decision with the rule evaluations that produced it,
so an operator can see exactly why the system did or did not act. Exit code 10
from ``defectvision check-retrain`` lets CI or a cron job branch on the result.
"""

from __future__ import annotations

import json
import operator
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ensure_parent, get, resolve
from ..logging_utils import get_logger

log = get_logger(__name__)

_OPS = {
    ">=": operator.ge, ">": operator.gt,
    "<=": operator.le, "<": operator.lt,
    "==": operator.eq,
}

#: Written after a retrain so the cooldown survives process restarts.
STATE_FILE = "monitoring/retrain_state.json"


def _load_state() -> dict[str, Any]:
    path = resolve(STATE_FILE)
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:  # pragma: no cover
            log.warning("Corrupt retrain state file at %s; ignoring", path)
    return {}


def record_retrain(reason: str = "manual") -> Path:
    """Stamp the cooldown clock. Call this after a retrain completes."""
    path = ensure_parent(STATE_FILE)
    state = _load_state()
    history = state.get("history", [])
    now = time.time()
    history.append({"ts": now,
                    "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                    "reason": reason})
    path.write_text(json.dumps({"last_retrain_ts": now, "history": history[-50:]}, indent=2),
                    encoding="utf-8")
    log.info("Recorded retrain event (%s)", reason)
    return path


def evaluate_rule(rule: dict[str, Any], metrics: dict[str, float]) -> dict[str, Any]:
    """Evaluate one rule against a window's flat metric dict."""
    name = str(rule["name"])
    metric = str(rule["metric"])
    op_symbol = str(rule.get("op", ">="))
    threshold = float(rule["threshold"])

    value = metrics.get(metric, float("nan"))
    if not np.isfinite(value):
        return {"rule": name, "metric": metric, "value": None, "threshold": threshold,
                "op": op_symbol, "breached": False, "severity": rule.get("severity", "medium"),
                "detail": f"{metric} unavailable in this window (no labels yet?)"}

    breached = bool(_OPS[op_symbol](value, threshold))
    return {
        "rule": name, "metric": metric, "value": float(value), "threshold": threshold,
        "op": op_symbol, "breached": breached,
        "severity": rule.get("severity", "medium"),
        "detail": f"{metric}={value:.4f} {op_symbol} {threshold:.4f} -> "
                  f"{'BREACH' if breached else 'ok'}",
    }


def _recommend_action(
    fired: list[dict[str, Any]],
    latest: dict[str, Any],
) -> tuple[str, str]:
    """Map the fired rules onto a concrete action and a one-line justification.

    The distinction that matters: drift *with* preserved accuracy is not a
    retraining case. The inputs moved but the model still handles them, so the
    cheap interventions (recalibrate, or just keep watching) are correct.
    """
    names = {f["rule"] for f in fired}
    accuracy_drop = latest["metrics"].get("accuracy_drop", float("nan"))
    psi_max = latest["metrics"].get("psi_max", 0.0)
    n_labeled = latest["performance"].get("n_labeled", 0)

    if "accuracy_degradation" in names:
        return ("retrain",
                "Measured accuracy has fallen below the reference on labelled traffic. "
                "This is confirmed degradation, not a leading indicator.")

    if "data_drift_psi" in names:
        if n_labeled >= 20 and np.isfinite(accuracy_drop) and accuracy_drop < 0.02:
            return ("monitor",
                    f"Inputs have shifted (PSI {psi_max:.3f}) but accuracy is intact "
                    f"({accuracy_drop:+.3f} vs reference). The model generalises to the new "
                    "regime; retraining now would spend compute for no measurable gain. "
                    "Keep watching and collect labels.")
        if psi_max >= 0.5:
            return ("investigate_capture",
                    f"Severe input drift (PSI {psi_max:.3f}). A shift this large usually means "
                    "a hardware or configuration change (lamp, lens, camera pose) rather than a "
                    "genuine change in the parts. Inspect the capture rig before retraining -- "
                    "a model retrained on a broken camera bakes the fault in.")
        return ("retrain",
                f"Sustained input drift (PSI {psi_max:.3f}) without evidence that accuracy "
                "is holding up. Retrain on data that includes the new regime.")

    if "confidence_collapse" in names or "review_queue_overflow" in names:
        return ("recalibrate_or_retrain",
                "The model is markedly less certain and the human-review queue is growing, "
                "but accuracy has not been confirmed to have dropped. Re-tune the operating "
                "threshold on recent labelled data first; retrain if that does not restore "
                "the review rate.")

    return ("none", "No rule fired.")


def evaluate_triggers(
    params: dict[str, Any],
    report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the configured trigger rules and return a structured decision."""
    if report is None:
        report_path = resolve("reports/monitoring_report.json")
        if not report_path.is_file():
            raise FileNotFoundError(
                f"{report_path} not found. Build it first:\n    defectvision monitor"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))

    cfg = get(params, "retraining")
    rules = list(cfg["rules"])
    required_consecutive = int(cfg.get("consecutive_windows", 2))
    cooldown_hours = float(cfg.get("cooldown_hours", 24))
    min_new_labels = int(cfg.get("min_new_labeled_samples", 300))

    windows: list[dict[str, Any]] = report.get("windows", [])
    if not windows:
        return {
            "should_retrain": False,
            "action": "none",
            "reason": "No complete monitoring window is available yet.",
            "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rules": [], "fired_rules": [], "windows_evaluated": 0,
        }

    # --- evaluate every rule on every window ------------------------------
    per_window: list[dict[str, Any]] = []
    for w in windows:
        evaluations = [evaluate_rule(rule, w["metrics"]) for rule in rules]
        per_window.append({
            "window_index": w.get("window_index"),
            "n_samples": w.get("n_samples"),
            "scenario": w.get("scenario"),
            "evaluations": evaluations,
            "breached": [e["rule"] for e in evaluations if e["breached"]],
        })

    # --- persistence: breached on the last N consecutive windows ----------
    recent = per_window[-required_consecutive:]
    fired: list[dict[str, Any]] = []
    if len(recent) >= required_consecutive:
        for rule in rules:
            name = str(rule["name"])
            if all(name in w["breached"] for w in recent):
                latest_eval = next(e for e in recent[-1]["evaluations"] if e["rule"] == name)
                fired.append({**latest_eval,
                              "consecutive_windows": required_consecutive})

    latest = windows[-1]
    action, reason = _recommend_action(fired, latest)

    # --- gating -----------------------------------------------------------
    blockers: list[str] = []
    should_retrain = action in ("retrain",)

    state = _load_state()
    last_retrain = state.get("last_retrain_ts")
    hours_since = ((time.time() - float(last_retrain)) / 3600.0
                   if last_retrain else float("inf"))
    if should_retrain and hours_since < cooldown_hours:
        blockers.append(
            f"cooldown active: {hours_since:.1f}h since the last retrain, "
            f"minimum is {cooldown_hours:.0f}h"
        )

    n_labeled_total = sum(w["performance"].get("n_labeled", 0) for w in windows)
    if should_retrain and n_labeled_total < min_new_labels:
        blockers.append(
            f"insufficient labelled data: {n_labeled_total} available, "
            f"{min_new_labels} required to retrain meaningfully"
        )

    if blockers:
        should_retrain = False

    decision = {
        "should_retrain": should_retrain,
        "action": action,
        "reason": reason,
        "blockers": blockers,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "policy": {
            "consecutive_windows_required": required_consecutive,
            "cooldown_hours": cooldown_hours,
            "min_new_labeled_samples": min_new_labels,
            "hours_since_last_retrain": None if not last_retrain else round(hours_since, 2),
            "labeled_samples_available": n_labeled_total,
        },
        "fired_rules": fired,
        "windows_evaluated": len(per_window),
        "latest_window": {
            "metrics": latest["metrics"],
            "alerts": latest["alerts"],
        },
        "per_window": per_window,
        "next_steps": _next_steps(action),
    }

    path = ensure_parent("reports/retraining_decision.json")
    path.write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")

    level = log.warning if should_retrain else log.info
    level("Retraining decision: action=%s should_retrain=%s -- %s",
          action, should_retrain, reason)
    for blocker in blockers:
        log.info("  blocked by: %s", blocker)
    log.info("Decision -> %s", path)
    return decision


def _next_steps(action: str) -> list[str]:
    """The concrete commands an operator should run for each action."""
    return {
        "retrain": [
            "defectvision data                 # re-ingest, including the new regime",
            "defectvision train --all          # retrain every arm on the refreshed dataset",
            "defectvision compare              # gate + promote only if the candidate wins",
            "defectvision reference-stats      # re-baseline monitoring on the new training set",
            "python -c \"from defectvision.monitoring.retrain_trigger import record_retrain;"
            " record_retrain('scheduled')\"    # start the cooldown clock",
        ],
        "recalibrate_or_retrain": [
            "Collect labels for recent traffic via POST /feedback",
            "Re-tune the threshold: set evaluate.threshold_strategy=target_recall in params.yaml",
            "defectvision train --model <current_model>   # cheap: re-tunes the operating point",
            "Escalate to a full retrain if the review rate stays above threshold",
        ],
        "investigate_capture": [
            "Inspect the capture rig: lamp output, lens cleanliness, camera pose, focus",
            "Compare reports/figures/drift/*.png against a known-good capture",
            "Only retrain once the rig is confirmed healthy, or the fault gets baked in",
        ],
        "monitor": [
            "No action required. Keep collecting labels via POST /feedback",
            "Re-run `defectvision monitor` after the next window closes",
        ],
        "none": ["No action required."],
    }.get(action, ["No action required."])
