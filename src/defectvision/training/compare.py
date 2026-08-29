"""Cross-run comparison, promotion gates, and the model registry (M3 -> M4 handoff).

Choosing a model is a decision with a rule behind it, not a glance at a table.
This module encodes the rule:

1. **Gates first.** A candidate must clear absolute floors on test F1 and defect
   recall, and a ceiling on p95 latency, before it is eligible at all. A model
   that is the best of a bad field should not reach production by default.
2. **Then rank.** Among eligible candidates, rank by the configured selection
   metric. Ties (differences inside the bootstrap confidence interval) are
   broken by latency, because when accuracy is statistically indistinguishable
   the cheaper model is the better engineering choice.

The promoted bundle is copied to ``models/production/`` -- a fixed path the
serving container reads -- and registered in the MLflow Model Registry so the
lineage from a running container back to a run, a commit and a dataset hash
stays intact.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import ensure_dir, ensure_parent, get, resolve
from ..logging_utils import banner, get_logger

log = get_logger(__name__)


@dataclass
class GateResult:
    """Whether one candidate cleared the promotion gates, and why not if it did not."""

    model_name: str
    eligible: bool
    failures: list[str]

    def reason(self) -> str:
        return "eligible" if self.eligible else "; ".join(self.failures)


def evaluate_gates(row: dict[str, Any], gates: dict[str, Any]) -> GateResult:
    """Apply the configured promotion gates to one candidate's test metrics."""
    failures: list[str] = []

    min_f1 = float(gates.get("min_test_f1", 0.0))
    if row.get("test_f1", 0.0) < min_f1:
        failures.append(f"test_f1 {row.get('test_f1'):.4f} < {min_f1:.4f}")

    min_recall = float(gates.get("min_test_recall_defect", 0.0))
    if row.get("test_recall", 0.0) < min_recall:
        failures.append(f"test_recall {row.get('test_recall'):.4f} < {min_recall:.4f}")

    max_latency = float(gates.get("max_latency_p95_ms", float("inf")))
    if row.get("latency_p95_ms", float("inf")) > max_latency:
        failures.append(f"latency_p95_ms {row.get('latency_p95_ms'):.1f} > {max_latency:.1f}")

    return GateResult(str(row["model"]), not failures, failures)


def build_comparison(results: list[Any]) -> pd.DataFrame:
    """Comparison table from in-memory :class:`TrainingResult` objects."""
    if not results:
        raise ValueError("No training results to compare")
    return pd.DataFrame([r.summary_row() for r in results])


def _metric(row: pd.Series, key: str, default: float = 0.0) -> float:
    """Read one metric from a ``search_runs`` row, tolerating missing *and* NaN.

    ``search_runs`` returns the union of every metric column across all runs, so
    a run that never logged a given metric still gets the column -- filled with
    NaN. A plain ``row.get(key, default)`` only substitutes the default when the
    *key* is absent, and ``value or default`` does not help either, because NaN
    is truthy in Python. Both leave NaN in place, which then blows up on
    ``int()``. This helper is the single place that gets it right.
    """
    value = row.get(key, default)
    if value is None or pd.isna(value):
        return float(default)
    return float(value)


def load_comparison_from_mlflow(params: dict[str, Any], max_runs: int = 100) -> pd.DataFrame:
    """Rebuild the comparison table by querying the MLflow tracking store.

    Used when comparing runs from *separate* invocations -- the normal case for
    a team where different members trained different arms, and the path the
    standalone ``defectvision compare`` and the DVC ``compare`` stage take.
    """
    from .train import setup_mlflow

    mlflow = setup_mlflow(params)
    experiment = mlflow.get_experiment_by_name(str(get(params, "mlflow.experiment_name")))
    if experiment is None:
        raise RuntimeError("MLflow experiment not found -- train at least one model first")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=max_runs,
    )
    if runs.empty:
        raise RuntimeError("No MLflow runs found in the experiment")

    rows = []
    for _, r in runs.iterrows():
        # Skip runs that never finished training.
        if pd.isna(r.get("metrics.test_f1")):
            continue

        # Skip runs that are not training candidates. The promotion step logs
        # the winner's test metrics onto its own run for lineage, so those runs
        # DO carry test_f1 and would otherwise be re-ingested here as phantom
        # candidates named "?" -- one more on every `compare`. Only a training
        # run logs `model_name`, which makes it the reliable discriminator.
        model_name = r.get("params.model_name")
        if model_name is None or pd.isna(model_name):
            continue

        rows.append({
            "model": str(model_name),
            "arch": str(r.get("params.arch") or "?"),
            "run_id": str(r["run_id"])[:8],
            "params_M": round(_metric(r, "metrics.total_params") / 1e6, 3),
            "threshold": round(_metric(r, "metrics.test_threshold", 0.5), 4),
            "test_f1": round(_metric(r, "metrics.test_f1"), 4),
            "test_recall": round(_metric(r, "metrics.test_recall"), 4),
            "test_precision": round(_metric(r, "metrics.test_precision"), 4),
            "test_accuracy": round(_metric(r, "metrics.test_accuracy"), 4),
            "test_roc_auc": round(_metric(r, "metrics.test_roc_auc"), 4),
            "latency_p95_ms": round(_metric(r, "metrics.latency_p95_ms"), 2),
            "throughput_ips": round(_metric(r, "metrics.latency_throughput_batched_ips"), 1),
            "train_s": round(_metric(r, "metrics.train_seconds"), 1),
            # 0 for arms with no epochs (the classical control), matching what
            # the in-memory path reports for the same run.
            "best_epoch": int(_metric(r, "metrics.best_epoch")),
        })

    if not rows:
        raise RuntimeError(
            "No completed training runs found in the experiment. "
            "Train at least one model first:  defectvision train --all"
        )

    df = pd.DataFrame(rows)

    # One row per model: keep its best run by the selection metric.
    #
    # `kind="mergesort"` is required, not cosmetic. Re-running a deterministic
    # arm produces a second run with an identical score, and pandas' default
    # quicksort is unstable -- so which of the tied runs survived varied between
    # invocations, and `compare` could promote a different run id each time it
    # was called on unchanged data. A stable sort preserves the incoming
    # start_time-DESC ordering, so ties resolve to the most recent run.
    return (df.sort_values("test_f1", ascending=False, kind="mergesort")
              .drop_duplicates(subset="model", keep="first")
              .reset_index(drop=True))


def select_best(
    table: pd.DataFrame,
    params: dict[str, Any],
    results: list[Any] | None = None,
) -> tuple[dict[str, Any] | None, list[GateResult], str]:
    """Apply gates, rank the survivors, and explain the choice.

    Returns ``(winning_row, gate_results, rationale)``.
    """
    gates = dict(get(params, "mlflow.promotion_gates"))
    metric = f"test_{get(params, 'train.monitor_metric')}"
    if metric not in table.columns:
        metric = "test_f1"

    gate_results = [evaluate_gates(row, gates) for row in table.to_dict("records")]
    eligible_names = {g.model_name for g in gate_results if g.eligible}

    for g in gate_results:
        (log.info if g.eligible else log.warning)("gate %-14s %s", g.model_name, g.reason())

    eligible = table[table["model"].isin(eligible_names)]
    if eligible.empty:
        return None, gate_results, (
            f"No candidate cleared the promotion gates {gates}. "
            "Nothing was promoted; investigate before deploying."
        )

    ranked = eligible.sort_values(metric, ascending=False).reset_index(drop=True)
    best = ranked.iloc[0].to_dict()

    # Tie-break on latency when the top scores are within the leader's
    # bootstrap CI -- a difference the test set cannot actually resolve.
    rationale = f"Highest {metric} ({best[metric]:.4f}) among candidates clearing the gates"
    ci_lo = None
    if results:
        by_name = {r.model_name: r for r in results}
        leader = by_name.get(best["model"])
        key = metric.replace("test_", "")
        if leader is not None and key in leader.test_ci:
            ci_lo = leader.test_ci[key].get("lo")

    if ci_lo is not None:
        contenders = ranked[ranked[metric] >= ci_lo]
        if len(contenders) > 1:
            cheapest = contenders.sort_values("latency_p95_ms", ascending=True).iloc[0].to_dict()
            if cheapest["model"] != best["model"]:
                rationale = (
                    f"{len(contenders)} candidates score within the leader's 95% CI "
                    f"(>= {ci_lo:.4f} {metric}); among those, {cheapest['model']} has the "
                    f"lowest p95 latency ({cheapest['latency_p95_ms']:.1f} ms vs "
                    f"{best['latency_p95_ms']:.1f} ms), so it wins on cost at equal accuracy"
                )
                best = cheapest
            else:
                rationale += (
                    f"; it is also the cheapest of the {len(contenders)} candidates "
                    f"statistically tied with it"
                )

    return best, gate_results, rationale


def _format_ci(test_ci: dict[str, Any], metric: str) -> str:
    """Render one bootstrap interval as ``point [lo, hi]``, or ``-`` if unavailable."""
    ci = test_ci.get(metric) or {}
    lo = ci.get("lo")
    if lo is None or lo != lo:  # missing, or NaN
        return "-"
    return f"{ci['point']:.4f} [{lo:.4f}, {ci['hi']:.4f}]"


def write_comparison_report(
    table: pd.DataFrame,
    params: dict[str, Any],
    best: dict[str, Any] | None,
    gate_results: list[GateResult],
    rationale: str,
    results: list[Any] | None = None,
    out_path: str = "reports/model_comparison.md",
) -> Path:
    """Render the model-comparison report required by the submission checklist."""
    gates = dict(get(params, "mlflow.promotion_gates"))
    metric = f"test_{get(params, 'train.monitor_metric')}"

    lines: list[str] = [
        "# Model Comparison Report",
        "",
        "*Generated by `defectvision compare`. Every number below comes from a tracked "
        "MLflow run; nothing is hand-entered.*",
        "",
        "## 1. Results",
        "",
        "All models were trained on the same manifest (same images, same folds), evaluated "
        "on the same held-out test split, and each used a threshold tuned on **validation** "
        "only. Test data influenced no decision.",
        "",
        table.to_markdown(index=False),
        "",
        "## 2. Promotion gates",
        "",
        "A candidate is eligible for deployment only if it clears every gate below "
        "(configured in `params.yaml` under `mlflow.promotion_gates`):",
        "",
        f"- `test_f1` >= **{gates.get('min_test_f1')}**",
        f"- `test_recall` (defect) >= **{gates.get('min_test_recall_defect')}**",
        f"- `latency_p95_ms` <= **{gates.get('max_latency_p95_ms')}**",
        "",
        "| model | eligible | detail |",
        "| --- | --- | --- |",
    ]
    for g in gate_results:
        lines.append(f"| {g.model_name} | {'yes' if g.eligible else 'NO'} | {g.reason()} |")

    lines += ["", "## 3. Selection", ""]
    if best is None:
        lines += [f"**No model was promoted.** {rationale}", ""]
    else:
        lines += [
            f"**Selected: `{best['model']}`** (run `{best.get('run_id', '?')}`)",
            "",
            f"Rationale: {rationale}.",
            "",
            f"Selection metric: `{metric}`.",
            "",
        ]

    if results:
        lines += ["## 4. Confidence intervals", "",
                  "95% percentile bootstrap over the test split. Overlapping intervals mean "
                  "the test set cannot resolve a difference between those models.", "",
                  "| model | F1 | recall | precision |", "| --- | --- | --- | --- |"]
        for r in results:
            cells = [_format_ci(r.test_ci, m) for m in ("f1", "recall", "precision")]
            lines.append(f"| {r.model_name} | {' | '.join(cells)} |")
        lines.append("")

    lines += [
        "## 5. Reproducing a run",
        "",
        "Each run logs its git commit, a full `params.yaml` snapshot, the dataset manifest "
        "SHA-256, and the resolved library versions under the `reproducibility/` artifact "
        "path. To rebuild any run:",
        "",
        "```bash",
        "mlflow ui --backend-store-uri sqlite:///mlflow.db   # open the run's reproducibility/",
        "git checkout <git_commit from the run tags>",
        "dvc repro                                           # rebuilds the exact dataset",
        "defectvision train --model <model_name>",
        "```",
        "",
    ]

    path = ensure_parent(out_path)
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Comparison report -> %s", path)
    return path


def promote(
    params: dict[str, Any],
    best: dict[str, Any],
    *,
    register: bool = True,
) -> Path:
    """Copy the winning bundle to the production path and register it in MLflow."""
    model_name = str(best["model"])
    src = resolve(f"models/candidates/{model_name}/model_bundle.pt")
    if not src.is_file():
        raise FileNotFoundError(
            f"No deployable bundle for {model_name!r} at {src}. "
            "Classical arms are comparison-only and cannot be promoted."
        )

    dest = resolve(get(params, "serving.model_bundle"))
    ensure_dir(dest.parent)
    shutil.copy2(src, dest)
    log.info("Promoted %s -> %s", model_name, dest)

    record = {
        "promoted_model": model_name,
        "source_bundle": str(src),
        "production_bundle": str(dest),
        "run_id": best.get("run_id"),
        "test_f1": best.get("test_f1"),
        "test_recall": best.get("test_recall"),
        "latency_p95_ms": best.get("latency_p95_ms"),
    }
    ensure_parent(dest.parent / "PROMOTED.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )

    if register:
        try:
            _register_in_mlflow(params, model_name, dest, best)
        except Exception as exc:
            # Registry unavailability must not break the pipeline: the bundle
            # on disk is what serving actually reads.
            log.warning("MLflow model registration skipped (%s: %s)", type(exc).__name__, exc)

    return dest


def _register_in_mlflow(params: dict[str, Any], model_name: str,
                        bundle_path: Path, best: dict[str, Any]) -> None:
    """Register the promoted model in the MLflow Model Registry.

    The registry needs a proper *MLflow Model* (a directory carrying an
    ``MLmodel`` descriptor), not an arbitrary artifact -- pointing
    ``register_model`` at a raw ``.pt`` file fails with "unable to find a logged
    model". So the promoted bundle is loaded and re-logged through
    ``mlflow.pytorch.log_model``, and the raw bundle is attached alongside it as
    the artifact the serving container actually consumes.
    """
    from ..bundle import load_bundle
    from .train import setup_mlflow

    mlflow = setup_mlflow(params)
    registered_name = str(get(params, "mlflow.registered_model_name"))
    loaded = load_bundle(bundle_path)

    with mlflow.start_run(run_name=f"promote-{model_name}") as run:
        mlflow.set_tags({
            "module": "M3->M4",
            "stage": "promotion",
            "promoted_model": model_name,
            "source_run_id": str(best.get("run_id", "")),
            "bundle_format_version": "1",
        })
        mlflow.log_metrics({
            k: float(best[k]) for k in ("test_f1", "test_recall", "test_precision",
                                        "test_accuracy", "latency_p95_ms")
            if k in best and best[k] == best[k]
        })
        # The self-contained bundle: this is what models/production holds and
        # what the container loads. Kept regardless of registry outcome.
        mlflow.log_artifact(str(bundle_path), artifact_path="production_bundle")
        mlflow.log_dict(loaded.info(), "production_bundle/model_card.json")

        # `serialization_format="pickle"` is explicit, not incidental. MLflow 3
        # defaults torch models to the traced-graph 'pt2' format, which requires
        # a TensorSpec-typed signature; a plain tensor input_example infers a
        # tensor-of-array signature that pt2 rejects. Pickle stores the module
        # as-is, needs no tracing, and is the right choice here anyway: the
        # bundle in models/production is what serving loads, so this registry
        # entry exists for lineage and discovery, not for MLflow-side inference.
        import torch

        spec = loaded.preprocess_spec
        example = torch.randn(1, loaded.in_channels, spec.resize, spec.resize).numpy()

        # MLflow 3 renamed `artifact_path` to `name`; accept either so the code
        # works across both major versions.
        try:
            info = mlflow.pytorch.log_model(
                loaded.model, name="model",
                input_example=example, serialization_format="pickle",
            )
        except TypeError:  # pragma: no cover - MLflow 2.x signature
            info = mlflow.pytorch.log_model(
                loaded.model, artifact_path="model",
                input_example=example, serialization_format="pickle",
            )

        model_uri = getattr(info, "model_uri", None) or f"runs:/{run.info.run_id}/model"
        version = mlflow.register_model(model_uri=model_uri, name=registered_name)
        log.info("Registered %s as %r version %s",
                 model_name, registered_name, getattr(version, "version", "?"))


def run_comparison(params: dict[str, Any], results: list[Any] | None = None,
                   do_promote: bool = True) -> dict[str, Any]:
    """Full compare -> gate -> select -> report -> promote flow."""
    banner(log, "M3 | Model comparison and promotion")

    table = build_comparison(results) if results else load_comparison_from_mlflow(params)
    log.info("Comparison table:\n%s", table.to_string(index=False))

    best, gate_results, rationale = select_best(table, params, results)
    report_path = write_comparison_report(table, params, best, gate_results, rationale, results)

    table_path = ensure_parent("reports/model_comparison.csv")
    table.to_csv(table_path, index=False)

    promoted_path = None
    if best is not None and do_promote:
        log.info("Selected %r: %s", best["model"], rationale)
        promoted_path = str(promote(params, best))
    elif best is None:
        log.error(rationale)

    return {
        "table": table,
        "best": best,
        "rationale": rationale,
        "gates": [{"model": g.model_name, "eligible": g.eligible, "detail": g.reason()}
                  for g in gate_results],
        "report_path": str(report_path),
        "table_path": str(table_path),
        "promoted_path": promoted_path,
    }
