"""``defectvision`` - one command-line entry point for the whole lifecycle.

Every stage of the pipeline is reachable here, and every DVC stage shells out to
this CLI rather than to a bespoke script. That means the command a developer
runs by hand and the command the pipeline runs in CI are the same command, so
there is no "works locally, fails in the pipeline" gap.

Run ``defectvision --help`` for the full list. The usual order is::

    defectvision data                 # M2: acquire -> validate -> split
    defectvision train --all          # M3: train every configured arm
    defectvision compare              # M3: gate, rank, promote
    defectvision serve                # M4: start the REST service
    defectvision simulate-drift       # M5: build shifted evaluation sets
    defectvision monitor              # M5: drift report + retraining decision
"""

from __future__ import annotations

import json
import sys

import typer

from .config import apply_overrides, coerce_scalar, get, load_params
from .logging_utils import banner, configure_logging, get_logger

app = typer.Typer(
    name="defectvision",
    help="End-to-end ML system for image-based casting defect classification.",
    add_completion=False,
    no_args_is_help=True,
)

log = get_logger("defectvision.cli")


def _params(overrides: list[str] | None = None, verbose: bool = False) -> dict:
    """Load params.yaml and apply any ``--set key=value`` overrides."""
    configure_logging("DEBUG" if verbose else "INFO")
    params = load_params()
    if overrides:
        parsed = {}
        for item in overrides:
            if "=" not in item:
                raise typer.BadParameter(f"--set expects key=value, got {item!r}")
            key, _, raw = item.partition("=")
            parsed[key.strip()] = coerce_scalar(raw.strip())
        params = apply_overrides(params, parsed)
        log.info("Applied overrides: %s", parsed)
    return params


SetOpt = typer.Option(None, "--set", "-s", help="Override a config key, e.g. -s train.epochs=2")
VerboseOpt = typer.Option(False, "--verbose", "-v", help="Debug-level logging")


# ===========================================================================
# M2 - Data
# ===========================================================================


@app.command()
def acquire(
    source: str | None = typer.Option(None, "--source", help="synthetic | kaggle | local"),
    preview: bool = typer.Option(True, "--preview/--no-preview", help="Write a sample contact sheet"),
    set_: list[str] | None = SetOpt,
    verbose: bool = VerboseOpt,
) -> None:
    """M2: fetch or generate the raw image corpus."""
    from .data.acquire import acquire as run_acquire
    from .data.acquire import preview_grid

    params = _params(set_, verbose)
    banner(log, "M2 | Data acquisition")
    record = run_acquire(params, source)
    if preview:
        preview_grid(params, "reports/figures/raw_samples.png")
    typer.echo(json.dumps(record, indent=2, default=str))


@app.command()
def validate(
    set_: list[str] | None = SetOpt,
    verbose: bool = VerboseOpt,
) -> None:
    """M2: validate the raw corpus against the configured quality gates."""
    from .data.validate import DataValidationError, run_validation

    params = _params(set_, verbose)
    banner(log, "M2 | Data validation")
    try:
        _, report = run_validation(params)
    except DataValidationError as exc:
        log.error("%s", exc)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Validation passed: {report.n_valid}/{report.n_files} images usable")


@app.command()
def split(
    set_: list[str] | None = SetOpt,
    verbose: bool = VerboseOpt,
) -> None:
    """M2: build the versioned train/val/test manifest."""
    from .data.split import build_manifest

    params = _params(set_, verbose)
    banner(log, "M2 | Dataset splitting")
    manifest = build_manifest(params)
    typer.echo(f"Manifest written with {len(manifest)} rows")


@app.command()
def data(
    source: str | None = typer.Option(None, "--source", help="synthetic | kaggle | local"),
    skip_acquire: bool = typer.Option(False, "--skip-acquire", help="Reuse existing data/raw"),
    set_: list[str] | None = SetOpt,
    verbose: bool = VerboseOpt,
) -> None:
    """M2: run the whole data pipeline (acquire -> validate -> split)."""
    from .data.acquire import acquire as run_acquire
    from .data.acquire import preview_grid
    from .data.split import build_manifest
    from .data.validate import DataValidationError, run_validation

    params = _params(set_, verbose)

    if not skip_acquire:
        banner(log, "M2 | Data acquisition")
        run_acquire(params, source)
        preview_grid(params, "reports/figures/raw_samples.png")

    banner(log, "M2 | Data validation")
    try:
        run_validation(params)
    except DataValidationError as exc:
        log.error("%s", exc)
        raise typer.Exit(code=1) from exc

    banner(log, "M2 | Dataset splitting")
    manifest = build_manifest(params)
    typer.echo(f"M2 complete: {len(manifest)} images in the versioned manifest")


# ===========================================================================
# M3 - Training
# ===========================================================================


@app.command()
def train(
    model: str | None = typer.Option(None, "--model", "-m", help="Model arm to train"),
    all_models: bool = typer.Option(False, "--all", help="Train every configured arm"),
    compare_after: bool = typer.Option(True, "--compare/--no-compare",
                                       help="Run comparison + promotion after training"),
    set_: list[str] | None = SetOpt,
    verbose: bool = VerboseOpt,
) -> None:
    """M3: train one arm or every arm, tracking each as an MLflow run."""
    from .training.compare import run_comparison
    from .training.train import train_model

    params = _params(set_, verbose)
    configured = list(get(params, "train.models").keys())

    if all_models:
        targets = configured
    elif model:
        if model not in configured:
            raise typer.BadParameter(f"Unknown model {model!r}. Configured: {configured}")
        targets = [model]
    else:
        raise typer.BadParameter("Pass --model <name> or --all")

    results = []
    for name in targets:
        try:
            results.append(train_model(params, name))
        except Exception as exc:
            log.exception("Training arm %r failed: %s", name, exc)
            if len(targets) == 1:
                raise typer.Exit(code=1) from exc

    if not results:
        log.error("No arm trained successfully")
        raise typer.Exit(code=1)

    typer.echo("\n=== Training summary ===")
    for r in results:
        typer.echo(json.dumps(r.summary_row(), indent=None))

    if compare_after and len(targets) > 1:
        run_comparison(params, results)


@app.command()
def compare(
    from_mlflow: bool = typer.Option(False, "--from-mlflow",
                                     help="Rebuild the table by querying the tracking store"),
    promote_best: bool = typer.Option(True, "--promote/--no-promote"),
    set_: list[str] | None = SetOpt,
    verbose: bool = VerboseOpt,
) -> None:
    """M3: compare tracked runs, apply promotion gates, promote the winner."""
    from .training.compare import run_comparison

    params = _params(set_, verbose)
    del from_mlflow  # comparison always reads MLflow when no in-memory results exist
    outcome = run_comparison(params, results=None, do_promote=promote_best)
    if outcome["best"] is None:
        raise typer.Exit(code=1)
    typer.echo(f"Promoted: {outcome['best']['model']} -> {outcome['promoted_path']}")


@app.command()
def package(
    model: str | None = typer.Option(None, "--model", "-m",
                                        help="Promote this arm instead of the gated winner"),
    set_: list[str] | None = SetOpt,
    verbose: bool = VerboseOpt,
) -> None:
    """M4: copy a candidate bundle to the production path used by the service."""
    from .training.compare import load_comparison_from_mlflow, promote, run_comparison

    params = _params(set_, verbose)
    banner(log, "M4 | Model packaging")

    if model:
        table = load_comparison_from_mlflow(params)
        rows = table[table["model"] == model]
        if rows.empty:
            raise typer.BadParameter(f"No tracked run found for model {model!r}")
        dest = promote(params, rows.iloc[0].to_dict())
    else:
        outcome = run_comparison(params, results=None, do_promote=True)
        if outcome["best"] is None:
            raise typer.Exit(code=1)
        dest = outcome["promoted_path"]

    typer.echo(f"Production bundle: {dest}")


# ===========================================================================
# M4 - Serving
# ===========================================================================


@app.command()
def serve(
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev only)"),
    set_: list[str] | None = SetOpt,
    verbose: bool = VerboseOpt,
) -> None:
    """M4: start the FastAPI inference service."""
    import uvicorn

    params = _params(set_, verbose)
    bind_host = host or str(get(params, "serving.host"))
    bind_port = int(port or get(params, "serving.port"))

    banner(log, f"M4 | Serving on http://{bind_host}:{bind_port}  (docs at /docs)")
    uvicorn.run(
        "defectvision.serving.app:app",
        host=bind_host,
        port=bind_port,
        reload=reload,
        log_level="debug" if verbose else "info",
    )


@app.command("benchmark")
def benchmark(
    n: int = typer.Option(200, "--n", help="Number of requests"),
    concurrency: int = typer.Option(4, "--concurrency", "-c"),
    url: str = typer.Option("http://127.0.0.1:8000", "--url"),
    set_: list[str] | None = SetOpt,
    verbose: bool = VerboseOpt,
) -> None:
    """M4: measure end-to-end API latency and throughput against a running service."""
    from .serving.benchmark import run_benchmark

    params = _params(set_, verbose)
    report = run_benchmark(params, base_url=url, n_requests=n, concurrency=concurrency)
    typer.echo(json.dumps(report, indent=2))


# ===========================================================================
# M5 - Monitoring
# ===========================================================================


@app.command("simulate-drift")
def simulate_drift(
    scenario: str | None = typer.Option(None, "--scenario",
                                           help="One scenario name; default is all"),
    send: bool = typer.Option(False, "--send",
                              help="POST the shifted images to a running service"),
    url: str = typer.Option("http://127.0.0.1:8000", "--url"),
    set_: list[str] | None = SetOpt,
    verbose: bool = VerboseOpt,
) -> None:
    """M5: generate distribution-shifted evaluation sets and score them."""
    from .monitoring.simulate_drift import run_simulation

    params = _params(set_, verbose)
    banner(log, "M5 | Drift simulation")
    report = run_simulation(params, scenario=scenario, send_to_api=send, base_url=url)
    typer.echo(json.dumps(report["summary"], indent=2, default=str))


@app.command()
def monitor(
    window: int | None = typer.Option(None, "--window", help="Rows per evaluation window"),
    set_: list[str] | None = SetOpt,
    verbose: bool = VerboseOpt,
) -> None:
    """M5: compute drift metrics from the prediction log and write the monitoring report."""
    from .monitoring.report import build_monitoring_report

    params = _params(set_, verbose)
    banner(log, "M5 | Monitoring report")
    report = build_monitoring_report(params, window=window)
    typer.echo(json.dumps(report["summary"], indent=2, default=str))


@app.command("reference-stats")
def reference_stats(
    set_: list[str] | None = SetOpt,
    verbose: bool = VerboseOpt,
) -> None:
    """M5: snapshot the training distribution as the drift reference baseline."""
    from .monitoring.reference import build_reference

    params = _params(set_, verbose)
    banner(log, "M5 | Building drift reference")
    ref = build_reference(params)
    typer.echo(f"Reference baseline written for {ref['n_samples']} training images")


@app.command("check-retrain")
def check_retrain(
    set_: list[str] | None = SetOpt,
    verbose: bool = VerboseOpt,
) -> None:
    """M5: evaluate the retraining trigger rules against current monitoring signals."""
    from .monitoring.retrain_trigger import evaluate_triggers

    params = _params(set_, verbose)
    banner(log, "M5 | Retraining trigger evaluation")
    decision = evaluate_triggers(params)
    typer.echo(json.dumps(decision, indent=2, default=str))
    raise typer.Exit(code=10 if decision["should_retrain"] else 0)


# ===========================================================================
# Utilities
# ===========================================================================


@app.command()
def info(
    set_: list[str] | None = SetOpt,
    verbose: bool = VerboseOpt,
) -> None:
    """Print environment, dataset and model status."""
    from .config import project_root, resolve
    from .data.acquire import kaggle_credentials_status
    from .training import reproducibility

    params = _params(set_, verbose)
    record = reproducibility.capture(params)
    bundle = resolve(get(params, "serving.model_bundle"))

    payload = {
        "project_root": str(project_root()),
        "data_source": get(params, "data.source"),
        "kaggle_credentials": kaggle_credentials_status(),
        "dataset": {k: record.get(k) for k in
                    ("manifest_sha256", "dataset_n_images", "dataset_split_counts")},
        "git": {k: record.get(k) for k in ("git_commit_short", "git_branch", "git_dirty")},
        "environment": {k: record.get(k) for k in
                        ("python_version", "torch_version", "mlflow_version", "device_name")},
        "production_bundle": {"path": str(bundle), "exists": bundle.is_file()},
    }
    typer.echo(json.dumps(payload, indent=2, default=str))


def main() -> None:  # pragma: no cover - console-script shim
    try:
        app()
    except KeyboardInterrupt:
        log.warning("Interrupted")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
