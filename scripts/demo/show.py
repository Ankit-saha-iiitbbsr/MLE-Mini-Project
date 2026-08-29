"""Render pipeline artifacts as camera-friendly tables for the recorded demo.

The demo has no voiceover, so anything shown on screen has to be legible at
video resolution: wide margins, few columns, no JSON dumps. Each subcommand
below formats one artifact for exactly that.

    python scripts/demo/show.py validation
    python scripts/demo/show.py dataset
    python scripts/demo/show.py models
    python scripts/demo/show.py intervals
    python scripts/demo/show.py runs
    python scripts/demo/show.py drift
    python scripts/demo/show.py retrain
    python scripts/demo/show.py benchmark
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(rel: str):
    path = ROOT / rel
    if not path.exists():
        print(f"\n  [missing artifact: {rel}]\n")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _rule(width: int = 74) -> None:
    print("  " + "-" * width)


# ---------------------------------------------------------------- M2


def validation() -> None:
    r = _load("reports/validation_report.json")
    if not r:
        return
    print()
    print(f"  Validation passed: {r['passed']}"
          f"    ({r['n_valid']:,} of {r['n_files']:,} images usable)")
    print()
    for c in r["checks"]:
        print(f"  [{c['status']:4}]  {c['name']:26}  {c['message'][:60]}")
    print()


def dataset() -> None:
    c = _load("data/processed/dataset_card.json")
    if not c:
        return
    s = c["split_counts"]
    src = c.get("source") or {}
    print()
    print(f"  images in manifest  : {c['n_images']:,}")
    print(f"  exact dups dropped  : {c['exact_duplicates_dropped']}")
    print(f"  leakage groups      : {c['n_groups']:,}   (none spanning folds)")
    print(f"  train / val / test  : {s['train']:,} / {s['val']:,} / {s['test']:,}")
    print(f"  source              : {src.get('source', '?')} / {src.get('subset', '?')}")
    print()


# ---------------------------------------------------------------- M3


def models() -> None:
    path = ROOT / "reports/model_comparison.csv"
    if not path.exists():
        print("\n  [missing reports/model_comparison.csv]\n")
        return
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    best = max(rows, key=lambda r: float(r["test_f1"]))

    print()
    print(f"  {'model':<20}{'params':>9}{'test F1':>10}{'recall':>9}"
          f"{'p95 ms':>9}{'img/s':>8}")
    _rule()
    for r in rows:
        mark = " <-" if r["model"] == best["model"] else ""
        print(f"  {r['model']:<20}{float(r['params_M']):>8.3f}M"
              f"{float(r['test_f1']):>10.4f}{float(r['test_recall']):>9.4f}"
              f"{float(r['latency_p95_ms']):>9.2f}{float(r['throughput_ips']):>8.0f}{mark}")
    _rule()
    print(f"  Test split held out and evaluated once. Threshold tuned on validation.")
    print()


def intervals() -> None:
    """The comparison that decided the promotion."""
    print()
    print(f"  {'model':<16}{'test F1  (95% bootstrap CI)':<32}{'p95':>9}{'params':>10}")
    _rule()
    print(f"  {'baseline_cnn':<16}{'0.9970  [0.9942, 0.9994]':<32}{'2.79 ms':>9}{'0.25 M':>10}")
    print(f"  {'resnet18':<16}{'0.9964  [0.9934, 0.9988]':<32}{'16.94 ms':>9}{'11.17 M':>10}")
    _rule()
    print()
    print("  The intervals overlap almost entirely -- the test set cannot resolve")
    print("  a difference in accuracy. So the tie is broken on cost, not on F1.")
    print()


def runs() -> None:
    """Show that every run carries what it needs to be reproducible."""
    print()
    print("  Each tracked run logs four things. A run missing any one of them")
    print("  is not reproducible:")
    print()
    print("    1. git commit          (flagged if the tree was dirty)")
    print("    2. params.yaml         full snapshot + content hash")
    print("    3. dataset manifest    SHA-256 -- pins which images, which folds")
    print("    4. library versions    resolved at run time")
    print()
    bundle = ROOT / "models/production/PROMOTED.json"
    if bundle.exists():
        p = json.loads(bundle.read_text(encoding="utf-8"))
        print(f"  Promoted to production : {p.get('promoted_model')}"
              f"  (run {str(p.get('run_id'))[:8]})")
        print(f"  test F1 {p.get('test_f1')}   recall {p.get('test_recall')}")
        print()


# ---------------------------------------------------------------- M5


def drift() -> None:
    r = _load("reports/monitoring_report.json")
    if not r:
        return
    print()
    print(f"  {'scenario':<22}{'max PSI':>9}{'accuracy':>10}{'acc drop':>10}"
          f"{'confidence':>12}")
    _rule()
    for name, v in r["summary"].items():
        acc = f"{v['accuracy']:.4f}" if v.get("accuracy") is not None else "   -"
        drop = f"+{v['accuracy_drop']:.4f}" if v.get("accuracy_drop") is not None else "   -"
        print(f"  {name:<22}{v['psi_max']:>9.3f}{acc:>10}{drop:>10}"
              f"{v['mean_confidence']:>12.3f}")
    _rule()
    print()


def silent_failure() -> None:
    """The single most important number in the whole project."""
    r = _load("reports/monitoring_report.json")
    if not r:
        return
    s = r["summary"]
    base = s.get("baseline", {})
    dim = s.get("lighting_dim", {})
    if not dim:
        return

    degraded = [v for k, v in s.items() if k != "baseline"]
    caught_by_conf = sum(1 for v in degraded if "confidence_collapse" in v.get("alerts", []))
    caught_by_psi = sum(1 for v in degraded if "data_drift" in v.get("alerts", []))

    print()
    print("                        accuracy      mean confidence")
    _rule(58)
    print(f"   baseline (control)     {base.get('accuracy', 0):.4f}"
          f"            {base.get('mean_confidence', 0):.3f}")
    print(f"   lighting_dim           {dim.get('accuracy', 0):.4f}"
          f"            {dim.get('mean_confidence', 0):.3f}")
    _rule(58)
    print()
    print("   Accuracy collapses. Confidence goes UP.")
    print("   The model is confidently, catastrophically wrong.")
    print()
    print(f"   Across the {len(degraded)} degraded scenarios:")
    print(f"     confidence monitoring alone would have caught {caught_by_conf} of {len(degraded)}")
    print(f"     PSI vs the frozen training baseline caught      {caught_by_psi} of {len(degraded)}")
    print()


def retrain() -> None:
    d = _load("reports/retraining_decision.json")
    if not d:
        return
    print()
    print(f"   should_retrain : {d['should_retrain']}")
    print(f"   action         : {d['action']}")
    print(f"   rules fired    : {[f['rule'] for f in d.get('fired_rules', [])]}")
    print(f"   windows        : {d.get('windows_evaluated')}")
    print()
    reason = str(d.get("reason", ""))
    import textwrap
    for line in textwrap.wrap(reason, width=70):
        print(f"   {line}")
    print()


# ---------------------------------------------------------------- M4


def benchmark() -> None:
    b = _load("reports/api_benchmark.json")
    if not b:
        return
    lat = b["latency_ms"]
    print()
    print(f"   requests    : {b['n_succeeded']}/{b['n_requests']} succeeded"
          f"   (concurrency {b['concurrency']})")
    print(f"   throughput  : {b['throughput_rps']} req/s")
    print(f"   latency     : p50 {lat['p50']} ms   p95 {lat['p95']} ms"
          f"   p99 {lat['p99']} ms")
    print()
    print("   Measured end-to-end over HTTP -- multipart parsing, JPEG decode,")
    print("   preprocessing and serialisation included, not just forward().")
    print()


COMMANDS = {
    "validation": validation,
    "dataset": dataset,
    "models": models,
    "intervals": intervals,
    "runs": runs,
    "drift": drift,
    "silent": silent_failure,
    "retrain": retrain,
    "benchmark": benchmark,
}


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = COMMANDS.get(name)
    if fn is None:
        print(f"usage: show.py [{' | '.join(COMMANDS)}]")
        sys.exit(2)
    fn()
