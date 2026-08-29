"""Drift detectors: PSI, two-sample KS, and chi-square.

Three tests because they answer different questions and fail in different ways:

**PSI (Population Stability Index)** is the workhorse. It bins the reference
distribution, compares bin *shares*, and returns a single interpretable number
with industry-standard bands (<0.10 stable, 0.10-0.25 moderate, >0.25
significant). Crucially it is **not** a hypothesis test, so its value does not
depend on sample size -- 10,000 requests will not flag a shift that 500 would
have called stable. That is what makes it safe to threshold on.

**KS (Kolmogorov-Smirnov)** is a proper hypothesis test on the full empirical
CDF, so it catches shape changes PSI's binning can smooth over. Its weakness is
the mirror image of PSI's strength: with enough samples it rejects on
differences too small to matter. It is therefore used as corroboration, never
as the sole trigger.

**Chi-square** handles the categorical signals -- predicted-class mix and
decision mix -- where the others do not apply.

The one detail that decides whether any of this works: **bin edges come from
the reference distribution and are frozen**. Recomputing quantile bins on each
window would make every window look like a perfect match to itself, and PSI
would sit near zero no matter what happened.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

EPS = 1e-6

#: Conventional PSI interpretation bands.
PSI_STABLE = 0.10
PSI_SIGNIFICANT = 0.25


@dataclass
class DriftResult:
    """Outcome of one feature's drift assessment."""

    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    reference_mean: float
    current_mean: float
    mean_shift: float
    severity: str  # none | moderate | significant
    n_reference: int
    n_current: int

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------


def quantile_bin_edges(reference: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Quantile bin edges from the reference sample, extended to +/-inf.

    Quantile (equal-frequency) bins rather than equal-width: image statistics
    are skewed, and equal-width bins would leave most of them empty, making PSI
    dominated by a handful of near-zero denominators.

    Duplicate edges (from a spiky distribution) are collapsed, so a feature with
    few distinct values yields fewer, valid bins instead of degenerate ones.
    """
    ref = np.asarray(reference, dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.array([-np.inf, np.inf])

    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(ref, quantiles))
    if edges.size < 2:
        # Constant feature: one bin spanning everything.
        return np.array([-np.inf, np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _bin_shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Fraction of *values* falling in each bin defined by *edges*."""
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros(len(edges) - 1, dtype=np.float64)
    counts, _ = np.histogram(vals, bins=edges)
    return counts.astype(np.float64) / max(counts.sum(), 1)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    edges: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """PSI between a reference and a current sample.

    Returns ``(psi, edges)`` so the caller can persist the edges and reuse them
    for every subsequent window.

    ``EPS`` floors both shares: an empty bin would otherwise send the log term
    to infinity and make PSI meaningless whenever one bin happens to be unused.
    """
    if edges is None:
        edges = quantile_bin_edges(reference, n_bins)

    ref_share = _bin_shares(reference, edges)
    cur_share = _bin_shares(current, edges)

    ref_share = np.clip(ref_share, EPS, None)
    cur_share = np.clip(cur_share, EPS, None)

    psi = float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))
    return psi, edges


def ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test. Returns ``(statistic, p_value)``."""
    from scipy import stats

    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size < 2 or cur.size < 2:
        return float("nan"), float("nan")
    result = stats.ks_2samp(ref, cur)
    return float(result.statistic), float(result.pvalue)


def chi_square_test(
    reference_counts: dict[Any, int],
    current_counts: dict[Any, int],
) -> tuple[float, float]:
    """Chi-square test of independence over a categorical distribution.

    Returns ``(statistic, p_value)``. Categories absent from one side are
    included with a zero count so the comparison stays aligned.
    """
    from scipy import stats

    categories = sorted(set(reference_counts) | set(current_counts), key=str)
    ref = np.array([reference_counts.get(c, 0) for c in categories], dtype=np.float64)
    cur = np.array([current_counts.get(c, 0) for c in categories], dtype=np.float64)

    # Drop categories empty on both sides; they contribute nothing and break
    # the test's degrees-of-freedom calculation.
    keep = (ref + cur) > 0
    ref, cur = ref[keep], cur[keep]
    if ref.size < 2 or ref.sum() == 0 or cur.sum() == 0:
        return float("nan"), float("nan")

    table = np.vstack([ref, cur])
    stat, p, _, _ = stats.chi2_contingency(table)
    return float(stat), float(p)


def classify_severity(psi: float, warn: float = PSI_STABLE,
                      alert: float = PSI_SIGNIFICANT) -> str:
    """Map a PSI value onto the ``none``/``moderate``/``significant`` bands."""
    if not np.isfinite(psi):
        return "unknown"
    if psi >= alert:
        return "significant"
    if psi >= warn:
        return "moderate"
    return "none"


# ---------------------------------------------------------------------------
# Feature-set assessment
# ---------------------------------------------------------------------------


def assess_feature_drift(
    reference: dict[str, Any],
    current: dict[str, np.ndarray],
    *,
    n_bins: int = 10,
    psi_warn: float = PSI_STABLE,
    psi_alert: float = PSI_SIGNIFICANT,
) -> dict[str, DriftResult]:
    """Run every detector over every monitored feature.

    ``reference`` is the payload written by
    :mod:`defectvision.monitoring.reference`: it carries the raw reference
    samples and the frozen bin edges per feature.
    """
    results: dict[str, DriftResult] = {}
    ref_features = reference.get("features", {})

    for name, cur_values in current.items():
        ref_entry = ref_features.get(name)
        if ref_entry is None:
            continue

        ref_values = np.asarray(ref_entry["samples"], dtype=np.float64)
        cur_values = np.asarray(cur_values, dtype=np.float64)
        cur_values = cur_values[np.isfinite(cur_values)]
        if cur_values.size == 0:
            continue

        stored_edges = ref_entry.get("bin_edges")
        edges = np.asarray(stored_edges, dtype=np.float64) if stored_edges else None
        psi, _ = population_stability_index(ref_values, cur_values, n_bins, edges)
        ks_stat, ks_p = ks_test(ref_values, cur_values)

        ref_mean = float(np.mean(ref_values)) if ref_values.size else float("nan")
        cur_mean = float(np.mean(cur_values))

        results[name] = DriftResult(
            feature=name,
            psi=psi,
            ks_statistic=ks_stat,
            ks_pvalue=ks_p,
            reference_mean=ref_mean,
            current_mean=cur_mean,
            mean_shift=cur_mean - ref_mean,
            severity=classify_severity(psi, psi_warn, psi_alert),
            n_reference=int(ref_values.size),
            n_current=int(cur_values.size),
        )

    return results


def summarise_drift(results: dict[str, DriftResult]) -> dict[str, Any]:
    """Roll per-feature results up into the numbers the trigger rules consume."""
    if not results:
        return {"psi_max": 0.0, "psi_mean": 0.0, "n_features": 0,
                "n_drifted": 0, "n_significant": 0, "drifted_features": [],
                "worst_feature": None}

    psis = {name: r.psi for name, r in results.items() if np.isfinite(r.psi)}
    drifted = [n for n, r in results.items() if r.severity in ("moderate", "significant")]
    significant = [n for n, r in results.items() if r.severity == "significant"]
    worst = max(psis, key=lambda k: psis[k]) if psis else None

    return {
        "psi_max": float(max(psis.values())) if psis else 0.0,
        "psi_mean": float(np.mean(list(psis.values()))) if psis else 0.0,
        "n_features": len(results),
        "n_drifted": len(drifted),
        "n_significant": len(significant),
        "drifted_features": sorted(drifted),
        "significant_features": sorted(significant),
        "worst_feature": worst,
    }
