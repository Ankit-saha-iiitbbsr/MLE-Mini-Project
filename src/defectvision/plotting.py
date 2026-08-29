"""Shared matplotlib helpers.

Every plotting site in this project needs the same three things: a non-interactive
backend (these run under DVC stages and in CI, where there is no display), a
consistent visual style, and the save-then-close-then-record dance. Centralising
them keeps the figure code in :mod:`defectvision.training.evaluate` and
:mod:`defectvision.monitoring.report` about the *plot* rather than about
matplotlib bookkeeping -- and guarantees figures are closed, which is the usual
cause of a pipeline slowly leaking memory over a few dozen plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: Palette used across every figure so reports read as one document.
COLOR_OK = "#3b7dd8"
COLOR_DEFECT = "#d1495b"
COLOR_ACCENT = "#e8a33d"
COLOR_NEUTRAL = "#6c757d"

DEFAULT_DPI = 130


def use_headless_backend() -> Any:
    """Select the Agg backend and return the ``pyplot`` module.

    Must run before the first pyplot import in the process, otherwise
    matplotlib picks an interactive backend and blocks (or crashes) on a
    headless CI runner.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_figure(fig: Any, path: str | Path, dpi: int = DEFAULT_DPI) -> str:
    """Write *fig* to *path*, close it, and return the path as a string.

    Closing is not optional: a stage that renders several figures per model per
    scenario will otherwise hold every one of them open until the process exits.
    """
    import matplotlib.pyplot as plt

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(dest, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(dest)
