"""Typed access to ``params.yaml``.

Everything in the pipeline reads its configuration through :func:`load_params`
so that a run is fully described by ``params.yaml`` + a git commit. The loader
resolves the project root by walking up from this file, which means the CLI
behaves the same whether it is invoked from the repo root, from ``src/``, or
from inside a DVC stage.
"""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------
# Project layout
# --------------------------------------------------------------------------

#: Repository root: ``src/defectvision/config.py`` -> up three levels.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Overridable so tests can point at a fixture params file.
PARAMS_ENV_VAR = "DEFECTVISION_PARAMS"


def project_root() -> Path:
    """Absolute path to the repository root."""
    return PROJECT_ROOT


def resolve(path: str | os.PathLike[str]) -> Path:
    """Resolve *path* against the project root unless it is already absolute."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def params_path() -> Path:
    """Location of the active params file."""
    override = os.environ.get(PARAMS_ENV_VAR)
    return Path(override).resolve() if override else PROJECT_ROOT / "params.yaml"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _load_cached(path_str: str, mtime: float) -> dict[str, Any]:
    """Parse the YAML file. Keyed on mtime so edits are picked up automatically."""
    del mtime  # part of the cache key only
    with open(path_str, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path_str} must contain a YAML mapping at the top level")
    return data


def load_params(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load ``params.yaml`` as a plain nested dict.

    A deep copy is returned so callers may mutate the result (e.g. to apply CLI
    overrides) without corrupting the cache shared by other callers.
    """
    p = Path(path).resolve() if path is not None else params_path()
    if not p.exists():
        raise FileNotFoundError(
            f"Params file not found: {p}\n"
            f"Run commands from the repository root, or set {PARAMS_ENV_VAR}."
        )
    return copy.deepcopy(_load_cached(str(p), p.stat().st_mtime))


def get(params: dict[str, Any], dotted_key: str, default: Any = ...) -> Any:
    """Fetch ``params["a"]["b"]["c"]`` via the dotted key ``"a.b.c"``.

    Raises :class:`KeyError` when the key is missing and no *default* is given,
    so a typo in a config path fails loudly instead of silently defaulting.
    """
    node: Any = params
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is ...:
                raise KeyError(f"Missing config key: {dotted_key!r}")
            return default
        node = node[part]
    return node


def apply_overrides(params: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *params* with dotted-key *overrides* applied.

    Used by the CLI (``--set train.epochs=2``) and by the test suite to shrink
    the pipeline without editing the committed config.
    """
    out = copy.deepcopy(params)
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise TypeError(f"Cannot override {dotted_key!r}: {part!r} is not a mapping")
        node[parts[-1]] = value
    return out


def coerce_scalar(text: str) -> Any:
    """Parse a CLI override value using YAML rules (so ``3``/``true``/``[1,2]`` work)."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


# --------------------------------------------------------------------------
# Derived helpers used across modules
# --------------------------------------------------------------------------


def class_names(params: dict[str, Any]) -> list[str]:
    """Ordered class names; index position is the integer label."""
    return list(get(params, "data.classes"))


def positive_class_index(params: dict[str, Any]) -> int:
    """Index of the class treated as 'positive' for recall/precision reporting.

    By convention the last class (``defect``) is positive: on a production line
    a missed defect is far more expensive than a false alarm, so 'positive'
    should mean 'the thing we must not miss'.
    """
    return len(class_names(params)) - 1


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create *path* (resolved against the project root) and return it."""
    p = resolve(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent(path: str | os.PathLike[str]) -> Path:
    """Create the parent directory of *path* and return the resolved path."""
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
