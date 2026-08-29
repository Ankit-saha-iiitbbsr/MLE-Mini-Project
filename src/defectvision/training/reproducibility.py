"""Capture everything needed to rebuild a run from its logged configuration.

"Reproducible" in the rubric means someone can pick a run out of MLflow months
later and recreate it. That needs four things recorded, and a run that is
missing any one of them is not reproducible:

1. **Code** - the git commit, plus whether the tree was dirty at launch. A run
   from a dirty tree is flagged, because its commit does not describe it.
2. **Config** - the exact ``params.yaml`` content hash and a full snapshot.
3. **Data** - the manifest hash. The manifest pins which images were in which
   fold, so this is the dataset version identifier.
4. **Environment** - interpreter, library versions, platform, thread counts.

All of it lands in MLflow as params/tags, and :func:`reproduction_command`
prints the command that recreates the run.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, params_path, resolve
from ..logging_utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------


def _git(*args: str) -> str | None:
    """Run a git command in the project root; return stripped stdout or None."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info() -> dict[str, Any]:
    """Commit, branch and dirty state of the working tree."""
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"git_available": False}

    status = _git("status", "--porcelain")
    dirty = bool(status)
    if dirty:
        log.warning(
            "Working tree has uncommitted changes -- this run is NOT reproducible "
            "from commit %s alone. Commit before a run you intend to cite.",
            commit[:8],
        )
    return {
        "git_available": True,
        "git_commit": commit,
        "git_commit_short": commit[:8],
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": dirty,
        "git_dirty_files": len(status.splitlines()) if status else 0,
    }


# ---------------------------------------------------------------------------
# Hashes
# ---------------------------------------------------------------------------


def file_hash(path: str | Path, algo: str = "sha256") -> str | None:
    """Content hash of a file, or None when it does not exist."""
    p = resolve(path)
    if not p.is_file():
        return None
    h = hashlib.new(algo)
    with open(p, "rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def dict_hash(payload: dict[str, Any]) -> str:
    """Stable hash of a config dict (key order independent)."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def dataset_fingerprint(params: dict[str, Any]) -> dict[str, Any]:
    """Identify the exact dataset version a run consumed."""
    from ..config import get

    processed = resolve(get(params, "data.processed_dir"))
    manifest = processed / "manifest.csv"
    card = processed / "dataset_card.json"

    info: dict[str, Any] = {
        "manifest_path": str(manifest),
        "manifest_sha256": file_hash(manifest),
        "dataset_card_sha256": file_hash(card),
    }
    if card.is_file():
        try:
            data = json.loads(card.read_text(encoding="utf-8"))
            info["dataset_n_images"] = data.get("n_images")
            info["dataset_split_counts"] = data.get("split_counts")
            info["dataset_source"] = (data.get("source") or {}).get("source")
        except json.JSONDecodeError:  # pragma: no cover
            pass
    return info


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def environment_info() -> dict[str, Any]:
    """Interpreter, key library versions, and hardware relevant to results."""
    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }
    for mod in ("numpy", "pandas", "torch", "torchvision", "sklearn", "PIL", "mlflow"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except ImportError:
            info[f"{mod}_version"] = "not-installed"

    try:
        import torch

        info["torch_threads"] = torch.get_num_threads()
        info["cuda_available"] = torch.cuda.is_available()
        info["device_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        )
    except ImportError:  # pragma: no cover
        pass
    return info


def capture(params: dict[str, Any]) -> dict[str, Any]:
    """Full reproducibility record for one run."""
    pp = params_path()
    return {
        **git_info(),
        **environment_info(),
        **dataset_fingerprint(params),
        "params_path": str(pp),
        "params_sha256": file_hash(pp),
        "params_config_hash": dict_hash(params),
        "seed": params.get("seed"),
    }


def reproduction_command(record: dict[str, Any], model_name: str) -> str:
    """A copy-pasteable recipe that recreates the run."""
    commit = record.get("git_commit_short", "<commit>")
    dirty = " (NOTE: tree was dirty; commit does not fully describe this run)" \
        if record.get("git_dirty") else ""
    return (
        f"git checkout {commit}{dirty}\n"
        f"pip install -r requirements.txt\n"
        f"dvc repro                       # rebuilds the dataset (manifest "
        f"sha256={str(record.get('manifest_sha256'))[:12]}...)\n"
        f"defectvision train --model {model_name}"
    )
