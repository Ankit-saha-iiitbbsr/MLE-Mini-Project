"""Structured logging + deterministic seeding.

Two concerns that every stage needs and that are easy to get subtly wrong:

* **Logging** - one configuration function, called once per entry point, so the
  pipeline logs look the same whether a stage runs under DVC, pytest, or
  uvicorn. The serving process emits JSON lines instead of human text so the
  logs can be shipped to a collector without re-parsing.
* **Seeding** - reproducibility (M3) needs more than ``random.seed``: Python
  hashing, NumPy, and every torch backend each carry their own RNG.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from typing import Any

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Render records as one JSON object per line (for the serving container)."""

    _RESERVED = frozenset(
        vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()
        | {"asctime", "message", "taskName"}
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Anything passed via `extra=` gets promoted to a top-level field.
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str | int = "INFO", json_output: bool | None = None) -> None:
    """Install a single stderr handler on the root logger.

    Idempotent: repeated calls (CLI -> library -> DVC stage) reconfigure the
    level but never stack duplicate handlers.
    """
    global _CONFIGURED

    if json_output is None:
        json_output = os.environ.get("DEFECTVISION_LOG_JSON", "0") == "1"

    root = logging.getLogger()
    root.setLevel(level)

    if _CONFIGURED:
        for h in root.handlers:
            h.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    root.handlers = [handler]

    # These libraries are chatty at INFO and drown out pipeline output.
    for noisy in ("matplotlib", "PIL", "urllib3", "git", "mlflow.utils", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Module-level logger; configures logging on first use."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool = True) -> int:
    """Seed every RNG the pipeline touches. Returns *seed* for convenient logging.

    ``deterministic=True`` also disables cuDNN autotuning. On CPU this costs
    nothing; on GPU it trades a little throughput for run-to-run identical
    results, which is the trade the reproducibility requirement asks for.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass

    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` so augmentation is reproducible per worker."""
    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    del worker_id


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------


@contextmanager
def timed(logger: logging.Logger, label: str, level: int = logging.INFO):
    """Log how long a block took. Reports on failure too, then re-raises."""
    start = time.perf_counter()
    try:
        yield
    except Exception:
        logger.log(level, "%s failed after %.2fs", label, time.perf_counter() - start)
        raise
    else:
        logger.log(level, "%s completed in %.2fs", label, time.perf_counter() - start)


def banner(logger: logging.Logger, text: str) -> None:
    """Visually separate pipeline stages in the console log."""
    line = "=" * 74
    logger.info(line)
    logger.info(text)
    logger.info(line)
