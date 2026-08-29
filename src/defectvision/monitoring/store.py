"""Prediction log -- the datastore every monitoring signal is derived from.

What gets logged is the design decision here. Logging only the prediction makes
drift undetectable: when accuracy falls you can see *that* it fell but never
*why*. So each row carries three groups of fields:

* **the decision** - probability, label, threshold, and whether it was routed to
  a human. Enables accuracy tracking once labels arrive, and confidence-based
  monitoring before they do.
* **the input's statistics** - the same seven features the reference baseline
  is built from, computed at request time. This is what makes *data* drift
  detectable without storing the images themselves.
* **the context** - model name, run id, latency, timestamp. Without the model
  identity a drift report cannot tell a genuine data shift from a deployment
  that swapped the model underneath it.

Ground truth is nullable and filled in later: on a real line, labels arrive
hours or days after the prediction, from teardown or customer returns. The
schema treats delayed labels as normal rather than as an afterthought.

SQLite is chosen deliberately: zero operational cost, transactional, and it is
the datastore the M5 lab exercise asks for. WAL mode plus one connection per
call keeps it safe under FastAPI's threadpool.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import ensure_parent, get
from ..features.image_stats import FEATURE_NAMES
from ..logging_utils import get_logger

log = get_logger(__name__)

SCHEMA_VERSION = 1

_STAT_COLUMNS = ",\n    ".join(f"{name} REAL" for name in FEATURE_NAMES)

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS predictions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id        TEXT    NOT NULL,
    ts                REAL    NOT NULL,
    ts_iso            TEXT    NOT NULL,

    -- context
    model_name        TEXT,
    model_run_id      TEXT,
    model_threshold   REAL,
    source            TEXT,          -- 'api' | 'drift_sim' | 'batch'
    scenario          TEXT,          -- drift scenario name, when simulated

    -- input descriptors
    filename          TEXT,
    width             INTEGER,
    height            INTEGER,
    file_bytes        INTEGER,
    {_STAT_COLUMNS},

    -- decision
    probability       REAL    NOT NULL,
    predicted_label   INTEGER NOT NULL,
    predicted_class   TEXT,
    decision          TEXT,          -- auto_accept | auto_reject | human_review
    latency_ms        REAL,

    -- delayed feedback
    ground_truth      INTEGER,
    ground_truth_ts   REAL,
    feedback_source   TEXT
);

CREATE INDEX IF NOT EXISTS idx_predictions_ts       ON predictions (ts);
CREATE INDEX IF NOT EXISTS idx_predictions_scenario ON predictions (scenario);
CREATE INDEX IF NOT EXISTS idx_predictions_model    ON predictions (model_name);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


@dataclass
class PredictionRecord:
    """One logged inference."""

    probability: float
    predicted_label: int
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = field(default_factory=time.time)
    model_name: str | None = None
    model_run_id: str | None = None
    model_threshold: float | None = None
    source: str = "api"
    scenario: str | None = None
    filename: str | None = None
    width: int | None = None
    height: int | None = None
    file_bytes: int | None = None
    predicted_class: str | None = None
    decision: str | None = None
    latency_ms: float | None = None
    ground_truth: int | None = None
    image_stats: dict[str, float] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        stats = row.pop("image_stats") or {}
        row["ts_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.ts))
        for name in FEATURE_NAMES:
            row[name] = float(stats[name]) if name in stats else None
        row["ground_truth_ts"] = None
        row["feedback_source"] = None
        return row


class PredictionStore:
    """SQLite-backed prediction log."""

    def __init__(self, db_path: str | Path) -> None:
        self.path = ensure_parent(db_path)
        self._init_schema()

    @contextmanager
    def _connect(self):
        """One short-lived connection per operation.

        Cheaper than it sounds for SQLite, and it sidesteps the
        thread-affinity rules that make a shared connection unsafe under
        FastAPI's request threadpool.
        """
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            # WAL lets the monitoring reader run while the service writes.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )

    # -- writes -----------------------------------------------------------

    def log(self, record: PredictionRecord) -> int:
        """Append one prediction; returns the row id."""
        row = record.to_row()
        cols = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row)
        with self._connect() as conn:
            cur = conn.execute(
                f"INSERT INTO predictions ({cols}) VALUES ({placeholders})", row
            )
            return int(cur.lastrowid or 0)

    def log_many(self, records: list[PredictionRecord]) -> int:
        """Bulk append -- used by the drift simulator, which writes thousands."""
        if not records:
            return 0
        rows = [r.to_row() for r in records]
        cols = ", ".join(rows[0].keys())
        placeholders = ", ".join(f":{k}" for k in rows[0])
        with self._connect() as conn:
            conn.executemany(
                f"INSERT INTO predictions ({cols}) VALUES ({placeholders})", rows
            )
        return len(rows)

    def attach_ground_truth(self, request_id: str, label: int,
                            source: str = "manual") -> bool:
        """Record a label that arrived after the prediction."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE predictions SET ground_truth = ?, ground_truth_ts = ?, "
                "feedback_source = ? WHERE request_id = ?",
                (int(label), time.time(), source, request_id),
            )
            return cur.rowcount > 0

    # -- reads ------------------------------------------------------------

    def count(self, scenario: str | None = None) -> int:
        with self._connect() as conn:
            if scenario is None:
                cur = conn.execute("SELECT COUNT(*) FROM predictions")
            else:
                cur = conn.execute(
                    "SELECT COUNT(*) FROM predictions WHERE scenario IS ?", (scenario,)
                )
            return int(cur.fetchone()[0])

    def fetch(
        self,
        *,
        limit: int | None = None,
        scenario: str | None = None,
        source: str | None = None,
        since: float | None = None,
        order: str = "ASC",
    ) -> pd.DataFrame:
        """Query the log as a DataFrame."""
        clauses: list[str] = []
        args: list[Any] = []
        if scenario is not None:
            clauses.append("scenario IS ?")
            args.append(scenario)
        if source is not None:
            clauses.append("source = ?")
            args.append(source)
        if since is not None:
            clauses.append("ts >= ?")
            args.append(since)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        direction = "DESC" if order.upper() == "DESC" else "ASC"
        limit_sql = f"LIMIT {int(limit)}" if limit else ""
        sql = f"SELECT * FROM predictions {where} ORDER BY ts {direction}, id {direction} {limit_sql}"

        with self._connect() as conn:
            return pd.read_sql_query(sql, conn, params=args)

    def latest_window(self, n: int, scenario: str | None = None) -> pd.DataFrame:
        """The most recent *n* rows, returned oldest-first."""
        df = self.fetch(limit=n, scenario=scenario, order="DESC")
        return df.iloc[::-1].reset_index(drop=True)

    def scenarios(self) -> list[str]:
        """Distinct non-null scenario labels present in the log."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT DISTINCT scenario FROM predictions WHERE scenario IS NOT NULL "
                "ORDER BY scenario"
            )
            return [r[0] for r in cur.fetchall()]

    def summary(self) -> dict[str, Any]:
        """Headline counters for the ``/metrics`` endpoint."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n, AVG(probability) AS mean_prob, "
                "AVG(latency_ms) AS mean_latency, MIN(ts) AS first_ts, MAX(ts) AS last_ts, "
                "SUM(CASE WHEN decision = 'human_review' THEN 1 ELSE 0 END) AS n_review, "
                "SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END) AS n_defect, "
                "SUM(CASE WHEN ground_truth IS NOT NULL THEN 1 ELSE 0 END) AS n_labeled "
                "FROM predictions"
            ).fetchone()
        n = int(row["n"] or 0)
        return {
            "n_predictions": n,
            "n_predicted_defect": int(row["n_defect"] or 0),
            "n_human_review": int(row["n_review"] or 0),
            "n_labeled": int(row["n_labeled"] or 0),
            "review_rate": (row["n_review"] or 0) / n if n else 0.0,
            "predicted_defect_rate": (row["n_defect"] or 0) / n if n else 0.0,
            "mean_probability": float(row["mean_prob"]) if row["mean_prob"] is not None else None,
            "mean_latency_ms": float(row["mean_latency"]) if row["mean_latency"] is not None else None,
            "first_ts": row["first_ts"],
            "last_ts": row["last_ts"],
        }

    def clear(self, scenario: str | None = None) -> int:
        """Delete rows (all, or one scenario). Used to reset a demo run."""
        with self._connect() as conn:
            if scenario is None:
                cur = conn.execute("DELETE FROM predictions")
            else:
                cur = conn.execute("DELETE FROM predictions WHERE scenario IS ?", (scenario,))
            return cur.rowcount

    def export_jsonl(self, path: str | Path, limit: int | None = None) -> Path:
        """Dump the log as JSON lines -- the submission's 'monitoring log' artifact."""
        df = self.fetch(limit=limit)
        dest = ensure_parent(path)
        with open(dest, "w", encoding="utf-8") as fh:
            for record in df.to_dict("records"):
                fh.write(json.dumps(record, default=str) + "\n")
        log.info("Exported %d prediction log rows -> %s", len(df), dest)
        return dest


def get_store(params: dict[str, Any]) -> PredictionStore:
    """Open the configured prediction store."""
    return PredictionStore(get(params, "monitoring.db_path"))
