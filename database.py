"""
SQLite persistence layer.
"""

import sqlite3
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from config import DB_PATH

log = logging.getLogger(__name__)


@dataclass
class Transmission:
    id: int
    timestamp: str          # ISO-8601 UTC
    transcript: str
    duration_seconds: float
    confidence: float
    flagged: bool           # keyword match
    keywords_matched: str   # comma-separated


def _connect(path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: str = DB_PATH) -> sqlite3.Connection:
    conn = _connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transmissions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT    NOT NULL,
            transcript       TEXT    NOT NULL,
            duration_seconds REAL    NOT NULL,
            confidence       REAL    NOT NULL,
            flagged          INTEGER NOT NULL DEFAULT 0,
            keywords_matched TEXT    NOT NULL DEFAULT ''
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON transmissions(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_flagged   ON transmissions(flagged)")
    conn.commit()
    log.info("Database ready: %s", path)
    return conn


def insert_transmission(
    conn: sqlite3.Connection,
    transcript: str,
    duration_seconds: float,
    confidence: float,
    flagged: bool = False,
    keywords_matched: list[str] | None = None,
    timestamp: datetime | None = None,
) -> int:
    ts = (timestamp or datetime.now(timezone.utc)).isoformat()
    kw = ",".join(keywords_matched or [])
    cur = conn.execute(
        """
        INSERT INTO transmissions
            (timestamp, transcript, duration_seconds, confidence, flagged, keywords_matched)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (ts, transcript, duration_seconds, float(confidence), int(flagged), kw),
    )
    conn.commit()
    return cur.lastrowid


def export_csv(path: str = DB_PATH, out_path: str | None = None) -> str:
    """Export all transmissions to CSV. Returns the output file path."""
    out = out_path or Path(path).with_suffix(".csv").as_posix()
    conn = _connect(path)
    rows = conn.execute("SELECT * FROM transmissions ORDER BY timestamp").fetchall()
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "timestamp", "transcript", "duration_seconds",
                         "confidence", "flagged", "keywords_matched"])
        for row in rows:
            writer.writerow(list(row))
    conn.close()
    log.info("Exported %d rows to %s", len(rows), out)
    return out
