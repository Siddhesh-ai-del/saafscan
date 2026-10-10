"""Append-only SQLite evidence log.

Design: evidence must be tamper-evident, so the store exposes no update or
delete API — only insert and read. One row per analysed sample.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from saafscan.verdict import Verdict

DB_PATH = Path("data/evidence.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    sample_name TEXT,
    verdict_json TEXT NOT NULL,
    image BLOB
);
"""


@dataclass(frozen=True)
class Entry:
    id: str
    timestamp: str
    verdict: Verdict
    sample_name: str | None
    image_bytes: bytes | None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "sample_name": self.sample_name,
            "verdict": self.verdict.to_dict(),
        }


class EvidenceStore:
    """Insert-only evidence log backed by SQLite."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def record(
        self,
        verdict: Verdict,
        sample_name: str | None = None,
        image_bytes: bytes | None = None,
    ) -> str:
        """Append one evidence row. Returns its id."""
        entry_id = uuid.uuid4().hex[:12]
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO evidence (id, created_at, sample_name, verdict_json, image)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    entry_id,
                    datetime.now(UTC).isoformat(),
                    sample_name,
                    json.dumps(verdict.to_dict()),
                    image_bytes,
                ),
            )
        return entry_id

    def list(self, limit: int = 50) -> list[Entry]:
        """Newest-first list of evidence entries."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, created_at, sample_name, verdict_json, image"
                " FROM evidence ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (max(int(limit), 1),),
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def get(self, entry_id: str) -> Entry | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, created_at, sample_name, verdict_json, image"
                " FROM evidence WHERE id = ?",
                (entry_id,),
            ).fetchone()
        return self._row_to_entry(row) if row else None

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> Entry:
        data = json.loads(row["verdict_json"])
        return Entry(
            id=row["id"],
            timestamp=row["created_at"],
            verdict=Verdict(
                adulterant=data["adulterant"],
                confidence=data["confidence"],
                severity=data["severity"],
                advice=data["advice"],
                evidence=data.get("evidence", {}),
            ),
            sample_name=row["sample_name"],
            image_bytes=row["image"],
        )
