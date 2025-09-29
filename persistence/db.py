"""SQLite persistence layer for Sparky.

Provides schema management and typed helper functions to record interactions,
parsed ticket fields, and downstream Jira outcomes. This is the foundation for
future learning/analytics features.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Iterable, Optional

DB_FILENAME = "sparky.db"
DEFAULT_DB_DIR = Path("data")

logger = logging.getLogger(__name__)


@dataclass
class InteractionRecord:
    interaction_id: int
    timestamp: str
    user_id: Optional[str]
    channel: Optional[str]
    message_text: str
    intent: Optional[str]
    parser_success: bool
    parser_error: Optional[str]
    jira_issue_key: Optional[str]
    jira_error: Optional[str]
    extra: dict[str, Any]


class Database:
    """Thread-safe SQLite wrapper with minimal ORM-style helpers."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = self._resolve_db_path(db_path)
        self._connection_lock = threading.Lock()
        self._ensure_schema()

    def _resolve_db_path(self, db_path: Optional[Path]) -> Path:
        if db_path is not None:
            return db_path

        base_dir = DEFAULT_DB_DIR
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir / DB_FILENAME

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        with self._connection_lock:
            connection = sqlite3.connect(self.db_path)
            connection.row_factory = sqlite3.Row
            try:
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT,
                    channel TEXT,
                    message_text TEXT NOT NULL,
                    intent TEXT,
                    parser_success INTEGER NOT NULL,
                    parser_error TEXT,
                    jira_issue_key TEXT,
                    jira_error TEXT,
                    extra_json TEXT
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS parsed_fields (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_id INTEGER NOT NULL,
                    field_name TEXT NOT NULL,
                    field_value TEXT,
                    FOREIGN KEY(interaction_id) REFERENCES interactions(id)
                        ON DELETE CASCADE
                )
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_interactions_timestamp
                ON interactions(timestamp DESC)
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_parsed_fields_field_name
                ON parsed_fields(field_name)
                """
            )

    def log_interaction(
        self,
        *,
        user_id: Optional[str],
        channel: Optional[str],
        message_text: str,
        intent: Optional[str],
        parser_success: bool,
        parser_error: Optional[str],
        jira_issue_key: Optional[str],
        jira_error: Optional[str],
        parsed_fields: Optional[dict[str, Any]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> int:
        extra_json = json.dumps(extra or {})
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO interactions (
                    user_id, channel, message_text, intent, parser_success,
                    parser_error, jira_issue_key, jira_error, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    channel,
                    message_text,
                    intent,
                    1 if parser_success else 0,
                    parser_error,
                    jira_issue_key,
                    jira_error,
                    extra_json,
                ),
            )
            interaction_id = cursor.lastrowid or -1

            if parsed_fields:
                cursor.executemany(
                    """
                    INSERT INTO parsed_fields (
                        interaction_id, field_name, field_value
                    ) VALUES (?, ?, ?)
                    """,
                    [
                        (
                            interaction_id,
                            field,
                            _safe_stringify(value),
                        )
                        for field, value in parsed_fields.items()
                    ],
                )

        logger.debug(
            "db_interaction_logged",
            extra={"interaction_id": interaction_id, "intent": intent},
        )
        return int(interaction_id)

    def fetch_recent_interactions(self, *, limit: int = 50) -> Iterable[InteractionRecord]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, timestamp, user_id, channel, message_text, intent,
                       parser_success, parser_error, jira_issue_key, jira_error,
                       extra_json
                FROM interactions
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            for row in cursor.fetchall():
                yield InteractionRecord(
                    interaction_id=row["id"],
                    timestamp=row["timestamp"],
                    user_id=row["user_id"],
                    channel=row["channel"],
                    message_text=row["message_text"],
                    intent=row["intent"],
                    parser_success=bool(row["parser_success"]),
                    parser_error=row["parser_error"],
                    jira_issue_key=row["jira_issue_key"],
                    jira_error=row["jira_error"],
                    extra=json.loads(row["extra_json"] or "{}"),
                )


def _safe_stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, (list, dict)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    return str(value)
