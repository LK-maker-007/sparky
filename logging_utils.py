# logging_utils.py
"""Logging configuration helpers for Sparky."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, ClassVar

_DEFAULT_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter for structured logs."""

    RESERVED_KEYS: ClassVar[set[str]] = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
    }

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            data["stack"] = record.stack_info

        for key, value in record.__dict__.items():
            if key not in self.RESERVED_KEYS and not key.startswith("_"):
                data[key] = value

        return json.dumps(data, default=str)


def _resolve_log_level(level_name: str) -> int:
    if not level_name:
        return logging.INFO
    if isinstance(level_name, str):
        numeric = logging.getLevelName(level_name.upper())
        if isinstance(numeric, int):
            return numeric
    return logging.INFO


def configure_logging(level_name: str, json_enabled: bool) -> None:
    """Configure root logging handler according to settings."""

    level = _resolve_log_level(level_name)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    formatter = JsonFormatter() if json_enabled else logging.Formatter(_DEFAULT_FORMAT)
    handler.setFormatter(formatter)
    root.addHandler(handler)
