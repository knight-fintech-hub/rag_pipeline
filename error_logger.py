
from __future__ import annotations

from logger import get_logger

logger = get_logger(__name__)


def init_error_logs_table() -> None:
    """No-op stub — replace with real DDL when an error_logs table is needed."""
    logger.debug("[ErrorLogger] init_error_logs_table called (stub).")


def log_error(
    error_type: str,
    message: str,
    source_module: str = "",
    source_function: str = "",
) -> None:
    """Console-only stub — logs to stdout instead of the DB."""
    logger.error(
        f"[ErrorLogger] type={error_type} module={source_module} "
        f"fn={source_function} | {message}"
    )
