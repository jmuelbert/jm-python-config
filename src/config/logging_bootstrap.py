# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: © 2025-present Jürgen Mülbert

from __future__ import annotations

import logging
import sys

import structlog
from structlog.dev import ConsoleRenderer
from structlog.stdlib import (
    ProcessorFormatter,
)


def bootstrap_logging() -> None:
    """
    Configure a minimal structlog setup.

    This setup is intended for immediate use during application startup,
    before the full, potentially more complex, logging configuration is loaded.
    It logs to stderr in a human-readable format.
    """
    if structlog.is_configured():
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )

    pre_chain_processors = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,  # Converts exc_info to string if present
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,  # Filter by level early in the pipeline
            *pre_chain_processors,  # Common processors
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,  # Hands off to standard logging
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        cache_logger_on_first_use=True,
    )

    console_handler = logging.StreamHandler(sys.stderr)

    console_formatter = ProcessorFormatter(
        processor=ConsoleRenderer(colors=True),
        foreign_pre_chain=[
            *pre_chain_processors,
            structlog.stdlib.PositionalArgumentsFormatter(),  # For standard log messages with args
        ],
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)
