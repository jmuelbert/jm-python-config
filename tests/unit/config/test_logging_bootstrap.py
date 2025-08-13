# SPDX-License-Identifier: EUPL-1.2
#
# SPDX-FileCopyrightText: © 2025-present Jürgen Mülbert

from __future__ import annotations

import inspect
import logging
from datetime import UTC, datetime

import pytest
import structlog
from structlog.stdlib import LoggerFactory

# Import your actual bootstrap_logging function (adjust import path if needed)
from checkconnect.config.logging_bootstrap import bootstrap_logging


class TestBootstrapLogging:
    @pytest.fixture(autouse=True)
    def reset_structlog_and_logging(self):
        """
        Automatically resets structlog and standard logging before and after each test,
        ensuring a clean environment and preventing side effects between tests.
        """
        structlog.reset_defaults()

        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        root_logger.setLevel(logging.NOTSET)

        yield

        structlog.reset_defaults()
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        root_logger.setLevel(logging.NOTSET)

    @pytest.fixture
    def mock_structlog_configure(self, mocker):
        """
        Provides a mocked version of structlog.configure to verify it is called
        and inspect its arguments without performing actual configuration.
        """
        return mocker.patch("structlog.configure")

    def test_bootstrap_logging_configures_structlog_once(self):
        """
        Ensures structlog is configured only once even if bootstrap_logging()
        is called multiple times.
        """
        assert not structlog.is_configured()

        bootstrap_logging()
        assert structlog.is_configured()

        bootstrap_logging()
        assert structlog.is_configured()

    def test_bootstrap_logging_sets_expected_processors(self, mock_structlog_configure):
        """
        Verifies that bootstrap_logging configures structlog with the expected processors
        in the correct order, including compatibility with older and newer versions.
        """
        bootstrap_logging()

        assert mock_structlog_configure.called
        _, kwargs = mock_structlog_configure.call_args

        processors = kwargs.get("processors")
        assert processors is not None

        # Map processor objects to names (function or class names)
        proc_names = [
            p.__name__
            if (inspect.isfunction(p) or inspect.isbuiltin(p) or inspect.isclass(p))
            else p.__class__.__name__
            for p in processors
        ]

        expected_order = [
            "filter_by_level",
            "add_logger_name",
            "add_log_level",
            "TimeStamper",
            "StackInfoRenderer",
            ("format_exc_info", "ExceptionRenderer"),  # accept both names
            "UnicodeDecoder",
            "wrap_for_formatter",
        ]

        idx = 0
        for expected in expected_order:
            if isinstance(expected, tuple):
                while idx < len(proc_names) and proc_names[idx] not in expected:
                    idx += 1
                assert idx < len(proc_names), f"Processor {expected} not found in processor list"
            else:
                while idx < len(proc_names) and proc_names[idx] != expected:
                    idx += 1
                assert idx < len(proc_names), f"Processor {expected} not found in processor list"
            idx += 1

        # Validate other structlog configuration options
        assert isinstance(kwargs.get("logger_factory"), LoggerFactory)
        wrapper_cls = kwargs.get("wrapper_class")
        assert wrapper_cls is not None
        assert "boundloggerfiltering" in wrapper_cls.__name__.lower()
        assert kwargs.get("context_class") is dict
        assert kwargs.get("cache_logger_on_first_use") is True

    def test_bootstrap_logging_configures_standard_logger_handlers(self, mocker):
        """
        Verifies that the root logger receives a StreamHandler with the ProcessorFormatter.
        """
        mock_stderr = mocker.patch("sys.stderr")
        mock_root_logger = mocker.MagicMock(spec=logging.Logger)
        mock_root_logger.handlers = []

        mock_get_logger = mocker.patch("logging.getLogger", return_value=mock_root_logger)
        mock_stream_handler = mocker.patch("logging.StreamHandler")
        mock_processor_formatter = mocker.patch("checkconnect.config.logging_bootstrap.ProcessorFormatter")

        bootstrap_logging()

        mock_get_logger.assert_called_once_with()
        mock_stream_handler.assert_called_once_with(mock_stderr)
        mock_processor_formatter.assert_called_once()

        handler_instance = mock_stream_handler.return_value
        formatter_instance = mock_processor_formatter.return_value

        handler_instance.setFormatter.assert_called_once_with(formatter_instance)
        mock_root_logger.addHandler.assert_called_once_with(handler_instance)
        mock_root_logger.setLevel.assert_called_once_with(logging.INFO)

    def test_bootstrap_logging_output_content(self, capsys):
        """
        Integration test: Verifies the output format, logger name, log level,
        timestamp, and exception rendering from structlog.
        """
        bootstrap_logging()
        log = structlog.get_logger("my_app.startup")
        log.info("Application starting up", version="1.0.0")

        captured = capsys.readouterr()
        assert "Application starting up" in captured.err
        assert "version=1.0.0" in captured.err or "version" in captured.err
        assert "info" in captured.err.lower()
        assert "my_app.startup" in captured.err
        assert str(datetime.now(UTC).date()) in captured.err

        # Test exception logging
        try:
            _ = 1 / 0
        except ZeroDivisionError as e:
            log.exception("Failed to divide!", exc_info=e)

        captured = capsys.readouterr()
        assert "Failed to divide!" in captured.err
        assert "ZeroDivisionError" in captured.err
        assert "traceback" in captured.err.lower() or "Traceback (most recent call last):" in captured.err

    def test_standard_logging_integration(self, capsys):
        """
        Verifies that standard logging calls (non-structlog) are properly formatted
        and filtered according to the configured logging level.
        """
        bootstrap_logging()
        std_logger = logging.getLogger("my_app.std_lib")

        std_logger.info("Standard log message.")
        std_logger.debug("This debug message should NOT appear.")  # Should be filtered out

        captured = capsys.readouterr()
        assert "Standard log message." in captured.err
        assert "info" in captured.err.lower()
        assert "my_app.std_lib" in captured.err
        assert str(datetime.now(UTC).date()) in captured.err
        assert "This debug message should NOT appear." not in captured.err
