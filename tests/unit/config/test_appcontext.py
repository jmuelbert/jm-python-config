# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: © 2025-present Jürgen Mülbert

"""
Pytest suite for the AppContext class and its initialization.

This module contains unit tests for the `AppContext` class and the
`initialize_app_context` function. It ensures their correct functionality,
initialization of dependencies, resource access, and proper handling of
various input scenarios, including default values and exception propagation.
Pytest-mock is used to isolate units under test by mocking external
dependencies.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
import structlog

# Assuming your AppContext is in 'checkconnect.config.appcontext'
from checkconnect.config.appcontext import AppContext
from checkconnect.config.settings_manager import SettingsManager
from checkconnect.config.translation_manager import TranslationManager

# --- Fixtures for Mocking Dependencies ---


@pytest.fixture
def mock_settings_manager() -> MagicMock:
    """Mocks the SettingsManager."""
    # Use string for spec if you're only patching the class in some tests,
    # or ensure the actual class is imported if you're creating real mocks.
    return MagicMock(spec=SettingsManager)


@pytest.fixture
def mocked_translation() -> MagicMock:
    """Mocks the Translator."""
    mock_translator = MagicMock(spec=TranslationManager)
    mock_translator.gettext.side_effect = lambda text: f"[mocked] {text}"
    mock_translator.translate.side_effect = lambda text: f"[mocked] {text}"

    return mock_translator


# --- Tests for AppContext ---


class TestAppContext:
    @pytest.mark.unit
    def test_app_context_initialization(self, mock_settings_manager: MagicMock, mocked_translation: MagicMock) -> None:
        """
        Tests that AppContext correctly stores the provided manager instances.
        """
        app_context = AppContext(settings=mock_settings_manager, translator=mocked_translation)

        assert app_context.settings is mock_settings_manager
        assert app_context.translator is mocked_translation

    @pytest.mark.unit
    @patch("structlog.get_logger")
    def test_get_module_logger_functionality(
        self,
        mock_structlog_get_logger: MagicMock,
        mock_settings_manager: MagicMock,
        mocked_translation: MagicMock,
    ) -> None:
        """
        Tests that get_module_logger returns a functional structlog BoundLogger
        and that it's correctly named.
        """
        module_name = "my.test.module.logger"

        # Get a standard Python logger that BoundLogger will wrap
        base_python_logger = logging.getLogger(module_name)

        # Ensure its level is low enough to capture messages
        base_python_logger.setLevel(logging.DEBUG)

        # --- FINAL CRITICAL FIX ---
        # Ensure no NullHandler is consuming logs directly on this logger.
        # Messages should propagate up to the root logger where `capture_logs` is active.
        for handler in base_python_logger.handlers[:]:
            base_python_logger.removeHandler(handler)
        # --- END FINAL CRITICAL FIX ---

        # Define processors for the manually instantiated BoundLogger
        # These MUST match the processors from your conftest.py's structlog_base_config
        # for `capture_logs` to correctly interpret the log entries.
        processors_for_bound_logger = [
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            structlog.processors.TimeStamper(fmt="iso", utc=True),  # Ensure this matches your conftest.py
            structlog.processors.StackInfoRenderer(),  # If you have these in conftest
            structlog.processors.format_exc_info,  # If you have these in conftest
            structlog.processors.UnicodeDecoder(),  # If you have these in conftest
            # Add ALL other processors from your conftest.py structlog_base_config here!
        ]

        real_bound_logger = structlog.stdlib.BoundLogger(
            base_python_logger, processors=processors_for_bound_logger, context={}
        )

        # Set the return value of the mocked structlog.get_logger
        mock_structlog_get_logger.return_value = real_bound_logger

        # from checkconnect.config.logging_bootstrap import bootstrap_logging
        # bootstrap_logging()

        app_context = AppContext(settings=mock_settings_manager, translator=mocked_translation)
        # module_name = __name__
        logger = app_context.get_module_logger(module_name)

        mock_structlog_get_logger.assert_called_once_with(module_name)
        assert isinstance(logger, structlog.stdlib.BoundLogger)

    @pytest.mark.unit
    def test_gettext(
        self,
        mock_settings_manager: MagicMock,
        mocked_translation: MagicMock,
    ) -> None:
        """
        Tests that gettext delegates to the translator and returns the translated message.
        """
        app_context = AppContext.create(settings_instance=mock_settings_manager, translator_instance=mocked_translation)

        message = "Hello, world!"
        translated_message = app_context.gettext(message)

        mocked_translation.gettext.assert_called_once_with(message)
        assert translated_message == f"[mocked] {message}"

    class TestCreateMethod:
        """
        Tests for the AppContext.create factory method.
        """

        @pytest.mark.unit
        @patch("checkconnect.config.logging_manager.LoggingManager")
        @patch("checkconnect.config.appcontext.TranslationManager")
        @patch("checkconnect.config.appcontext.SettingsManager")
        def test_create_does_not_instantiate_logging_manager(
            self,
            mock_settings_manager: MagicMock,  # Corresponds to the bottom-most @patch
            mock_translation_manager: MagicMock,  # Corresponds to the middle @patch
            mock_logging_manager: MagicMock,  # Corresponds to the top-most @patch
        ) -> None:
            """
            Tests that AppContext.create does NOT instantiate LoggingManager directly,
            as its configuration is assumed to be handled externally.
            """
            # This test runs in isolation, so the AppContext's code is paramount.
            # The configure_structlog_for_tests fixture for this class is auto-applied.
            AppContext.create(
                settings_instance=mock_settings_manager.return_value,
                translator_instance=mock_translation_manager.return_value,
            )

            mock_logging_manager.assert_not_called()  # The key assertion
            # Also verify that other managers are still created/used correctly
            mock_settings_manager.assert_not_called()
            mock_translation_manager.assert_not_called()
