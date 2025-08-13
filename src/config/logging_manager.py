# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: © 2025-present Jürgen Mülbert

"""
The LoggingManager.

A comprehensive logging manager for cross-platform logging with structured logging,
file rotation, and JSON output, built on `structlog`.

The `LoggingManager` class encapsulates the setup and management of application
logging, integrating Python's standard logging system with `structlog` for
structured log records. It supports flexible output to both the console (with
color formatting) and rotating log files, and outputs logs in JSON format for
easy machine readability and analysis.

Features:
---------
- **Console Logging:** Colorized output for enhanced readability in development.
- **File Logging:** Automatic log file rotation based on size, ensuring logs don't consume excessive disk space.
- **Structured Logging:** Uses `structlog` to produce rich, context-aware log
  entries, ideal for querying and analysis in log management systems.
- **Configurable:** Settings are loaded from a TOML configuration file,
  allowing for granular control over log levels, formats, and destinations.
- **Context Manager:** Designed to be used as a context manager for proper
  resource cleanup (e.g., closing file handlers) at application exit.
- **Easy Integration:** Provides a straightforward API for logging messages
  throughout the application.

Configuration is handled via a TOML file (e.g., `config.toml`), which allows
specifying:
- The global log level (e.g., `INFO`, `DEBUG`, `ERROR`).
- Whether console logging is enabled and its specific format.
- Whether file logging is enabled, its location, and rotation parameters
  (max size, backup count).

Example Usage:
--------------
```python
from checkconnect.config.appcontext import AppContext
from checkconnect.config.translation_manager import TranslationManagerSingleton
from checkconnect.logging_manager import LoggingManagerSingleton
from pathlib import Path


# Assume AppContext and its settings are already initialized
# For a real example, settings would be loaded from a config file
class MockSettingsManager:
    def get_section(self, section_name: str) -> dict:
        if section_name == "logger":
            return {"level": "INFO", "log_directory": "logs"}
        if section_name == "console_handler":
            return {"enabled": True}
        if section_name == "file_handler":
            return {"enabled": True, "file_name": "app.log"}
        if section_name == "limited_file_handler":
            return {"enabled": True, "file_name": "limited.log", "max_bytes": 1048576, "backup_count": 3}
        return {}


class MockAppContext:
    settings = MockSettingsManager()
    translator = TranslationManagerSingleton.get_instance()  # Assuming this is set up


# Initialize the logging manager at the application's startup
app_context = MockAppContext()
LoggingManagerSingleton.initialize_from_context(app_context=app_context, enable_console_logging=True)

# Get the main application logger
logger = LoggingManagerSingleton.get_instance().get_logger(__app_name__)

# Log messages with structured data
logger.info("Application started successfully", version="1.0.0", environment="development")
logger.debug("Database connection details", host="localhost", port=5432)

try:
    result = 10 / 0
except ZeroDivisionError as e:
    logger.exception("An unhandled error occurred during division", error_type=str(type(e).__name__), details=str(e))

logger.warning("Configuration file not found, using default settings.")
logger.error("Failed to connect to external service.", service="AuthAPI", status_code=500)

# The context manager ensures proper shutdown for the LoggingManager instance
# (though for the singleton, it's typically managed globally)
# with LoggingManagerSingleton.get_instance() as logger_instance:
#     log = logger_instance.get_logger("my_module")
#     log.info("Inside context manager")
"""

from __future__ import annotations

import logging
import logging.config
import logging.handlers
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Self

import structlog
from rich.console import Console
from structlog.stdlib import ProcessorFormatter

from checkconnect.__about__ import __app_name__
from checkconnect.exceptions import (  # Assuming these custom exceptions exist
    InvalidLogLevelError,
    LogDirectoryError,
    LogHandlerError,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    from structlog.typing import Processor

    from checkconnect.config.appcontext import AppContext


# --- Global Logging Setup (Bootstrap) ---
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s: %(message)s")

# Console for critical errors before full logging is operational
_error_console = Console(file=sys.stderr)  # This also fits better outside the class

# --- Module-Level Constants ---
APP_NAME: Final[str] = __app_name__.lower()
DEFAULT_LOG_FILENAME: Final[str] = f"{APP_NAME}.log"
DEFAULT_LIMITED_LOG_FILENAME: Final[str] = f"limited_{APP_NAME}.log"
DEFAULT_MAX_BYTE: Final[int] = 1024 * 1024  # 1MB
DEFAULT_BACKUP_COUNT: Final[int] = 3

VERBOSITY_LEVELS: dict[int, int] = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}


# --- Logging Manager ---
class LoggingManager:
    """
    Manage the full configuration and lifecycle of the application's logging system.

    Integrates Python's standard logging with `structlog` for structured,
    context-rich logging to console and rotating files.
    """

    # Class-level attribute for the main logger instance, initialized after full config.
    _logger: structlog.stdlib.BoundLogger | None = None
    _internal_errors: list[str]
    effective_log_level: int

    # Type definition for the translation function
    _translate_func: Callable[[str], str]

    def __init__(self) -> None:
        """
        Initialize LoggingManager attributes in a lightweight manner.

        Full logging configuration is deferred until `apply_configuration()` is called.
        """
        self._internal_errors: list[str] = []
        self._logger = None  # Will be set after structlog is fully configured

        # Attributes that will be set by apply_configuration
        self.cli_log_level: int | None = None
        self.enable_console_logging: bool = False
        self.log_config: dict[str, Any] = {}
        self.effective_log_level = logging.NOTSET  # Default to lowest level initially
        self.translator: Any = None  # Placeholder for a translator instance
        self._translate_func = lambda x: x  # Default no-op translator

        # Set a temporary structlog logger for messages *during* the initial setup phase
        # This logger will output to the pre-configured basic logging.
        self._logger = structlog.get_logger("LoggingManagerInit")

    @property
    def internal_errors(self) -> list[str]:
        """Return the list of internal errors."""
        return self._internal_errors

    def apply_configuration(
        self,
        *,
        cli_log_level: int | None = None,
        enable_console_logging: bool,
        log_config: dict[str, Any],
        translator: Any,
    ) -> None:
        """
        Apply the comprehensive logging configuration to the manager instance.

        This method should be called once by the `LoggingManagerSingleton` to
        set up all logging handlers and `structlog` processors based on
        application settings and CLI overrides.

        Args:
            cli_log_level (int | None): The log level specified via CLI arguments.
                                        If provided, it overrides the config file
                                        level if more verbose (lower numerical value).
            enable_console_logging (bool): Flag to explicitly enable/disable console logging.
            log_config (dict[str, Any]): A dictionary containing all logging-related
                                         configuration sections from the application settings.
            translator (Any): An instance of a translation manager to translate
                              log messages where applicable.

        Raises:
            LogHandlerError: If any critical logging handler fails to initialize.
            InvalidLogLevelError: If a log level specified in config is invalid.
        """
        self._internal_errors.clear()  # Clear previous errors for a fresh attempt

        self.cli_log_level = cli_log_level

        self.enable_console_logging = enable_console_logging
        self.log_config = log_config
        self.translator = translator
        self._translate_func = self.translator.translate  # Convenient alias for translation

        self._logger.debug(self._translate_func("Applying full logging configuration..."))

        try:
            self._setup_logging_pipeline()
            self._logger.info(self._translate_func("Full logging configuration applied successfully."))
        except (InvalidLogLevelError, LogDirectoryError, LogHandlerError) as e:
            error_msg = self._translate_func("Critical error during logging configuration:")
            self._internal_errors.append(f"{error_msg} {e}")
            self._logger.exception(error_msg, exc_info=e)
            raise  # Re-raise for the singleton to catch as a critical error
        except Exception as e:
            error_msg = self._translate_func("An unexpected error occurred during logging configuration:")
            self._internal_errors.append(f"{error_msg} {e}")
            self._logger.exception(error_msg, exc_info=e)
            raise

    def shutdown(self) -> None:
        """
        Shut down all active logging handlers.

        Ensuring logs are flushed and resources
        (like file handles) are properly released.

        This method is crucial for clean application termination, especially
        in testing environments or long-running applications.
        """
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:  # Iterate over a copy to safely modify
            try:
                handler.close()
                root_logger.removeHandler(handler)
            except (OSError, ValueError, AttributeError, TypeError) as e:
                # Catch specific I/O-related errors, which are the most likely failure modes for .close()
                _error_console.print(
                    f"[bold red]Error[/bold red]: Failed to close log handler {handler.__class__.__name__} due to an I/O error: {e}"
                )

        if self._logger:
            self._logger.debug(self._translate_func("All logging handlers shut down."))

    def get_logger(self, name: str | None = None) -> structlog.stdlib.BoundLogger:
        """
        Retrieve a configured `structlog` logger instance.

        Args:
            name (str | None): The name of the logger to retrieve. If `None`,
                               the root `structlog` logger is returned (typically
                               bound to the application's main logger).

        Returns:
            BoundLogger: A `structlog` bound logger instance ready for use.
        """
        # Ensure the logger is configured before returning it.
        # This implicitly relies on apply_configuration being called first.
        if not self._logger:
            _error_console.print(
                "[bold red]Warning[/bold red]: Attempted to get logger before configuration. Returning basic logger."
            )
            return structlog.get_logger(name if name else APP_NAME)  # Fallback to a basic structlog logger

        return structlog.get_logger(name if name else APP_NAME)

    def info(self, msg: str, **kwargs: Any) -> None:
        """Log an info-level message with optional structured data."""
        self.get_logger().info(self._translate_func(msg), **kwargs)

    def debug(self, msg: str, **kwargs: Any) -> None:
        """Log a debug-level message with optional structured data."""
        self.get_logger().debug(self._translate_func(msg), **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        """Log a warning-level message with optional structured data."""
        self.get_logger().warning(self._translate_func(msg), **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        """Log an error-level message with optional structured data."""
        self.get_logger().error(self._translate_func(msg), **kwargs)

    def critical(self, msg: str, **kwargs: Any) -> None:
        """Log a critical-level message with optional structured data."""
        self.get_logger().critical(self._translate_func(msg), **kwargs)

    def exception(self, msg: str, **kwargs: Any) -> None:
        """
        Log an exception-level message.

        That automatically including current exception information.

        Args:
            msg (str): The primary message describing the exception.
            **kwargs: Additional structured data to include in the log record.
        """
        self.get_logger().exception(self._translate_func(msg), **kwargs)

    def get_instance_errors(self) -> list[str]:
        """
        Retrieve a list of non-critical errors.

        That list was encountered during the
        `LoggingManager`'s configuration.

        These errors indicate issues that did not prevent the logger from
        being initialized but might affect its full functionality (e.g.,
        a specific handler failing to set up).

        Returns:
            list[str]: A list of error messages.
        """
        return list(self._internal_errors)

    # --- Private Helper Methods for Configuration ---

    def _clear_existing_handlers(self, root_logger: logging.Logger) -> None:
        """Remove all existing handlers from the root logger."""
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        self._logger.debug(self._translate_func("Cleared existing root logger handlers."))

    def _get_effective_log_level(self, logger_main_settings: dict[str, Any]) -> int:
        """
        Determine the effective log level.

        Prioritizing CLI verbosity overconfiguration file settings.

        Args:
            logger_main_settings (dict[str, Any]): The 'logger' section of the config.

        Returns:
            int: The calculated effective log level (e.g., `logging.INFO`, `logging.DEBUG`).

        Raises:
            InvalidLogLevelError: If the log level specified in the config is not valid.
        """
        settings_level_str = logger_main_settings.get("level", "INFO").upper()
        effective_level = getattr(logging, settings_level_str, None)

        if not isinstance(effective_level, int):
            error_msg = self._translate_func(
                f"Invalid log level '{settings_level_str}' in config. Falling back to INFO."
            )
            self._internal_errors.append(error_msg)
            self._logger.warning(error_msg, level_from_config=settings_level_str)
            effective_level = logging.INFO
            # Consider raising InvalidLogLevelError if you want this to be a critical setup failure
            # raise InvalidLogLevelError(error_msg)

        if self.cli_log_level is not None:
            # A lower numerical value means higher verbosity. `min` correctly selects
            # the more verbose (lower number) level between CLI and config.
            original_effective_level = effective_level
            effective_level = min(effective_level, self.cli_log_level)
            self._logger.info(
                self._translate_func("CLI log level applied, potentially increasing verbosity."),
                cli_level=logging.getLevelName(self.cli_log_level),
                config_level=logging.getLevelName(original_effective_level),
                final_level=logging.getLevelName(effective_level),
            )
        else:
            self._logger.debug(self._translate_func("No CLI log level provided, using config/default."))

        self._logger.debug(
            self._translate_func("Final effective log level calculated."),
            final_level_name=logging.getLevelName(effective_level),
        )
        return effective_level

    def _get_structlog_processors_pre_chain(self) -> list[Processor]:
        """
        Define the common `structlog` processors.structlog.

        That run before log events are handed off to standard logging handlers.

        These processors enrich the log event dictionary before formatting.
        """
        return [
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,  # Converts exc_info to string if present
            structlog.processors.UnicodeDecoder(),
        ]

    def _setup_console_handler(
        self,
        root_logger: logging.Logger,
        pre_chain_processors: list[Processor],
        console_handler_settings: dict[str, Any],
        effective_log_level: int,
    ) -> None:
        """Set up the console log handler if enabled in configuration."""
        if self.enable_console_logging or console_handler_settings.get("enabled"):
            try:
                console_handler = logging.StreamHandler(sys.stdout)
                # ProcessorFormatter uses structlog's processors for formatting
                console_formatter = ProcessorFormatter(
                    processor=structlog.dev.ConsoleRenderer(colors=True),  # Formats for human readability in console
                    foreign_pre_chain=[
                        *pre_chain_processors,
                        structlog.stdlib.PositionalArgumentsFormatter(),  # For standard log messages with args
                    ],
                )
                console_handler.setFormatter(console_formatter)
                console_handler.setLevel(effective_log_level)
                root_logger.addHandler(console_handler)
                self._logger.debug(
                    self._translate_func("Console handler added."), level=logging.getLevelName(effective_log_level)
                )
            except Exception as e:
                msg = self._translate_func("Failed to set up console handler.")
                self._internal_errors.append(f"{msg} {e}")
                self._logger.exception(msg, exc_info=e)
                raise LogHandlerError(msg) from e

    def _setup_file_handler(
        self,
        root_logger: logging.Logger,
        pre_chain_processors: list[Processor],
        file_handler_settings: dict[str, Any],
        logger_main_settings: dict[str, Any],
        effective_log_level: int,
    ) -> None:
        """Set up the main file log handler if enabled."""
        if file_handler_settings.get("enabled"):
            log_dir_str = logger_main_settings.get("log_directory")
            if not log_dir_str:
                msg = self._translate_func("Log directory not specified in settings for file handler.")
                raise LogHandlerError(msg)

            try:
                log_dir = Path(log_dir_str)
                log_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

                file_name = file_handler_settings.get("file_name", DEFAULT_LOG_FILENAME)
                log_file_path = log_dir / file_name
                file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
                file_formatter = ProcessorFormatter(
                    processor=structlog.processors.JSONRenderer(),  # Renders log events as JSON
                    foreign_pre_chain=[
                        *pre_chain_processors,
                        structlog.stdlib.PositionalArgumentsFormatter(),  # Apply last for file
                    ],
                )
                file_handler.setFormatter(file_formatter)
                file_handler.setLevel(effective_log_level)
                root_logger.addHandler(file_handler)
                self._logger.debug(
                    self._translate_func("Main file handler added."),
                    path=str(log_file_path),
                    level=logging.getLevelName(effective_log_level),
                )
            except Exception as e:
                msg = self._translate_func("Failed to set up main file handler.")
                self._internal_errors.append(f"{msg} {e}")
                self._logger.exception(msg, exc_info=e)
                raise LogHandlerError(msg) from e

    def _setup_limited_file_handler(
        self,
        root_logger: logging.Logger,
        pre_chain_processors: list[Processor],
        limited_file_handler_settings: dict[str, Any],
        logger_main_settings: dict[str, Any],
        effective_log_level: int,
    ) -> None:
        """Set up the rotating file log handler for limited logs. if enabled."""
        if limited_file_handler_settings.get("enabled"):
            log_dir_str = logger_main_settings.get("log_directory")
            if not log_dir_str:
                msg = self._translate_func("Log directory not specified in settings for limited file handler.")
                raise LogHandlerError(msg)

            try:
                log_dir = Path(log_dir_str)
                log_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

                file_name = limited_file_handler_settings.get("file_name", DEFAULT_LIMITED_LOG_FILENAME)
                max_bytes = limited_file_handler_settings.get("max_bytes", DEFAULT_MAX_BYTE * 5)  # Default 5MB
                backup_count = limited_file_handler_settings.get("backup_count", DEFAULT_BACKUP_COUNT)

                limited_log_file_path = log_dir / file_name

                rotating_handler = logging.handlers.RotatingFileHandler(
                    limited_log_file_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
                )
                rotating_formatter = ProcessorFormatter(
                    processor=structlog.processors.JSONRenderer(),
                    foreign_pre_chain=[
                        *pre_chain_processors,
                        structlog.stdlib.PositionalArgumentsFormatter(),
                    ],
                )
                rotating_handler.setFormatter(rotating_formatter)
                rotating_handler.setLevel(effective_log_level)  # Often, limited logs are for errors/critical only
                root_logger.addHandler(rotating_handler)
                self._logger.debug(
                    self._translate_func("Limited file handler added."),
                    path=str(limited_log_file_path),
                    level=logging.getLevelName(effective_log_level),
                )
            except Exception as e:
                msg = self._translate_func("Failed to set up limited file handler.")
                self._internal_errors.append(f"{msg} {e}")
                self._logger.exception(msg, exc_info=e)
                raise LogHandlerError(msg) from e

    def _setup_logging_pipeline(self) -> None:
        """
        Configure the core `structlog` pipeline.

        Attaches standard logging handlers based on application settings.
        """
        root_logger = logging.getLogger()
        self._clear_existing_handlers(root_logger)

        if not self.log_config:
            self._internal_errors.append(self._translate_func("Logging configuration dictionary is empty."))
            self._logger.warning(self._translate_func("No logging configuration provided."))
            return  # Exit if no config to apply

        logger_main_settings = self.log_config.get("logger", {})
        console_handler_settings = self.log_config.get("console_handler", {})
        file_handler_settings = self.log_config.get("file_handler", {})
        limited_file_handler_settings = self.log_config.get("limited_file_handler", {})

        self.effective_log_level = self._get_effective_log_level(logger_main_settings)
        root_logger.setLevel(self.effective_log_level)
        self._logger.debug(
            self._translate_func("Root logger level set."),
            effective_level_name=logging.getLevelName(self.effective_log_level),
        )
        # Configure the core structlog behavior. This setup defines how
        # `structlog.get_logger()` instances will behave and how they hand off
        # events to the standard logging module.
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,  # Filter by level early in the pipeline
                *self._get_structlog_processors_pre_chain(),  # Common processors
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,  # Hands off to standard logging
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        self._logger.debug(self._translate_func("Structlog core configured."))

        # Set up individual handlers
        self._setup_console_handler(
            root_logger, self._get_structlog_processors_pre_chain(), console_handler_settings, self.effective_log_level
        )
        self._setup_file_handler(
            root_logger,
            self._get_structlog_processors_pre_chain(),
            file_handler_settings,
            logger_main_settings,
            self.effective_log_level,
        )
        self._setup_limited_file_handler(
            root_logger,
            self._get_structlog_processors_pre_chain(),
            limited_file_handler_settings,
            logger_main_settings,
            self.effective_log_level,
        )

        # Re-fetch the manager's internal logger to ensure it's using the newly configured structlog
        self._logger = structlog.get_logger("LoggingManager")
        self._logger.debug(self._translate_func("LoggingManager internal logger re-initialized with full config."))

    def __enter__(self) -> Self:
        """
        Enters the runtime context of the LoggingManager.

        While the singleton manages initialization, this can be used to ensure
        clean shutdown if the LoggingManager instance is used directly in a `with` statement.
        """
        # For a singleton, __enter__ might not do much beyond returning self
        # as setup is typically done via initialize_from_context.
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the runtime context, triggering a shutdown of logging resources."""
        self.shutdown()


class LoggingManagerSingleton:
    """
    Singleton class for `LoggingManager`.

    Ensures a single instance manages application logging and handles its
    controlled initialization and access.
    """

    _instance: LoggingManager | None = None
    _initialization_errors: ClassVar[list[str]] = []
    _is_configured: ClassVar[bool] = False  # Track if the instance has been configured

    @classmethod
    def get_instance(cls) -> LoggingManager:
        """
        Return the single, initialized instance of `LoggingManager`.

        Raises:
            RuntimeError: If `initialize_from_context` has not been called yet.
        """
        if cls._instance is None:
            msg = "LoggingManager has not been initialized. Call initialize_from_context first."
            raise RuntimeError(msg)
        return cls._instance

    @classmethod
    def initialize_from_context(
        cls, *, app_context: AppContext, cli_log_level: int | None = None, enable_console_logging: bool = True
    ) -> None:
        """
        Initialize the `LoggingManagerSingleton`.

        Using application context settings and optional CLI parameters.

        This method should be called once at the application's startup.
        Subsequent calls will be ignored if the manager is already configured.

        Args:
            app_context (AppContext): The application's context object,
                                      providing access to settings and translator.
            cli_log_level (int | None): An optional log level override from CLI,
                                        e.g., `logging.DEBUG` for `--verbose`.
            enable_console_logging (bool): Flag to explicitly enable/disable console output.

        Raises:
            LogHandlerError: If a critical logging handler cannot be set up.
            InvalidLogLevelError: If an invalid log level is provided in config.
            Exception: For any other unexpected errors during initialization.
        """
        if cls._is_configured:
            # Use the already configured logger if available, otherwise bootstrap
            current_logger = cls._instance.get_logger() if cls._instance else logging.getLogger(__name__)
            current_logger.warning(
                "Attempted to re-initialize LoggingManagerSingleton, but it's already configured. Ignoring."
            )
            cls._initialization_errors.append("LoggingManagerSingleton already configured. Cannot re-configure.")
            return

        # Create the lightweight instance if it doesn't exist yet
        if cls._instance is None:
            try:
                cls._instance = LoggingManager()  # Lightweight creation
            except Exception as e:
                cls._initialization_errors.append(f"Error creating LoggingManager instance: {e}")
                cls._instance = None
                raise  # Re-raise critical creation error

        cls._initialization_errors.clear()  # Clear errors for a fresh initialization attempt

        try:
            # Extract logging config specific sections from AppContext settings
            logging_config_for_manager = {
                "logger": app_context.settings.get_section("logger"),
                "console_handler": app_context.settings.get_section("console_handler"),
                "file_handler": app_context.settings.get_section("file_handler"),
                "limited_file_handler": app_context.settings.get_section("limited_file_handler"),
            }

            cls._instance.apply_configuration(
                cli_log_level=cli_log_level,
                enable_console_logging=enable_console_logging,
                log_config=logging_config_for_manager,
                translator=app_context.translator,
            )

            # Collect any non-critical setup errors from the instance
            cls._initialization_errors.extend(cls._instance.internal_errors)
            cls._is_configured = True  # Mark as configured only on success

        except (LogHandlerError, InvalidLogLevelError) as e:
            cls._initialization_errors.append(str(e))
            raise  # Re-raise known critical errors
        except Exception as e:
            cls._initialization_errors.append(f"Unexpected error during LoggingManager configuration: {e}")
            raise  # Re-raise any other unexpected errors

    @classmethod
    def get_initialization_errors(cls) -> list[str]:
        """
        Retrieve a consolidated list of errors.

        That list was encountered during the singleton's initialization and
        subsequent `LoggingManager` configuration.

        Returns:
            list[str]: A list of unique error messages.
        """
        errors = list(cls._initialization_errors)
        if cls._instance:
            errors.extend(cls._instance.internal_errors)
        return list(set(errors))  # Return unique errors

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance and its configuration.

        This method is primarily intended for testing purposes to ensure a
        clean state between test runs. It properly shuts down logging
        resources and clears any accumulated errors.
        """
        if cls._instance:
            cls._instance.shutdown()  # Call the instance's shutdown method
        cls._instance = None
        cls._initialization_errors.clear()
        cls._is_configured = False
