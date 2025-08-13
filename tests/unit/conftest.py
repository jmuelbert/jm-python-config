# conftest.py
# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: © 2025-present Jürgen Mülbert

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
import structlog
import tomli_w
from PySide6.QtWidgets import QApplication
from typer.testing import CliRunner

# Import your application's singletons and core classes
# Example for __about__.py (e.g., if it's in checkconnect/__about__.py)
import checkconnect.__about__ as about_module

# Example for cli.main (e.g., if it's in checkconnect/cli/main.py)
import checkconnect.cli.main as cli_main_module

# Example for cli.options (e.g., if it's in checkconnect/cli/options.py)
import checkconnect.cli.options as cli_options_module

# Example for cli.run_app (e.g., if it's in checkconnect/cli/run_app.py)
from checkconnect.config.appcontext import AppContext
from checkconnect.config.logging_manager import LoggingManager, LoggingManagerSingleton
from checkconnect.config.settings_manager import SettingsManager, SettingsManagerSingleton
from checkconnect.config.translation_manager import TranslationManager, TranslationManagerSingleton

# Assuming these exist in your project, if not, adjust paths or remove
from checkconnect.core.checkconnect import CheckConnect  # For CheckConnect mocking

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator
    from typing import Literal

    from pytest_mock import MockerFixture
    from structlog.typing import EventDict


# --- Core Logging Setup Fixture ---
# This MUST run before any of your application code gets its first logger.
# It ensures `structlog.get_logger()` returns a properly configured BoundLogger.
@pytest.fixture(autouse=True)
def structlog_base_config() -> Generator[None, None, None]:
    """
    Fixture to set up and tear down a robust structlog configuration for each test function.
    Ensures that structlog.get_logger() returns a concrete BoundLogger, not a LazyProxy.
    """
    # 1. Reset structlog and standard logging to a clean slate
    structlog.reset_defaults()
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    root_logger.setLevel(logging.NOTSET)

    # 2. Add a basic StreamHandler to the root logger for structlog.stdlib to use
    test_handler = logging.StreamHandler(sys.stdout)  # You can change this to sys.stderr or a NullHandler
    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    test_handler.setFormatter(formatter)
    root_logger.addHandler(test_handler)
    root_logger.setLevel(logging.DEBUG)  # Set a low level so all messages are processed

    # 3. Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,  # IMPORTANT: Ensures concrete BoundLogger immediately
    )
    yield

    # --- Teardown Phase ---
    structlog.reset_defaults()
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    root_logger.setLevel(logging.NOTSET)
    logging.disable(logging.NOTSET)


# --- Logging Assertion Fixtures ---
@pytest.fixture
def caplog_structlog() -> list[EventDict]:
    """
    Captures `structlog` events for the duration of a test.
    Requires `structlog` to be configured beforehand (e.g., by `structlog_base_config`).
    """
    with structlog.testing.capture_logs() as captured_events:
        yield captured_events


@pytest.fixture
def assert_log_contains() -> Any:
    """
    Provides a helper function to assert that a `structlog` capture contains specific log entries.
    """

    def _assert(log: list[EventDict], text: str, level: str | None = None) -> None:
        matches = [
            entry
            for entry in log
            if text in entry["event"] and (level is None or entry["level"].lower() == level.lower())
        ]
        assert matches, f"No log entry found with text '{text}' and level '{level}'"

    return _assert


# --- 1. Environment Isolation Fixture ---
@pytest.fixture
def isolated_test_env(mocker: MockerFixture, tmp_path: Path) -> dict[str, Path]:
    """
    Sets up a completely isolated temporary environment for tests:
    - Mocks platformdirs.user_config_dir and site_config_dir to a sub-directory within tmp_path.
    - Changes the current working directory to a unique sub-directory within tmp_path,
      ensuring no existing 'config.toml' interferes with the local lookup.
    """
    original_cwd = Path.cwd()
    try:
        # Create a unique temporary directory for the CWD of this test
        test_cwd = tmp_path / "test_run_cwd"
        test_cwd.mkdir()
        os.chdir(test_cwd)

        # Mock the platformdirs to point to a controlled temp location
        mock_user_config_path = tmp_path / "mock_user_config" / "checkconnect"
        mock_user_config_path.mkdir(parents=True, exist_ok=True)

        mocker.patch(
            "platformdirs.user_config_dir",
            return_value=str(mock_user_config_path),
        )
        mocker.patch(
            "platformdirs.site_config_dir",
            return_value=str(tmp_path / "mock_site_config" / "checkconnect"),
        )

        yield {"user_config_dir": mock_user_config_path, "current_working_dir": test_cwd}
    finally:
        os.chdir(original_cwd)


# --- 2. Global Dependency Mocks Fixture ---
@pytest.fixture
def mock_dependencies(  # noqa: PLR0915
    mocker: MockerFixture,
    isolated_test_env: dict[str, Path],  # This fixture creates temporary directories for config/data/reports
) -> dict[str, Any]:
    """
    Mocks external dependencies to isolate the CLI logic during tests.
    """

    # --- 0. CRUCIAL: RESET SINGLETON INTERNAL STATE ---
    # Singletons retain state across tests. This ensures each test starts with a clean slate.
    # Adjust attribute names if they are different in your actual singleton classes.
    SettingsManagerSingleton._instance = None  # noqa: SLF001
    SettingsManagerSingleton._initialization_errors = []  # noqa: SLF001
    # If cli_main_module has its own aliased reference to the singleton, reset that too.
    try:
        cli_main_module.SettingsManagerSingleton._instance = None  # noqa: SLF001
        cli_main_module.SettingsManagerSingleton._initialization_errors = []  # noqa: SLF001
    except AttributeError:
        # Catch if cli_main_module doesn't directly expose these attributes
        pass

    LoggingManagerSingleton._instance = None  # noqa: SLF001
    LoggingManagerSingleton._initialization_errors = []  # noqa: SLF001
    try:
        cli_main_module.LoggingManagerSingleton._instance = None  # noqa: SLF001
        cli_main_module.LoggingManagerSingleton._initialization_errors = []  # noqa: SLF001
    except AttributeError:
        pass

    TranslationManagerSingleton._instance = None  # noqa: SLF001
    TranslationManagerSingleton._initialization_errors = []  # noqa: SLF001
    try:
        cli_main_module.TranslationManagerSingleton._instance = None  # noqa: SLF001
        cli_main_module.TranslationManagerSingleton._initialization_errors = []  # noqa: SLF001
    except AttributeError:
        pass

    # 1. Mock __about__ module
    mocker.patch.object(about_module, "__app_name__", "checkconnect")
    mocker.patch.object(about_module, "__app_org_id__", "MyAwesomeOrg")
    mocker.patch.object(about_module, "__version__", "0.1.0")

    # 2. Mock Typer option definition functions (no changes here)
    def mock_get_option_definition() -> Any:
        import typer

        return typer.Option(None)

    mocker.patch.object(cli_options_module, "get_config_option_definition", return_value=mock_get_option_definition())
    mocker.patch.object(cli_options_module, "get_language_option_definition", return_value=mock_get_option_definition())
    mocker.patch.object(cli_options_module, "get_verbose_option_definition", return_value=mock_get_option_definition())
    mocker.patch.object(
        cli_options_module, "get_report_dir_option_definition", return_value=mock_get_option_definition()
    )
    mocker.patch.object(cli_options_module, "get_data_dir_option_definition", return_value=mock_get_option_definition())

    # 3. Mock LoggingManagerSingleton
    mock_logging_manager_instance = MagicMock(
        spec=LoggingManager
    )  # Using logging.Logger as spec as it's the underlying
    # implementation for structlog's BoundLoggerLazyProxy
    # Ensure apply_configuration is callable on the mock
    mock_logging_manager_instance.apply_configuration.return_value = (
        None  # Or another mock if apply_configuration returns something
    )
    mock_logging_manager_instance.info.return_value = None
    mock_logging_manager_instance.debug.return_value = None
    mock_logging_manager_instance.warning.return_value = None
    mock_logging_manager_instance.error.return_value = None
    mock_logging_manager_instance.critical.return_value = None
    mock_logging_manager_instance.exception.return_value = None

    # Side effect for initialize_from_context, to simulate real behavior on mock
    def mock_logging_initialize_from_context_side_effect(
        *, app_context: AppContext, cli_log_level: int, enable_console_logging: bool
    ):
        # Simulate the real initialize_from_context calling apply_configuration
        mock_logging_manager_instance.apply_configuration(
            cli_log_level=cli_log_level,
            enable_console_logging=enable_console_logging,
            log_config=app_context.settings.get_section("logger"),
            translator=app_context.translator,
        )
        return mock_logging_manager_instance

    mocker.patch.object(
        LoggingManagerSingleton, "initialize_from_context", side_effect=mock_logging_initialize_from_context_side_effect
    )
    mocker.patch.object(
        cli_main_module.LoggingManagerSingleton,
        "initialize_from_context",
        side_effect=mock_logging_initialize_from_context_side_effect,
    )
    mocker.patch.object(LoggingManagerSingleton, "get_instance", return_value=mock_logging_manager_instance)
    mocker.patch.object(
        cli_main_module.LoggingManagerSingleton, "get_instance", return_value=mock_logging_manager_instance
    )
    mocker.patch.object(LoggingManagerSingleton, "get_initialization_errors", return_value=[])
    mocker.patch.object(cli_main_module.LoggingManagerSingleton, "get_initialization_errors", return_value=[])

    # 4. Mock SettingsManagerSingleton
    mock_settings_instance = MagicMock(spec=SettingsManager)
    settings_data = {
        "logger": {"level": "DEBUG"},
        "general": {"default_language": "en"},
        "data": {"directory": str(isolated_test_env["user_config_dir"] / "data_dir")},
        "network": {"timeout": 5, "ntp_servers": ["pool.ntp.org"], "urls": ["https://www.example.com"]},
        "reports": {"directory": str(isolated_test_env["user_config_dir"] / "reports_dir")},
    }
    mock_settings_instance.get_all_settings.return_value = settings_data
    mock_settings_instance.get_section.side_effect = (
        lambda section_name: mock_settings_instance.get_all_settings.return_value.get(section_name, {})
    )
    mock_settings_instance.get.side_effect = (
        lambda section, key, default=None: mock_settings_instance.get_all_settings.return_value.get(section, {}).get(
            key, default
        )
    )

    # --- CRITICAL FIX: Mock load_settings directly on the mock instance ---
    # This prevents the real load_settings logic from running (which involves file I/O)
    # and explicitly sets the loaded_config_file on the mock.
    def mock_settings_instance_load_settings_side_effect(config_path_from_cli: Path | None = None) -> None:
        if config_path_from_cli:
            mock_settings_instance.loaded_config_file = config_path_from_cli
        else:
            mock_settings_instance.loaded_config_file = None  # Or a suitable default for no config file
        # Ensure _settings is populated as if it were loaded
        mock_settings_instance._settings = settings_data.copy()  # noqa: SLF001

    mock_settings_instance.load_settings.side_effect = mock_settings_instance_load_settings_side_effect
    # mock_settings_instance._save_default_config.return_value = None # No longer strictly needed if load_settings is fully mocked

    # Side effect for initialize_from_context, which calls load_settings on the instance
    def mock_settings_singleton_initialize_from_context_side_effect(
        *,
        config_path: Path | None = None,
    ):
        # This calls the *mocked* load_settings on mock_settings_instance
        mock_settings_instance.load_settings(config_path_from_cli=config_path)
        return mock_settings_instance

    mocker.patch.object(
        SettingsManagerSingleton,
        "initialize_from_context",
        side_effect=mock_settings_singleton_initialize_from_context_side_effect,
    )
    mocker.patch.object(
        cli_main_module.SettingsManagerSingleton,
        "initialize_from_context",
        side_effect=mock_settings_singleton_initialize_from_context_side_effect,
    )

    mocker.patch.object(SettingsManagerSingleton, "get_instance", return_value=mock_settings_instance)
    mocker.patch.object(cli_main_module.SettingsManagerSingleton, "get_instance", return_value=mock_settings_instance)
    mocker.patch.object(SettingsManagerSingleton, "get_initialization_errors", return_value=[])

    # 5. Mock TranslationManagerSingleton
    mock_translator_instance = MagicMock(spec=TranslationManager)
    mock_translator_instance.gettext.side_effect = lambda x: x
    mock_translator_instance.translate.side_effect = lambda x: x
    mock_translator_instance.current_language.return_value = "en"

    # Side effect for configure_instance for TranslationManagerSingleton
    def mock_translation_configure_instance_side_effect(
        *, language: str | None = None, translation_domain: str | None = None, locale_dir: Path | None = None
    ):
        mock_translator_instance.configure(
            language=language, translation_domain=translation_domain, locale_dir=locale_dir
        )
        return mock_translator_instance

    mocker.patch.object(
        TranslationManagerSingleton, "configure_instance", side_effect=mock_translation_configure_instance_side_effect
    )
    mocker.patch.object(
        cli_main_module.TranslationManagerSingleton,
        "configure_instance",
        side_effect=mock_translation_configure_instance_side_effect,
    )
    mocker.patch.object(TranslationManagerSingleton, "get_instance", return_value=mock_translator_instance)
    mocker.patch.object(
        cli_main_module.TranslationManagerSingleton, "get_instance", return_value=mock_translator_instance
    )
    mocker.patch.object(TranslationManagerSingleton, "get_initialization_errors", return_value=[])

    # 6. Mock AppContext class (Static method 'create')
    mock_app_context_instance = MagicMock(spec=AppContext)
    mock_app_context_instance.settings = mock_settings_instance
    mock_app_context_instance.translator = mock_translator_instance
    mock_app_context_instance.gettext.side_effect = mock_translator_instance.gettext
    # Mock get_module_logger to return a simple mock structlog logger
    mock_app_context_instance.get_module_logger.side_effect = lambda name: structlog.get_logger(name)

    # Patch the `create` *class method* of AppContext.
    mocker.patch.object(AppContext, "create", return_value=mock_app_context_instance)
    # mocker.patch(f"{cli_main_module.__name__}.AppContext.create", return_value=mock_app_context_instance) # If needed

    # 7. Mock CheckConnect core logic
    mock_check_connect_instance = MagicMock(spec=CheckConnect)
    mock_check_connect_instance.run_all_checks.return_value = None
    # If cli_run_app_module directly imports and uses CheckConnect (e.g., in a 'run' command)
    # Ensure this path is correct if CheckConnect is imported elsewhere.
    # import checkconnect.cli.run_app as cli_run_app_module # Example import
    # mocker.patch(f"{cli_run_app_module.__name__}.CheckConnect", return_value=mock_check_connect_instance)

    return {
        "logging_manager_instance": mock_logging_manager_instance,
        "settings_manager_instance": mock_settings_instance,
        "translation_manager_instance": mock_translator_instance,
        "app_context_instance": mock_app_context_instance,
        "check_connect_instance": mock_check_connect_instance,
    }


@pytest.fixture
def mock_importlib_files(mocker: Any, tmp_path: Path) -> Path:
    """
    Fixture to mock importlib.resources.files to return a Path object.
    """
    mock_resource_path = tmp_path / "resources"
    mock_resource_path.mkdir()
    mocker.patch("importlib.resources.files", return_value=mock_resource_path)
    return mock_resource_path


@pytest.fixture
def mock_platformdirs_paths(mocker: Any, tmp_path: Path) -> dict[str, Path]:
    """
    Fixture to mock platformdirs to return paths within a temporary directory.

    Returns:
        dict[str, Path]: A dictionary containing the mocked user and site config paths.
    """
    mock_user_config_dir = tmp_path / "user_config"
    mock_site_config_dir = tmp_path / "site_config"
    mock_user_data_dir = tmp_path / "user_data"
    mock_site_data_dir = tmp_path / "site_data"

    mocker.patch("platformdirs.user_config_dir", return_value=str(mock_user_config_dir))
    mocker.patch("platformdirs.site_config_dir", return_value=str(mock_site_config_dir))
    mocker.patch("platformdirs.user_data_dir", return_value=str(mock_user_data_dir))
    mocker.patch("platformdirs.site_data_dir", return_value=str(mock_site_data_dir))

    # Ensure these directories exist for tests that expect them to be writable
    mock_user_config_dir.mkdir(parents=True, exist_ok=True)
    mock_site_config_dir.mkdir(parents=True, exist_ok=True)
    mock_user_data_dir.mkdir(parents=True, exist_ok=True)
    mock_site_data_dir.mkdir(parents=True, exist_ok=True)

    return {
        "user_config": mock_user_config_dir,
        "site_config": mock_site_config_dir,
        "user_data": mock_user_data_dir,
        "site_data": mock_site_data_dir,
    }


@pytest.fixture
def setup_default_config_locations(
    tmp_path: Path,
    mock_platformdirs_paths: dict[str, Path],
    mock_importlib_files: Path,
) -> None:
    """
    Set up the CONFIG_LOCATIONS for each test to use temporary paths.
    """
    # Dynamically set CONFIG_LOCATIONS to use tmp_path for predictable behavior
    SettingsManager.DEFAULT_SETTINGS_LOCATIONS = [
        tmp_path / "conf.toml",  # Relative path, will be in current working dir of test
        mock_platformdirs_paths["user_config"] / SettingsManager.CONF_NAME,
        mock_platformdirs_paths["site_config"] / SettingsManager.CONF_NAME,
        mock_importlib_files / SettingsManager.CONF_NAME,
    ]
    # Ensure the default config file exists in the mocked resource path for some tests
    # (mock_importlib_files / SettingsManager.CONF_NAME).write_text("dummy_content = 'default'")
    # Mock Path.exists for the initial "config.toml" in current dir
    # mocker.patch.object(Path(SettingsManager.CONFIG_LOCATIONS[0]), "exists", return_value=False)


@pytest.fixture
def setup_default_config(
    tmp_path: Path,
):
    SettingsManager.DEFAULT_CONFIG = {
        "logger": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "log_directory": str(tmp_path),
            "output_format": "console",
        },
        "console_handler": {"enabled": True},
        "gui": {
            "enabled": True,
        },
        "reports": {"directory": str(tmp_path)},
        "data": {"directory": str(tmp_path)},
        "network": {
            "timeout": 5,
            "ntp_servers": ["pool.ntp.org"],
            "urls": ["https://example.com"],
        },
    }


@pytest.fixture(autouse=True)
def cleanup_singletons() -> Generator[None, None, None]:
    """
    Resets all application singletons to ensure clean state between tests.
    """
    LoggingManagerSingleton._instance = None  # noqa: SLF001
    LoggingManagerSingleton._initialization_errors.clear()  # noqa: SLF001

    SettingsManagerSingleton._instance = None  # noqa: SLF001
    SettingsManagerSingleton._initialization_errors.clear()  # noqa: SLF001

    TranslationManagerSingleton._instance = None  # noqa: SLF001
    TranslationManagerSingleton._initialization_errors.clear()  # noqa: SLF001

    yield  # Test runs here

    # Post-test cleanup (optional, but good for robustness if singletons have complex teardown)
    if LoggingManagerSingleton._instance:  # noqa: SLF001
        LoggingManagerSingleton.reset()  # Assuming a reset method


# --- Settings ---
@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """
    Creates a temporary test configuration file with predefined content.

    Args:
    ----
        tmp_path: The `pytest` fixture for creating temporary directories.

    Returns:
    -------
        The path to the created temporary configuration file.
    """
    default_config: dict[str, dict[str, Any]] = {
        "logger": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "log_directory": str(tmp_path),
            "output_format": "console",
        },
        "console_handler": {"enabled": True},
        "file_handler": {
            "enabled": False,
            "file_name": "test_file.log",
        },
        "limited_file_handler": {
            "enabled": False,
            "file_name": "limited_test_file.log",
            "max_bytes": 1024,
            "backup_count": 5,
        },
        "gui": {
            "enabled": True,
        },
        "reports": {"directory": str(tmp_path)},
        "data": {"directory": str(tmp_path)},
        "network": {
            "timeout": 5,
            "ntp_servers": ["pool.ntp.org"],
            "urls": ["https://example.com"],
        },
    }

    config_path = tmp_path / "config.toml"
    with config_path.open("wb") as f:
        tomli_w.dump(default_config, f)
    return config_path


@pytest.fixture
def mock_settings_with_defaults(tmp_path: Path, mocker: Any) -> Any:
    """
    Provides a realistic mock of the SettingsManager with complete default config.

    It supports both settings.get("section", "key", fallback)
    and settings.get_section("section") access patterns,
    returning real dicts and values for Pydantic validation.
    """
    app_name = "checkconnect"
    report_dir = tmp_path / "reports"
    data_dir = tmp_path / "data"
    log_dir = tmp_path / "logs"

    for directory in [report_dir, data_dir, log_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    default_config: dict[str, dict[str, Any]] = {
        "logger": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "log_directory": str(log_dir),
            "output_format": "console",
        },
        "console_handler": {"enabled": True},
        "file_handler": {
            "enabled": False,
            "file_name": app_name + ".log",
        },
        "limited_file_handler": {
            "enabled": False,
            "file_name": "limited_" + app_name + ".log",
            "max_bytes": 1024,
            "backup_count": 5,
        },
        "gui": {"enabled": True},
        "reports": {
            "directory": str(report_dir),
        },
        "data": {
            "directory": str(data_dir),
        },
        "network": {
            "timeout": 5,
            "ntp_servers": ["pool.ntp.org"],
            "urls": ["https://example.com"],
        },
        "general": {
            "default_language": "en",
        },
    }

    mock_settings = mocker.Mock()
    mock_settings.get_all_settings.return_value = default_config

    # ✅ direct methods on settings (not settings.config)
    mock_settings.get.side_effect = lambda section, key=None, fallback=None: (
        default_config.get(section, fallback) if key is None else default_config.get(section, {}).get(key, fallback)
    )
    mock_settings.get_section.side_effect = lambda section: default_config.get(section, {})

    return mock_settings


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Creates a sample configuration dictionary."""
    return {
        "logger": {"level": "INFO"},
        "network": {"timeout": 5, "ntp_servers": ["pool.ntp.org"]},
        "results": {"directory": "test_reports"},
    }


# --- File System Fixtures ---
@pytest.fixture
def temp_config_dir(mocker):
    """
    Mocks user_config_dir to a temporary directory and ensures it's cleaned up.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_config_path = Path(tmpdir) / "checkconnect"
        mock_config_path.mkdir(parents=True, exist_ok=True)

        mocker.patch(
            "checkconnect.config.settings_manager.SettingsManager.DEFAULT_SETTINGS_LOCATIONS",
            return_value=str(mock_config_path),
        )
        mocker.patch("checkconnect.cli.main.__about__.__app_name__", "checkconnect")
        mocker.patch("checkconnect.cli.main.__about__.__app_org_id__", "MyAwesomeOrg")
        mocker.patch("checkconnect.cli.main.__about__.__version__", "0.1.0")  # Also patch version here for consistency

        yield mock_config_path


# --- AppContext ---
@pytest.fixture
def mock_app_context(mocker):
    """
    Provides a realistic mocked AppContext with gettext and logger.
    Note: If you truly use structlog/loguru, AppContext's logger attribute
    might need to be an actual structlog.BoundLogger in your real code,
    but for this mock context, a simple mock is fine unless its methods are called.
    """
    mock_ctx = mocker.Mock()
    mock_ctx.gettext.side_effect = lambda x: x
    mock_ctx.translator = mocker.Mock()  # Ensure translator is mocked
    mock_ctx.translator.gettext.side_effect = lambda x: x
    return mock_ctx


@pytest.fixture
def app_context_fixture(
    mocker: MockerFixture,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> AppContext:
    """
    Returns an `AppContext` mock, defaulting to 'full' configuration,
    but can be set to 'simple' via parametrization.

    This fixture allows tests to easily obtain a mock `AppContext` configured
    either minimally ("simple") or with more detailed settings ("full"),
    simulating different application states or configurations.

    Args:
    ----
        mocker: The `pytest-mock` fixture for mocking objects.
        tmp_path: The `pytest` fixture for creating temporary directories.
        request: The `pytest` fixture request object, used to access parametrization.

    Returns:
    -------
        A mocked `AppContext` instance.
    """
    # Determine the desired level: 'simple' if explicitly requested via parametrize,
    # 'full' by default if no parametrization is applied to the fixture request.
    level: Literal["simple", "full"] = (
        request.param if hasattr(request, "param") and request.param is not None else "full"
    )

    # --- Setup the mock logger instance for the AppContext ---
    mock_logger_instance_for_context = mocker.Mock()
    mock_logger_instance_for_context.info.return_value = None
    mock_logger_instance_for_context.warning.return_value = None
    mock_logger_instance_for_context.error.return_value = None
    # Patch structlog.get_logger globally *if* your application code calls it directly
    # and you want to control its return value within tests.
    # If your app always gets loggers via context.get_module_logger, this global patch isn't strictly needed.
    mocker.patch("structlog.get_logger", return_value=mock_logger_instance_for_context)

    mock_translator = mocker.Mock(spec=TranslationManager)
    mock_translator.gettext.side_effect = lambda text: f"[mocked] {text}"
    mock_translator.translate.side_effect = lambda text: f"[mocked] {text}"

    context = mocker.Mock(spec=AppContext)
    context.translator = mock_translator
    context.gettext = mock_translator.gettext
    # Ensure get_module_logger returns the mock logger instance
    context.get_module_logger.side_effect = lambda name: mock_logger_instance_for_context  # noqa: ARG005

    # Get a logger *from the context* for messages within the fixture itself
    # This ensures consistency with how your application uses loggers.
    # For fixture-specific internal logging, you might also use a dedicated logger or print statements.
    fixture_logger = context.get_module_logger(__name__)  # Use the logger provided by the mock context

    if level == "simple":
        mock_config = mocker.Mock(spec=SettingsManager)
        mock_config.get_section.return_value.get.return_value = None
        mock_config.get.side_effect = lambda section, key, default=None: (
            default if not (section == "reports" and key == "directory") else None
        )
        context.settings = mock_config
        fixture_logger.debug("AppContext fixture providing 'simple' configuration.")
        return context

    # If level is 'full' (either by default or explicitly requested)
    mock_config = mocker.Mock(spec=SettingsManager)

    mock_network_section = mocker.Mock()
    mock_network_section.get.side_effect = lambda key, default=None: {
        "ntp_servers": ["time.google.com", "time.cloudflare.com"],
        "urls": ["https://example.com", "https://google.com"],
        "timeout": 10,
    }.get(key, default)

    def get_section_side_effect(section_name: str) -> MagicMock:
        if section_name == "network":
            return mock_network_section
        return mocker.Mock()

    mock_config.get_section.side_effect = get_section_side_effect

    def config_get_top_level(section: str, key: str, default: Any = None) -> Any:
        if section == "reports" and key == "directory":
            return str(tmp_path / "test_reports_from_config")
        if section == "data" and key == "directory":
            return str(tmp_path / "data")
        if section == "network" and key == "timeout":
            return 10
        return default

    mock_config.get.side_effect = config_get_top_level

    context.settings = mock_config
    fixture_logger.debug("AppContext fixture providing 'full' configuration.")
    return context


@pytest.fixture
def patch_checkconnect(mocker):
    """
    Patches the CheckConnect class for isolation from actual network calls.
    """
    return mocker.patch("checkconnect.cli.run_app.CheckConnect")


# You can also use Pytest's built-in `tmp_path` fixture for general temporary paths if preferred.


# --- CLI Runner Fixture ---
@pytest.fixture(scope="session")  # Often safe to be session scoped if it's stateless
def runner() -> CliRunner:
    """
    Provides an instance of `typer.testing.CliRunner` for testing CLI applications.
    """
    return CliRunner()


# --- Network Mocking Fixtures ---
@pytest.fixture
def mock_network_calls(mocker: MockerFixture) -> None:
    """
    Mocks network-related functionality (ntplib, requests) for isolated testing.
    """
    mock_ntp_response = mocker.Mock()
    mock_ntp_response.tx_time = 1234567890
    mocker.patch("ntplib.NTPClient.request", return_value=mock_ntp_response)

    mock_http_response = mocker.Mock()
    mock_http_response.status_code = 200
    mocker.patch("requests.get", return_value=mock_http_response)


# --- GUI-specific Fixtures (if applicable to your CLI project) ---
@pytest.fixture(scope="session")
def q_app() -> Iterator[QApplication]:
    """
    Provides a fresh QApplication instance for GUI tests.
    Ensures no QApplication instance leaks across tests.
    """
    app = QApplication.instance()
    created = False
    if not app:
        app = QApplication([])
        created = True
    yield app
    if created:
        app.quit()
        del app


@pytest.fixture
def mock_qapplication_class(mocker: MockerFixture) -> MagicMock:
    """
    Patches QApplication in your GUI startup module for testing.
    """
    mock_app_instance = mocker.MagicMock(spec=QApplication)
    mock_app_instance.exec.return_value = 0
    mock_app_instance.quit.return_value = None
    mock_qapp_ctor = mocker.patch("checkconnect.gui.startup.QApplication", return_value=mock_app_instance)
    mock_qapp_ctor.instance.return_value = None  # Force creation of new app
    return mock_qapp_ctor


@pytest.fixture
def _always_mock_qapp():
    """Ensures QApplication is always mocked for GUI-related tests."""
    return


# --- Data Fixtures (Examples) ---


@pytest.fixture
def sample_ntp_results() -> list[str]:
    """Provides sample NTP check results."""
    return ["NTP Server 1: OK", "NTP Server 2: FAILED"]


@pytest.fixture
def sample_url_results() -> list[str]:
    """Provides sample URL check results."""
    return ["https://example.com: OK", "https://bad-url.invalid: ERROR"]
