# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: © 2025-present Jürgen Mülbert

"""
CheckConnect Application Context Module.

This module provides the core application context for CheckConnect, offering
a centralized way to manage shared resources like the logger, translation manager,
and application configuration. The `AppContext` class handles the
application state, ensuring all components can access common resources
consistently and, where applicable, in a thread-safe manner.

The `AppContext` serves as the central point of access for:
- **Application Logging**: Accessible via `get_module_logger()`.
- **Translation Management**: Accessible via `gettext()`.
- **Configuration Management**: Handled by the `config` attribute.

Responsibilities:
-----------------
- Store references to pre-initialized application managers: settings, translation.
- Provide unified access methods for getting module-specific loggers and translations.
- Act as a facade for other application components to access core services.

Usage Example (within a subcommand):
-----------------------------------
```python
# In a Typer subcommand, after AppContext is passed via ctx.obj:
from jm_python_config.appcontext import AppContext
import typer


@typer_app.command(...)
def my_command(ctx: typer.Context):
    app_context: AppContext = ctx.obj["app_context"]  # Assuming you store it here

    logger = app_context.get_module_logger(__name__)
    logger.info(app_context.gettext("Starting CheckConnect..."))

    config_value = app_context.settings.get_setting("network", "timeout")
    logger.debug(f"Network timeout: {config_value}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from jm_python_config.settings_manager import SettingsManager
from jm_python_config.translation_manager import TranslationManager

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

    from jm_python_config.settings_manager import SettingsManager
    from jm_python_config.translation_manager import TranslationManager

# Global logger for main.py (will be reconfigured by LoggingManagerSingleton)
log: structlog.stdlib.BoundLogger
log = structlog.get_logger(__name__)


@dataclass
class AppContext:
    """
    Manage shared application context.

    Providing access to pre-initialized configuration and translation services.

    This class serves as a central hub for essential application components,
    ensuring consistent access to global resources across the application.
    It receives already-initialized singleton instances from the application's
    main startup logic.

    Attributes
    ----------
    translator : TranslationManager
        Manages message translations for the UI and CLI. This is an
        already initialized instance.
    settings : SettingsManager
        An instance of the settings manager for application configuration.
        This is an already initialized instance.
    """

    # These attributes directly hold the instances of your managers
    settings: SettingsManager
    translator: TranslationManager

    def get_module_logger(self, name: str) -> BoundLogger:
        """
        Retrieve a `structlog` logger instance for a specific module.

        This method leverages the globally configured `structlog` system,
        which is set up by the `LoggingManagerSingleton` during application startup.

        Parameters
        ----------
        name : str
            The name of the module for which to retrieve the logger (e.g., `__name__`).

        Returns
        -------
        structlog.stdlib.BoundLogger
            A bound logger instance for the specified module.
        """
        # structlog.get_logger() will automatically use the global configuration
        # applied by LoggingManagerSingleton.
        return structlog.get_logger(name)

    def gettext(self, message: str) -> str:
        """
        Translate a given message string using the active translation manager.

        Parameters
        ----------
        message : str
            The message string to be translated.

        Returns
        -------
        str
            The translated string.
        """
        return self.translator.gettext(message)

    @classmethod
    def create(
        cls,
        settings_instance: SettingsManager,  # Changed to accept an instance
        translator_instance: TranslationManager,  # Changed to accept an instance
    ) -> AppContext:
        """
        Create an `AppContext` instance from pre-initialized managers.

        That's the factory method for creating an `AppContext` instance.

        Parameters
        ----------
        settings_instance : SettingsManager
            An already initialized `SettingsManager` instance.
        translator_instance : TranslationManager
            An already initialized `TranslationManager` instance.

        Returns
        -------
        AppContext
            A fully initialized `AppContext` instance.
        """
        # No internal creation here, just passing them to the constructor
        return cls(settings=settings_instance, translator=translator_instance)
