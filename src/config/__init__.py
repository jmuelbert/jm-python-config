# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: © 2025-present Jürgen Mülbert

"""
CheckConnect Configuration Module.

This module handles the initialization and loading of various configuration
components for the CheckConnect application, including logger settings,
translation management, and application context. It provides the necessary
infrastructure for configuring the application, loading settings from
external sources like TOML files, and managing resources like logging,
translations, and network configurations.

The module is structured to ensure that settings, logging configurations,
and translation resources are accessible throughout the application.

Components of this module:
--------------------------------
- `config.toml`: The primary configuration file for CheckConnect containing
  network settings (such as NTP servers and URLs) and output directory
  options.
- `appcontext.py`: Contains the `AppContext` class which is responsible for
  managing the shared application state, including the logger, translation
  manager, and configurations.
- `logger.py`: Provides the `LoggingManager` class for setting up and
  managing logging configurations, including file-based and console logging,
  with support for JSON output and rotation.
- `settings_manager.py`: Holds application settings, with helper functions to load
  and parse configuration data, particularly for network-related settings.
- `translation_manager.py`: Contains the `TranslationManager` class which loads
  and manages translation files for different languages, enabling multi-language
  support.

Main responsibilities:
-----------------------
1. Initialize and load application configurations from external TOML files.
2. Set up and configure the logging system using `LoggingManager`.
3. Set up the translation manager using `TranslationManager` to handle
   internationalization.
4. Provide a shared `AppContext` for the application, encapsulating
   configurations, translators, and loggers.

Usage:
------
To initialize the configuration and start using the application context,
the following steps are typically performed:
1. Load the application context from the `config.toml` file using
   `initialize_app_context()`.
2. Access configuration values through the context, such as network settings
   and output paths.
3. Use the contexts logger and translation manager across the application.
"""
