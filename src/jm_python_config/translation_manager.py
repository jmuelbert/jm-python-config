# SPDX-License-Identifier: EUPL-1.2

# SPDX-FileCopyrightText: © 2025-present Jürgen Mülbert

"""
translation.py — TranslationManager for internationalization.

This module provides the TranslationManager class, which handles loading and
managing gettext-based translations for the CheckConnect project. It attempts
to use language preferences from a configuration file or system locale, and
falls back to English if no translation files are found.

Typical usage:
--------------
>>> tm = TranslationManager(language="de")
>>> print(tm.gettext("Hello"))

The TranslationManager automatically falls back to default translation
files if the specified ones are missing.
"""

from __future__ import annotations

import gettext
import importlib.resources
import locale
import os
import sys
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final

import structlog

from checkconnect.__about__ import __app_name__
from checkconnect.config.settings_manager import SettingsManagerSingleton

if TYPE_CHECKING:
    from collections.abc import Callable

# Global logger for main.py (will be reconfigured by LoggingManagerSingleton)
log: structlog.stdlib.BoundLogger
log = structlog.get_logger(__name__)


class TranslationManager:
    """
    Manage translations for the CheckConnect project.

    This class initializes and manages gettext-based translation files (.mo)
    for multiple languages, based on project domain, language code, and locale
    directory.

    Attributes
    ----------
    domain : str
        The gettext domain name, typically the project name (e.g., "checkconnect").
    locale_dir : str
        The directory where the compiled `.mo` translation files are located.
    language : str
        The current language code in use (e.g., "en", "de").
    translation : gettext.NullTranslations
        The active gettext translation object.
    _ : Callable
        A shorthand alias for gettext's translation function.

    """

    _internal_errors: list[str]

    # Type definition for the translation function
    _translate_func: Callable[[str], str]

    APP_NAME: Final[str] = __app_name__.lower()
    LOCALES_DIR_NAME: Final[str] = "locales"

    def __init__(self) -> None:
        """
        Initialize TranslationManager attributes.

        Does NOT load translations yet.
        Call .configure() to set up translations.
        """
        self._translation_domain: str = self.APP_NAME
        self.locale_dir: Path | None = None
        self._current_language: str | None = None
        self._translation: gettext.NullTranslations = gettext.NullTranslations()
        self._translate_func = self._translation.gettext
        self._internal_errors: list[str] = []  # Errors specific to this instance's setup

    @property
    def internal_errors(self) -> list[str]:
        """Return the list of internal errors."""
        return self._internal_errors

    @property
    def has_errors(self) -> bool:
        """Return True if there are any internal errors."""
        return bool(self._internal_errors)

    @property
    def last_error(self) -> str | None:
        """Return the last internal error."""
        return self._internal_errors[-1] if self._internal_errors else None

    def configure(
        self,
        language: str | None = None,
        translation_domain: str | None = None,
        locale_dir: str | None = None,
    ) -> None:
        """
        Configure and load translations for the TranslationManager.

        Args:
            language: Explicit language code (e.g. 'en', 'de').
            translation_domain: Translation domain name (defaults to app name).
            locale_dir: Path to translation .mo files (optional).

        Raises:
        OSError
            If the translation file (.mo) cannot be loaded.
        """
        if language is None:
            language = self._get_default_language()

        if self._current_language == language:
            return  # Already configure

        self._internal_errors.clear()
        self._translation_domain = translation_domain or self.APP_NAME
        self.locale_dir = Path(locale_dir) if locale_dir else self._default_locale_dir()
        resolved_language = self._resolve_language(language)

        try:
            self._translation = gettext.translation(
                self._translation_domain,
                localedir=self.locale_dir,
                languages=[resolved_language],
                fallback=True,
            )
            self._translate_func = self._translation.gettext
            self._current_language = resolved_language
        except OSError as error:
            log.exception("Translation files not found. Using fallback.", exc_info=error)
            self._translator = gettext.NullTranslations()
            self._translator.install()
            # Add this:
            if self.current_language is None:
                try:
                    lang, encoding = locale.getlocale()
                    self.current_language = f"{lang}.{encoding}" if lang and encoding else "en_US.UTF-8"
                except locale.Error:
                    self.current_language = "en_US.UTF-8"

        try:
            locale.setlocale(locale.LC_ALL, resolved_language)
        except locale.Error as e:
            self._handle_translation_error(e, resolved_language)

    def _get_default_language(self) -> str:
        """
        Safely determines the system default language code.

        Returns:
            The default locale language string, like 'en_US'.
            Falls back to 'en' if detection fails.
        """
        try:
            try:
                locale.setlocale(locale.LC_MESSAGES, "")
            except locale.Error:
                locale.setlocale(locale.LC_ALL, "")
            lang, _ = locale.getlocale(locale.LC_MESSAGES)

        except locale.Error as e:
            log.exception("Failed to determine default language. Going back to English.", exc_info=e)
            return "en"
        else:
            return lang if lang else "en"

    def _resolve_language(self, explicit_lang: str | None) -> str:
        """Resolve the language to use, using explicit input, settings, or system default."""
        if explicit_lang:
            return explicit_lang

        try:
            from checkconnect.config.settings_manager import SettingsManagerSingleton

            settings_lang = SettingsManagerSingleton.get_instance().get_setting("general", "default_language")
            return settings_lang or self._get_system_language() or "en"
        except (ImportError, AttributeError, KeyError, RuntimeError) as e:
            self._internal_errors.append(f"Could not resolve language: {e}")
            log.exception("Language resolution failed", exc_info=e)
            return "en"

    def _handle_translation_error(self, error: Exception, language: str) -> None:
        """Handle translation setup errors."""
        msg = f"Failed to load translations for '{language}': {error}"
        self._internal_errors.append(msg)
        log.exception(msg, exc_info=error)

    def _default_locale_dir(self) -> Path:
        """
        Determine the default directory for translation files.

        Returns
        -------
        Path
            Path to the default "locales" directory inside the project.

        """
        locales_dir = Path(__file__).parent.parent / self.LOCALES_DIR_NAME
        if locales_dir.exists():
            return locales_dir

        return Path(self._package_locale_dir())

    def _package_locale_dir(self) -> str:
        """Fallback: use the translations from PyPI-Package."""
        try:
            return str(
                importlib.resources.files(self.APP_NAME) / self.LOCALES_DIR_NAME,
            )
        except OSError as e:
            self._internal_errors.append(f"Failed to resolve package locale directory for '{self.APP_NAME}': {e}")
            log.exception("Failed to resolve package locale directory", app_name=self.APP_NAME, exc_info=e)
            # Fallback for when importlib.resources.files might fail
            return str(Path(__file__).parent.parent / self.LOCALES_DIR_NAME)
        except ValueError as e:
            self._internal_errors.append(f"Failed to resolve package locale directory for '{self.APP_NAME}': {e}")
            log.exception("Failed to resolve package locale directory", app_name=self.APP_NAME, exc_info=e)
            # Fallback for when importlib.resources.files might fail
            return str(Path(__file__).parent.parent / self.LOCALES_DIR_NAME)
        except TypeError as e:
            self._internal_errors.append(f"Failed to resolve package locale directory for '{self.APP_NAME}': {e}")
            log.exception("Failed to resolve package locale directory", app_name=self.APP_NAME, exc_info=e)
            # Fallback for when importlib.resources.files might fail
            return str(Path(__file__).parent.parent / self.LOCALES_DIR_NAME)

    @staticmethod
    def _extract_two_letter_lang(full_locale: str) -> str:
        """
        Extract the two-letter language code from a full locale string.

        Handles formats like 'en_US.UTF-8', 'en.UTF-8', 'en_US', 'en', 'C'.
        Returns 'en' for 'C' or empty strings as a default.
        """
        if not full_locale or full_locale.upper() == "C":
            return "en"  # Default to 'en' if locale is 'C' or empty/unparsable

        # Split by underscore first (e.g., 'en_US.UTF-8' -> 'en_US.UTF-8')
        # Then split by dot (e.g., 'en_US.UTF-8' -> 'en_US')
        # Then take the first two characters and lowercase
        # This handles cases like 'zh_Hans.UTF-8' -> 'zh'
        lang_part = full_locale.split("_")[0]
        base_lang = lang_part.split(".")[0]
        return base_lang.lower()

    @staticmethod
    def _normalize_locale_string(lang_code: str) -> str:
        """
        Normalize a language code into a full locale string.

        Attempts to use system locale normalization, ensures UTF-8 encoding,
        and falls back to a basic constructed format if necessary.

        Args:
            lang_code: The short language code like 'en', 'de', etc.

        Returns:
            A normalized locale string like 'en_US.UTF-8' or 'C' as fallback.
        """
        if not lang_code:
            return "C"

        lang_code = lang_code.lower()

        # First, try locale.normalize (e.g. 'fr' -> 'fr_FR.ISO8859-1')
        normalized = locale.normalize(lang_code)

        if "utf-8" in normalized.lower():
            return normalized

        # Try locale.alias as a fallback
        alias = locale.locale_alias.get(lang_code)
        if alias:
            if "." not in alias:
                alias += ".UTF-8"
            return alias

        # Final fallback
        return f"{lang_code}_{lang_code.upper()}.UTF-8"

    @staticmethod
    def _get_locale_from_getlocale_attempts() -> str | None:
        """
        Attempt to get the locale from `locale.getlocale()`.

        Directly or after a `locale.setlocale(LC_ALL, '')` attempt.
        Returns a full locale string (e.g., 'en_US.UTF-8') or None.

        Returns:
            str | None: The full locale string or None if not found.
        """
        try:
            lang_code, encoding = locale.getlocale(locale.LC_CTYPE)
            if lang_code:
                return f"{lang_code}.{encoding}" if encoding else f"{lang_code}.UTF-8"

            # If initial getlocale didn't return a language, try setting LC_ALL to empty string
            with suppress(locale.Error):
                locale.setlocale(locale.LC_ALL, "")
                lang_code, encoding = locale.getlocale(locale.LC_CTYPE)
                if lang_code:
                    return f"{lang_code}.{encoding}" if encoding else f"{lang_code}.UTF-8"
        except locale.Error as e:
            log.warning("Initial locale.getlocale attempts failed: %s", e)
        except Exception as e:
            log.exception("Unexpected error during initial locale attempts.", exc_info=e)
        return None

    @staticmethod
    def _get_locale_from_macos_workaround() -> str | None:
        """
        Attempt a macOS-specific locale workaround by setting 'en_US.UTF-8'.

        Returns 'en_US.UTF-8' if successful, otherwise None.

        Returns
        -------
        str | None
        """
        if sys.platform == "darwin":
            try:
                locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
                log.info("Successfully set locale to en_US.UTF-8 as macOS workaround.")

            except locale.Error as e:
                log.exception("macOS specific locale workaround 'en_US.UTF-8' also failed.", exc_info=e)
            else:
                return "en_US.UTF-8"

        return None

    @staticmethod
    def _get_locale_from_environment_variables() -> str | None:
        """
        Check common environment variables for locale information.

        Returns a normalized full locale string (e.g., 'en_US.UTF-8') or None.

        Returns
        -------
        str | None
            A normalized full locale string (e.g., 'en_US.UTF-8') if found,
            otherwise None.
        """
        for var in ["LANG", "LC_ALL", "LC_CTYPE", "LANGUAGE"]:
            env_val = os.getenv(var)
            if env_val:
                # Extract two-letter code then normalize it to a full locale string
                return TranslationManager._normalize_locale_string(TranslationManager._extract_two_letter_lang(env_val))
        return None

    def _get_system_language(self) -> str | None:
        """
        Attempt to determine the system's default language using a series of strategies.

        Returns:
            str | None: A full locale string (e.g., 'en_US.UTF-8') if found,
                        otherwise None.
        """
        # Strategy 1: Direct locale.getlocale attempts
        system_locale = self._get_locale_from_getlocale_attempts()
        if system_locale:
            return system_locale

        # Strategy 2: macOS-specific workaround
        system_locale = self._get_locale_from_macos_workaround()
        if system_locale:
            return system_locale

        # Strategy 3: Check environment variables
        system_locale = self._get_locale_from_environment_variables()
        if system_locale:
            return system_locale

        log.warning("Could not determine system default language from any source.")
        return None  # If no locale can be determined by any strategy

    def _set_language(self) -> None:
        """
        Set the language for translations.

        Tries (1) explicit language (from `self._current_language` if already set),
        (2) config setting, (3) system locale, (4) fallback to 'en'.
        """
        lang_for_gettext: str  # This will always store the two-letter code (e.g., "en", "es")
        full_locale_string_for_setlocale: str  # This will always store the full locale string (e.g., "en_US.UTF-8")

        # Priority 1: Use language explicitly passed to configure()
        if self._current_language:  # This is the input_language from configure()
            lang_for_gettext = self._current_language
            full_locale_string_for_setlocale = self._normalize_locale_string(lang_for_gettext)
            # Removed the redundant locale.setlocale call here.

        else:
            # Priority 2: Fallback to settings
            settings_lang = SettingsManagerSingleton.get_instance().get_setting("general", "default_language")
            if settings_lang:
                lang_for_gettext = settings_lang
                full_locale_string_for_setlocale = self._normalize_locale_string(settings_lang)
            else:
                # Priority 3: Fallback to system locale
                system_full_locale = self._get_system_language()  # This returns a full locale string or None
                if system_full_locale:
                    lang_for_gettext = self._extract_two_letter_lang(system_full_locale)
                    full_locale_string_for_setlocale = system_full_locale  # Use the full string from system
                else:
                    # Priority 4: Ultimate fallback to "en"
                    lang_for_gettext = "en"
                    full_locale_string_for_setlocale = self._normalize_locale_string(
                        "en"
                    )  # Ensure this is a full string

        # Always store the two-letter code in _current_language
        self._current_language = lang_for_gettext

        try:
            # Configure gettext translation (uses two-letter code)
            self._translation = gettext.translation(
                self._translation_domain,
                localedir=self.locale_dir,
                languages=[lang_for_gettext],
                fallback=True,
            )
            self._translate_func = self._translation.gettext

            # Set the locale for the entire process (uses full locale string)
            locale.setlocale(locale.LC_ALL, full_locale_string_for_setlocale)

        except locale.Error as e:
            msg = f"Failed to set system locale to '{full_locale_string_for_setlocale}': {e}. Falling back to default gettext."
            log.exception(
                "Failed to set system locale.",
                full_locale_string_for_setlocale=full_locale_string_for_setlocale,
                exc_info=e,
            )
            self._internal_errors.append(msg)
            raise
        except OSError as e:
            msg = f"Translation for '{lang_for_gettext}' failed to load from '{self.locale_dir}': {e}. Falling back to default gettext."
            log.exception(
                "Translation failed to load from locel directory. Falling back to default gettext.",
                locale_dir=self.locale_dir,
                exc_info=e,
            )
            self._internal_errors.append(msg)
            raise
        except TypeError as e:
            msg = f"Translation configuration for '{lang_for_gettext}' failed due to type error: {e}. Falling back to default gettext."
            log.exception(
                "Translation configuration failed due to type error. Falling back to default gettext.",
                lang_for_gettext=lang_for_gettext,
                exc_info=e,
            )
            self._internal_errors.append(msg)
            self._translate_func = gettext.gettext
        except Exception as e:
            msg = f"An unexpected error occurred during translation setup for '{lang_for_gettext}': {e}. Falling back to default gettext."
            log.exception(
                "An unexpected error occurred during translation setup. Falling back to default gettext.",
                lang_for_gettext=lang_for_gettext,
                exc_info=e,
            )
            self._internal_errors.append(msg)
            self._translate_func = gettext.gettext

    def translate(self, text: str) -> str:
        """Translate a given string."""
        return self._translation.gettext(text)

    def translate_plural(self, singular: str, plural: str, count: int) -> str:
        """Translate a given string with plural forms."""
        return self.ngettext(singular, plural, count)

    def translate_context(self, context: str, text: str) -> str:
        """
        Translate the context (like pgettext).

        Convention: Context and text are separeated with a `|` in the .po-files.
        """
        return self._translation.gettext(f"{context}|{text}")

    def gettext(self, message: str) -> str:
        """Translate a single message string."""
        if not self._translation:
            msg = "TranslationManager is not configured."
            raise RuntimeError(msg)
        return self._translation.gettext(message)

    def ngettext(self, singular: str, plural: str, count: int) -> str:
        """Translate a message with plural forms."""
        if not self._translation:
            msg = "TranslationManager is not configured."
            raise RuntimeError(msg)
        return self._translation.ngettext(singular, plural, count)

    def set_language(self, language: str) -> None:
        """
        Change the active language and reload translations.

        Parameters
        ----------
        language : str
            New language code to activate (e.g., "fr", "de").

        """
        self._current_language = language
        self._set_language()

    @property
    def current_language(self) -> str:
        """Return the current language code."""
        return self._current_language

    @current_language.setter
    def current_language(self, language: str) -> None:
        """
        Change the active language and reload translations.

        Parameters
        ----------
        language : str
            New language code to activate (e.g., "fr", "de").

        """
        self._current_language = language
        self._set_language()


class TranslationManagerSingleton:
    """
    Singleton class for TranslationManager.

    Ensures a single instance manages application translations
    and handles its controlled initialization.
    """

    _instance: TranslationManager | None = None
    _initialization_errors: ClassVar[list[str]] = []
    _is_configured: ClassVar[bool] = False  # Track if the instance has been configured

    @classmethod
    def get_instance(cls) -> TranslationManager:
        """
        Return the single instance of TranslationManager.

        Raises RuntimeError if not yet initialized.
        """
        if cls._instance is None:
            try:
                cls._instance = TranslationManager()
            except Exception as e:
                cls._initialization_errors.append(f"Error creating TranslationManager instance: {e}")
                cls._instance = None
                raise
        return cls._instance

    @classmethod
    def configure_instance(
        cls,
        language: str | None = None,
        translation_domain: str | None = None,
        locale_dir: str | None = None,
    ) -> None:
        """
        Configure the TranslationManager instance.

        This should be called
        once during application startup after the instance is obtained.

        Raises:
        ------
        OSError
            If the translation file (.mo) cannot be loaded.
        """
        if cls._is_configured:
            # Optional: Decide if you want to allow re-configuration or log a warning
            # For now, let's allow it to potentially re-run configure on the instance
            # but clear singleton errors for this attempt.
            # If you want strict single-time configuration for the app lifecycle:
            cls._initialization_errors.append("TranslationManagerSingleton already configured. Cannot re-configure.")
            return

        instance = cls.get_instance()  # Ensure instance exists

        cls._initialization_errors.clear()  # Clear errors for a fresh configuration attempt
        try:
            instance.configure(
                language=language,
                translation_domain=translation_domain,
                locale_dir=locale_dir,
            )
            # Add errors reported by the instance itself
            cls._initialization_errors.extend(instance.internal_errors)
            cls._is_configured = True
        except Exception as e:
            msg = f"Critical error during TranslationManager configuration: {e}"
            cls._initialization_errors.append(msg)
            raise  # Re-raise if configuration failed critically

    @classmethod
    def get_initialization_errors(cls) -> list[str]:
        """Exposes initialization errors for testing/debugging."""
        errors = list(cls._initialization_errors)
        if cls._instance:
            errors.extend(cls._instance.internal_errors)  # Now calling public method
        return list(set(errors))  # Return unique errors

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance and its configuration state.

        Primarily for testing.
        """
        cls._instance = None
        cls._initialization_errors.clear()
        cls._is_configured = False
