# SPDX-License-Identifier: EUPL-1.2
#
# SPDX-FileCopyrightText: © 2025-present Jürgen Mülbert


from __future__ import annotations

import gettext
import locale
from typing import TYPE_CHECKING, Final
from unittest.mock import MagicMock, patch

import pytest

from checkconnect.config.translation_manager import TranslationManager, TranslationManagerSingleton

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture(autouse=True)
def reset_translation_singleton():
    TranslationManagerSingleton.reset()
    yield
    TranslationManagerSingleton.reset()


@pytest.fixture
def manager() -> TranslationManager:
    return TranslationManager()


class TestTranslationManager:
    def test_initial_state(self, manager: TranslationManager) -> None:
        """
        Test the initial state of the TranslationManager.

        Args:
            manager: TranslationManager instance.

        """
        assert manager._translation_domain == "checkconnect"  # noqa: SLF001
        assert manager._translation is not None  # noqa: SLF001
        assert isinstance(manager._translation, gettext.NullTranslations)  # noqa: SLF001
        assert manager.internal_errors == []

    def test_configure_explicit_language(self, monkeypatch, manager: TranslationManager, mocker) -> None:
        """
        Test configuring the translation manager with an explicit language.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
            manager: TranslationManager instance.
            mocker: Pytest mocker fixture.

        """
        fake_translation = mocker.MagicMock()
        fake_translation.gettext = lambda x: f"translated:{x}"

        monkeypatch.setattr(gettext, "translation", lambda *args, **kwargs: fake_translation)  # noqa: ARG005
        monkeypatch.setattr(locale, "setlocale", lambda category, value: None)  # noqa: ARG005

        manager.configure(language="de")
        assert manager.current_language == "de"
        assert manager.gettext("Hello") == "translated:Hello"
        assert manager.internal_errors == []

    def test_configure_locale_fallback(self, monkeypatch, manager: TranslationManager) -> None:
        """
        Test that the locale is configured correctly when the default locale is not available.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
            manager: TranslationManager instance.
        """
        monkeypatch.setattr(gettext, "translation", lambda *a, **k: gettext.NullTranslations())  # noqa: ARG005
        monkeypatch.setattr(locale, "setlocale", lambda *a, **k: None)  # noqa: ARG005
        monkeypatch.setattr(locale, "getlocale", lambda *a: ("fr_FR", "UTF-8"))  # noqa: ARG005

        manager.configure()
        assert manager.current_language == "fr_FR"

    def test_configure_oserror(self, monkeypatch, manager: TranslationManager) -> None:
        """
        Test that the locale is configured correctly when the default locale is not available.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
            manager: TranslationManager instance.
        """
        monkeypatch.setattr(gettext, "translation", lambda *a, **k: (_ for _ in ()).throw(OSError("fail")))  # noqa: ARG005
        monkeypatch.setattr(locale, "setlocale", lambda *a, **k: None)  # noqa: ARG005

        with pytest.raises(OSError, match="fail"):
            manager.configure(language="es")

        assert manager.gettext("Text") == "Text"
        assert any("failed" in e.lower() for e in manager.internal_errors)

    def test_plural_translation(self, manager: TranslationManager) -> None:
        """
        Test that plural translations are handled correctly.

        Args:
            manager: TranslationManager instance.
        """
        mock_translator = MagicMock(spec=TranslationManager)
        mock_translator.ngettext.side_effect = lambda s, p, n: f"{n} apple" if n == 1 else f"{n} apples"  # noqa: ARG005
        manager._translation = mock_translator  # noqa: SLF001

        result = manager.translate_plural("apple", "apples", 2)
        assert result == "2 apples"

    def test_context_translation(self, manager: TranslationManager) -> None:
        """
        Test that context translations are handled correctly.

        Args:
            manager: TranslationManager instance.
        """
        mock_translator = MagicMock(spec=TranslationManager)
        mock_translator.gettext.side_effect = lambda x: {"menu|File": "Datei"}.get(x, x)
        manager._translation = mock_translator  # noqa: SLF001

        result = manager.translate_context("menu", "File")
        assert result == "Datei"

    def test_set_language(self, monkeypatch, manager: TranslationManager) -> None:
        """
        Test that the language can be set correctly.

        Args:
            monkeypatch: Monkeypatch fixture.
            manager: TranslationManager instance.
        """
        monkeypatch.setattr(gettext, "translation", lambda *a, **k: gettext.NullTranslations())  # noqa: ARG005
        monkeypatch.setattr(locale, "setlocale", lambda *a, **k: None)  # noqa: ARG005

        manager.set_language("en")
        assert manager.current_language == "en"

    def test_system_language_env(self, monkeypatch, manager: TranslationManager) -> None:
        """
        Test that the system language is correctly detected.

        Args:
            monkeypatch: Monkeypatch fixture.
            manager: TranslationManager instance.
        """
        monkeypatch.delenv("LANG", raising=False)
        monkeypatch.setenv("LANG", "de_DE.UTF-8")

        lang = manager._get_system_language()  # noqa: SLF001
        assert lang == "de_DE.UTF-8"

    @patch("checkconnect.config.translation_manager.locale.normalize")
    def test_normalize_locale_string_mocked(self, mock_normalize) -> None:
        """
        Test the _normalize_locale_string method with mocked locale.normalize.

        Args:
            mock_normalize (Mock): Mocked locale.normalize function.
        """
        mock_normalize.side_effect = lambda loc: {
            "en": "en_US.UTF-8",
            "de": "de_DE.UTF-8",
            "fr": "fr_FR.UTF-8",
            "es": "es_ES.UTF-8",
            "it": "it_IT.UTF-8",
            "": "C",
        }[loc]

        assert TranslationManager._normalize_locale_string("en") == "en_US.UTF-8"  # noqa: SLF001
        assert TranslationManager._normalize_locale_string("de") == "de_DE.UTF-8"  # noqa: SLF001
        assert TranslationManager._normalize_locale_string("fr") == "fr_FR.UTF-8"  # noqa: SLF001
        assert TranslationManager._normalize_locale_string("es") == "es_ES.UTF-8"  # noqa: SLF001
        assert TranslationManager._normalize_locale_string("it") == "it_IT.UTF-8"  # noqa: SLF001
        assert TranslationManager._normalize_locale_string("") == "C"  # noqa: SLF001

    def test_extract_two_letter_lang(self) -> None:
        """Test the extraction of two-letter language codes from locale strings."""
        assert TranslationManager._extract_two_letter_lang("en_US.UTF-8") == "en"  # noqa: SLF001
        assert TranslationManager._extract_two_letter_lang("de_DE") == "de"  # noqa: SLF001
        assert TranslationManager._extract_two_letter_lang("C") == "en"  # noqa: SLF001
        assert TranslationManager._extract_two_letter_lang("") == "en"  # noqa: SLF001


class TestTranslationManagerSingleton:
    def test_singleton_returns_same_instance(self) -> None:
        """Test that the singleton returns the same instance."""
        a = TranslationManagerSingleton.get_instance()
        b = TranslationManagerSingleton.get_instance()
        assert a is b

    def test_singleton_configuration(self, monkeypatch) -> None:
        """
        Test that the singleton configuration works as expected.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setattr(gettext, "translation", lambda *a, **k: gettext.NullTranslations())  # noqa: ARG005
        monkeypatch.setattr(locale, "setlocale", lambda *a, **k: None)  # noqa: ARG005

        TranslationManagerSingleton.configure_instance(language="en")
        instance = TranslationManagerSingleton.get_instance()
        assert instance.current_language == "en"
        assert TranslationManagerSingleton.get_initialization_errors() == []

    def test_configure_adds_internal_error(self, monkeypatch) -> None:
        """
        Test that configure handles translation errors gracefully.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
        """
        monkeypatch.setattr(gettext, "translation", lambda *a, **k: (_ for _ in ()).throw(OSError("no .mo")))  # noqa: ARG005
        monkeypatch.setattr(locale, "setlocale", lambda *a, **k: None)  # noqa: ARG005

        tm = TranslationManager()
        with pytest.raises(OSError, match="no .mo"):
            tm.configure(language="de", translation_domain="dummy", locale_dir="/invalid/path")

        assert any("no .mo" in msg for msg in tm._internal_errors)  # noqa: SLF001

    def test_singleton_reset(self) -> None:
        """
        Test that the singleton reset works as expected.

        Args:
            None
        """
        old = TranslationManagerSingleton.get_instance()
        TranslationManagerSingleton.reset()
        new = TranslationManagerSingleton.get_instance()
        assert old is not new

    def test_get_initialization_errors_aggregates_from_instance(
        self,
        mocker: MockerFixture,
    ) -> None:
        """
        Test that get_initialization_errors aggregates errors from the instance
        and ensures the test is properly isolated.
        """
        expected_unique_errors: Final[int] = 3

        # ARRANGE
        # Mock the instance that the singleton will hold.
        mock_instance = mocker.MagicMock(spec=TranslationManager)

        # Corrected: We are setting the value of the 'internal_errors' property,
        # not the return value of a method.
        mock_instance.internal_errors = ["Instance Error 1", "Instance Error 2"]

        # Use patch.object to replace the class attributes for the duration of this test.
        # mocker will automatically revert these changes after the test is done.
        mocker.patch.object(TranslationManagerSingleton, "_instance", new=mock_instance)
        mocker.patch.object(TranslationManagerSingleton, "_initialization_errors", new=["Singleton Error 1"])

        # ACT
        # Call the method under test.
        errors = TranslationManagerSingleton.get_initialization_errors()

        # ASSERT
        # Check that the returned errors are a set of the expected unique values.
        assert set(errors) == {"Singleton Error 1", "Instance Error 1", "Instance Error 2"}

        # Check the length to ensure no duplicates.
        assert len(errors) == expected_unique_errors
