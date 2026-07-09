# tests/unit/test_internet_search.py
"""Юнит-тесты чистых функций интернет-поиска (src/services/internet_search.py)."""

from src.services.internet_search import needs_translation, validate_vlm_answer


class TestNeedsTranslation:
    """Тесты определения необходимости перевода."""

    def test_english_text_needs_translation(self):
        assert needs_translation("Eiffel Tower in Paris") is True

    def test_russian_text_no_translation_needed(self):
        assert needs_translation("Эйфелева башня в Париже") is False

    def test_empty_string_no_translation_needed(self):
        assert needs_translation("") is False

    def test_mixed_text_high_cyrillic_ratio(self):
        # Более 30% кириллицы — перевод не нужен
        text = "Исаакиевский собор (Saint Isaac's Cathedral)"
        assert needs_translation(text) is False


class TestValidateVlmAnswer:
    """Тесты валидации ответа VLM на запрос названия."""

    def test_valid_name_passes(self):
        assert validate_vlm_answer("Eiffel Tower") == "Eiffel Tower"

    def test_unknown_returns_none(self):
        assert validate_vlm_answer("unknown") is None

    def test_empty_string_returns_none(self):
        assert validate_vlm_answer("") is None

    def test_too_long_answer_returns_none(self):
        # Более 8 слов — скорее всего не название
        long = "This is a very long answer that is not a landmark name at all"
        assert validate_vlm_answer(long) is None

    def test_quotes_are_stripped(self):
        assert validate_vlm_answer('"Notre-Dame"') == "Notre-Dame"
