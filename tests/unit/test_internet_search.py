# tests/unit/test_internet_search.py
"""Юнит-тесты чистых функций интернет-поиска (src/services/internet_search.py)."""

from src.services.internet_search import (
    build_vlm_messages,
    extract_clean_name,
    filter_wiki_results,
    needs_translation,
    validate_vlm_answer,
)


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


class TestFilterWikiResults:
    """Тесты фильтрации результатов Wikipedia."""

    def test_empty_description_removed(self):
        assert filter_wiki_results({"Eiffel Tower": ""}) == {}

    def test_noise_token_name_removed(self):
        """Название с шумовым токеном (tripadvisor) отбрасывается."""
        res = filter_wiki_results(
            {
                "Notre-Dame tripadvisor page": "A cathedral in Paris.",
                "Notre-Dame Cathedral": "A Gothic cathedral in Paris.",
            }
        )
        assert "Notre-Dame tripadvisor page" not in res
        assert "Notre-Dame Cathedral" in res

    def test_vlm_exact_match_short_circuits(self):
        """Точное совпадение с vlm_name → возвращаем только его."""
        res = filter_wiki_results(
            {
                "Statue of Liberty": "A statue in New York.",
                "National Monument": "A monument in the center.",
            },
            vlm_name="Statue of Liberty",
        )
        assert res == {"Statue of Liberty": "A statue in New York."}

    def test_hint_relevance_filters_unrelated(self):
        """Статья без общего значимого слова с подсказками отбрасывается."""
        res = filter_wiki_results(
            {
                "Cologne Cathedral": "A Gothic cathedral in Cologne.",
                "Sydney Opera House": "A performing arts venue in Sydney.",
            },
            query_hints=["Cologne Cathedral Germany"],
        )
        assert "Cologne Cathedral" in res
        assert "Sydney Opera House" not in res


class TestExtractCleanName:
    """Тесты очистки названия от мусорных хвостов."""

    def test_strips_double_colon_suffix(self):
        assert extract_clean_name("Hagia Sophia :: My Blog") == "Hagia Sophia"

    def test_strips_site_suffix_after_dash(self):
        assert extract_clean_name("Hagia Sophia - Klook Australia") == "Hagia Sophia"

    def test_keeps_architectural_name(self):
        assert (
            extract_clean_name("Notre-Dame Cathedral | Klook") == "Notre-Dame Cathedral"
        )

    def test_truncates_long_name_without_arch_term(self):
        out = extract_clean_name("alpha beta gamma delta epsilon zeta eta theta")
        assert out == "alpha beta gamma delta"


class TestBuildVlmMessages:
    """Тесты построения промпта извлечения названия."""

    def test_no_hint_structure(self):
        msgs = build_vlm_messages("data:image/jpeg;base64,AAAA")
        assert len(msgs) == 1
        content = msgs[0]["content"]
        assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,AAAA"
        # текст-вопрос просит короткое название и упоминает 'unknown'
        assert "unknown" in content[-1]["text"].lower()
        # без подсказок блока Hint нет
        assert "Hint" not in content[-1]["text"]

    def test_hint_included(self):
        msgs = build_vlm_messages("data:uri", hint="- Colosseum\n- Rome")
        text = msgs[0]["content"][-1]["text"]
        assert "Hint" in text
        assert "Colosseum" in text
