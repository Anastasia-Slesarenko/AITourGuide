# tests/unit/test_ai_tour_guide.py
"""
Юнит-тесты логики AITourGuide.
Тестируем чистые методы без обращения к моделям и внешним сервисам.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.services.ai_tour_guide import AITourGuide
from src.services.internet_search import (
    needs_translation,
    validate_vlm_answer,
)
from src.services.scoring import (
    parse_logprobs_p_yes,
    text_response_to_p_yes,
)

# ---------------------------------------------------------------------------
# Фикстура: минимальный AITourGuide без реальных моделей
# ---------------------------------------------------------------------------


@pytest.fixture
def guide():
    """
    AITourGuide с замоканными тяжёлыми зависимостями.
    Позволяет тестировать чистую логику без загрузки SigLIP и vLLM.
    """
    with (
        patch("src.services.ai_tour_guide.LandmarkRetriever") as mock_retriever_cls,
        patch("src.services.ai_tour_guide.VLLMClient") as mock_vllm_cls,
    ):
        mock_retriever_cls.from_index_dir.return_value = MagicMock()
        mock_vllm_cls.return_value = MagicMock()

        g = AITourGuide(index_dir="/tmp/fake_index")
        yield g


# ---------------------------------------------------------------------------
# _parse_logprobs_p_yes
# ---------------------------------------------------------------------------


class TestParseLogprobsPYes:
    """Тесты парсинга P(yes) из logprobs vLLM."""

    def _make_response(self, yes_logprob=None, no_logprob=None):
        """Вспомогательный метод: строим структуру ответа vLLM."""
        top_lp = []
        if yes_logprob is not None:
            top_lp.append({"token": "Yes", "logprob": yes_logprob})
        if no_logprob is not None:
            top_lp.append({"token": "No", "logprob": no_logprob})

        return {"choices": [{"logprobs": {"content": [{"top_logprobs": top_lp}]}}]}

    def test_equal_logprobs_give_p_yes_05(self):
        """Если logit_yes == logit_no, P(yes) должен быть 0.5."""
        response = self._make_response(yes_logprob=-1.0, no_logprob=-1.0)
        p = parse_logprobs_p_yes(response)
        assert p == pytest.approx(0.5, abs=1e-6)

    def test_high_yes_logprob_gives_p_yes_near_1(self):
        """Если logit_yes >> logit_no, P(yes) должен быть близок к 1."""
        response = self._make_response(yes_logprob=0.0, no_logprob=-10.0)
        p = parse_logprobs_p_yes(response)
        assert p > 0.99

    def test_only_yes_token_gives_p_yes_1(self):
        """Если есть только Yes-токен, P(yes) = 1.0."""
        resp = self._make_response(yes_logprob=-0.5)
        assert parse_logprobs_p_yes(resp) == 1.0

    def test_only_no_token_gives_p_yes_0(self):
        """Если есть только No-токен, P(yes) = 0.0."""
        resp = self._make_response(no_logprob=-0.5)
        assert parse_logprobs_p_yes(resp) == 0.0

    def test_empty_logprobs_returns_none(self):
        """Если logprobs отсутствуют, метод возвращает None."""
        resp: dict = {"choices": [{"logprobs": {}}]}
        assert parse_logprobs_p_yes(resp) is None

    def test_no_yes_or_no_tokens_returns_none(self):
        """Если в top_logprobs нет ни Yes, ни No — возвращаем None."""
        response = {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {"top_logprobs": [{"token": "Maybe", "logprob": -1.0}]}
                        ]
                    }
                }
            ]
        }
        assert parse_logprobs_p_yes(response) is None


# ---------------------------------------------------------------------------
# _text_response_to_p_yes
# ---------------------------------------------------------------------------


class TestTextResponseToPYes:
    """Тесты текстового fallback для P(yes)."""

    def _make_text_response(self, text: str) -> dict:
        return {"choices": [{"message": {"content": text}}]}

    def test_yes_response(self):
        assert text_response_to_p_yes(self._make_text_response("Yes")) == 0.9

    def test_no_response(self):
        assert text_response_to_p_yes(self._make_text_response("No")) == 0.1

    def test_unknown_response(self):
        assert text_response_to_p_yes(self._make_text_response("Maybe")) == 0.5


# ---------------------------------------------------------------------------
# _needs_translation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _validate_vlm_answer
# ---------------------------------------------------------------------------


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
