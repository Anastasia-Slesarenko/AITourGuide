# tests/unit/test_scoring.py
"""Юнит-тесты извлечения P(yes) из ответа VLM (src/services/scoring.py)."""

import pytest

from src.services.scoring import parse_logprobs_p_yes, text_response_to_p_yes


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
