# src/services/scoring.py
"""Извлечение вероятности P(yes) из ответа VLM на бинарный yes/no-вопрос."""

import math

# Варианты токенов Yes/No для парсинга logprobs
YES_VARIANTS = frozenset({"yes", "Yes", "YES", " yes", " Yes"})
NO_VARIANTS = frozenset({"no", "No", "NO", " no", " No"})


def parse_logprobs_p_yes(response: dict) -> float | None:
    """
    Извлекает p_yes из logprobs ответа vLLM.

    Единая точка парсинга logprobs для VLM reranking и верификации
    интернет-результата.

    Returns:
        p_yes ∈ [0, 1] или None если logprobs недоступны
        (None — вызывающий код должен использовать текстовый fallback).
    """
    logprobs_data = response.get("choices", [{}])[0].get("logprobs", {})
    if not logprobs_data or not logprobs_data.get("content"):
        return None

    top_lp = logprobs_data["content"][0].get("top_logprobs", [])

    logit_yes = None
    logit_no = None
    for item in top_lp:
        token = item.get("token", "")
        logprob = item.get("logprob", -100)
        if logit_yes is None and token in YES_VARIANTS:
            logit_yes = logprob
        elif logit_no is None and token in NO_VARIANTS:
            logit_no = logprob

    if logit_yes is not None and logit_no is not None:
        max_logit = max(logit_yes, logit_no)
        exp_yes = math.exp(logit_yes - max_logit)
        exp_no = math.exp(logit_no - max_logit)
        return exp_yes / (exp_yes + exp_no)
    elif logit_yes is not None:
        return 1.0
    elif logit_no is not None:
        return 0.0
    return None


def text_response_to_p_yes(response: dict) -> float:
    """
    Fallback: извлекает p_yes из текстового ответа VLM (без logprobs).

    Returns:
        0.9 если ответ начинается с "yes", 0.1 если "no", иначе 0.5.
    """
    text = (
        str(response.get("choices", [{}])[0].get("message", {}).get("content", ""))
        .strip()
        .lower()
    )
    if text.startswith("yes"):
        return 0.9
    elif text.startswith("no"):
        return 0.1
    return 0.5
