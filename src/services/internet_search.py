# src/services/internet_search.py
"""
Обработка результатов интернет-поиска достопримечательностей.

Чистые функции без состояния сервиса: фильтрация Wikipedia-результатов,
очистка названий, проверка ответов VLM и построение промпта для извлечения
названия. Используются оркестратором AITourGuide в fallback-поиске.
"""

import logging
import re

from .search_filters import (
    ARCHITECTURAL_TERMS,
    DESC_NOISE_PREFIXES,
    SEARCH_NOISE_TOKENS,
    SITE_INDICATORS,
    TAIL_NOISE_RE,
    VLM_RESPONSE_NOISE,
)

logger = logging.getLogger(__name__)


def filter_wiki_results(
    wiki_result: dict[str, str],
    query_hints: list[str] | None = None,
    vlm_name: str | None = None,
) -> dict[str, str]:
    """
    Фильтрует результаты Wikipedia:
    - убирает шумовые названия (SEARCH_NOISE_TOKENS)
    - убирает описания которые явно не про достопримечательность
    - если переданы query_hints — отбрасывает статьи чьё название
      не имеет ни одного общего значимого слова с подсказками
      (защита от нерелевантных fulltext-результатов)
    - статья с именем vlm_name всегда проходит hint_words фильтр
      (VLM уже подтвердил релевантность)
    - даёт приоритет архитектурным объектам (ARCHITECTURAL_TERMS)

    Возвращает пустой словарь если все результаты отфильтрованы.

    Args:
        wiki_result: Результаты Wikipedia {название: описание}
        query_hints: Список pageTitle для проверки релевантности
        vlm_name: Название от VLM — пропускает hint_words фильтр
    """
    # Собираем значимые слова из подсказок (длина >= 4, не стоп-слова)
    hint_words: set = set()
    if query_hints:
        for hint in query_hints:
            for w in hint.lower().split():
                if len(w) >= 4:
                    hint_words.add(w.strip(".,!?;:\"'"))

    # Нормализованное vlm_name для сравнения
    vlm_name_lower = vlm_name.lower().strip() if vlm_name else None

    filtered = {}
    for name, desc in wiki_result.items():
        if not desc:
            continue
        # Фильтр по названию
        if any(x in name.lower() for x in SEARCH_NOISE_TOKENS):
            logger.debug(f"Отфильтровано по названию: '{name}'")
            continue
        # Фильтр по первому предложению описания
        first_sentence = desc.split(".")[0].lower()
        if any(x in first_sentence for x in DESC_NOISE_PREFIXES):
            logger.debug(
                f"Отфильтровано по описанию: '{name}' → '{first_sentence[:80]}'"
            )
            continue
        # Фильтр по релевантности к подсказкам:
        # название статьи должно иметь хотя бы одно общее слово
        # с pageTitle (защита от нерелевантных fulltext-результатов).
        # Исключение: статья с именем от VLM всегда проходит —
        # VLM уже подтвердил её релевантность визуально.
        is_vlm_match = (
            vlm_name_lower is not None and name.lower().strip() == vlm_name_lower
        )
        if hint_words and not is_vlm_match:
            name_words = {
                w.strip(".,!?;:\"'") for w in name.lower().split() if len(w) >= 4
            }
            if name_words and not name_words & hint_words:
                logger.debug(
                    f"Отфильтровано по релевантности: '{name}' "
                    f"(нет общих слов с подсказками)"
                )
                continue
        filtered[name] = desc

    # Если vlm_name точно есть в filtered — возвращаем только его.
    # Это предотвращает выбор общих статей ('National monument')
    # вместо конкретного названия от VLM ('Statue of Liberty').
    if vlm_name_lower:
        vlm_exact = {
            name: desc
            for name, desc in filtered.items()
            if name.lower().strip() == vlm_name_lower
        }
        if vlm_exact:
            return vlm_exact

    # Приоритет архитектурным объектам у которых есть собственное имя
    # (название длиннее одного архитектурного термина).
    # Исключаем чисто родовые названия типа 'National monument',
    # 'Memorial', 'Museum' — они не несут конкретики.
    priority = {}
    for name, desc in filtered.items():
        name_lower = name.lower()
        for term in ARCHITECTURAL_TERMS:
            if term in name_lower:
                # Проверяем что есть слова помимо самого термина
                other_words = name_lower.replace(term, "").split()
                other_words = [
                    w.strip(".,!?;:\"'")
                    for w in other_words
                    if len(w.strip(".,!?;:\"'")) > 2
                ]
                if other_words:
                    priority[name] = desc
                break
    if priority:
        return priority
    # Возвращаем filtered (может быть пустым) — не откатываемся к
    # нефильтрованному wiki_result чтобы не пропустить мусор
    return filtered


def needs_translation(text: str) -> bool:
    """
    Определяет нужен ли перевод текста на русский.

    Считает долю кириллических букв среди всех букв.
    Если < 30% — текст считается английским и требует перевода.

    Простая проверка "есть ли хоть одна кириллица" не работает:
    Wikipedia EN может содержать транслитерацию в скобках
    (напр. "Saint Isaac's Cathedral (Russian: Исаакиевский собор)").
    """
    if not text:
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    cyrillic = sum(1 for c in letters if "Ѐ" <= c <= "ӿ")
    ratio = cyrillic / len(letters)
    return ratio < 0.3


def extract_clean_name(raw_name: str) -> str:
    """
    Очищает название от мусорных хвостов и сокращает до сути.

    Обрабатывает случаи когда WikipediaService возвращает исходный
    pageTitle как ключ (например "Hagia Sophia Ticket - Klook Australia"),
    а не реальное название Wikipedia-статьи.
    """
    name = raw_name

    # 1. Убираем суффиксы после :: (личные блоги, авторы)
    name = re.split(r"\s*::\s*", name)[0].strip()

    # 2. Убираем суффиксы после | (сайты, агрегаторы)
    name = re.split(r"\s*\|\s*", name)[0].strip()

    # 3. Убираем суффиксы после - (агрегаторы, сайты, страны)
    #    Но только если вторая часть — сайт/страна или первая содержит
    #    архитектурный термин (чтобы не обрезать "Notre-Dame")
    parts = re.split(r"\s+[-–—]\s+", name)
    if len(parts) > 1:
        first = parts[0].strip()
        second = parts[1].strip().lower()
        if any(s in second for s in SITE_INDICATORS):
            name = first
        elif any(t in first.lower() for t in ARCHITECTURAL_TERMS):
            name = first

    # 4. Убираем слова-мусор в конце (ticket, билет, tour и т.д.)
    name = TAIL_NOISE_RE.sub("", name).strip()

    # 5. Если есть архитектурный термин — берём контекст вокруг него.
    #    Берём до 2 слов до термина + сам термин (без слова после —
    #    чтобы не захватить лишнее: "Hagia Sophia Grand Mosque" → "Sophia
    #    Grand Mosque" вместо "Hagia Sophia").
    words = name.split()
    for i, w in enumerate(words):
        if w.lower() in ARCHITECTURAL_TERMS:
            start = max(0, i - 2)
            arch_name = " ".join(words[start : i + 1]).strip()
            return arch_name if len(arch_name) >= 3 else name

    # 6. Если нет архитектурного термина но строка длинная (> 6 слов) —
    #    берём первые 4 слова. Длинные ключи Wikipedia типа
    #    "Istanbul meta turistica... Hagia sophia, Istanbul, Byzantine"
    #    не должны возвращаться целиком.
    if len(words) > 6:
        name = " ".join(words[:4]).strip()

    return name


def validate_vlm_answer(answer: str) -> str | None:
    """
    Проверяет и очищает ответ VLM на запрос названия достопримечательности.

    Returns:
        Очищенное название или None если ответ является мусором.
    """
    answer = answer.strip("\"'«»").strip()
    if not answer or answer.lower() in ("unknown", "none", ""):
        return None
    # Фильтр туристического мусора
    answer_lower = answer.lower()
    if any(noise in answer_lower for noise in VLM_RESPONSE_NOISE):
        logger.warning(f"VLM вернул туристический мусор: {answer!r}")
        return None
    # Слишком длинный ответ — скорее всего не название
    if len(answer.split()) > 8:
        logger.warning(
            f"VLM вернул слишком длинный ответ "
            f"({len(answer.split())} слов): {answer!r}"
        )
        return None
    return answer


def build_vlm_messages(image_uri: str, hint: str | None = None) -> list[dict]:
    """
    Формирует промпт для извлечения названия достопримечательности.

    Args:
        image_uri: base64 data URI изображения
        hint: Подсказки из pageTitle (None = без подсказок)

    Returns:
        Список сообщений в формате OpenAI API
    """
    extra = (
        f"Hint — reverse image search page titles (may contain noise):\n{hint}\n\n"
        if hint
        else ""
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Photo:"},
                {
                    "type": "image_url",
                    "image_url": {"url": image_uri},
                },
                {
                    "type": "text",
                    "text": (
                        f"{extra}"
                        "What is the exact name of the landmark "
                        "shown in the photo? Reply with a short, "
                        "precise name suitable for a Wikipedia "
                        "search (e.g. 'Notre-Dame de Paris'). "
                        "If you cannot identify the landmark, "
                        "reply with 'unknown'."
                    ),
                },
            ],
        }
    ]
