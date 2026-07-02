# src/services/search_filters.py
"""
Константы и утилиты для фильтрации результатов интернет-поиска.

Используются в YandexSearchService, WikipediaService и AITourGuide.
"""

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Числовые константы для поиска
# ---------------------------------------------------------------------------

OPENSEARCH_LIMIT: int = 5
FULLTEXT_SEARCH_LIMIT: int = 5
MAX_QUERY_WORDS_FOR_VARIANTS: int = 4
MIN_RELEVANCE_RATIO: float = 0.5

# ---------------------------------------------------------------------------
# Стоп-слова для проверки релевантности
# ---------------------------------------------------------------------------

STOPWORDS_EN: frozenset = frozenset(
    {
        "this",
        "that",
        "there",
        "their",
        "what",
        "which",
        "from",
        "with",
        "without",
        "have",
        "been",
        "were",
        "will",
        "would",
        "could",
        "should",
    }
)

STOPWORDS_RU: frozenset = frozenset(
    {
        "это",
        "что",
        "который",
        "такой",
        "также",
        "более",
        "было",
        "есть",
        "были",
        "будет",
        "может",
        "должен",
    }
)

STOPWORDS: frozenset = STOPWORDS_EN | STOPWORDS_RU

# ---------------------------------------------------------------------------
# Шумовые токены — результаты с такими подстроками в названии отбрасываются
# ---------------------------------------------------------------------------

SEARCH_NOISE_TOKENS: frozenset = frozenset(
    {
        # Файлы и медиа
        ".jpg",
        ".jpeg",
        ".png",
        # Агрегаторы и сервисы
        "panoramio",
        "georama",
        "greekreporter",
        "youtube",
        "cnn ",
        "klook",
        "viator",
        "getyourguide",
        "tripadvisor",
        # Туристический мусор
        "honeymoon",
        "travel",
        "tourist",
        "туризм в",
        "экскурси",
        "экскурсии",
        "ticket",
        "билет",
        "билеты",
        "opening time",
        "when i can visit",
        "amazing ancient cities",
        "города и страны",
        # Политика, религия, абстракции
        "lgbtq",
        "religious beliefs",
        "religion in",
        "генеральный план",
        "администрации",
        # Медиа-контент
        "слайд-шоу",
        "гимн",
        "background for slides",
        "фон для слайдов",
        "time period",
        "период времени",
        # Книги, романы, фильмы, игры
        "роман",
        "novel",
        "фантастический",
        "science fiction",
        "фильм",
        "сериал",
        "video game",
        "игра",
        "альбом",
        "album",
        "песня",
        "song",
        "симфония",
        "symphony",
        "опера",
        "opera no",
        "концерт",
        "concerto",
        # Личные блоги и соцсети
        ":: ",
        " - youtube",
        "yandex maps",
        "yandex.maps",
        "instagram",
        "facebook",
    }
)

# ---------------------------------------------------------------------------
# Архитектурные термины — результаты с такими словами получают приоритет
# ---------------------------------------------------------------------------

ARCHITECTURAL_TERMS: frozenset = frozenset(
    {
        # Английские
        "cathedral",
        "church",
        "temple",
        "mosque",
        "synagogue",
        "palace",
        "castle",
        "fortress",
        "tower",
        "monument",
        "memorial",
        "museum",
        "bridge",
        "gate",
        "basilica",
        "colosseum",
        "amphitheater",
        "amphitheatre",
        "arena",
        "forum",
        "pantheon",
        "acropolis",
        "parthenon",
        "pyramid",
        # Русские
        "собор",
        "церковь",
        "храм",
        "мечеть",
        "синагога",
        "дворец",
        "замок",
        "крепость",
        "башня",
        "мост",
        "памятник",
        "мемориал",
        "музей",
        "монастырь",
        "часовня",
        "ворота",
        "площадь",
        "кремль",
        "цитадель",
        "колизей",
        "амфитеатр",
        "форум",
        "пантеон",
        "акрополь",
        "пирамида",
    }
)

# ---------------------------------------------------------------------------
# Паттерны исключения для _is_likely_landmark()
# ---------------------------------------------------------------------------

LANDMARK_EXCLUDE_PATTERNS: tuple = (
    # Музыка и искусство (не архитектура)
    r"\bsymphony\b",
    r"\bopera\s+no\b",
    r"\bconcerto\b",
    r"\bsonata\b",
    r"\bquartet\b",
    r"\bphilharmonic\b",
    r"\borchestra\b",
    r"\bconductor\b",
    # Люди (имена с фамилиями без архитектурных терминов)
    r"^(karl|charles|ludwig|friedrich|wilhelm|johann|franz)\s+\w+$",
    r"^(карл|шарль|людвиг|фридрих|вильгельм|иоганн|франц)\s+\w+$",
    # Книги, игры и публикации
    r"\bvolume\b",
    r"\bedition\b",
    r"\bchapter\b",
    r"\bbook\b",
    r"\bgame\b",
    r"\bvideo game\b",
    r"\brole.?playing\b",
    r"\bstock (image|photo)\b",
    r"стоковое изображение",
    # Общие категории и сервисы
    r"panoramio",
    r"geograph\.org",
    r"wikimedia",
    r"honeymoon",
    r"travel guide",
    r"tourism",
    # Абстрактные концепции
    r"\breligion in\b",
    r"\breligious beliefs\b",
    r"\blgbtq\b",
    r"\bhistory of\b",
    # Улицы и адреса (обычно не достопримечательности)
    r"^\d+\s+\w+\s+(street|avenue|road|boulevard)",
    r"^улица\s+",
    r"^проспект\s+",
    r"^бульвар\s+",
)

# ---------------------------------------------------------------------------
# Позитивные индикаторы достопримечательностей для _is_likely_landmark()
# ---------------------------------------------------------------------------

LANDMARK_POSITIVE_PATTERNS: tuple = (
    # Типы зданий
    r"\b(cathedral|church|temple|mosque|synagogue|chapel)\b",
    r"\b(собор|церковь|храм|мечеть|синагога|часовня)\b",
    r"\b(palace|castle|fortress|tower|bridge|gate)\b",
    r"\b(дворец|замок|крепость|башня|мост|ворота)\b",
    r"\b(museum|gallery|theater|theatre|opera house)\b",
    r"\b(музей|галерея|театр|оперный)\b",
    r"\b(monument|memorial|statue|square|plaza)\b",
    r"\b(памятник|мемориал|статуя|площадь)\b",
    # Архитектурные стили
    r"\b(gothic|baroque|renaissance|romanesque|neoclassical)\b",
    r"\b(готический|барокко|ренессанс|романский|неоклассический)\b",
)

# ---------------------------------------------------------------------------
# Шумовые префиксы первого предложения Wikipedia-описания
# ---------------------------------------------------------------------------

DESC_NOISE_PREFIXES: frozenset = frozenset(
    {
        "туризм",
        "tourism",
        "путешестви",
        "travel",
        "экономик",
        "economy",
        "отрасль",
        "industry",
        "список",
        "list of",
        "категория",
        "category",
        "история ",
        "history of",
    }
)

# ---------------------------------------------------------------------------
# Мусорные слова в ответе VLM — туристические фразы вместо названия
# ---------------------------------------------------------------------------

VLM_RESPONSE_NOISE: frozenset = frozenset(
    {
        "экскурси",
        "тур ",
        "туры",
        "посетить",
        "visit",
        "tour ",
        "tours",
        "tickets",
        "билет",
        "билеты",
        "расписание",
        "schedule",
        "opening",
        "hours",
        "как добраться",
        "getting there",
        "отзыв",
        "review",
        "купить",
        "buy ",
        "price",
        "цена",
        "стоимость",
    }
)

# ---------------------------------------------------------------------------
# Индикаторы сайтов/агрегаторов в суффиксах названий
# ---------------------------------------------------------------------------

SITE_INDICATORS: frozenset = frozenset(
    {
        "klook",
        "viator",
        "getyourguide",
        "tripadvisor",
        "wikipedia",
        "wikimedia",
        "youtube",
        "instagram",
        "australia",
        "russia",
        "turkey",
        "france",
        "italy",
        "россия",
        "турция",
        "франция",
        "италия",
    }
)

# ---------------------------------------------------------------------------
# Regex для удаления мусорных хвостов из названий (компилируем один раз)
# ---------------------------------------------------------------------------

TAIL_NOISE_RE: re.Pattern = re.compile(
    r"\s+(ticket|tickets|билет|билеты|tour|tours|"
    r"тур|туры|visit|посетить|купить|buy|price|цена).*$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Специальные ключевые слова для проверки релевантности Wikipedia
# ---------------------------------------------------------------------------

RELEVANCE_SPECIAL_KEYWORDS: frozenset = frozenset(
    {
        "cathedral",
        "church",
        "temple",
        "mosque",
        "synagogue",
        "palace",
        "castle",
        "fortress",
        "tower",
        "bridge",
        "собор",
        "церковь",
        "храм",
        "мечеть",
        "синагога",
        "дворец",
        "замок",
        "крепость",
        "башня",
        "мост",
    }
)

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

MIN_WORD_LENGTH_FOR_RELEVANCE: int = 4


def is_likely_landmark(name: str) -> bool:
    """
    Проверяет, похоже ли название на достопримечательность.
    Фильтрует людей, музыку, книги и другие нерелевантные результаты.

    Args:
        name: Очищенное название

    Returns:
        True если название похоже на достопримечательность
    """
    name_lower = name.lower()

    for pattern in LANDMARK_EXCLUDE_PATTERNS:
        if re.search(pattern, name_lower, re.IGNORECASE):
            return False

    for pattern in LANDMARK_POSITIVE_PATTERNS:
        if re.search(pattern, name_lower, re.IGNORECASE):
            return True

    # Нет явных индикаторов и нет исключений — пропускаем как возможное место
    return True


def is_relevant(
    query: str,
    extract: str,
    min_relevance_ratio: float = 0.5,
) -> bool:
    """
    Проверяет релевантность найденного Wikipedia-описания запросу.

    Args:
        query: Поисковый запрос
        extract: Найденное описание
        min_relevance_ratio: Минимальная доля совпадающих слов

    Returns:
        True если описание релевантно запросу
    """
    extract_lower = extract.lower()

    # Проверяем что имя собственное из запроса есть в первом предложении
    first_sentence = extract_lower.split(".")[0]
    query_words_all = query.split()
    proper_words = [
        w.lower()
        for w in query_words_all[:2]
        if len(w) >= MIN_WORD_LENGTH_FOR_RELEVANCE
    ]
    if proper_words:
        found_in_first = any(w in first_sentence for w in proper_words)
        if not found_in_first:
            logger.debug(
                f"Отклонено '{query}': имя '{proper_words}' "
                f"не найдено в первом предложении: '{first_sentence[:100]}'"
            )
            return False

    # Извлекаем значимые слова из запроса
    pattern = rf"\b[a-zа-я]{{{MIN_WORD_LENGTH_FOR_RELEVANCE},}}\b"
    query_words = set(re.findall(pattern, query.lower()))
    query_words = query_words - STOPWORDS

    if not query_words:
        return True

    # Проверяем специальные ключевые слова
    query_special = query_words & RELEVANCE_SPECIAL_KEYWORDS
    if query_special:
        special_found = [w for w in query_special if w in extract_lower]
        if not special_found:
            logger.debug(
                f"Отклонено '{query}': специальные слова "
                f"{query_special} не найдены в описании. "
                f"Описание начинается: {extract[:100]}..."
            )
            return False

    # Подсчитываем долю совпадающих слов
    matching_words = sum(1 for word in query_words if word in extract_lower)
    relevance_ratio = matching_words / len(query_words)
    is_rel = relevance_ratio >= min_relevance_ratio

    if not is_rel:
        logger.debug(
            f"Отклонено: релевантность {relevance_ratio:.2f} < "
            f"{min_relevance_ratio} "
            f"({matching_words}/{len(query_words)} слов)"
        )

    return is_rel


def clean_landmark_name(name: str) -> str:
    """
    Очищает название достопримечательности от лишнего текста.

    Удаляет префиксы File:/Image:, расширения файлов, суффиксы Wikipedia,
    содержимое в скобках, типичные хвосты после тире/пайпа.

    Args:
        name: Сырое название

    Returns:
        Очищенное название
    """
    # Удаляем префиксы File:, Image:, Category: и т.д.
    name = re.sub(
        r"^(File|Image|Category|Template):",
        "",
        name,
        flags=re.IGNORECASE,
    )

    # Удаляем расширения файлов
    name = re.sub(
        r"\.(jpg|jpeg|png|gif|svg|webp)$",
        "",
        name,
        flags=re.IGNORECASE,
    )

    # Удаляем "Wikimedia Commons", "Wikipedia", и подобное
    name = re.sub(
        r"\s*[-–—]\s*(Wikimedia Commons|Wikipedia|Wiki).*$",
        "",
        name,
        flags=re.IGNORECASE,
    )

    # Удаляем содержимое в скобках в конце
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name)

    # Удаляем типичные хвосты (после тире, пайпа, скобок)
    name = re.sub(
        r"\s*(—.*?|\|\s*.*?|\s*\[(.*?)\]|\s*—.*)$",
        "",
        name,
    )

    # Удаляем лишние пробелы и подчеркивания
    name = re.sub(r"[_\s]+", " ", name).strip()

    # Убираем дублирующиеся части: "A A, B A" → "A"
    if "," in name:
        parts = [p.strip() for p in name.split(",")]
        first = parts[0]
        if any(first.lower() in p.lower() for p in parts[1:]):
            name = first

    return name


def generate_search_variants(
    query: str,
    max_words: int = 4,
) -> list[str]:
    """
    Генерирует варианты поискового запроса для Wikipedia.

    Args:
        query: Исходный запрос
        max_words: Максимум слов в сокращённом варианте

    Returns:
        Список вариантов запроса (без дубликатов)
    """
    clean = re.sub(r"['\"]", "", query)
    clean = re.sub(r"[.,!?;:]", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    variants = [
        clean,
        " ".join(clean.split()[:max_words]),
    ]
    return [v for v in dict.fromkeys(variants) if v]
