# src/services/internet_search.py
"""
Обработка результатов интернет-поиска достопримечательностей.

Чистые функции без состояния сервиса: фильтрация Wikipedia-результатов,
очистка названий, проверка ответов VLM и построение промпта для извлечения
названия. Используются оркестратором AITourGuide в fallback-поиске.
"""

import asyncio
import logging
import re
from io import BytesIO

import httpx
from PIL import Image

from .image_utils import image_to_base64_data_uri, to_pil_image
from .scoring import parse_logprobs_p_yes, text_response_to_p_yes
from .search_filters import (
    ARCHITECTURAL_TERMS,
    DESC_NOISE_PREFIXES,
    SEARCH_NOISE_TOKENS,
    SITE_INDICATORS,
    TAIL_NOISE_RE,
    VLM_RESPONSE_NOISE,
)
from .yandex_search import WikipediaService

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
                f"Отфильтровано по описанию: '{name}' -> '{first_sentence[:80]}'"
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
    #    чтобы не захватить лишнее: "Hagia Sophia Grand Mosque" -> "Sophia
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


class InternetSearchService:
    """
    Fallback-поиск достопримечательности в интернете, когда реранкер не уверен.

    Пайплайн: Yandex Image Search -> уточнение названия через VLM ->
    Wikipedia -> выбор статьи -> VLM-верификация найденного объекта.
    Держит ссылки на vLLM-клиент, YandexSearchService и конфиг сервиса.
    """

    DEFAULT_MAX_RESULTS = 5

    def __init__(self, vllm_client, yandex_service, config) -> None:
        self.vllm_client = vllm_client
        self.yandex_service = yandex_service
        self.config = config

    def _yandex_search_sync(self, image: "str | bytes | Image.Image") -> set | None:
        """Синхронный вызов Yandex Image Search (переиспользуемая сессия)."""
        if self.yandex_service is None:
            return None
        return self.yandex_service.search_by_image(
            image, num_results=self.DEFAULT_MAX_RESULTS
        )

    async def search(
        self,
        image: "Image.Image | str | bytes",
        retrieved_descs: list[str],
        retrieved_names: list[str],
        fallback_name: str,
        timeout: float = 90.0,
    ) -> dict:
        """
        Ищет информацию о достопримечательности через Yandex + Wikipedia.

        Возвращает словарь с полями found, name, description, query, confidence.
        При провале — top-1 из retrieval (is_fallback_retrieval=True).
        """
        result = {
            "found": False,
            "name": fallback_name,
            "description": "",
            "query": None,
            "confidence": 0.5,
            # True если интернет-поиск провалился и вернули top-1 из retrieval.
            "is_fallback_retrieval": False,
        }

        if self.yandex_service is None:
            logger.warning("Yandex API ключи не настроены, поиск пропущен")
            return result

        # Нормализуем image в PIL для VLM (None, если прочитать не удалось)
        try:
            pil_image: Image.Image | None = to_pil_image(image)
        except Exception:
            pil_image = None

        try:
            found = await asyncio.wait_for(
                self._do_search(image, pil_image), timeout=timeout
            )
            if found:
                result["found"] = True
                result["name"] = found["name"]
                result["description"] = found["description"]
                result["query"] = found["query"]
                result["confidence"] = found["confidence"]
                return result
        except asyncio.TimeoutError:
            logger.warning(f"Таймаут интернет-поиска ({timeout}с)")
        except Exception as e:
            logger.error(f"Ошибка интернет-поиска: {e}")

        # Fallback: возвращаем top-1 из retrieval
        if retrieved_names and retrieved_names[0]:
            result["found"] = True
            result["name"] = retrieved_names[0]
            result["query"] = [retrieved_names[0]]
            result["description"] = retrieved_descs[0] if retrieved_descs else ""
            # confidence ниже чем при слабом интернет-результате
            # т.к. retrieval-fallback означает полный провал интернет-поиска
            result["confidence"] = self.config.internet_confidence_fallback_retrieval
            result["is_fallback_retrieval"] = True

        return result

    async def _do_search(
        self,
        image: "Image.Image | str | bytes",
        pil_image: "Image.Image | None",
    ) -> dict | None:
        """
        Yandex + VLM-уточнение названия + Wikipedia + выбор best_key.

        Вызывается под таймаутом через asyncio.wait_for().

        Returns:
            Словарь {name, description, query, confidence} или None.
        """
        # Кодируем query-изображение в base64 один раз —
        # используется и в этапе 1, и в этапе 2 VLM.
        image_uri: str | None = (
            image_to_base64_data_uri(pil_image) if pil_image is not None else None
        )

        # 1. Yandex Image Search (в потоке) и VLM этап 1 (без подсказок)
        #    запускаются параллельно — этап 1 не зависит от Yandex.
        #    Если skip_internet_search_stage1=True — этап 1 пропускается.
        async def _vlm_stage1_no_hints() -> str | None:
            """VLM этап 1: название без подсказок."""
            if image_uri is None or self.config.skip_internet_search_stage1:
                return None
            try:
                response = await self.vllm_client.chat_completion(
                    messages=build_vlm_messages(image_uri, hint=None),
                    max_tokens=48,
                    temperature=0.0,
                )
                raw = str(response["choices"][0]["message"]["content"]).strip()
                logger.debug(f"VLM этап 1 (без подсказок, параллельно): {raw!r}")
                return validate_vlm_answer(raw)
            except Exception as e:
                logger.warning(f"Ошибка VLM этап 1 (параллельно): {e}")
                return None

        yandex_task = asyncio.create_task(
            asyncio.to_thread(self._yandex_search_sync, image)
        )
        vlm_stage1_task = asyncio.create_task(_vlm_stage1_no_hints())

        names_raw, vlm_stage1_result = await asyncio.gather(
            yandex_task, vlm_stage1_task
        )

        if not names_raw:
            logger.info("Yandex не вернул результатов")
            return None

        # Сортируем по длине: короткие названия обычно точнее
        page_titles = sorted(names_raw, key=len)

        # 2. Определяем vlm_name: этап 1 или (если пусто) этап 2 с pageTitle.
        if image_uri is None:
            logger.warning(
                "pil_image недоступен, VLM-шаг пропущен — "
                "поиск по pageTitle без уточнения от Qwen"
            )
        vlm_name: str | None = None
        if vlm_stage1_result is not None:
            vlm_name = vlm_stage1_result
            logger.info(f"VLM извлёк название (этап 1, параллельно): {vlm_name!r}")
        elif image_uri is not None:
            if not page_titles:
                logger.info("VLM не смог определить название (нет pageTitle)")
            else:
                titles_str = "\n".join(f"- {t}" for t in page_titles[:10])
                try:
                    response2 = await self.vllm_client.chat_completion(
                        messages=build_vlm_messages(image_uri, hint=titles_str),
                        max_tokens=48,
                        temperature=0.0,
                    )
                    raw2 = str(response2["choices"][0]["message"]["content"]).strip()
                    logger.debug(f"VLM этап 2 (с подсказками): {raw2!r}")
                    result2 = validate_vlm_answer(raw2)
                    if result2:
                        vlm_name = result2
                        logger.info(f"VLM извлёк название (этап 2): {vlm_name!r}")
                    else:
                        logger.info(
                            "VLM не смог определить название достопримечательности"
                        )
                except Exception as e:
                    logger.warning(f"Ошибка VLM этап 2: {e}")

        # 3. Wikipedia-поиск. vlm_name (если есть) первым, pageTitle — запасные.
        search_names: set = (
            {vlm_name} | set(page_titles) if vlm_name else set(page_titles)
        )
        async with WikipediaService(language="ru") as wiki:
            wiki_result = await wiki.get_landmark_info(search_names)

        if not any(wiki_result.values()):
            return None

        # pageTitle — подсказки для фильтрации; vlm_name пропускает hint-фильтр.
        filtered = filter_wiki_results(
            wiki_result,
            query_hints=page_titles,
            vlm_name=vlm_name,
        )
        if not filtered:
            return None

        # best_key с дифференцированным confidence:
        # точное совпадение с vlm_name / частичное / ключ с минимумом слов.
        best_key: str
        match_confidence: float
        if vlm_name and vlm_name in filtered:
            best_key = vlm_name
            match_confidence = self.config.internet_confidence_exact
        else:
            partial_match: str | None = None
            if vlm_name:
                for key in filtered:
                    vl = vlm_name.lower()
                    kl = key.lower()
                    if vl in kl or kl in vl:
                        partial_match = key
                        break
            if partial_match is not None:
                best_key = partial_match
                match_confidence = self.config.internet_confidence_partial
            else:
                best_key = min(filtered.keys(), key=lambda n: len(n.split()))
                match_confidence = self.config.internet_confidence_fallback_wiki
                logger.debug(
                    f"Fallback к min(len): выбран '{best_key}' "
                    f"из {list(filtered.keys())}"
                )

        # Даже vlm_name может содержать мусорные суффиксы — чистим всегда.
        best_name = extract_clean_name(best_key)
        logger.info(f"Интернет-поиск: best_key='{best_key}' -> best_name='{best_name}'")

        return {
            "name": best_name,
            "description": filtered[best_key],
            "query": page_titles,
            "confidence": match_confidence,
        }

    async def _fetch_image_from_url(self, url: str) -> "Image.Image | None":
        """
        Скачивает изображение по URL и возвращает PIL Image (или None).

        User-Agent совместим с политикой доступа Wikimedia
        (upload.wikimedia.org блокирует запросы без корректного User-Agent).
        """
        try:
            headers = {
                "User-Agent": (
                    "AITourGuide/1.0 (slesarenko221999@gmail.com) educational bot"
                )
            }
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(15.0),
                headers=headers,
                follow_redirects=True,
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            logger.warning(f"Не удалось скачать thumbnail {url}: {e}")
            return None

    async def verify(
        self,
        image: "Image.Image",
        landmark_name: str,
        description: str,
        wiki_service: "WikipediaService | None" = None,
    ) -> float:
        """
        Верифицирует найденный объект через VLM, возвращает p_yes ∈ [0, 1].

        Если доступен Wikipedia thumbnail — промпт Query+Candidate (формат
        обучения реранкера, точнее). Иначе — фото + текстовое описание.
        landmark_name передаётся до перевода (VLM обучен на английском).
        """
        try:
            query_uri = image_to_base64_data_uri(image)
            caption = (
                description[: self.config.caption_max_length] if description else ""
            )

            candidate_uri: str | None = None
            if wiki_service is not None:
                thumb_url = await wiki_service.get_thumbnail_url(landmark_name)
                if thumb_url:
                    thumb_img = await self._fetch_image_from_url(thumb_url)
                    if thumb_img is not None:
                        candidate_uri = image_to_base64_data_uri(thumb_img)
                        logger.info(f"Wikipedia thumbnail скачан для '{landmark_name}'")
                    else:
                        logger.warning(
                            f"Не удалось скачать thumbnail для "
                            f"'{landmark_name}': {thumb_url}"
                        )

            if candidate_uri is not None:
                # Формат обучения: Query Photo + Candidate Photo (thumbnail).
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Query Photo:"},
                            {"type": "image_url", "image_url": {"url": query_uri}},
                            {"type": "text", "text": "Candidate Photo:"},
                            {"type": "image_url", "image_url": {"url": candidate_uri}},
                            {
                                "type": "text",
                                "text": (
                                    f"Question: Are these photos showing the same "
                                    f'landmark: "{landmark_name}"?\n'
                                    f"Candidate details: {caption}\n"
                                    f"Answer only with Yes or No."
                                ),
                            },
                        ],
                    }
                ]
                logger.info(
                    f"VLM верификация: thumbnail режим "
                    f"(Query+Candidate) для '{landmark_name}'"
                )
            else:
                # Thumbnail недоступен: одно фото + название и описание из Wikipedia.
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Photo:"},
                            {
                                "type": "image_url",
                                "image_url": {"url": query_uri},
                            },
                            {
                                "type": "text",
                                "text": (
                                    f"Question: Is the landmark shown in "
                                    f'this photo "{landmark_name}"?\n'
                                    f"Description: {caption}\n"
                                    f"Answer only with Yes or No."
                                ),
                            },
                        ],
                    }
                ]
                logger.info(
                    f"VLM верификация: текстовый режим "
                    f"(нет thumbnail) для '{landmark_name}'"
                )

            response = await self.vllm_client.chat_completion(
                messages=messages,
                max_tokens=1,
                temperature=0.0,
                logprobs=True,
                top_logprobs=20,
            )

            p_yes = parse_logprobs_p_yes(response)
            if p_yes is None:
                p_yes = text_response_to_p_yes(response)
                logger.info(
                    f"VLM верификация (logprobs недоступны, текст fallback): "
                    f"'{landmark_name}' -> p_yes={p_yes}"
                )
            else:
                logger.info(
                    f"VLM верификация (logprobs): '{landmark_name}' -> p_yes={p_yes:.4f}"
                )
            return p_yes

        except Exception as e:
            logger.warning(f"Ошибка VLM верификации для '{landmark_name}': {e}")
            return 0.5
