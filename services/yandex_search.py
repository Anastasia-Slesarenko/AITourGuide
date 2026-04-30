import base64
import logging
from typing import Dict, Set, Optional, Union
import hashlib
import re
import os
import asyncio

import aiohttp
import requests
from cachetools import TTLCache

logger = logging.getLogger(__name__)

# Константы для конфигурации
MIN_LANDMARK_NAME_LENGTH = 3
DEFAULT_CACHE_TTL = 3600
DEFAULT_TIMEOUT = 100
DEFAULT_MAX_RESULTS = 5
MAX_CACHE_SIZE = 1000
MAX_CONCURRENT_REQUESTS = 5

# Константы для поиска и валидации
MIN_WORD_LENGTH_FOR_RELEVANCE = 4
OPENSEARCH_LIMIT = 5
FULLTEXT_SEARCH_LIMIT = 5
MAX_QUERY_WORDS_FOR_VARIANTS = 4
MIN_RELEVANCE_RATIO = 0.5  # Минимальная доля совпадающих слов

# Стоп-слова для проверки релевантности
STOPWORDS_EN = {
    'this', 'that', 'there', 'their', 'what', 'which',
    'from', 'with', 'without', 'have', 'been', 'were',
    'will', 'would', 'could', 'should'
}
STOPWORDS_RU = {
    'это', 'что', 'который', 'такой', 'также', 'более',
    'было', 'есть', 'были', 'будет', 'может', 'должен'
}
STOPWORDS = STOPWORDS_EN | STOPWORDS_RU


class YandexSearchService:
    """
    Сервис для поиска названий достопримечательностей
    по изображениям через Yandex API Search.
    """
    _URL_SEARCH_BY_IMAGE = (
        "https://searchapi.api.cloud.yandex.net/v2/image/"
        "search_by_image"
    )

    def __init__(
        self,
        yc_folder_id: str,
        yc_api_key: str,
        timeout: int = DEFAULT_TIMEOUT,
        max_results: int = DEFAULT_MAX_RESULTS,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL
    ) -> None:
        # Валидация обязательных параметров
        if not yc_folder_id or not yc_api_key:
            raise ValueError(
                "yc_folder_id и yc_api_key обязательны"
            )

        self.yc_folder_id = yc_folder_id
        self.yc_api_key = yc_api_key
        self.timeout = timeout
        self.max_results = max_results
        self.cache_ttl_seconds = cache_ttl_seconds

        # Кэш для результатов поиска (thread-safe TTLCache)
        self._search_cache: Union[TTLCache, Dict] = TTLCache(
            maxsize=MAX_CACHE_SIZE,
            ttl=cache_ttl_seconds
        )

        # HTTP сессия для переиспользования соединений
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Api-Key {self.yc_api_key}",
            "Content-Type": "application/json"
        })

    def close(self) -> None:
        """Явное закрытие HTTP сессии."""
        if hasattr(self, '_session') and self._session:
            self._session.close()
            self._session = None

    def __del__(self) -> None:
        """Закрытие HTTP сессии при удалении объекта."""
        self.close()

    def __enter__(self) -> 'YandexSearchService':
        """Context manager support."""
        return self

    def __exit__(self, *args) -> None:
        """Закрытие сессии при выходе из контекста."""
        self.close()

    def search_by_image(
        self,
        image: Union[str, bytes],
        num_results: int = DEFAULT_MAX_RESULTS
    ) -> Optional[Set[str]]:
        """
        Находит названия достопримечательностей как заголовки
        страниц с похожими изображениями.

        Args:
            image: Путь к изображению или изображение в байтах.
            num_results: Ограничение на кол-во результатов

        Returns:
            Названия найденных достопримечательностей
        """
        if not self.yc_api_key:
            logger.warning(
                "Yandex Search API ключ не настроен"
            )
            return None

        num_results = num_results or self.max_results
        
        # Валидация num_results
        if num_results <= 0:
            raise ValueError("num_results должен быть положительным")

        try:
            # Всегда кодируем в base64
            if isinstance(image, str):
                # Проверка существования файла
                if not os.path.exists(image):
                    logger.error(f"Файл не найден: {image}")
                    return None
                
                with open(image, "rb") as image_file:
                    image_bytes = image_file.read()
            elif isinstance(image, bytes):
                image_bytes = image
            else:
                # PIL Image - конвертируем в байты
                from io import BytesIO
                buffer = BytesIO()
                image.save(buffer, format='JPEG')
                image_bytes = buffer.getvalue()

            encoded_string = base64.b64encode(image_bytes).decode("utf-8")
            # Генерируем ключ кэша (по хешу изображения)
            image_hash = hashlib.sha256(
                encoded_string.encode()
            ).hexdigest()
            # Проверяем кэш (TTLCache thread-safe)
            if image_hash in self._search_cache:
                logger.info("Возврат из кэша")
                return self._search_cache[image_hash]

            payload = {
                "folderId": self.yc_folder_id,
                "data": encoded_string,
                # "page": 0  # Исправлено: число вместо строки
            }
            
            # Отправка POST-запроса
            logger.info("Отправка запроса к Yandex Search API")
            
            if self._session:
                response = self._session.post(
                    self._URL_SEARCH_BY_IMAGE,
                    json=payload,
                    timeout=self.timeout
                )
            else:
                # Fallback без сессии
                headers = {
                    "Authorization": f"Api-Key {self.yc_api_key}",
                    "Content-Type": "application/json"
                }
                response = requests.post(
                    self._URL_SEARCH_BY_IMAGE,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
            
            if response.status_code != 200:
                logger.error(
                    f"HTTP ошибка: {response.status_code} - "
                    f"{response.text[:200]}"
                )
                return None

            result_data = response.json()
            landmark_names = self._extract_landmark_name(
                result_data, num_results
            )
            if landmark_names:
                # Сохраняем в кэш (TTLCache thread-safe)
                self._search_cache[image_hash] = landmark_names
                logger.info(
                    f"Найдено {len(landmark_names)} названий "
                    f"достопримечательностей"
                )
                return landmark_names
            else:
                logger.warning("Нет результатов от Yandex Search API")
                return None

        except FileNotFoundError as e:
            logger.error(f"Файл изображения не найден: {e}")
            return None
        except requests.exceptions.Timeout:
            logger.error("Превышен таймаут Yandex API")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка сети при запросе к Yandex API: {e}")
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка при поиске: {e}")
            return None

    def _extract_landmark_name(
        self,
        response_data: Dict,
        num_results: int = 5
    ) -> Set[str]:
        """
        Внутрення функция: пытается достать названия
        достопримечательностей как заголовки страниц
        с похожими изображениями.

        Args:
            response_data: Ответ от Yandex API.
            num_results: Ограничение на кол-во результатов

        Returns:
            Названия найденных достопримечательностей
        """
        landmark_names: Set[str] = set()

        for query in response_data.get("images", [])[:num_results]:
            page_title = query.get("pageTitle")
            if page_title:
                # Очистка названия от лишнего текста
                clean_name = self._clean_landmark_name(page_title)
                if clean_name and len(clean_name) > MIN_LANDMARK_NAME_LENGTH:
                    # Проверяем, что название похоже на достопримечательность
                    if self._is_likely_landmark(clean_name):
                        landmark_names.add(clean_name)
                    else:
                        logger.debug(
                            f"Filtered out non-landmark: '{clean_name}'"
                        )

        return landmark_names

    def _clean_landmark_name(self, name: str) -> str:
        """Очищает название достопримечательности от лишнего текста."""
        
        # Удаляем префиксы File:, Image:, Category: и т.д.
        name = re.sub(
            r"^(File|Image|Category|Template):",
            "",
            name,
            flags=re.IGNORECASE
        )
        
        # Удаляем расширения файлов
        name = re.sub(
            r"\.(jpg|jpeg|png|gif|svg|webp)$",
            "",
            name,
            flags=re.IGNORECASE
        )
        
        # Удаляем "Wikimedia Commons", "Wikipedia", и подобное
        name = re.sub(
            r"\s*[-–—]\s*(Wikimedia Commons|Wikipedia|Wiki).*$",
            "",
            name,
            flags=re.IGNORECASE
        )
        
        # Удаляем содержимое в скобках в конце
        name = re.sub(r"\s*\([^)]*\)\s*$", "", name)
        
        # Удаляем типичные хвосты
        name = re.sub(
            r"\s*(—.*?|\|\s*.*?|\s*\[(.*?)\]|\s*—.*)$",
            "",
            name
        )

        # Удаляем лишние пробелы и подчеркивания
        name = re.sub(r"[_\s]+", " ", name).strip()

        return name
    
    def _is_likely_landmark(self, name: str) -> bool:
        """
        Проверяет, похоже ли название на достопримечательность.
        Фильтрует людей, музыку, книги и другие нерелевантные результаты.
        
        Args:
            name: Очищенное название
            
        Returns:
            True если название похоже на достопримечательность
        """
        name_lower = name.lower()
        
        # Паттерны для исключения
        exclude_patterns = [
            # Музыка и искусство (не архитектура)
            r'\bsymphony\b', r'\bopera\s+no\b', r'\bconcerto\b',
            r'\bsonata\b', r'\bquartet\b', r'\bphilharmonic\b',
            r'\borchestra\b', r'\bconductor\b',
            # Люди (имена с фамилиями без архитектурных терминов)
            r'^(karl|charles|ludwig|friedrich|wilhelm|johann|franz)\s+\w+$',
            r'^(карл|шарль|людвиг|фридрих|вильгельм|иоганн|франц)\s+\w+$',
            # Книги, игры и публикации
            r'\bvolume\b', r'\bedition\b', r'\bchapter\b', r'\bbook\b',
            r'\bgame\b', r'\bvideo game\b', r'\brole.?playing\b',
            r'\bstock (image|photo)\b', r'стоковое изображение',
            # Общие категории и сервисы
            r'panoramio', r'geograph\.org', r'wikimedia',
            r'honeymoon', r'travel guide', r'tourism',
            # Абстрактные концепции
            r'\breligion in\b', r'\breligious beliefs\b',
            r'\blgbtq\b', r'\bhistory of\b',
            # Улицы и адреса (обычно не достопримечательности)
            r'^\d+\s+\w+\s+(street|avenue|road|boulevard)',
            r'^улица\s+', r'^проспект\s+', r'^бульвар\s+',
        ]
        
        # Проверяем паттерны исключения
        for pattern in exclude_patterns:
            if re.search(pattern, name_lower, re.IGNORECASE):
                return False
        
        # Позитивные индикаторы достопримечательностей
        landmark_indicators = [
            # Типы зданий
            r'\b(cathedral|church|temple|mosque|synagogue|chapel)\b',
            r'\b(собор|церковь|храм|мечеть|синагога|часовня)\b',
            r'\b(palace|castle|fortress|tower|bridge|gate)\b',
            r'\b(дворец|замок|крепость|башня|мост|ворота)\b',
            r'\b(museum|gallery|theater|theatre|opera house)\b',
            r'\b(музей|галерея|театр|оперный)\b',
            r'\b(monument|memorial|statue|square|plaza)\b',
            r'\b(памятник|мемориал|статуя|площадь)\b',
            # Архитектурные стили
            r'\b(gothic|baroque|renaissance|romanesque|neoclassical)\b',
            r'\b(готический|барокко|ренессанс|романский|неоклассический)\b',
        ]
        
        # Если есть позитивные индикаторы - точно достопримечательность
        for pattern in landmark_indicators:
            if re.search(pattern, name_lower, re.IGNORECASE):
                return True
        
        # Если нет явных индикаторов, но и нет исключений - пропускаем
        # (может быть названием места)
        return True


class WikipediaService:
    """
    Сервис для поиска описания достопримечательностей по названию.
    Использует прямой REST API Wikipedia для лучшей производительности
    и thread-safety.
    """
    
    BASE_URL = "https://{lang}.wikipedia.org/w/api.php"

    def __init__(
        self,
        language: str = "ru",
        fallback_lang: str = "en",
        max_concurrent: int = MAX_CONCURRENT_REQUESTS,
        timeout: int = 30
    ) -> None:
        self.language = language
        self.fallback_lang = fallback_lang
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        
        # TTL кэш для автоматического управления (thread-safe)
        self._cache: Union[TTLCache, Dict[str, str]] = TTLCache(
            maxsize=MAX_CACHE_SIZE,
            ttl=DEFAULT_CACHE_TTL
        )
        
        # Async сессия (создается при первом использовании)
        self._aiohttp_session: Optional[aiohttp.ClientSession] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        
        # Sync сессия для requests
        self._requests_session = requests.Session()
        self._requests_session.headers.update({
            'User-Agent': 'AITourGuide/1.0'
        })

    def close(self) -> None:
        """Явное закрытие sync сессии."""
        if hasattr(self, '_requests_session') and self._requests_session:
            self._requests_session.close()
            self._requests_session = None

    async def aclose(self) -> None:
        """Явное закрытие async сессии."""
        if self._aiohttp_session:
            await self._aiohttp_session.close()
            self._aiohttp_session = None

    def __del__(self) -> None:
        """Закрытие сессий при удалении объекта."""
        self.close()

    def __enter__(self) -> 'WikipediaService':
        """Context manager для sync использования."""
        return self

    def __exit__(self, *args) -> None:
        """Закрытие sync сессии."""
        self.close()

    async def __aenter__(self) -> 'WikipediaService':
        """Context manager для async использования."""
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        headers = {
            'User-Agent': (
                'AITourGuide/1.0 '
                '(slesarenko221999@gmail.com) educational bot'
            )
        }
        self._aiohttp_session = aiohttp.ClientSession(
            timeout=timeout, headers=headers
        )
        return self

    async def __aexit__(self, *args) -> None:
        """Закрытие async сессии."""
        await self.aclose()

    def get_landmark_info_sync(
        self,
        landmark_names: Set[str]
    ) -> Dict[str, str]:
        """
        Синхронная версия: последовательно ищет описание
        достопримечательности сначала на предпочтительном языке,
        если не удаётся — на запасном.

        Args:
            landmark_names: Названия достопримечательностей
                (на любом языке).

        Returns:
            Описания или сообщение об ошибке.
        """
        return self._get_landmark_info_sequential(landmark_names)

    async def get_landmark_info_async(
        self,
        landmark_names: Set[str]
    ) -> Dict[str, str]:
        """
        Асинхронная версия: параллельно ищет описание
        достопримечательности сначала на предпочтительном языке,
        если не удаётся — на запасном.

        Args:
            landmark_names: Названия достопримечательностей
                (на любом языке).

        Returns:
            Описания или сообщение об ошибке.
        """
        return await self._get_landmark_info_async(landmark_names)

    def _get_landmark_info_sequential(
        self,
        landmark_names: Set[str]
    ) -> Dict[str, str]:
        """Последовательная обработка запросов."""
        results: Dict[str, str] = {}

        for query in landmark_names:
            # Определяем язык запроса
            lang = self._detect_language(query)
            cache_key = f"{query}:{lang}"
            
            # TTLCache thread-safe
            if cache_key in self._cache:
                results[query] = self._cache[cache_key]
                continue

            # Первичный язык
            description = self._try_get_summary_sync(query, lang)
            if description:
                results[query] = description
                self._cache[cache_key] = description
                logger.info(
                    f"Описание найдено для '{query}' ({lang})"
                )
                continue

            # Запасной язык
            desc_fallback = self._try_get_summary_sync(
                query, self.fallback_lang
            )
            if desc_fallback:
                results[query] = desc_fallback
                cache_key_fallback = f"{query}:{self.fallback_lang}"
                self._cache[cache_key_fallback] = desc_fallback
                logger.info(
                    f"Описание найдено (Fallback) для '{query}' "
                    f"({self.fallback_lang})"
                )
                continue

            logger.warning(f"Не найдено описание для '{query}'")

        return results

    async def _get_landmark_info_async(
        self,
        landmark_names: Set[str]
    ) -> Dict[str, str]:
        """Асинхронная обработка запросов для ускорения."""
        results: Dict[str, str] = {}
        queries_to_fetch = []

        # Проверяем кэш (TTLCache thread-safe)
        for query in landmark_names:
            cache_key = f"{query}:{self.language}"
            if cache_key in self._cache:
                results[query] = self._cache[cache_key]
            else:
                queries_to_fetch.append(query)

        if not queries_to_fetch:
            return results

        # Создаем семафор для ограничения одновременных запросов
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)

        # Создаем временную сессию если нет постоянной
        close_session = False
        if self._aiohttp_session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {
                'User-Agent': 'AITourGuide/1.0 '
                '(slesarenko221999@gmail.com)'
            }
            self._aiohttp_session = aiohttp.ClientSession(
                timeout=timeout, headers=headers
            )
            close_session = True

        try:
            # Асинхронная обработка оставшихся запросов
            tasks = [
                self._fetch_single_landmark_async(query)
                for query in queries_to_fetch
            ]

            # Ждем завершения всех задач
            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Обрабатываем результаты
            for query, result in zip(queries_to_fetch, task_results):
                if isinstance(result, Exception):
                    logger.error(f"Ошибка при обработке '{query}': {result}")
                    continue

                description, lang_used = result
                if description:
                    results[query] = description
                    cache_key = f"{query}:{lang_used}"
                    self._cache[cache_key] = description
                    logger.info(
                        f"Описание найдено для '{query}' ({lang_used})"
                    )
                else:
                    logger.warning(f"Не найдено описание для '{query}'")
        finally:
            # Закрываем временную сессию
            if close_session and self._aiohttp_session:
                await self._aiohttp_session.close()
                self._aiohttp_session = None

        return results

    async def _fetch_single_landmark_async(
        self,
        query: str
    ) -> tuple[Optional[str], str]:
        """
        Асинхронно получает описание для одной достопримечательности.
        Возвращает (описание, использованный_язык).
        """
        async with self._semaphore:
            lang = self._detect_language(query)
            # Первичный язык
            description = await self._try_get_summary_async(
                query, lang
            )
            if description:
                return description, lang

            # Запасной язык
            desc_fallback = await self._try_get_summary_async(
                query, self.fallback_lang
            )
            if desc_fallback:
                return desc_fallback, self.fallback_lang

            return None, self.language

    async def _try_get_summary_async(
        self,
        query: str,
        lang: str,
        try_search: bool = True
    ) -> Optional[str]:
        """
        Асинхронно получает описание через REST API Wikipedia.
        
        Args:
            query: Название достопримечательности
            lang: Язык Wikipedia
            try_search: Пытаться ли искать если не найдено напрямую
        
        Returns:
            Описание или None
        """
        url = self.BASE_URL.format(lang=lang)
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "exsentences": "5",  # Ограничение до 5 предложений
            "titles": query,
            "redirects": "1",
            "formatversion": "2"
        }

        try:
            async with self._aiohttp_session.get(
                url, params=params
            ) as response:
                if response.status != 200:
                    logger.debug(
                        f"Wikipedia API вернул статус {response.status} "
                        f"для '{query}'"
                    )
                    return None

                data = await response.json()
                pages = data.get("query", {}).get("pages", [])

                if pages and len(pages) > 0:
                    page = pages[0]
                    if "missing" not in page:
                        extract = page.get("extract")
                        if extract:
                            if self._is_relevant(query, extract):
                                logger.debug(f"Найдено напрямую: '{query}'")
                                return extract
                            else:
                                logger.debug(
                                    f"Описание для '{query}' не прошло "
                                    f"проверку релевантности"
                                )

                # Если не нашли напрямую, пробуем поиск
                if try_search:
                    logger.debug(f"Пробуем поиск для: '{query}'")
                    return await self._search_and_get_async(query, lang)
                
                return None

        except aiohttp.ClientError as e:
            logger.warning(f"Ошибка при запросе к Wikipedia: {e}")
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}")
            return None
    
    def _is_relevant(self, query: str, extract: str) -> bool:
        """
        Проверяет релевантность найденного описания запросу.
        
        Args:
            query: Поисковый запрос
            extract: Найденное описание
            
        Returns:
            True если описание релевантно запросу
        """
        # Извлекаем значимые слова из запроса
        pattern = rf'\b[a-zа-я]{{{MIN_WORD_LENGTH_FOR_RELEVANCE},}}\b'
        query_words = set(re.findall(pattern, query.lower()))
        query_words = query_words - STOPWORDS
        
        if not query_words:
            return True  # Если нет значимых слов, считаем релевантным
        
        extract_lower = extract.lower()
        
        # Специальные слова, которые должны присутствовать если есть в запросе
        special_keywords = {
            'cathedral', 'church', 'temple', 'mosque', 'synagogue',
            'palace', 'castle', 'fortress', 'tower', 'bridge',
            'собор', 'церковь', 'храм', 'мечеть', 'синагога',
            'дворец', 'замок', 'крепость', 'башня', 'мост'
        }
        
        # Проверяем специальные ключевые слова
        query_special = query_words & special_keywords
        if query_special:
            # Если в запросе есть специальные слова, они должны быть в описании
            special_found = [w for w in query_special if w in extract_lower]
            if not special_found:
                logger.debug(
                    f"Отклонено '{query}': специальные слова "
                    f"{query_special} не найдены в описании. "
                    f"Описание начинается: {extract[:100]}..."
                )
                return False
            else:
                logger.debug(
                    f"Специальные слова найдены: {special_found}"
                )
        
        # Подсчитываем долю совпадающих слов
        matching_words = sum(
            1 for word in query_words if word in extract_lower
        )
        relevance_ratio = matching_words / len(query_words)
        
        is_relevant = relevance_ratio >= MIN_RELEVANCE_RATIO
        
        if not is_relevant:
            logger.debug(
                f"Отклонено: релевантность {relevance_ratio:.2f} < "
                f"{MIN_RELEVANCE_RATIO} "
                f"({matching_words}/{len(query_words)} слов)"
            )
        
        return is_relevant
    
    def _generate_search_variants(self, query: str) -> list[str]:
        """
        Генерирует варианты поискового запроса.
        
        Args:
            query: Исходный запрос
            
        Returns:
            Список вариантов запроса
        """
        # Очистка запроса от апострофов и лишней пунктуации
        clean = re.sub(r"['\"]", "", query)
        clean = re.sub(r"[.,!?;:]", "", clean)
        clean = re.sub(r"\s+", " ", clean).strip()

        # Генерируем варианты
        variants = [
            clean,  # Полный очищенный запрос
            " ".join(clean.split()[:MAX_QUERY_WORDS_FOR_VARIANTS]),
        ]
        
        # Убираем дубликаты и пустые строки
        return [v for v in dict.fromkeys(variants) if v]
    
    async def _search_and_get_async(
        self, query: str, lang: str
    ) -> Optional[str]:
        """
        Поиск статьи через opensearch и полнотекстовый поиск.
        
        Args:
            query: Поисковый запрос
            lang: Язык Wikipedia
            
        Returns:
            Описание или None
        """
        url = self.BASE_URL.format(lang=lang)
        if self._aiohttp_session is None:
            return None

        variants = self._generate_search_variants(query)

        # 1. Opensearch для каждого варианта
        for variant in variants:
            params_os = {
                "action": "opensearch",
                "format": "json",
                "search": variant,
                "limit": OPENSEARCH_LIMIT,
                "namespace": 0,
                "suggest": "true",
            }
            try:
                async with self._aiohttp_session.get(
                    url, params=params_os
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        titles = data[1] if len(data) > 1 else []
                        if titles:
                            logger.debug(
                                f"Opensearch with '{variant}' → {titles}"
                            )
                            for title in titles:
                                desc = await self._try_get_summary_async(
                                    title, lang, try_search=False
                                )
                                if desc:
                                    logger.info(
                                        f"Found via opensearch: '{title}'"
                                    )
                                    return desc
            except Exception as e:
                logger.debug(f"Opensearch error for '{variant}': {e}")

        # 2. Полнотекстовый поиск для каждого варианта
        for variant in variants:
            params_full = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": variant,
                "srwhat": "text",
                "srlimit": FULLTEXT_SEARCH_LIMIT,
                "formatversion": 2,
            }
            try:
                async with self._aiohttp_session.get(
                    url, params=params_full
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        pages = data.get("query", {}).get("search", [])
                        if pages:
                            titles = [p['title'] for p in pages]
                            logger.debug(
                                f"Fulltext with '{variant}' → {titles}"
                            )
                            for title in titles:
                                desc = await self._try_get_summary_async(
                                    title, lang, try_search=False
                                )
                                if desc:
                                    logger.info(
                                        f"Found via fulltext: '{title}'"
                                    )
                                    return desc
            except Exception as e:
                logger.debug(f"Fulltext error for '{variant}': {e}")

        return None

    def _try_get_summary_sync(
        self,
        query: str,
        lang: str,
        try_search: bool = True
    ) -> Optional[str]:
        """
        Синхронно получает описание через REST API Wikipedia.
        
        Args:
            query: Название достопримечательности
            lang: Язык Wikipedia
            try_search: Пытаться ли искать если не найдено напрямую
        
        Returns:
            Описание или None
        """
        url = self.BASE_URL.format(lang=lang)
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "exsentences": "5",  # Ограничение до 5 предложений
            "titles": query,
            "redirects": "1",
            "formatversion": "2"
        }

        try:
            if self._requests_session:
                response = self._requests_session.get(
                    url, params=params, timeout=self.timeout
                )
            else:
                response = requests.get(
                    url, params=params, timeout=self.timeout
                )

            if response.status_code != 200:
                logger.debug(
                    f"Wikipedia API вернул статус {response.status_code} "
                    f"для '{query}'"
                )
                return None

            data = response.json()
            pages = data.get("query", {}).get("pages", [])

            if pages and len(pages) > 0:
                page = pages[0]
                if "missing" not in page:
                    extract = page.get("extract")
                    if extract:
                        if self._is_relevant(query, extract):
                            logger.debug(f"Найдено напрямую: '{query}'")
                            return extract
                        else:
                            logger.debug(
                                f"Описание для '{query}' не прошло "
                                f"проверку релевантности"
                            )

            # Если не нашли напрямую, пробуем поиск
            if try_search:
                logger.debug(f"Пробуем поиск для: '{query}'")
                return self._search_and_get_sync(query, lang)
            
            return None

        except requests.exceptions.RequestException as e:
            logger.warning(f"Ошибка при запросе к Wikipedia: {e}")
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}")
            return None

    def _search_and_get_sync(
        self, query: str, lang: str
    ) -> Optional[str]:
        """
        Синхронный поиск статьи через opensearch и полнотекстовый поиск.
        
        Args:
            query: Поисковый запрос
            lang: Язык Wikipedia
            
        Returns:
            Описание или None
        """
        url = self.BASE_URL.format(lang=lang)
        variants = self._generate_search_variants(query)

        # 1. Opensearch
        for variant in variants:
            params = {
                "action": "opensearch",
                "format": "json",
                "search": variant,
                "limit": OPENSEARCH_LIMIT,
                "namespace": 0,
                "suggest": "true",
            }
            try:
                resp = self._requests_session.get(
                    url, params=params, timeout=self.timeout
                )
                if resp.status_code == 200:
                    data = resp.json()
                    titles = data[1] if len(data) > 1 else []
                    logger.debug(f"Opensearch titles: {titles}")
                    for title in titles:
                        desc = self._try_get_summary_sync(
                            title, lang, try_search=False
                        )
                        if desc:
                            logger.info(
                                f"Found via opensearch: '{title}'"
                            )
                            return desc
            except Exception as e:
                logger.debug(f"Opensearch error: {e}")

        # 2. Fulltext
        for variant in variants:
            params_full = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": variant,
                "srwhat": "text",
                "srlimit": FULLTEXT_SEARCH_LIMIT,
                "formatversion": 2,
            }
            try:
                resp = self._requests_session.get(
                    url, params=params_full, timeout=self.timeout
                )
                if resp.status_code == 200:
                    data = resp.json()
                    pages = data.get("query", {}).get("search", [])
                    logger.debug(
                        f"Fulltext results: {[p['title'] for p in pages]}"
                    )
                    for page in pages:
                        title = page.get("title")
                        if title:
                            desc = self._try_get_summary_sync(
                                title, lang, try_search=False
                            )
                            if desc:
                                logger.info(
                                    f"Found via fulltext: '{title}'"
                                )
                                return desc
            except Exception as e:
                logger.debug(f"Fulltext search error: {e}")

        return None

    def _detect_language(self, query: str) -> str:
        """
        Определяет язык запроса по наличию кириллицы.
        
        Args:
            query: Поисковый запрос
            
        Returns:
            Код языка ('ru' или 'en')
        """
        if re.search(r'[а-яА-ЯёЁ]', query):
            return self.language
        else:
            return self.fallback_lang


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Включаем логирование для отладки
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s: %(message)s'
    )

    # Пример использования
    print("🧪 Тест Yandex + Wikipedia Services")
    print("=" * 70)

    # Получаем переменные окружения
    folder_id = os.environ.get("YC_FOLDER_ID")
    api_key = os.environ.get("YC_API_KEY")

    if not folder_id or not api_key:
        print("Ошибка: YC_FOLDER_ID и YC_API_KEY должны быть установлены")
    else:
        # Тест с context managers для автоматического закрытия ресурсов
        test_image = "images/475_3.jpg"

        if os.path.exists(test_image):
            # Синхронная версия
            print("\n📝 Синхронная версия:")
            print("-" * 70)
            
            with YandexSearchService(
                yc_folder_id=folder_id,
                yc_api_key=api_key,
            ) as yandex_service:
                names = yandex_service.search_by_image(test_image)

                if names:
                    print("\nНайденные названия:")
                    for name in names:
                        print(f"  • {name}")

                    # Синхронный поиск описаний
                    with WikipediaService(
                        language="ru",
                        fallback_lang="en"
                    ) as wiki_service:
                        print("\nПолучение описаний (sync)...")
                        descriptions = wiki_service.get_landmark_info_sync(
                            names
                        )

                        for i, name in enumerate(descriptions, 1):
                            desc = descriptions[name]
                            print(f"\n{i}. {name}:")
                            if len(desc) > 200:
                                print(desc[:200] + "...")
                            else:
                                print(desc)
                else:
                    print("Не найдено названий достопримечательностей")

            # Асинхронная версия
            print("\n\n⚡ Асинхронная версия:")
            print("-" * 70)

            async def test_async():
                with YandexSearchService(
                    yc_folder_id=folder_id,
                    yc_api_key=api_key,
                ) as yandex_service:
                    names = yandex_service.search_by_image(test_image)

                    if names:
                        print("\nНайденные названия:")
                        for name in names:
                            print(f"  • {name}")

                        # Асинхронный поиск описаний
                        async with WikipediaService(
                            language="ru",
                            fallback_lang="en"
                        ) as wiki_service:
                            print("\nПолучение описаний (async)...")
                            descriptions = (
                                await wiki_service.get_landmark_info_async(
                                    names
                                )
                            )

                            for i, name in enumerate(descriptions, 1):
                                desc = descriptions[name]
                                print(f"\n{i}. {name}:")
                                if len(desc) > 200:
                                    print(desc[:200] + "...")
                                else:
                                    print(desc)

            asyncio.run(test_async())
        else:
            print(f"Тестовый файл не найден: {test_image}")
