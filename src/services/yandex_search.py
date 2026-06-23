import base64
import hashlib
import logging
import os
import re
import asyncio
from typing import Any, Dict, Set, Optional, Union

import aiohttp
import requests
from cachetools import TTLCache

from .search_filters import (
    OPENSEARCH_LIMIT,
    FULLTEXT_SEARCH_LIMIT,
    is_likely_landmark,
    is_relevant,
    clean_landmark_name,
    generate_search_variants,
)

logger = logging.getLogger(__name__)

# Константы для конфигурации
MIN_LANDMARK_NAME_LENGTH = 3
DEFAULT_CACHE_TTL = 3600
DEFAULT_TIMEOUT = 100
DEFAULT_MAX_RESULTS = 5
MAX_CACHE_SIZE = 1000
MAX_CONCURRENT_REQUESTS = 5


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
        if not yc_folder_id or not yc_api_key:
            raise ValueError(
                "yc_folder_id и yc_api_key обязательны"
            )

        self.yc_folder_id = yc_folder_id
        self.yc_api_key = yc_api_key
        self.timeout = timeout
        self.max_results = max_results
        self.cache_ttl_seconds = cache_ttl_seconds

        self._search_cache: Union[TTLCache, Dict] = TTLCache(
            maxsize=MAX_CACHE_SIZE,
            ttl=cache_ttl_seconds
        )

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
        self.close()

    def __enter__(self) -> 'YandexSearchService':
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def search_by_image(
        self,
        image: Union[str, bytes, Any],
        num_results: int = DEFAULT_MAX_RESULTS
    ) -> Optional[Set[str]]:
        """
        Находит названия достопримечательностей как заголовки
        страниц с похожими изображениями.

        Args:
            image: Путь к изображению, байты или PIL Image.
            num_results: Ограничение на кол-во результатов

        Returns:
            Названия найденных достопримечательностей
        """
        if not self.yc_api_key:
            logger.warning("Yandex Search API ключ не настроен")
            return None

        num_results = num_results or self.max_results

        if num_results <= 0:
            raise ValueError("num_results должен быть положительным")

        try:
            if isinstance(image, str):
                if not os.path.exists(image):
                    logger.error(f"Файл не найден: {image}")
                    return None
                with open(image, "rb") as image_file:
                    image_bytes = image_file.read()
            elif isinstance(image, bytes):
                image_bytes = image
            else:
                # PIL Image
                from io import BytesIO
                buffer = BytesIO()
                image.save(buffer, format='JPEG')
                image_bytes = buffer.getvalue()

            encoded_string = base64.b64encode(image_bytes).decode("utf-8")
            image_hash = hashlib.sha256(
                encoded_string.encode()
            ).hexdigest()

            if image_hash in self._search_cache:
                logger.info("Возврат из кэша")
                return self._search_cache[image_hash]

            payload = {
                "folderId": self.yc_folder_id,
                "data": encoded_string,
            }

            logger.info("Отправка запроса к Yandex Search API")

            if self._session:
                response = self._session.post(
                    self._URL_SEARCH_BY_IMAGE,
                    json=payload,
                    timeout=self.timeout
                )
            else:
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
            landmark_names = self._extract_landmark_names(
                result_data, num_results
            )
            if landmark_names:
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

    def _extract_landmark_names(
        self,
        response_data: Dict,
        num_results: int = 5
    ) -> Set[str]:
        """
        Извлекает названия достопримечательностей из ответа Yandex API.

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
                cleaned = clean_landmark_name(page_title)
                if cleaned and len(cleaned) > MIN_LANDMARK_NAME_LENGTH:
                    if is_likely_landmark(cleaned):
                        landmark_names.add(cleaned)
                    else:
                        logger.debug(
                            f"Filtered out non-landmark: '{cleaned}'"
                        )

        return landmark_names


class WikipediaService:
    """
    Сервис для поиска описания достопримечательностей по названию.
    Использует прямой REST API Wikipedia.

    Только async-интерфейс. Для sync-использования оберните через
    asyncio.run() или asyncio.get_event_loop().run_until_complete().
    """

    BASE_URL = "https://{lang}.wikipedia.org/w/api.php"

    def __init__(
        self,
        language: str = "ru",
        fallback_lang: str = "en",
        max_concurrent: int = MAX_CONCURRENT_REQUESTS,
        timeout: int = 30,
        min_relevance_ratio: float = 0.5,
    ) -> None:
        self.language = language
        self.fallback_lang = fallback_lang
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.min_relevance_ratio = min_relevance_ratio

        self._cache: Union[TTLCache, Dict[str, str]] = TTLCache(
            maxsize=MAX_CACHE_SIZE,
            ttl=DEFAULT_CACHE_TTL
        )

        self._aiohttp_session: Optional[aiohttp.ClientSession] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    async def aclose(self) -> None:
        """Явное закрытие async сессии."""
        if self._aiohttp_session:
            await self._aiohttp_session.close()
            self._aiohttp_session = None

    def __del__(self) -> None:
        # Не закрываем async сессию из __del__ — небезопасно
        pass

    async def __aenter__(self) -> 'WikipediaService':
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
        await self.aclose()

    async def get_landmark_info(
        self,
        landmark_names: Set[str]
    ) -> Dict[str, str]:
        """
        Параллельно ищет описание достопримечательностей.
        Сначала на предпочтительном языке, если не удаётся — на запасном.

        Args:
            landmark_names: Названия достопримечательностей (на любом языке).

        Returns:
            Словарь {название: описание}
        """
        return await self._get_landmark_info_async(landmark_names)

    # Оставляем алиас для обратной совместимости
    async def get_landmark_info_async(
        self,
        landmark_names: Set[str]
    ) -> Dict[str, str]:
        """Алиас для get_landmark_info (обратная совместимость)."""
        return await self.get_landmark_info(landmark_names)

    async def _get_landmark_info_async(
        self,
        landmark_names: Set[str]
    ) -> Dict[str, str]:
        """Асинхронная обработка запросов для ускорения."""
        results: Dict[str, str] = {}
        queries_to_fetch = []

        for query in landmark_names:
            cache_key = f"{query}:{self.language}"
            if cache_key in self._cache:
                results[query] = self._cache[cache_key]
            else:
                queries_to_fetch.append(query)

        if not queries_to_fetch:
            return results

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)

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
            tasks = [
                self._fetch_single_landmark(query)
                for query in queries_to_fetch
            ]
            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            for query, result in zip(queries_to_fetch, task_results):
                if isinstance(result, Exception):
                    logger.error(f"Ошибка при обработке '{query}': {result}")
                    continue

                wiki_title, description, lang_used = result
                if description:
                    key = wiki_title if wiki_title else query
                    results[key] = description
                    cache_key = f"{key}:{lang_used}"
                    self._cache[cache_key] = description
                    logger.info(
                        f"Описание найдено для '{key}' ({lang_used})"
                    )
                else:
                    logger.warning(f"Не найдено описание для '{query}'")
        finally:
            if close_session and self._aiohttp_session:
                await self._aiohttp_session.close()
                self._aiohttp_session = None

        return results

    async def _fetch_single_landmark(
        self,
        query: str
    ) -> tuple[Optional[str], Optional[str], str]:
        """
        Асинхронно получает описание для одной достопримечательности.

        Returns:
            (wiki_title, описание, использованный_язык)
        """
        async with self._semaphore:
            lang = self._detect_language(query)

            description = await self._try_get_summary(query, lang)
            if description:
                return query, description, lang

            desc_fallback = await self._try_get_summary(
                query, self.fallback_lang
            )
            if desc_fallback:
                return query, desc_fallback, self.fallback_lang

            return None, None, self.language

    async def _try_get_summary(
        self,
        query: str,
        lang: str,
        try_search: bool = True
    ) -> Optional[str]:
        """
        Получает описание через REST API Wikipedia.

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
            "exsentences": "5",
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
                            if is_relevant(
                                query, extract, self.min_relevance_ratio
                            ):
                                logger.debug(f"Найдено напрямую: '{query}'")
                                return extract
                            else:
                                logger.debug(
                                    f"Описание для '{query}' не прошло "
                                    f"проверку релевантности"
                                )

                if try_search:
                    logger.debug(f"Пробуем поиск для: '{query}'")
                    return await self._search_and_get(query, lang)

                return None

        except aiohttp.ClientError as e:
            logger.warning(f"Ошибка при запросе к Wikipedia: {e}")
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}")
            return None

    async def _search_and_get(
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

        variants = generate_search_variants(query)

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
                                desc = await self._try_get_summary(
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
                                desc = await self._try_get_summary(
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

    import logging as _logging
    _logging.basicConfig(
        level=_logging.DEBUG,
        format='%(levelname)s: %(message)s'
    )

    print("🧪 Тест Yandex + Wikipedia Services")
    print("=" * 70)

    folder_id = os.environ.get("YC_FOLDER_ID")
    api_key = os.environ.get("YC_API_KEY")

    if not folder_id or not api_key:
        print("Ошибка: YC_FOLDER_ID и YC_API_KEY должны быть установлены")
    else:
        test_image = "images/475_3.jpg"

        if os.path.exists(test_image):
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

                        async with WikipediaService(
                            language="ru",
                            fallback_lang="en"
                        ) as wiki_service:
                            print("\nПолучение описаний...")
                            descriptions = (
                                await wiki_service.get_landmark_info(names)
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

            asyncio.run(test_async())
        else:
            print(f"Тестовый файл не найден: {test_image}")
