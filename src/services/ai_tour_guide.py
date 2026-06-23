# src/services/ai_tour_guide.py
# -*- coding: utf-8 -*-
"""
Сервис распознавания достопримечательностей.

Пайплайн:
  1. SigLIP + FAISS — поиск top-K кандидатов
  2. VLM Reranking через SGLang (попарное сравнение, P(yes))
  3. Расчёт уверенности
  4. Интернет-поиск при низкой уверенности (Yandex + Wikipedia)
"""

import math
import os
import re
import time
import logging
import asyncio
import base64
import uuid
import traceback
import httpx
from io import BytesIO
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime
from dotenv import load_dotenv
from src.rag.landmark_retriever import LandmarkRetriever, LandmarkRetrievalResult

from .yandex_search import (
    YandexSearchService,
    WikipediaService,
    SEARCH_NOISE_TOKENS,
    ARCHITECTURAL_TERMS,
)
from .translator import YandexTranslator

logger = logging.getLogger(__name__)

load_dotenv()


class PredictionSource(Enum):
    """Источник итогового предсказания."""
    RETRIEVAL = "retrieval"
    INTERNET = "internet"
    FALLBACK = "fallback"


@dataclass
class AITourGuideConfig:
    """Конфигурация сервиса AITourGuide."""

    # Обязательный параметр — путь к индексу
    index_dir: str

    # SGLang сервер
    sglang_base_url: str = "http://localhost:30000/v1"
    sglang_model_name: str = "qwen2-vl-2b-r16"
    sglang_timeout: float = 30.0
    sglang_max_retries: int = 3

    # Локальный путь к SigLIP модели (пустая строка = загрузка с HuggingFace)
    siglip_model_path: str = ""

    # Базовая директория изображений галереи.
    # image_path в gallery_metadata.json — просто имя файла (photo.jpg),
    # полный путь = images_base_dir / image_path
    images_base_dir: str = ""

    # Retrieval
    top_k_retrieval: int = 10
    faiss_k: int = 100

    # VLM параметры
    caption_max_length: int = 300
    max_new_tokens: int = 256
    temperature: float = 0.0

    # Уверенность
    # vlm_threshold — порог на p_yes от VLM для решения об интернет-поиске.
    # Берётся из experiments/find_th_and_recompute_metrics.py (opt_t).
    # confidence в result["confidence"] == p_yes напрямую.
    vlm_threshold: float = 0.5

    # Интернет-поиск
    enable_internet_search: bool = True

    # Устройство (не используется с SGLang, для совместимости)
    device: str = "cuda"

    def to_dict(self) -> Dict:
        """Конвертирует конфигурацию в словарь."""
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Метрики производительности сервиса."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_confidence: float = 0.0
    internet_search_rate: float = 0.0
    avg_retrieval_time: float = 0.0
    avg_generation_time: float = 0.0
    avg_total_time: float = 0.0
    last_updated: Optional[str] = None

    # Накопительные суммы (не отображаются в repr)
    _sum_confidence: float = field(default=0.0, repr=False)
    _sum_retrieval_time: float = field(default=0.0, repr=False)
    _sum_generation_time: float = field(default=0.0, repr=False)
    _sum_total_time: float = field(default=0.0, repr=False)
    _internet_searches: int = field(default=0, repr=False)

    def update(self, result: Dict):
        """Обновляет метрики на основе результата предсказания."""
        self.total_requests += 1

        if result.get("error"):
            self.failed_requests += 1
        else:
            self.successful_requests += 1

            conf = result.get("confidence", 0.0)
            self._sum_confidence += conf
            self.avg_confidence = self._sum_confidence / self.successful_requests

            timing = result.get("timing", {})
            if "retrieval" in timing:
                self._sum_retrieval_time += timing["retrieval"]
                self.avg_retrieval_time = (
                    self._sum_retrieval_time / self.successful_requests
                )

            if "vlm_generation" in timing:
                self._sum_generation_time += timing["vlm_generation"]
                self.avg_generation_time = (
                    self._sum_generation_time / self.successful_requests
                )

            total_time = sum(timing.values())
            self._sum_total_time += total_time
            self.avg_total_time = (
                self._sum_total_time / self.successful_requests
            )

            if result.get("source") == PredictionSource.INTERNET.value:
                self._internet_searches += 1
            self.internet_search_rate = (
                self._internet_searches / self.successful_requests
            )

        self.last_updated = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """Возвращает метрики в виде словаря (без приватных полей)."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (
                self.successful_requests / self.total_requests
                if self.total_requests > 0 else 0.0
            ),
            "avg_confidence": round(self.avg_confidence, 4),
            "internet_search_rate": round(self.internet_search_rate, 4),
            "avg_retrieval_time": round(self.avg_retrieval_time, 3),
            "avg_generation_time": round(self.avg_generation_time, 3),
            "avg_total_time": round(self.avg_total_time, 3),
            "last_updated": self.last_updated,
        }


class SGLangClient:
    """HTTP-клиент для SGLang сервера (совместимый с OpenAI API)."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_keepalive_connections=10, max_connections=20
            ),
        )
        logger.info(f"SGLang клиент инициализирован: {base_url}")

    async def health_check(self) -> bool:
        """Проверяет доступность SGLang сервера."""
        try:
            # SGLang healthcheck на /health (без /v1 префикса)
            base = self.base_url.rstrip("/")
            if base.endswith("/v1"):
                base = base[:-3]
            response = await self.client.get(f"{base}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"SGLang health check failed: {e}")
            return False

    async def chat_completion(
        self,
        messages: List[Dict],
        max_tokens: int = 256,
        temperature: float = 0.0,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
    ) -> Dict:
        """
        Отправляет запрос к SGLang серверу через OpenAI API.

        Args:
            messages: Список сообщений в формате OpenAI
            max_tokens: Максимум новых токенов
            temperature: Температура генерации
            logprobs: Возвращать ли logprobs
            top_logprobs: Количество top logprobs

        Returns:
            Ответ от SGLang сервера
        """
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        if logprobs:
            payload["logprobs"] = True
            if top_logprobs:
                payload["top_logprobs"] = top_logprobs

        last_exception: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = await self.client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                last_exception = e
                logger.warning(
                    f"SGLang HTTP ошибка "
                    f"(попытка {attempt + 1}/{self.max_retries}): "
                    f"{e.response.status_code}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

            except httpx.RequestError as e:
                last_exception = e
                logger.warning(
                    f"SGLang ошибка запроса "
                    f"(попытка {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        raise RuntimeError(
            f"SGLang запрос не выполнен после {self.max_retries} попыток: "
            f"{last_exception}"
        )

    async def close(self):
        """Закрывает HTTP-клиент."""
        await self.client.aclose()


class AITourGuide:
    """
    Сервис распознавания достопримечательностей по изображениям.

    Использует SGLang сервер для VLM reranking через OpenAI API.
    """

    MAX_CONTEXT_LENGTH = 200
    DEFAULT_MAX_RESULTS = 5

    # Варианты токенов Yes/No для парсинга logprobs
    _YES_VARIANTS: frozenset = frozenset({"yes", "Yes", "YES", " yes", " Yes"})
    _NO_VARIANTS: frozenset = frozenset({"no", "No", "NO", " no", " No"})

    def __init__(
        self,
        config: Union["AITourGuideConfig", Dict, None] = None,
        **kwargs,
    ):
        """
        Args:
            config: Конфигурация (AITourGuideConfig, dict или None)
            **kwargs: Параметры конфигурации (если config=None)
        """
        if config is None:
            config = AITourGuideConfig(**kwargs)
        elif isinstance(config, dict):
            config = AITourGuideConfig(**config)

        self.config = config
        self.vlm_threshold = config.vlm_threshold
        self.top_k_retrieval = config.top_k_retrieval
        self.faiss_k = config.faiss_k
        self.caption_max_length = config.caption_max_length
        # Базовая директория изображений галереи (может быть пустой строкой)
        self.images_base_dir = Path(config.images_base_dir) if config.images_base_dir else None

        self.yc_folder_id = os.getenv("YC_FOLDER_ID")
        self.yc_api_key = os.getenv("YC_API_KEY")

        self.translator = YandexTranslator(
            yc_folder_id=self.yc_folder_id,
            yc_api_key=self.yc_api_key,
        )

        # Переиспользуемый экземпляр YandexSearchService
        # (создаётся один раз, сессия requests.Session живёт всё время)
        self._yandex_service: Optional[YandexSearchService] = None
        if self.yc_folder_id and self.yc_api_key:
            self._yandex_service = YandexSearchService(
                yc_folder_id=self.yc_folder_id,
                yc_api_key=self.yc_api_key,
            )

        self.metrics = PerformanceMetrics()
        self._is_ready = False

        logger.info("Загрузка LandmarkRetriever...")
        from src.rag.indexing_v2 import IndexConfig
        index_config = IndexConfig(
            model_name=config.siglip_model_path or "",
            embedder_type="siglip",
            device=config.device,
        )
        self.retriever = LandmarkRetriever.from_index_dir(
            index_dir=config.index_dir,
            index_config=index_config,
        )

        logger.info("Инициализация SGLang клиента...")
        self.sglang_client = SGLangClient(
            base_url=config.sglang_base_url,
            model_name=config.sglang_model_name,
            timeout=config.sglang_timeout,
            max_retries=config.sglang_max_retries,
        )

        self._is_ready = True
        logger.info("AITourGuide готов")
        logger.info(f"  SGLang: {config.sglang_base_url}")
        logger.info(f"  Модель: {config.sglang_model_name}")
        logger.info(f"  VLM порог (p_yes): {config.vlm_threshold}")

    # ------------------------------------------------------------------
    # Health check и метрики
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """Проверяет состояние сервиса и его компонентов."""
        safe_config = {
            k: v for k, v in self.config.to_dict().items()
            if k != "index_dir"
        }
        health: Dict[str, Any] = {
            "status": "healthy" if self._is_ready else "not_ready",
            "ready": self._is_ready,
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "config": safe_config,
            "metrics": self.metrics.to_dict(),
        }

        try:
            health["components"]["retriever"] = {
                "status": "ok" if hasattr(self, "retriever") else "error",
                "index_size": (
                    len(self.retriever.gallery_metadata)
                    if hasattr(self, "retriever") else 0
                ),
            }

            sglang_ok = await self.sglang_client.health_check()
            health["components"]["sglang"] = {
                "status": "ok" if sglang_ok else "error",
                "base_url": self.config.sglang_base_url,
                "model": self.config.sglang_model_name,
            }

            if not sglang_ok:
                health["status"] = "degraded"
                health["error"] = "SGLang сервер недоступен"

        except Exception as e:
            health["status"] = "degraded"
            health["error"] = str(e)

        return health

    def get_metrics(self) -> Dict:
        """Возвращает метрики производительности."""
        return self.metrics.to_dict()

    def reset_metrics(self):
        """Сбрасывает метрики производительности."""
        self.metrics = PerformanceMetrics()
        logger.info("Метрики сброшены")

    # ------------------------------------------------------------------
    # VLM через SGLang
    # ------------------------------------------------------------------

    def _image_to_base64_data_uri(
        self,
        image: Image.Image,
        max_size: int = 448,
        quality: int = 85,
    ) -> str:
        """Конвертирует PIL Image в base64 data URI для OpenAI API.

        Ресайзит до max_size px по большей стороне чтобы не превышать
        лимит payload SGLang.
        """
        # Ресайз с сохранением пропорций
        w, h = image.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        with BytesIO() as buf:
            image.save(buf, format="JPEG", quality=quality)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"

    def prepare_vlm_messages(
        self,
        query_image: Union[Image.Image, str, bytes],
        candidate_image: str,
        candidate_caption: str,
        candidate_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Формирует сообщения в формате OpenAI API для VLM reranking.

        Args:
            query_image: Запросное изображение (PIL, путь или bytes)
            candidate_image: Путь к изображению кандидата
            candidate_caption: Описание кандидата
            candidate_name: Название кандидата

        Returns:
            Список сообщений для OpenAI API
        """
        # Загружаем query-изображение
        if isinstance(query_image, Image.Image):
            query_img = query_image.convert("RGB")
        elif isinstance(query_image, str):
            with Image.open(query_image) as img:
                query_img = img.convert("RGB")
        elif isinstance(query_image, bytes):
            query_img = Image.open(BytesIO(query_image)).convert("RGB")
        else:
            raise ValueError(
                f"Неподдерживаемый тип query_image: {type(query_image)}"
            )

        # Загружаем изображение кандидата
        # image_path в метаданных — просто имя файла (photo.jpg),
        # поэтому добавляем images_base_dir если он задан
        candidate_path = candidate_image
        if self.images_base_dir is not None:
            candidate_path = str(self.images_base_dir / candidate_image)
        with Image.open(candidate_path) as img:
            candidate_img = img.convert("RGB")

        query_uri = self._image_to_base64_data_uri(query_img)
        candidate_uri = self._image_to_base64_data_uri(candidate_img)
        caption = candidate_caption[:self.caption_max_length]

        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Query Photo:"},
                    {"type": "image_url", "image_url": {"url": query_uri}},
                    {"type": "text", "text": "Candidate Photo:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": candidate_uri},
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Question: Are these photos showing the same "
                            f"landmark: \"{candidate_name}\"?\n"
                            f"Candidate details: {caption}\n"
                            f"Answer only with Yes or No."
                        ),
                    },
                ],
            }
        ]

    async def _generate_with_vlm(
        self,
        image: Image.Image,
        candidate_image: str,
        candidate_caption: str,
        candidate_name: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Генерирует ответ VLM для одного кандидата."""
        messages = self.prepare_vlm_messages(
            image, candidate_image, candidate_caption, candidate_name
        )
        response = await self.sglang_client.chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return str(response["choices"][0]["message"]["content"]).strip()

    # ------------------------------------------------------------------
    # Интернет-поиск
    # ------------------------------------------------------------------

    def _yandex_search_sync(
        self, image: Union[str, bytes, Image.Image]
    ) -> Optional[set]:
        """
        Синхронный wrapper для Yandex Image Search.
        Использует переиспользуемый экземпляр _yandex_service.
        """
        if self._yandex_service is None:
            return None
        return self._yandex_service.search_by_image(
            image, num_results=self.DEFAULT_MAX_RESULTS
        )

    def _filter_wiki_results(
        self, wiki_result: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Фильтрует результаты Wikipedia:
        - убирает шумовые названия (SEARCH_NOISE_TOKENS)
        - даёт приоритет архитектурным объектам (ARCHITECTURAL_TERMS)
        """
        filtered = {
            name: desc
            for name, desc in wiki_result.items()
            if desc and not any(
                x in name.lower() for x in SEARCH_NOISE_TOKENS
            )
        }

        # Приоритет архитектурным объектам
        priority = {
            name: desc
            for name, desc in filtered.items()
            if any(x in name.lower() for x in ARCHITECTURAL_TERMS)
        }
        if priority:
            return priority
        return filtered if filtered else wiki_result

    def _extract_clean_name(self, raw_name: str) -> str:
        """
        Очищает название от мусорных хвостов (тире, пайп, двоеточие)
        и сокращает до архитектурного термина если он есть.
        """
        # Обрезаем хвосты после разделителей
        clean = re.split(r'\s*[-–—::|]\s*', raw_name)[0].strip()
        words = clean.split()
        for i, w in enumerate(words):
            if w.lower() in ARCHITECTURAL_TERMS:
                # берём до 2 слов перед термином + сам термин
                arch_name = " ".join(words[max(0, i - 2):i + 1]).strip()
                return arch_name if len(arch_name) >= 3 else clean
        return clean

    async def _vlm_extract_landmark_name(
        self,
        image: Image.Image,
        page_titles: List[str],
    ) -> Optional[str]:
        """
        Использует Qwen (через SGLang) для извлечения точного названия
        достопримечательности из изображения и списка pageTitle от Yandex.

        pageTitle может содержать мусор ("Лучшие места Парижа", "File:..."),
        поэтому просим модель сгенерировать чистое название, пригодное
        для поиска в Wikipedia, а не выбирать из списка.

        Args:
            image: PIL-изображение запроса
            page_titles: Список pageTitle от Yandex (после базовой очистки)

        Returns:
            Чистое название достопримечательности или None если не удалось.
        """
        if not page_titles:
            return None

        titles_str = "\n".join(
            f"- {t}" for t in page_titles[:10]  # не более 10 подсказок
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Photo:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self._image_to_base64_data_uri(image)
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Reverse image search returned these page titles "
                            "(may contain noise):\n"
                            f"{titles_str}\n\n"
                            "What is the exact name of the landmark shown "
                            "in the photo? Reply with a short, precise name "
                            "suitable for a Wikipedia search "
                            "(e.g. 'Notre-Dame de Paris'). "
                            "If you cannot identify the landmark, "
                            "reply with 'unknown'."
                        ),
                    },
                ],
            }
        ]
        try:
            response = await self.sglang_client.chat_completion(
                messages=messages,
                max_tokens=48,
                temperature=0.0,
            )
            answer = str(
                response["choices"][0]["message"]["content"]
            ).strip()
            logger.debug(f"VLM landmark name extraction: {answer!r}")

            if not answer or answer.lower() in ("unknown", "none", ""):
                logger.info(
                    "VLM не смог определить название достопримечательности"
                )
                return None

            # Убираем кавычки если модель их добавила
            answer = answer.strip("\"'«»").strip()

            # Проверяем что ответ не является туристическим мусором
            # (Qwen иногда повторяет pageTitle вместо названия)
            _vlm_noise = {
                "экскурси", "тур ", "туры", "посетить", "visit",
                "tour ", "tours", "tickets", "билет", "расписание",
                "schedule", "opening", "hours", "как добраться",
                "getting there", "отзыв", "review",
            }
            answer_lower = answer.lower()
            if any(noise in answer_lower for noise in _vlm_noise):
                logger.warning(
                    f"VLM вернул туристический мусор: {answer!r}, "
                    f"пропускаем"
                )
                return None

            # Слишком длинный ответ — скорее всего не название
            if len(answer.split()) > 8:
                logger.warning(
                    f"VLM вернул слишком длинный ответ ({len(answer.split())} "
                    f"слов): {answer!r}, пропускаем"
                )
                return None

            logger.info(f"VLM извлёк название: {answer!r}")
            return answer

        except Exception as e:
            logger.warning(f"Ошибка VLM извлечения названия: {e}")
            return None

    async def _search_internet(
        self,
        image: Union[Image.Image, str, bytes],
        retrieved_scores: List[float],
        retrieved_descs: List[str],
        retrieved_names: List[str],
        fallback_name: str,
        timeout: float = 90.0,
    ) -> Dict:
        """
        Ищет информацию о достопримечательности через Yandex + Wikipedia.

        Пайплайн:
          1. Yandex Image Search → список pageTitle
          2. Qwen извлекает чистое название из изображения + pageTitle
          3. Wikipedia ищет по чистому названию от Qwen
          4. Если Qwen не смог — Wikipedia ищет по всем pageTitle (fallback)

        Возвращает словарь с полями: found, name, description, confidence.
        """
        result = {
            "found": False,
            "name": fallback_name,
            "description": "",
            "query": None,
            "confidence": 0.5,
        }

        if self._yandex_service is None:
            logger.warning("Yandex API ключи не настроены, поиск пропущен")
            return result

        # Нормализуем image в PIL для VLM
        pil_image: Optional[Image.Image] = None
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, bytes):
            try:
                pil_image = Image.open(BytesIO(image)).convert("RGB")
            except Exception:
                pass
        elif isinstance(image, str):
            try:
                pil_image = Image.open(image).convert("RGB")
            except Exception:
                pass

        async def _do_search() -> Optional[Dict]:
            """Внутренняя корутина — весь поиск под одним таймаутом."""
            # 1. Yandex Image Search (синхронный вызов в потоке)
            names = await asyncio.to_thread(
                self._yandex_search_sync, image
            )
            if not names:
                logger.info("Yandex не вернул результатов")
                return None

            page_titles = list(names)

            # 2. Qwen извлекает чистое название из фото + pageTitle
            if pil_image is None:
                logger.warning(
                    "pil_image недоступен, VLM-шаг пропущен — "
                    "поиск по pageTitle без уточнения от Qwen"
                )
            vlm_name: Optional[str] = None
            if pil_image is not None:
                vlm_name = await self._vlm_extract_landmark_name(
                    pil_image, page_titles
                )

            # 3. Wikipedia-поиск
            # Если Qwen дал название — ищем сначала только по нему.
            # pageTitle используем как fallback если Qwen-запрос пустой.
            if vlm_name:
                async with WikipediaService(language="ru") as wiki:
                    wiki_result = await wiki.get_landmark_info_async(
                        {vlm_name}
                    )
                # Если по названию от Qwen ничего не нашли —
                # добавляем pageTitle как fallback
                if not any(wiki_result.values()):
                    logger.debug(
                        f"Wikipedia не нашла '{vlm_name}', "
                        f"добавляем pageTitle как fallback"
                    )
                    async with WikipediaService(language="ru") as wiki:
                        extra = await wiki.get_landmark_info_async(
                            set(page_titles)
                        )
                    wiki_result.update(extra)
            else:
                async with WikipediaService(language="ru") as wiki:
                    wiki_result = await wiki.get_landmark_info_async(
                        set(page_titles)
                    )

            if not any(wiki_result.values()):
                return None

            filtered = self._filter_wiki_results(wiki_result)
            if not filtered:
                return None

            # Приоритет: название от Qwen если есть в filtered
            if vlm_name and vlm_name in filtered:
                best_key = vlm_name
                best_name = vlm_name
            else:
                # Ищем частичное совпадение с vlm_name
                best_key = None
                if vlm_name:
                    for key in filtered:
                        vl = vlm_name.lower()
                        kl = key.lower()
                        if vl in kl or kl in vl:
                            best_key = key
                            break
                # Если не нашли — берём по минимальной длине
                if best_key is None:
                    best_key = min(
                        filtered.keys(),
                        key=lambda n: len(n.split())
                    )
                    logger.debug(
                        f"Fallback к min(len): выбран '{best_key}' "
                        f"из {list(filtered.keys())}"
                    )
                # best_key гарантированно str после min() выше
                assert isinstance(best_key, str)
                best_name = self._extract_clean_name(best_key)

            # query возвращаем в словаре — не мутируем result внутри замыкания
            return {
                "name": best_name,
                "description": filtered[best_key],
                "query": page_titles,
            }

        try:
            found = await asyncio.wait_for(_do_search(), timeout=timeout)
            if found:
                result["found"] = True
                result["name"] = found["name"]
                result["description"] = found["description"]
                result["query"] = found["query"]
                result["confidence"] = 0.85
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
            result["description"] = (
                retrieved_descs[0] if retrieved_descs else ""
            )
            result["confidence"] = 0.75

        return result

    # ------------------------------------------------------------------
    # Основной пайплайн предсказания
    # ------------------------------------------------------------------

    def _init_result(self) -> Dict:
        """Возвращает пустую структуру результата."""
        return {
            "name": "",
            "description": "",
            "confidence": 0.0,
            "source": PredictionSource.RETRIEVAL.value,
            "retrieved_names": [],
            "retrieved_scores": [],
            "retrieved_images": [],
            "retrieved_captions": [],
            "search_query": None,
            "error": None,
            "timing": {},
        }

    async def _validate_and_load_image(
        self,
        image_input: Union[str, Path, bytes],
        timing: Dict[str, float],
    ) -> Tuple[Union[Path, str], Image.Image]:
        """Загружает изображение из пути или байтов."""
        t0 = time.time()

        if isinstance(image_input, bytes):
            try:
                image = Image.open(BytesIO(image_input)).convert("RGB")
                timing["image_load"] = round(time.time() - t0, 3)
                return "bytes_image", image
            except Exception as e:
                raise RuntimeError(f"Не удалось загрузить изображение: {e}")

        image_path = Path(image_input)
        try:
            image_path = image_path.resolve(strict=True)
        except (ValueError, OSError, RuntimeError) as e:
            raise ValueError(f"Недоступный путь: {e}")

        if not image_path.is_file():
            raise ValueError(f"Не является файлом: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Не удалось открыть изображение: {e}")

        timing["image_load"] = round(time.time() - t0, 3)
        return image_path, image

    async def _retrieve_candidates(
        self,
        image: Image.Image,
        timing: Dict[str, float],
    ) -> List[LandmarkRetrievalResult]:
        """Ищет кандидатов через SigLIP + FAISS."""
        t0 = time.time()
        retrieved = await asyncio.to_thread(
            self.retriever.retrieve,
            image,
            top_k=self.top_k_retrieval,
            faiss_k=self.faiss_k,
        )
        timing["retrieval"] = round(time.time() - t0, 3)

        if not retrieved:
            raise RuntimeError("Кандидаты не найдены")

        return retrieved

    async def _generate_vlm_prediction(
        self,
        image: Image.Image,
        retrieved: List[LandmarkRetrievalResult],
        timing: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Выбирает лучшего кандидата через VLM reranking.

        Для каждого кандидата вычисляет P(yes) через SGLang logprobs,
        возвращает кандидата с максимальной вероятностью.
        """
        t0 = time.time()

        # Собираем кандидатов с изображениями
        candidates = []
        for cand in retrieved:
            top_image = cand.get_top_image()
            if top_image:
                # Предпочитаем guide_description, затем caption_landmark
                description = (
                    top_image.guide_description
                    or top_image.caption_landmark
                    or top_image.caption
                )
                candidates.append({
                    "landmark_id": cand.landmark_id,
                    "landmark_name": cand.landmark_name,
                    "image_path": top_image.image_path,
                    "caption": top_image.caption,
                    "description": description,
                })

        if not candidates:
            raise RuntimeError("Нет кандидатов для VLM reranking")

        async def _score_candidate(cand: Dict) -> Dict:
            """Вычисляет P(yes) для одного кандидата."""
            try:
                messages = self.prepare_vlm_messages(
                    image,
                    cand["image_path"],
                    cand["caption"],
                    cand["landmark_name"],
                )
                response = await self.sglang_client.chat_completion(
                    messages=messages,
                    max_tokens=1,
                    temperature=0.0,
                    logprobs=True,
                    top_logprobs=20,
                )

                logprobs_data = (
                    response.get("choices", [{}])[0].get("logprobs", {})
                )
                if not logprobs_data or not logprobs_data.get("content"):
                    # logprobs не вернулись — fallback на текстовый ответ
                    text = str(
                        response.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    ).strip().lower()
                    logger.debug(
                        f"logprobs пусты, текст: {text!r} "
                        f"кандидат: {cand['landmark_name']!r}"
                    )
                    if text.startswith("yes"):
                        return {**cand, "p_yes": 0.9}
                    elif text.startswith("no"):
                        return {**cand, "p_yes": 0.1}
                    return {**cand, "p_yes": 0.0}

                top_lp = logprobs_data["content"][0].get("top_logprobs", [])

                logit_yes = None
                logit_no = None
                for item in top_lp:
                    token = item.get("token", "")
                    logprob = item.get("logprob", -100)
                    if logit_yes is None and token in self._YES_VARIANTS:
                        logit_yes = logprob
                    elif logit_no is None and token in self._NO_VARIANTS:
                        logit_no = logprob

                if logit_yes is not None and logit_no is not None:
                    max_logit = max(logit_yes, logit_no)
                    exp_yes = math.exp(logit_yes - max_logit)
                    exp_no = math.exp(logit_no - max_logit)
                    p_yes = exp_yes / (exp_yes + exp_no)
                elif logit_yes is not None:
                    p_yes = 1.0
                elif logit_no is not None:
                    p_yes = 0.0
                else:
                    p_yes = 0.0

                return {**cand, "p_yes": p_yes}

            except Exception as e:
                logger.warning(
                    f"Ошибка VLM reranking для {cand['landmark_id']}: {e}"
                )
                return {**cand, "p_yes": 0.0}

        # Параллельные запросы с ограничением параллелизма
        # (SGLang T4 не справляется с 10 одновременными запросами с изображениями)
        semaphore = asyncio.Semaphore(3)

        async def _score_with_sem(cand: Dict) -> Dict:
            async with semaphore:
                return await _score_candidate(cand)

        scored = await asyncio.gather(
            *[_score_with_sem(c) for c in candidates]
        )
        results = list(scored)

        # Выбираем кандидата с максимальным P(yes)
        results.sort(key=lambda x: x["p_yes"], reverse=True)
        best = results[0]

        timing["vlm_generation"] = round(time.time() - t0, 3)

        return {
            "name": best["landmark_name"],
            "description": best["description"],
            "p_yes": best["p_yes"],
        }

    async def _enhance_with_internet_search(
        self,
        image: Image.Image,
        image_path: Union[Path, str],
        retrieved_scores: List[float],
        retrieved_names: List[str],
        retrieved_descs: List[str],
        result: Dict,
        timing: Dict[str, float],
    ):
        """Улучшает результат через интернет-поиск."""
        search_input = (
            image if image_path == "bytes_image" else str(image_path)
        )

        t0 = time.time()
        search_result = await self._search_internet(
            image=search_input,
            retrieved_scores=retrieved_scores,
            retrieved_descs=retrieved_descs,
            retrieved_names=retrieved_names,
            fallback_name=result["name"],
        )
        timing["internet_search"] = round(time.time() - t0, 3)

        if search_result.get("found"):
            result["source"] = PredictionSource.INTERNET.value
            result["search_query"] = search_result["query"]
            result["confidence"] = round(search_result["confidence"], 4)

            # Переводим название и описание на русский язык,
            # только если текст не содержит кириллицы (т.е. на английском)
            name = search_result["name"]
            description = search_result["description"]
            t_translate = time.time()
            try:
                if not re.search(r'[а-яА-ЯёЁ]', name):
                    translated_name = self.translator.translate(
                        name, target_language="ru", source_language="en"
                    )
                    if translated_name:
                        name = translated_name

                if description and not re.search(r'[а-яА-ЯёЁ]', description):
                    translated_desc = self.translator.translate(
                        description, target_language="ru", source_language="en"
                    )
                    if translated_desc:
                        description = translated_desc
            except Exception as e:
                logger.warning(f"Ошибка перевода: {e}")
            timing["translation"] = round(time.time() - t_translate, 3)

            result["name"] = name
            result["description"] = description

    async def predict(
        self,
        image_input: Union[str, Path, bytes],
        use_internet_search: bool = True,
    ) -> Dict:
        """
        Распознаёт достопримечательность на изображении.

        Args:
            image_input: Путь к изображению или байты
            use_internet_search: Включить поиск при низкой уверенности

        Returns:
            Словарь с name, description, confidence, source, timing
        """
        timing: Dict[str, float] = {}
        result = self._init_result()
        correlation_id = str(uuid.uuid4())[:8]

        input_id = (
            f"bytes ({len(image_input)} bytes)"
            if isinstance(image_input, bytes)
            else str(image_input)
        )
        logger.info(f"[{correlation_id}] Предсказание для {input_id}")

        try:
            # 1. Загрузка изображения
            try:
                image_path, image = await self._validate_and_load_image(
                    image_input, timing
                )
                logger.info(
                    f"[{correlation_id}] Изображение загружено: {image.size}"
                )
            except (ValueError, FileNotFoundError, RuntimeError) as e:
                result["error"] = str(e)
                result["timing"] = timing
                logger.error(f"[{correlation_id}] Ошибка загрузки: {e}")
                return result

            # 2. Retrieval кандидатов
            try:
                retrieved = await self._retrieve_candidates(image, timing)
                retrieved_scores = []
                retrieved_names = []
                retrieved_images = []
                retrieved_captions = []

                for candidate in retrieved:
                    top_image = candidate.get_top_image()
                    if not top_image:
                        continue
                    retrieved_scores.append(candidate.aggregated_score)
                    retrieved_names.append(candidate.landmark_name)
                    retrieved_images.append(top_image.image_path)
                    retrieved_captions.append(top_image.caption)

                result["retrieved_scores"] = retrieved_scores
                result["retrieved_names"] = retrieved_names
                result["retrieved_images"] = retrieved_images
                result["retrieved_captions"] = retrieved_captions

                logger.info(
                    f"[{correlation_id}] Найдено {len(retrieved)} кандидатов, "
                    f"top score: {retrieved_scores[0]:.4f}"
                )
            except RuntimeError as e:
                result["error"] = str(e)
                result["timing"] = timing
                logger.error(f"[{correlation_id}] Ошибка retrieval: {e}")
                return result

            # 3. VLM reranking
            try:
                parsed = await self._generate_vlm_prediction(
                    image, retrieved, timing
                )
                result["name"] = parsed["name"]
                result["description"] = parsed["description"]
                logger.info(
                    f"[{correlation_id}] VLM выбрал: {parsed['name']} "
                    f"(P(yes)={parsed.get('p_yes', 0):.4f})"
                )
            except Exception as e:
                result["error"] = f"VLM ошибка: {e}"
                result["timing"] = timing
                logger.error(f"[{correlation_id}] VLM ошибка: {e}")
                return result

            # 4. Уверенность = p_yes напрямую
            p_yes_val = parsed.get("p_yes", 0.0)
            result["confidence"] = round(p_yes_val, 4)
            logger.info(
                f"[{correlation_id}] P(yes)={p_yes_val:.4f}"
            )

            # 5. Интернет-поиск при низком P(yes) от VLM
            if use_internet_search and p_yes_val < self.vlm_threshold:
                logger.info(
                    f"[{correlation_id}] P(yes)={p_yes_val:.4f} "
                    f"< vlm_threshold={self.vlm_threshold}, "
                    f"запускаем интернет-поиск"
                )
                await self._enhance_with_internet_search(
                    image=image,
                    image_path=image_path,
                    retrieved_scores=retrieved_scores,
                    retrieved_names=retrieved_names,
                    retrieved_descs=retrieved_captions,
                    result=result,
                    timing=timing,
                )

            result["timing"] = timing
            self.metrics.update(result)

            logger.info(
                f"[{correlation_id}] Готово за "
                f"{sum(timing.values()):.3f}с"
            )
            return result

        except Exception as e:
            result["error"] = "Внутренняя ошибка"
            result["timing"] = timing
            logger.error(
                f"[{correlation_id}] Неожиданная ошибка: {e}\n"
                f"{traceback.format_exc()}"
            )
            self.metrics.update(result)
            return result

    # ------------------------------------------------------------------
    # Пакетная обработка
    # ------------------------------------------------------------------

    async def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        use_internet_search: bool = True,
        max_concurrency: int = 4,
    ) -> List[Dict]:
        """
        Пакетная обработка изображений с ограниченным параллелизмом.

        Args:
            image_paths: Список путей к изображениям
            use_internet_search: Включить интернет-поиск
            max_concurrency: Максимум одновременных запросов
        """
        total = len(image_paths)
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _predict_with_sem(i: int, path: Union[str, Path]) -> Dict:
            async with semaphore:
                logger.info(f"Прогресс: {i + 1}/{total}")
                return await self.predict(path, use_internet_search)

        tasks = [
            _predict_with_sem(i, path)
            for i, path in enumerate(image_paths)
        ]
        return list(await asyncio.gather(*tasks))

    # ------------------------------------------------------------------
    # Очистка ресурсов
    # ------------------------------------------------------------------

    async def cleanup(self):
        """Освобождает ресурсы сервиса."""
        await self._cleanup_resources()

    async def _cleanup_resources(self):
        """Закрывает HTTP-клиент, YandexSearchService и удаляет retriever."""
        logger.info("Освобождение ресурсов...")
        if hasattr(self, "sglang_client"):
            await self.sglang_client.close()
        if hasattr(self, "_yandex_service") and self._yandex_service:
            self._yandex_service.close()
            self._yandex_service = None
        if hasattr(self, "retriever"):
            del self.retriever

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._cleanup_resources()


if __name__ == "__main__":
    pass