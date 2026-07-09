# src/services/ai_tour_guide.py
"""
Сервис распознавания достопримечательностей.

Пайплайн:
  1. SigLIP + FAISS — поиск top-K кандидатов
  2. VLM Reranking через vLLM (попарное сравнение, P(yes))
  3. Расчёт уверенности
  4. Интернет-поиск при низкой уверенности (Yandex + Wikipedia)
"""

import asyncio
import hashlib
import logging
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from cachetools import LRUCache, TTLCache
from PIL import Image

from src.core.metrics import METRICS
from src.rag.landmark_retriever import (
    LandmarkRetrievalResult,
    LandmarkRetriever,
)

from .calibration import ConfidenceCalibrator
from .image_utils import image_to_base64_data_uri, to_pil_image
from .internet_search import InternetSearchService, needs_translation
from .scoring import parse_logprobs_p_yes, text_response_to_p_yes
from .translator import YandexTranslator
from .vllm_client import VLLMClient
from .yandex_search import (
    WikipediaService,
    YandexSearchService,
)

logger = logging.getLogger(__name__)


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

    # vLLM сервер
    vllm_base_url: str = "http://localhost:30000/v1"
    vllm_model_name: str = "qwen2-vl-2b-r16"
    vllm_timeout: float = 30.0
    vllm_max_retries: int = 3

    # Локальный путь к SigLIP модели (пустая строка = загрузка с HuggingFace)
    siglip_model_path: str = ""

    # Базовая директория изображений галереи.
    # image_path в gallery_metadata.json — просто имя файла (photo.jpg),
    # полный путь = images_base_dir / image_path
    images_base_dir: str = ""

    # Yandex Cloud API (перевод и поиск по изображению)
    yc_folder_id: str = ""
    yc_api_key: str = ""

    # Retrieval
    top_k_retrieval: int = 10
    faiss_k: int = 100

    # Максимум параллельных VLM-запросов при reranking
    vlm_semaphore_limit: int = 10

    # VLM параметры
    caption_max_length: int = 300
    max_new_tokens: int = 256
    temperature: float = 0.0

    # Уверенность
    # vlm_threshold — порог на сыром p_yes для accept/reject (known/unknown) и
    # запуска интернет-поиска. Youden-оптимальный порог LoRA из open-set анализа
    # (recompute_e2e_metrics.py, sweep по TPR+TNR). Должен совпадать с
    # ACCEPT_THRESHOLD в calibration.py. Калибруется только отдаваемое число.
    vlm_threshold: float = 0.472656

    # Калибровка отдаваемой уверенности (isotonic, фит на val).
    # Прод показывает калиброванный P(correct), а не сырой P(yes) реранкера
    # (сырой переуверен: ECE 0.16 → 0.006 после isotonic). Преобразование
    # монотонно — порог отсечки и решение known/unknown не меняет.
    calibrate_confidence: bool = True
    calibration_curve_path: str = "data/calibration/isotonic_reranker.json"

    # Бэнды уверенности для UI (по калиброванному P(correct)). Границы подобраны
    # на accepted-тесте: низкая (<0.3) факт.точность ~27%, средняя (0.3–0.6) ~44%,
    # высокая (>=0.6) ~78% — подпись честно отражает вероятность верного ответа.
    confidence_band_high: float = 0.6
    confidence_band_medium: float = 0.3

    # Confidence-уровни интернет-поиска:
    #   exact   — vlm_name точно совпал с ключом Wikipedia
    #   partial — vlm_name частично совпал с ключом Wikipedia
    #   fallback_wiki — выбран ключ с минимальным числом слов
    #   fallback_retrieval — интернет-поиск полностью провалился,
    #                        возвращаем top-1 из retrieval
    internet_confidence_exact: float = 0.90
    internet_confidence_partial: float = 0.80
    internet_confidence_fallback_wiki: float = 0.70
    internet_confidence_fallback_retrieval: float = 0.65

    # Интернет-поиск
    enable_internet_search: bool = True
    # Пропустить этап 1 VLM (без подсказок) — полезно для слабых моделей
    # которые почти всегда возвращают "unknown" без контекста
    skip_internet_search_stage1: bool = False

    # Устройство для SigLIP-энкодера (cpu/cuda)
    device: str = "cuda"

    def to_dict(self) -> dict:
        """Конвертирует конфигурацию в словарь."""
        return asdict(self)

    @classmethod
    def from_settings(cls, settings) -> "AITourGuideConfig":
        """Собирает конфиг сервиса из app-настроек (Settings) — единая точка
        маппинга, чтобы поля не расходились между окружением и сервисом."""
        return cls(
            index_dir=str(settings.index_dir_abs),
            vllm_base_url=settings.vllm_base_url,
            vllm_model_name=settings.vllm_model_name,
            vllm_timeout=settings.vllm_timeout,
            vllm_max_retries=settings.vllm_max_retries,
            device=settings.device,
            siglip_model_path=settings.siglip_model_path,
            images_base_dir=settings.images_base_dir,
            top_k_retrieval=settings.top_k_retrieval,
            vlm_threshold=settings.confidence_threshold,
            enable_internet_search=settings.enable_internet_search,
            calibrate_confidence=settings.calibrate_confidence,
            calibration_curve_path=str(settings.calibration_curve_path_abs),
            yc_folder_id=settings.yc_folder_id,
            yc_api_key=settings.yc_api_key,
        )


# Собственные счётчики для /v1/health и /v1/info — дублируют Prometheus,
# но не зависят от его приватного API.
_stats: dict[str, float] = {
    "successful": 0.0,
    "failed": 0.0,
    "internet": 0.0,
    "confidence_last": 0.0,
}


def _metrics_to_dict() -> dict:
    """
    Возвращает снимок метрик в виде словаря для /v1/health и /v1/info.

    Читает значения из собственных счётчиков _stats — не зависит
    от внутреннего API prometheus_client.
    """
    successful = _stats["successful"]
    failed = _stats["failed"]
    total = successful + failed
    return {
        "total_requests": int(total),
        "successful_requests": int(successful),
        "failed_requests": int(failed),
        "success_rate": round(successful / total, 4) if total > 0 else 0.0,
        "internet_searches": int(_stats["internet"]),
        "confidence_last": round(_stats["confidence_last"], 4),
    }


def _update_metrics(result: dict, image_size_bytes: int = 0) -> None:
    """Обновляет METRICS и собственные счётчики _stats на основе результата predict()."""
    if result.get("error"):
        METRICS.requests_total.labels(status="error").inc()
        _stats["failed"] += 1
        return

    METRICS.requests_total.labels(status="success").inc()
    _stats["successful"] += 1

    conf = result.get("confidence", 0.0)
    METRICS.confidence_last.set(conf)
    METRICS.confidence_histogram.observe(conf)
    _stats["confidence_last"] = conf

    # Источник ответа
    source = result.get("source", "retrieval")
    METRICS.source_total.labels(source=source).inc()

    # Флаг unknown
    if result.get("unknown"):
        METRICS.unknown_total.inc()

    # Интернет-поиск
    if source == PredictionSource.INTERNET.value:
        METRICS.internet_searches_total.inc()
        _stats["internet"] += 1

    # Тайминги
    timing = result.get("timing", {})
    if "retrieval" in timing:
        METRICS.retrieval_duration.observe(timing["retrieval"])
    if "vlm_generation" in timing:
        METRICS.vlm_duration.observe(timing["vlm_generation"])
    if "internet_search" in timing:
        METRICS.internet_search_duration.observe(timing["internet_search"])

    total_time = sum(timing.values())
    if total_time > 0:
        METRICS.total_duration.observe(total_time)

    # Размер изображения
    if image_size_bytes > 0:
        METRICS.image_size_bytes.observe(image_size_bytes)


class AITourGuide:
    """
    Сервис распознавания достопримечательностей по изображениям.

    Использует vLLM сервер для VLM reranking через OpenAI API.
    """

    def __init__(self, config: "AITourGuideConfig"):
        """
        Args:
            config: Конфигурация сервиса
        """
        self.config = config
        self.vlm_threshold = config.vlm_threshold
        self.top_k_retrieval = config.top_k_retrieval
        self.faiss_k = config.faiss_k
        self.caption_max_length = config.caption_max_length
        # Базовая директория изображений галереи (может быть пустой строкой)
        self.images_base_dir = (
            Path(config.images_base_dir) if config.images_base_dir else None
        )

        # Ключи Yandex Cloud берём из конфига (единая точка конфигурации)
        yc_folder_id = config.yc_folder_id or None
        yc_api_key = config.yc_api_key or None

        self.translator = YandexTranslator(
            yc_folder_id=yc_folder_id,
            yc_api_key=yc_api_key,
        )

        # Переиспользуемый экземпляр YandexSearchService
        # (создаётся один раз, сессия requests.Session живёт всё время)
        self._yandex_service: YandexSearchService | None = None
        if yc_folder_id and yc_api_key:
            self._yandex_service = YandexSearchService(
                yc_folder_id=yc_folder_id,
                yc_api_key=yc_api_key,
            )

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

        logger.info("Инициализация vLLM клиента...")
        self.vllm_client = VLLMClient(
            base_url=config.vllm_base_url,
            model_name=config.vllm_model_name,
            timeout=config.vllm_timeout,
            max_retries=config.vllm_max_retries,
        )

        # Сервис fallback-поиска в интернете (Yandex + Wikipedia + верификация)
        self.internet_search = InternetSearchService(
            vllm_client=self.vllm_client,
            yandex_service=self._yandex_service,
            config=config,
        )

        # Калибратор отдаваемой уверенности (isotonic-кривая, фит на val).
        # Без кривой отдаём сырой p_yes; монотонность порог и решения не меняет.
        self.calibrator = ConfidenceCalibrator(
            curve_path=(
                config.calibration_curve_path
                if config.calibrate_confidence
                else None
            ),
            band_high=config.confidence_band_high,
            band_medium=config.confidence_band_medium,
        )

        # base64 data URI изображений галереи, заполняется лениво.
        self._gallery_image_cache: LRUCache = LRUCache(maxsize=500)

        # Результаты predict по MD5 входных байтов, TTL 1 час.
        self._predict_cache: TTLCache = TTLCache(maxsize=200, ttl=3600)

        self._is_ready = True
        logger.info("AITourGuide готов")
        logger.info(f"  vLLM: {config.vllm_base_url}")
        logger.info(f"  Модель: {config.vllm_model_name}")
        logger.info(f"  VLM порог (p_yes): {config.vlm_threshold}")

    # Health check и метрики

    async def health_check(self) -> dict[str, Any]:
        """Проверяет состояние сервиса и его компонентов."""
        safe_config = {
            k: v for k, v in self.config.to_dict().items() if k != "index_dir"
        }
        health: dict[str, Any] = {
            "status": "healthy" if self._is_ready else "not_ready",
            "ready": self._is_ready,
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "config": safe_config,
            "metrics": _metrics_to_dict(),
        }

        try:
            health["components"]["retriever"] = {
                "status": "ok",
                "index_size": len(self.retriever.gallery_metadata),
            }

            vllm_ok = await self.vllm_client.health_check()
            health["components"]["vllm"] = {
                "status": "ok" if vllm_ok else "error",
                "base_url": self.config.vllm_base_url,
                "model": self.config.vllm_model_name,
            }

            if not vllm_ok:
                health["status"] = "degraded"
                health["error"] = "vLLM сервер недоступен"

        except Exception as e:
            health["status"] = "degraded"
            health["error"] = str(e)

        return health

    # VLM через vLLM

    def _load_and_encode_gallery_image(self, candidate_image: str) -> str:
        """
        Синхронно загружает изображение галереи и кодирует в base64 data URI.

        Вызывается через asyncio.to_thread() чтобы не блокировать event loop.

        Args:
            candidate_image: Имя файла или относительный путь изображения

        Returns:
            base64 data URI строка
        """
        candidate_path = candidate_image
        if self.images_base_dir is not None:
            candidate_path = str(self.images_base_dir / candidate_image)

        with Image.open(candidate_path) as img:
            pil_img = img.convert("RGB")
            return image_to_base64_data_uri(pil_img)

    async def _get_candidate_image_uri(self, candidate_image: str) -> str:
        """
        Возвращает base64 data URI для изображения кандидата из галереи.

        Результат кэшируется в LRU-кэше (_gallery_image_cache).
        Чтение файла выполняется в отдельном потоке через asyncio.to_thread()
        чтобы не блокировать event loop.

        Args:
            candidate_image: Имя файла или относительный путь изображения

        Returns:
            base64 data URI строка
        """
        if candidate_image in self._gallery_image_cache:
            return cast(str, self._gallery_image_cache[candidate_image])

        uri = await asyncio.to_thread(
            self._load_and_encode_gallery_image, candidate_image
        )
        self._gallery_image_cache[candidate_image] = uri
        logger.debug(
            f"Закэшировано gallery-изображение: {candidate_image} "
            f"(кэш: {len(self._gallery_image_cache)} записей)"
        )
        return uri

    async def prepare_vlm_messages(
        self,
        query_image: Image.Image | str | bytes,
        candidate_image: str,
        candidate_caption: str,
        candidate_name: str,
        query_uri: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Формирует сообщения в формате OpenAI API для VLM reranking.

        Args:
            query_image: Запросное изображение (PIL, путь или bytes)
            candidate_image: Путь к изображению кандидата
            candidate_caption: Описание кандидата
            candidate_name: Название кандидата
            query_uri: Готовый base64 data URI query-изображения. Если передан —
                повторное кодирование пропускается. При reranking query-картинка
                одна на всех кандидатов, поэтому кодируется один раз в вызывающем
                коде (см. _generate_vlm_prediction).

        Returns:
            Список сообщений для OpenAI API
        """
        if query_uri is None:
            query_img = to_pil_image(query_image)
            query_uri = image_to_base64_data_uri(query_img)
        # Изображение кандидата берём из кэша (lazy load + кэш по пути)
        candidate_uri = await self._get_candidate_image_uri(candidate_image)
        caption = candidate_caption[: self.caption_max_length]

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
                            f'landmark: "{candidate_name}"?\n'
                            f"Candidate details: {caption}\n"
                            f"Answer only with Yes or No."
                        ),
                    },
                ],
            }
        ]

    # Основной пайплайн предсказания

    def _init_result(self) -> dict:
        """Возвращает пустую структуру результата."""
        return {
            "name": "",
            "description": "",
            "confidence": 0.0,
            "unknown": False,
            "source": PredictionSource.RETRIEVAL.value,
            "winner_images": [],  # все изображения победителя из базы
            "winner_landmark_id": "",  # landmark_id победителя
            "retrieved_names": [],
            "retrieved_scores": [],
            "retrieved_p_yes": [],  # p_yes от VLM для каждого кандидата
            "retrieved_images": [],
            "retrieved_captions": [],
            "search_query": None,
            "error": None,
            "timing": {},
        }

    async def _validate_and_load_image(
        self,
        image_input: str | Path | bytes,
        timing: dict[str, float],
    ) -> tuple[Path | str, Image.Image]:
        """Загружает изображение из пути или байтов."""
        t0 = time.time()

        if isinstance(image_input, bytes):
            try:
                image = Image.open(BytesIO(image_input)).convert("RGB")
                timing["image_load"] = round(time.time() - t0, 3)
                return "bytes_image", image
            except Exception as e:
                raise RuntimeError(f"Не удалось загрузить изображение: {e}") from e

        image_path = Path(image_input)
        try:
            image_path = image_path.resolve(strict=True)
        except (ValueError, OSError, RuntimeError) as e:
            raise ValueError(f"Недоступный путь: {e}") from e

        if not image_path.is_file():
            raise ValueError(f"Не является файлом: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Не удалось открыть изображение: {e}") from e

        timing["image_load"] = round(time.time() - t0, 3)
        return image_path, image

    async def _retrieve_candidates(
        self,
        image: Image.Image,
        timing: dict[str, float],
    ) -> list[LandmarkRetrievalResult]:
        """Ищет кандидатов через SigLIP + FAISS."""
        t0 = time.time()
        retrieved = await asyncio.to_thread(
            self.retriever.retrieve,
            image,
            top_k=self.top_k_retrieval,
            faiss_k=self.faiss_k,
        )
        timing["retrieval"] = round(time.time() - t0, 3)
        logger.debug(
            f"retrieval: duration={timing['retrieval']}s, "
            f"candidates_found={len(retrieved)}"
        )

        if not retrieved:
            raise RuntimeError("Кандидаты не найдены")

        return retrieved

    async def _generate_vlm_prediction(
        self,
        image: Image.Image,
        retrieved: list[LandmarkRetrievalResult],
        timing: dict[str, float],
    ) -> dict[str, Any]:
        """
        Выбирает лучшего кандидата через VLM reranking.

        Для каждого кандидата вычисляет P(yes) через vLLM logprobs,
        возвращает кандидата с максимальной вероятностью.
        """
        t0 = time.time()

        # Собираем кандидатов с изображениями
        candidates = []
        for cand in retrieved:
            top_image = cand.get_top_image()
            if top_image:
                description = (
                    top_image.guide_description
                    or top_image.caption_landmark
                    or top_image.caption
                )
                # Русское название для отображения пользователю
                display_name = top_image.landmark_name_ru or cand.landmark_name
                candidates.append(
                    {
                        "landmark_id": cand.landmark_id,
                        "landmark_name": cand.landmark_name,
                        "landmark_name_ru": display_name,
                        "image_path": top_image.image_path,
                        "caption": top_image.caption,
                        "description": description,
                    }
                )

        if not candidates:
            raise RuntimeError("Нет кандидатов для VLM reranking")

        METRICS.vlm_candidates_count.observe(len(candidates))

        # Query одно на всех кандидатов — кодируем в base64 один раз: меньше
        # работы CPU и попадание в prefix-кэш vLLM по токенам query-картинки.
        query_img = to_pil_image(image)
        query_uri = await asyncio.to_thread(image_to_base64_data_uri, query_img)

        async def _score_candidate(cand: dict) -> dict:
            """Вычисляет P(yes) для одного кандидата."""
            try:
                messages = await self.prepare_vlm_messages(
                    image,
                    cand["image_path"],
                    cand["caption"],
                    cand["landmark_name"],
                    query_uri=query_uri,
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
                    # logprobs не вернулись — fallback на текстовый ответ
                    p_yes = text_response_to_p_yes(response)
                    logger.debug(
                        f"logprobs пусты, текст fallback: "
                        f"p_yes={p_yes} "
                        f"кандидат: {cand['landmark_name']!r}"
                    )

                return {**cand, "p_yes": p_yes}

            except Exception as e:
                logger.warning(f"Ошибка VLM reranking для {cand['landmark_id']}: {e}")
                return {**cand, "p_yes": 0.0}

        # Параллельные запросы с лимитом config.vlm_semaphore_limit — vLLM их батчит.
        semaphore = asyncio.Semaphore(self.config.vlm_semaphore_limit)

        async def _score_with_sem(cand: dict) -> dict:
            async with semaphore:
                return await _score_candidate(cand)

        scored = await asyncio.gather(*[_score_with_sem(c) for c in candidates])
        results = list(scored)

        # Выбираем кандидата с максимальным P(yes)
        results.sort(key=lambda x: x["p_yes"], reverse=True)
        best = results[0]

        timing["vlm_generation"] = round(time.time() - t0, 3)
        logger.debug(
            f"vlm_reranking: candidates={len(candidates)}, "
            f"best_landmark={best['landmark_name']!r}, "
            f"best_p_yes={round(best['p_yes'], 4)}, "
            f"duration={timing['vlm_generation']}s"
        )

        # Строим словарь landmark_name → p_yes для всех кандидатов
        # (используется для отображения в UI)
        all_p_yes = {r["landmark_name"]: round(r["p_yes"], 4) for r in results}

        return {
            "name": best["landmark_name_ru"],
            "description": best["description"],
            "p_yes": best["p_yes"],
            "all_p_yes": all_p_yes,
        }

    async def _enhance_with_internet_search(
        self,
        image: Image.Image,
        image_path: Path | str,
        retrieved_names: list[str],
        retrieved_descs: list[str],
        result: dict,
        timing: dict[str, float],
    ):
        """Улучшает результат через интернет-поиск."""
        search_input = image if image_path == "bytes_image" else str(image_path)

        t0 = time.time()
        search_result = await self.internet_search.search(
            image=search_input,
            retrieved_descs=retrieved_descs,
            retrieved_names=retrieved_names,
            fallback_name=result["name"],
        )
        timing["internet_search"] = round(time.time() - t0, 3)

        if not search_result.get("found"):
            return

        # Интернет-поиск провалился и вернул top-1 из retrieval —
        # result уже содержит правильные name, description, confidence
        # и winner_images от VLM reranking. Ничего не меняем.
        if search_result.get("is_fallback_retrieval", False):
            logger.info("Интернет-поиск провалился, используем top-1 из retrieval")
            return

        result["source"] = PredictionSource.INTERNET.value
        result["search_query"] = search_result["query"]

        # Очищаем winner_images: фото из базы относятся к другому объекту
        # (тому что был top-1 при retrieval), а не к найденному в интернете.
        result["winner_images"] = []
        result["winner_landmark_id"] = ""

        # Переводим название и описание на русский, если нужно (см. _needs_translation).
        name = search_result["name"]
        description = search_result["description"]
        t_translate = time.time()
        try:
            if needs_translation(name):
                translated_name = await self.translator.translate(
                    name, target_language="ru", source_language="en"
                )
                if translated_name:
                    name = translated_name

            if description and needs_translation(description):
                translated_desc = await self.translator.translate(
                    description, target_language="ru", source_language="en"
                )
                if translated_desc:
                    description = translated_desc
        except Exception as e:
            logger.warning(f"Ошибка перевода: {e}")
        timing["translation"] = round(time.time() - t_translate, 3)

        result["name"] = name
        result["description"] = description

        # Верификация через VLM по оригинальному (английскому) названию —
        # даёт p_yes в той же шкале, что и reranking.
        t_verify = time.time()
        # Открываем WikipediaService для получения thumbnail —
        # промпт будет идентичен prepare_vlm_messages() (обучение).
        async with WikipediaService(language="ru", fallback_lang="en") as wiki_svc:
            verified_p_yes = await self.internet_search.verify(
                image=image,
                landmark_name=search_result["name"],
                description=search_result["description"],
                wiki_service=wiki_svc,
            )
        timing["vlm_verification"] = round(time.time() - t_verify, 3)
        result["confidence"] = round(verified_p_yes, 4)
        logger.info(
            f"Интернет-поиск завершён: "
            f"name='{search_result['name']}' "
            f"(переведено: '{result['name']}') "
            f"confidence={result['confidence']}"
        )

    async def predict(
        self,
        image_input: str | Path | bytes,
        use_internet_search: bool = True,
    ) -> dict:
        """
        Распознаёт достопримечательность на изображении.

        Args:
            image_input: Путь к изображению или байты
            use_internet_search: Включить поиск при низкой уверенности

        Returns:
            Словарь с name, description, confidence, source, timing
        """
        timing: dict[str, float] = {}
        result = self._init_result()
        correlation_id = str(uuid.uuid4())[:8]
        _image_size = len(image_input) if isinstance(image_input, bytes) else 0

        input_id = (
            f"bytes ({_image_size} bytes)"
            if isinstance(image_input, bytes)
            else str(image_input)
        )

        # Кешируем только bytes-вход (файл под тем же путём может измениться).
        # use_internet_search входит в ключ: с поиском и без — разные результаты.
        _cache_key: str | None = None
        if isinstance(image_input, bytes):
            _cache_key = (
                f"{hashlib.md5(image_input).hexdigest()}"
                f":internet={int(use_internet_search)}"
            )
            if _cache_key in self._predict_cache:
                logger.info(
                    f"[{correlation_id}] Возврат из predict-кеша "
                    f"(md5={_cache_key[:8]}…)"
                )
                return dict(self._predict_cache[_cache_key])

        logger.info(f"[{correlation_id}] Предсказание для {input_id}")

        try:
            # 1. Загрузка изображения
            try:
                image_path, image = await self._validate_and_load_image(
                    image_input, timing
                )
                logger.info(f"[{correlation_id}] Изображение загружено: {image.size}")
            except (ValueError, FileNotFoundError, RuntimeError) as e:
                result["error"] = str(e)
                result["timing"] = timing
                logger.error(f"[{correlation_id}] Ошибка загрузки: {e}")
                return result

            # 2. Retrieval кандидатов
            try:
                retrieved = await self._retrieve_candidates(image, timing)
                retrieved_scores = []
                retrieved_names = []  # RU-названия для UI
                retrieved_names_en = []  # EN-названия для маппинга p_yes
                retrieved_images = []
                retrieved_captions = []

                for candidate in retrieved:
                    top_image = candidate.get_top_image()
                    if not top_image:
                        continue
                    retrieved_scores.append(candidate.aggregated_score)
                    # Русское название для отображения пользователю
                    display_name = top_image.landmark_name_ru or candidate.landmark_name
                    retrieved_names.append(display_name)
                    # EN-название для маппинга p_yes из VLM
                    retrieved_names_en.append(candidate.landmark_name)
                    retrieved_images.append(top_image.image_path)
                    retrieved_captions.append(top_image.caption)

                result["retrieved_scores"] = retrieved_scores
                result["retrieved_names"] = retrieved_names
                result["retrieved_images"] = retrieved_images
                result["retrieved_captions"] = retrieved_captions

                # Все изображения top-1 кандидата для слайдера
                if retrieved:
                    winner = retrieved[0]
                    result["winner_landmark_id"] = winner.landmark_id
                    result["winner_images"] = [
                        img.image_path
                        for _, img in sorted(
                            winner.gallery_images,
                            key=lambda x: x[0],
                            reverse=True,
                        )
                    ]

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
                parsed = await self._generate_vlm_prediction(image, retrieved, timing)
                result["name"] = parsed["name"]
                result["description"] = parsed["description"]
                # Заполняем p_yes для каждого кандидата в том же порядке
                # что retrieved_names — для отображения в UI. Значения
                # калибруем (isotonic P(correct)), как и главную уверенность;
                # монотонность сохраняет порядок сортировки кандидатов.
                # Маппинг строится по EN-именам (all_p_yes ключи — EN).
                all_p_yes = parsed.get("all_p_yes", {})
                result["retrieved_p_yes"] = [
                    round(self.calibrator.calibrate(all_p_yes.get(name_en, 0.0)), 4)
                    for name_en in retrieved_names_en
                ]
                logger.info(
                    f"[{correlation_id}] VLM выбрал: {parsed['name']} "
                    f"(P(yes)={parsed.get('p_yes', 0):.4f})"
                )
            except Exception as e:
                result["error"] = f"VLM ошибка: {e}"
                result["timing"] = timing
                logger.error(f"[{correlation_id}] VLM ошибка: {e}")
                return result

            # 4. Уверенность = сырой p_yes (все решения ниже — на нём).
            p_yes_val = parsed.get("p_yes", 0.0)
            result["confidence"] = round(p_yes_val, 4)
            logger.info(f"[{correlation_id}] P(yes)={p_yes_val:.4f}")

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
                    retrieved_names=retrieved_names,
                    retrieved_descs=retrieved_captions,
                    result=result,
                    timing=timing,
                )

            # 6. Определяем флаг unknown:
            # достопримечательность не распознана если confidence
            # всё ещё ниже порога после всех этапов (VLM reranking +
            # интернет-поиск + VLM верификация).
            # При unknown=True очищаем name/description и всё из базы: фото
            # победителя и кандидаты принадлежат неверному top-1 объекту.
            # Пользователь должен видеть только своё загруженное фото.
            if result["confidence"] < self.vlm_threshold:
                result["unknown"] = True
                result["name"] = ""
                result["description"] = ""
                result["winner_images"] = []
                result["winner_landmark_id"] = ""
                result["retrieved_images"] = []
                result["retrieved_names"] = []
                result["retrieved_scores"] = []
                result["retrieved_p_yes"] = []
                result["retrieved_captions"] = []
                logger.info(
                    f"[{correlation_id}] Достопримечательность "
                    f"не распознана (confidence="
                    f"{result['confidence']:.4f} < "
                    f"{self.vlm_threshold})"
                )

            # 7-8. Калибровка + бэнд отдаваемой уверенности. Здесь confidence —
            # всегда reranker p_yes (БД-путь или интернет-верификация в той же
            # шкале), поэтому калибруем единообразно; решения уже приняты на сыром.
            if result.get("unknown"):
                result["confidence_band"] = None
            else:
                result["confidence"] = round(
                    self.calibrator.calibrate(result["confidence"]), 4
                )
                result["confidence_band"] = self.calibrator.band(result["confidence"])

            result["timing"] = timing
            _update_metrics(result, image_size_bytes=_image_size)

            # Кешируем только успешные результаты (без error)
            if _cache_key and not result.get("error"):
                self._predict_cache[_cache_key] = result
                logger.debug(
                    f"[{correlation_id}] Результат закешован "
                    f"(md5={_cache_key[:8]}…, "
                    f"кэш: {len(self._predict_cache)} записей)"
                )

            logger.info(f"[{correlation_id}] Готово за {sum(timing.values()):.3f}с")
            return result

        except Exception as e:
            result["error"] = "Внутренняя ошибка"
            result["timing"] = timing
            logger.error(
                f"[{correlation_id}] Неожиданная ошибка: {e}\n{traceback.format_exc()}"
            )
            _update_metrics(result, image_size_bytes=_image_size)
            return result

    # Очистка ресурсов

    async def cleanup(self):
        """Закрывает HTTP-клиент, YandexSearchService и удаляет retriever."""
        logger.info("Освобождение ресурсов...")
        await self.vllm_client.close()
        if self._yandex_service:
            self._yandex_service.close()
            self._yandex_service = None
        del self.retriever

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()


if __name__ == "__main__":
    pass
