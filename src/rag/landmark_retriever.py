# src/rag/landmark_retriever.py
"""
Retriever достопримечательностей на основе FAISS-индекса.

Пайплайн:
  1. Кодирование запросного изображения через SigLIP
  2. Поиск top-K изображений в FAISS-индексе
  3. Агрегация оценок по landmark_id
  4. Возврат отсортированного списка кандидатов
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from PIL import Image

try:
    from src.rag.indexing_v2 import IndexBuilder, IndexConfig
except ImportError:
    from indexing_v2 import IndexBuilder, IndexConfig  # type: ignore

logger = logging.getLogger(__name__)


# Метаданные изображения галереи


@dataclass
class GalleryImageMetadata:
    """
    Метаданные одного изображения в FAISS-индексе.

    Каждое изображение принадлежит конкретной достопримечательности
    и содержит описание, координаты и прочие атрибуты.
    """

    image_id: int
    image_path: str
    landmark_id: str
    landmark_name: str
    caption_landmark: str  # Сводное описание достопримечательности
    caption: str  # Описание конкретного изображения

    # Русское название для отображения пользователю
    landmark_name_ru: str = ""

    # Оценки качества
    confidence: float = 0.0
    mean_conf: float = 0.0
    max_conf: float = 0.0

    # Описание гида (основной текст для пользователя)
    guide_description: str = ""

    # Дополнительные метаданные
    wikidata_id: str = ""
    coordinates: dict[str, float] = field(default_factory=dict)
    country_ru: str = ""
    country_en: str = ""
    city_ru: str = ""
    city_en: str = ""
    landmark_type: dict[str, str] = field(default_factory=dict)
    year_built: str = ""
    architectural_style_ru: list[str] = field(default_factory=list)
    architectural_style_en: list[str] = field(default_factory=list)
    heritage_status_ru: str = ""
    heritage_status_en: str = ""
    wikipedia_url_ru: str = ""
    wikipedia_url_en: str = ""
    website: str = ""
    wikidata_description_ru: str = ""
    wikidata_description_en: str = ""

    # Поля для обучения (нужны для корректной перезагрузки индекса)
    row_idx: int = -1
    num_gallery_images: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Сериализует метаданные в словарь для JSON."""
        return asdict(self)


# Результат retrieval


@dataclass
class LandmarkRetrievalResult:
    """
    Результат поиска для одной достопримечательности.

    Содержит агрегированную оценку и все найденные изображения галереи.
    """

    landmark_id: str
    landmark_name: str
    aggregated_score: float
    gallery_images: list[tuple[float, GalleryImageMetadata]]
    rank: int = 0  # Позиция в результатах (с 1)

    def get_top_image(self) -> GalleryImageMetadata | None:
        """Возвращает изображение с наибольшей оценкой."""
        if not self.gallery_images:
            return None
        return max(self.gallery_images, key=lambda x: x[0])[1]

    def get_metadata(self) -> dict[str, Any]:
        """Возвращает метаданные лучшего изображения."""
        top = self.get_top_image()
        return top.to_dict() if top else {}

    def to_dict(self) -> dict[str, Any]:
        """Сериализует результат в словарь."""
        top = self.get_top_image()
        return {
            "landmark_id": self.landmark_id,
            "landmark_name": self.landmark_name,
            "score": float(self.aggregated_score),
            "rank": self.rank,
            "num_gallery_images": len(self.gallery_images),
            "metadata": self.get_metadata() if top else {},
        }


# Агрегация оценок


def aggregate_scores(
    scores: list[float],
    mode: str = "weighted_top2",
    alpha: float = 0.7,
) -> float:
    """
    Агрегирует оценки нескольких изображений в одну оценку landmark.

    Args:
        scores: Список оценок схожести
        mode: Стратегия агрегации (max / top2_mean / weighted_top2)
        alpha: Вес первой оценки для weighted_top2

    Returns:
        Агрегированная оценка
    """
    if not scores:
        return 0.0

    sorted_scores = sorted(scores, reverse=True)

    if mode == "max":
        return sorted_scores[0]

    if mode == "top2_mean":
        if len(sorted_scores) == 1:
            return sorted_scores[0]
        return (sorted_scores[0] + sorted_scores[1]) / 2.0

    if mode == "weighted_top2":
        if len(sorted_scores) == 1:
            return sorted_scores[0]
        return alpha * sorted_scores[0] + (1 - alpha) * sorted_scores[1]

    # Fallback — неизвестный mode, возвращаем максимум
    return sorted_scores[0]


# Retriever


class LandmarkRetriever:
    """
    Retriever достопримечательностей на основе FAISS-индекса.

    Использование:
        retriever = LandmarkRetriever.from_index_dir("data/index/siglip")
        results = retriever.retrieve(query_image, top_k=10)
    """

    def __init__(
        self,
        index_builder: IndexBuilder,
        gallery_index: faiss.Index,
        gallery_metadata: list[GalleryImageMetadata],
        aggregation_mode: str = "weighted_top2",
        aggregation_alpha: float = 0.7,
    ):
        self.index_builder = index_builder
        self.gallery_index = gallery_index
        self.gallery_metadata = gallery_metadata
        self.aggregation_mode = aggregation_mode
        self.aggregation_alpha = aggregation_alpha

        logger.info(
            f"LandmarkRetriever инициализирован: "
            f"{len(gallery_metadata)} изображений, "
            f"агрегация={aggregation_mode}"
        )

    def _encode_query(self, image: Image.Image) -> np.ndarray | None:
        """Кодирует запросное изображение через SigLIP."""
        try:
            results = self.index_builder.encoder.encode_batch([image])
            if results and results[0] is not None:
                embedding, _, _ = results[0]
                return embedding
            return None
        except Exception as e:
            logger.error(f"Ошибка кодирования изображения: {e}")
            return None

    def _search_and_aggregate(
        self,
        query_embedding: np.ndarray,
        k: int = 50,
    ) -> list[LandmarkRetrievalResult]:
        """
        Ищет в FAISS и агрегирует результаты по landmark_id.

        Args:
            query_embedding: Эмбеддинг запроса
            k: Количество изображений для поиска в FAISS

        Returns:
            Список результатов, отсортированных по убыванию оценки
        """
        # Нормализуем запрос для косинусного сходства
        query_norm = query_embedding.copy()
        faiss.normalize_L2(query_norm.reshape(1, -1))

        distances, indices = self.gallery_index.search(
            query_norm.reshape(1, -1),
            min(k, self.gallery_index.ntotal),
        )

        # Группируем по landmark_id
        landmark_scores: dict[str, list[tuple[float, GalleryImageMetadata]]] = {}

        for dist, idx in zip(distances[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self.gallery_metadata):
                continue
            meta = self.gallery_metadata[idx]
            lid = meta.landmark_id
            if lid not in landmark_scores:
                landmark_scores[lid] = []
            landmark_scores[lid].append((float(dist), meta))

        # Агрегируем оценки по каждому landmark
        results = []
        for lid, scores_and_metas in landmark_scores.items():
            scores = [s for s, _ in scores_and_metas]
            agg_score = aggregate_scores(
                scores,
                mode=self.aggregation_mode,
                alpha=self.aggregation_alpha,
            )
            landmark_name = scores_and_metas[0][1].landmark_name
            results.append(
                LandmarkRetrievalResult(
                    landmark_id=lid,
                    landmark_name=landmark_name,
                    aggregated_score=agg_score,
                    gallery_images=scores_and_metas,
                )
            )

        results.sort(key=lambda x: x.aggregated_score, reverse=True)
        for rank, result in enumerate(results, start=1):
            result.rank = rank

        return results

    def retrieve(
        self,
        query_image: Image.Image,
        top_k: int = 10,
        faiss_k: int = 50,
    ) -> list[LandmarkRetrievalResult]:
        """
        Возвращает top-k достопримечательностей для запросного изображения.

        Args:
            query_image: PIL Image
            top_k: Количество возвращаемых достопримечательностей
            faiss_k: Количество изображений для поиска в FAISS
        """
        embedding = self._encode_query(query_image)
        if embedding is None:
            logger.error("Не удалось закодировать изображение")
            return []

        results = self._search_and_aggregate(embedding, k=faiss_k)
        return results[:top_k]

    @classmethod
    def from_index_dir(
        cls,
        index_dir: str,
        index_config: IndexConfig | None = None,
        aggregation_mode: str = "weighted_top2",
        aggregation_alpha: float = 0.7,
    ) -> LandmarkRetriever:
        """
        Загружает retriever из директории с готовым индексом.

        Ожидаемая структура директории:
            index_dir/
                gallery_index.faiss
                gallery_metadata.json

        Args:
            index_dir: Путь к директории индекса
            index_config: Конфигурация энкодера (по умолчанию SigLIP)
            aggregation_mode: Стратегия агрегации оценок
            aggregation_alpha: Вес для weighted_top2
        """
        index_dir_path = Path(index_dir)

        index_path = index_dir_path / "gallery_index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS индекс не найден: {index_path}")

        logger.info(f"Загрузка FAISS индекса из {index_path}")
        gallery_index = faiss.read_index(str(index_path))

        metadata_path = index_dir_path / "gallery_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Метаданные не найдены: {metadata_path}")

        logger.info(f"Загрузка метаданных из {metadata_path}")
        with open(metadata_path, encoding="utf-8") as f:
            metadata_dicts = json.load(f)

        gallery_metadata = [
            GalleryImageMetadata(
                image_id=d.get("image_id", 0),
                image_path=d.get("image_path", ""),
                landmark_id=d.get("landmark_id", ""),
                landmark_name=d.get("landmark_name", ""),
                landmark_name_ru=d.get("landmark_name_ru", ""),
                caption=d.get("caption", ""),
                caption_landmark=d.get("caption_landmark", ""),
                confidence=d.get("confidence", 0.0),
                mean_conf=d.get("mean_conf", 0.0),
                max_conf=d.get("max_conf", 0.0),
                guide_description=d.get("guide_description", ""),
                wikidata_id=d.get("wikidata_id", ""),
                coordinates=d.get("coordinates", {}),
                country_ru=d.get("country_ru", ""),
                country_en=d.get("country_en", ""),
                city_ru=d.get("city_ru", ""),
                city_en=d.get("city_en", ""),
                landmark_type=d.get("landmark_type", {}),
                year_built=d.get("year_built", ""),
                architectural_style_ru=d.get("architectural_style_ru", []),
                architectural_style_en=d.get("architectural_style_en", []),
                heritage_status_ru=d.get("heritage_status_ru", ""),
                heritage_status_en=d.get("heritage_status_en", ""),
                wikipedia_url_ru=d.get("wikipedia_url_ru", ""),
                wikipedia_url_en=d.get("wikipedia_url_en", ""),
                website=d.get("website", ""),
                wikidata_description_ru=d.get("wikidata_description_ru", ""),
                wikidata_description_en=d.get("wikidata_description_en", ""),
            )
            for d in metadata_dicts
        ]

        if index_config is None:
            index_config = IndexConfig()

        index_builder = IndexBuilder(index_config)

        logger.info(
            f"Загружено {len(gallery_metadata)} изображений, "
            f"FAISS размер: {gallery_index.ntotal}"
        )

        return cls(
            index_builder=index_builder,
            gallery_index=gallery_index,
            gallery_metadata=gallery_metadata,
            aggregation_mode=aggregation_mode,
            aggregation_alpha=aggregation_alpha,
        )
