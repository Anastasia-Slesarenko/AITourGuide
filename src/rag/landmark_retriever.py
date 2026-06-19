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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import faiss
import numpy as np
from PIL import Image

try:
    from src.rag.indexing_v2 import IndexBuilder, IndexConfig
except ImportError:
    from indexing_v2 import IndexBuilder, IndexConfig  # type: ignore

logger = logging.getLogger(__name__)


# ==============================
# Метаданные изображения галереи
# ==============================

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
    caption: str           # Описание конкретного изображения

    # Оценки качества
    confidence: float = 0.0
    mean_conf: float = 0.0
    max_conf: float = 0.0

    # Описание гида (основной текст для пользователя)
    guide_description: str = ""

    # Дополнительные метаданные
    wikidata_id: str = ""
    coordinates: Dict[str, float] = field(default_factory=dict)
    country_ru: str = ""
    country_en: str = ""
    city_ru: str = ""
    city_en: str = ""
    landmark_type: Dict[str, str] = field(default_factory=dict)
    year_built: str = ""
    architectural_style_ru: List[str] = field(default_factory=list)
    architectural_style_en: List[str] = field(default_factory=list)
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

    def to_dict(self) -> Dict[str, Any]:
        """Сериализует метаданные в словарь для JSON."""
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "landmark_id": self.landmark_id,
            "landmark_name": self.landmark_name,
            "caption": self.caption,
            "caption_landmark": self.caption_landmark,
            "confidence": self.confidence,
            "mean_conf": self.mean_conf,
            "max_conf": self.max_conf,
            "guide_description": self.guide_description,
            "wikidata_id": self.wikidata_id,
            "coordinates": self.coordinates,
            "country_ru": self.country_ru,
            "country_en": self.country_en,
            "city_ru": self.city_ru,
            "city_en": self.city_en,
            "landmark_type": self.landmark_type,
            "year_built": self.year_built,
            "architectural_style_ru": self.architectural_style_ru,
            "architectural_style_en": self.architectural_style_en,
            "heritage_status_ru": self.heritage_status_ru,
            "heritage_status_en": self.heritage_status_en,
            "wikipedia_url_ru": self.wikipedia_url_ru,
            "wikipedia_url_en": self.wikipedia_url_en,
            "website": self.website,
            "wikidata_description_ru": self.wikidata_description_ru,
            "wikidata_description_en": self.wikidata_description_en,
            "row_idx": self.row_idx,
            "num_gallery_images": self.num_gallery_images,
        }


# ==============================
# Результат retrieval
# ==============================

@dataclass
class LandmarkRetrievalResult:
    """
    Результат поиска для одной достопримечательности.

    Содержит агрегированную оценку и все найденные изображения галереи.
    """

    landmark_id: str
    landmark_name: str
    aggregated_score: float
    gallery_images: List[Tuple[float, GalleryImageMetadata]]
    rank: int = 0  # Позиция в результатах (с 1)

    def get_top_image(self) -> Optional[GalleryImageMetadata]:
        """Возвращает изображение с наибольшей оценкой."""
        if not self.gallery_images:
            return None
        return max(self.gallery_images, key=lambda x: x[0])[1]

    def get_metadata(self) -> Dict[str, Any]:
        """Возвращает метаданные лучшего изображения."""
        top = self.get_top_image()
        return top.to_dict() if top else {}

    def to_dict(self) -> Dict[str, Any]:
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


# ==============================
# Агрегация оценок
# ==============================

class ScoreAggregator:
    """
    Агрегирует оценки нескольких изображений в одну оценку landmark.

    Стратегии:
    - max: максимальная оценка
    - top2_mean: среднее двух лучших
    - weighted_top2: взвешенное среднее двух лучших
    """

    @staticmethod
    def aggregate(
        scores: List[float],
        mode: str = "weighted_top2",
        alpha: float = 0.7,
    ) -> float:
        """
        Агрегирует список оценок.

        Args:
            scores: Список оценок схожести
            mode: Стратегия агрегации
            alpha: Вес первой оценки для weighted_top2
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

        # Fallback
        return sorted_scores[0]


# ==============================
# Retriever
# ==============================

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
        gallery_metadata: List[GalleryImageMetadata],
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

    def _encode_query(self, image: Image.Image) -> Optional[np.ndarray]:
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
    ) -> List[LandmarkRetrievalResult]:
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
        landmark_scores: Dict[
            str, List[Tuple[float, GalleryImageMetadata]]
        ] = {}

        for dist, idx in zip(distances[0], indices[0]):
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
            agg_score = ScoreAggregator.aggregate(
                scores,
                mode=self.aggregation_mode,
                alpha=self.aggregation_alpha,
            )
            landmark_name = scores_and_metas[0][1].landmark_name
            results.append(LandmarkRetrievalResult(
                landmark_id=lid,
                landmark_name=landmark_name,
                aggregated_score=agg_score,
                gallery_images=scores_and_metas,
            ))

        results.sort(key=lambda x: x.aggregated_score, reverse=True)
        for rank, result in enumerate(results, start=1):
            result.rank = rank

        return results

    def retrieve(
        self,
        query_image: Image.Image,
        top_k: int = 10,
        faiss_k: int = 50,
    ) -> List[LandmarkRetrievalResult]:
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

    def retrieve_batch(
        self,
        query_images: List[Image.Image],
        top_k: int = 10,
        faiss_k: int = 50,
    ) -> List[List[LandmarkRetrievalResult]]:
        """Возвращает результаты для списка изображений."""
        return [
            self.retrieve(img, top_k=top_k, faiss_k=faiss_k)
            for img in query_images
        ]

    @classmethod
    def from_index_dir(
        cls,
        index_dir: str,
        index_config: Optional[IndexConfig] = None,
        aggregation_mode: str = "weighted_top2",
        aggregation_alpha: float = 0.7,
    ) -> "LandmarkRetriever":
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
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_dicts = json.load(f)

        gallery_metadata = [
            GalleryImageMetadata(
                image_id=d.get("image_id", 0),
                image_path=d.get("image_path", ""),
                landmark_id=d.get("landmark_id", ""),
                landmark_name=d.get("landmark_name", ""),
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

    def save_index(self, output_dir: Path) -> None:
        """
        Сохраняет FAISS-индекс и метаданные в директорию.

        Args:
            output_dir: Директория для сохранения
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        index_path = out / "gallery_index.faiss"
        logger.info(f"Сохранение FAISS индекса в {index_path}")
        faiss.write_index(self.gallery_index, str(index_path))

        metadata_path = out / "gallery_metadata.json"
        logger.info(f"Сохранение метаданных в {metadata_path}")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                [m.to_dict() for m in self.gallery_metadata],
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(f"Индекс сохранён в {out}")


# ==============================
# Вспомогательная функция
# ==============================

def build_index_from_landmarks(
    landmarks_json_path: Path,
    image_base_dir: Path,
    output_dir: Path,
    index_config: Optional[IndexConfig] = None,
    max_images_per_landmark: int = 5,
) -> LandmarkRetriever:
    """
    Строит FAISS-индекс из JSON-файла с достопримечательностями.

    Для продакшена используйте LandmarkRetriever.from_index_dir()
    с заранее построенным индексом.

    Args:
        landmarks_json_path: Путь к JSON с данными
        image_base_dir: Базовая директория изображений
        output_dir: Директория для сохранения индекса
        index_config: Конфигурация энкодера
        max_images_per_landmark: Максимум изображений на landmark
    """
    from tqdm import tqdm

    logger.info(f"Загрузка данных из {landmarks_json_path}")
    with open(landmarks_json_path, "r", encoding="utf-8") as f:
        landmarks = json.load(f)

    if index_config is None:
        index_config = IndexConfig()

    index_builder = IndexBuilder(index_config)

    # Собираем данные галереи
    gallery_data = []
    image_id = 0

    for landmark in tqdm(landmarks, desc="Сбор изображений"):
        landmark_id = landmark.get("landmark_id", "")
        landmark_name = (
            landmark.get("name_en", "").strip()
            or landmark.get("name_ru", "").strip()
            or landmark.get("name", "").strip()
        )
        valid_images = landmark.get("valid_images", [])

        for img_data in valid_images[:max_images_per_landmark]:
            if isinstance(img_data, dict):
                img_path = img_data.get("path", "")
                img_caption = img_data.get("caption", landmark_name)
            else:
                img_path = img_data
                img_caption = landmark_name

            if not img_path:
                continue

            gallery_data.append({
                "image_id": image_id,
                "image_path": img_path,
                "landmark_id": landmark_id,
                "landmark_name": landmark_name,
                "caption": img_caption,
                "caption_landmark": landmark.get(
                    "landmark_summary_caption", landmark_name
                ),
                "metadata": landmark,
            })
            image_id += 1

    logger.info(f"Собрано {len(gallery_data)} изображений")

    # Кодируем батчами
    batch_size = getattr(index_config, "batch_size", 32)
    embeddings_list = []
    metadata_list = []

    for batch_start in tqdm(
        range(0, len(gallery_data), batch_size),
        desc="Кодирование",
        unit="batch",
    ):
        batch_items = gallery_data[batch_start: batch_start + batch_size]

        batch_images: List[Optional[Image.Image]] = []
        for item in batch_items:
            img_path = Path(image_base_dir) / item["image_path"]
            if not img_path.exists():
                logger.warning(f"Изображение не найдено: {img_path}")
                batch_images.append(None)
                continue
            try:
                batch_images.append(Image.open(img_path).convert("RGB"))
            except Exception as e:
                logger.debug(f"Ошибка загрузки {img_path}: {e}")
                batch_images.append(None)

        valid_indices = [
            i for i, img in enumerate(batch_images) if img is not None
        ]
        valid_images = [batch_images[i] for i in valid_indices]

        if not valid_images:
            for img in batch_images:
                if img is not None:
                    img.close()
            continue

        try:
            results = index_builder.encoder.encode_batch(valid_images)
        except Exception as e:
            logger.error(f"Ошибка кодирования батча: {e}")
            results = [None] * len(valid_images)

        for result_idx, item_idx in enumerate(valid_indices):
            item = batch_items[item_idx]
            result = (
                results[result_idx] if result_idx < len(results) else None
            )
            if result is None:
                continue
            embedding, _, _ = result
            embeddings_list.append(embedding)

            lm = item["metadata"]
            metadata_list.append(GalleryImageMetadata(
                image_id=item["image_id"],
                image_path=item["image_path"],
                landmark_id=item["landmark_id"],
                landmark_name=item["landmark_name"],
                caption=item["caption"],
                caption_landmark=item["caption_landmark"],
                confidence=lm.get("confidence", 0.0),
                mean_conf=lm.get("mean_conf", 0.0),
                max_conf=lm.get("max_conf", 0.0),
                guide_description=lm.get("guide_description", ""),
                wikidata_id=lm.get("wikidata_id", ""),
                coordinates=lm.get("coordinates", {}),
                country_ru=lm.get("country_ru", ""),
                country_en=lm.get("country_en", ""),
                city_ru=lm.get("city_ru", ""),
                city_en=lm.get("city_en", ""),
                landmark_type=lm.get("landmark_type", {}),
                year_built=lm.get("year_built", ""),
                architectural_style_ru=lm.get("architectural_style_ru", []),
                architectural_style_en=lm.get("architectural_style_en", []),
                heritage_status_ru=lm.get("heritage_status_ru", ""),
                heritage_status_en=lm.get("heritage_status_en", ""),
                wikipedia_url_ru=lm.get("wikipedia_url_ru", ""),
                wikipedia_url_en=lm.get("wikipedia_url_en", ""),
                website=lm.get("website", ""),
                wikidata_description_ru=lm.get("wikidata_description_ru", ""),
                wikidata_description_en=lm.get("wikidata_description_en", ""),
            ))

        for img in batch_images:
            if img is not None:
                img.close()

    if not embeddings_list:
        raise ValueError("Не удалось закодировать ни одного изображения")

    # Строим FAISS-индекс
    logger.info("Построение FAISS индекса...")
    embeddings = np.vstack(embeddings_list)
    dim = embeddings.shape[1]
    gallery_index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    gallery_index.add(embeddings)

    logger.info(f"FAISS индекс построен: {gallery_index.ntotal} векторов")

    retriever = LandmarkRetriever(
        index_builder=index_builder,
        gallery_index=gallery_index,
        gallery_metadata=metadata_list,
    )
    retriever.save_index(output_dir)

    return retriever
