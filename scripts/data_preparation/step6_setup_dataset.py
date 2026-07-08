"""
Шаг 6: Скрипт генерации датасета для распознавания достопримечательностей
с hard negative mining на основе FAISS.

Скрипт наследуется от src.rag.landmark_retriever и добавляет логику,
специфичную для обучения:
- Hard negative mining
- Генерация меток (target_idx)
- Разбиение на train/val/test
- Генерация сэмплов с долей unknown

Предварительные требования:
    - Требуется landmarks_with_guide_descriptions.json из шага 5
    - Входной файл содержит валидированные изображения с caption'ами

Этот скрипт:
1. Загружает данные достопримечательностей с валидированными изображениями
2. Распределяет изображения по ролям gallery/query (корректно для продакшена)
3. Строит FAISS-индекс только из gallery-изображений
4. Генерирует обучающие сэмплы с hard negatives
5. Сохраняет сэмплы и обучающий индекс для повторного использования

Повторное использование обучающего индекса:
    Скрипт сохраняет обучающий gallery-индекс (только gallery-изображения),
    который можно переиспользовать для экспериментов с разными параметрами:

    - training_gallery_index.faiss: FAISS-индекс
    - training_gallery_metadata.json: метаданные изображений
    - training_gallery_embeddings.npy: SigLIP embeddings

    Чтобы переиспользовать существующий индекс (по умолчанию):
        config.reuse_training_index = True
        config.force_rebuild_index = False

    Чтобы принудительно перестроить:
        config.force_rebuild_index = True

    ВНИМАНИЕ: Это НЕ продакшен-индекс. Для продакшена используйте
    LandmarkRetriever.build_index_from_landmarks(), который включает ВСЕ
    изображения.

Использование:
    python step6_setup_dataset.py
"""

from __future__ import annotations

# ---------------------------------------------------------------
# Предотвращение segfault на macOS: ВСЕ переменные окружения для
# потоков/процессов ДОЛЖНЫ быть заданы до импорта любой нативной
# библиотеки (numpy, torch, faiss).
# ---------------------------------------------------------------
import os
import platform as _platform

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if _platform.system() == "Darwin":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# ---------------------------------------------------------------

import json
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Добавляем корень проекта в sys.path для импортов из src/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Импортируем из продакшен-модулей
from src.rag.indexing_v2 import IndexBuilder, IndexConfig
from src.rag.landmark_retriever import (
    GalleryImageMetadata,
    aggregate_scores,
)

# ======================
# ЛОГИРОВАНИЕ
# ======================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ======================
# КОНФИГУРАЦИЯ
# ======================
@dataclass
class DatasetConfig:
    """Конфигурация для генерации датасета."""

    # Пути к файлам
    data_path: str = "data/processed/landmarks_with_guide_descriptions_filtred.json"
    output_dir: Path = Path("data/processed/")
    image_base_dir: Path = Path("images")

    # Бэкенд энкодера: "siglip" или "dinov2"
    # "siglip" — google/siglip-so400m-patch14-384 с классификацией exterior/interior
    #            и фильтрацией объектов.
    # "dinov2" — facebook/dinov2-base без классификации,
    #            все изображения кодируются с одинаковым весом.
    embedder_type: str = "dinov2"
    embedder_model: str = ""  # Пустая строка = использовать дефолтную модель

    # Переиспользование индекса
    reuse_training_index: bool = True  # Загружать существующий индекс если есть
    force_rebuild_index: bool = False  # Принудительно перестроить индекс

    # Параметры сэмплирования
    max_candidates: int = 15
    faiss_k: int = 100
    max_gallery_per_landmark: int = 5

    # Разнообразие кандидатов
    enforce_type_diversity: bool = True
    type_diversity_strictness: float = 0.7
    min_candidates_before_diversity: int = 3

    # Разбивка на выборки
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Разбивка для редких объектов (2-3 изображения)
    two_image_train_ratio: float = 0.8
    two_image_val_ratio: float = 0.1
    two_image_test_ratio: float = 0.1

    three_image_train_ratio: float = 0.7
    three_image_val_ratio: float = 0.15
    three_image_test_ratio: float = 0.15

    # Размер батча при кодировании
    encoding_batch_size: int = 32

    # Аугментация галереи (применяется при построении индекса)
    # Аугментированные версии увеличивают покрытие для редких объектов
    gallery_augmentations: int = 5  # N аугментированных версий на изображение
    gallery_augmentation_threshold: int = 5  # Аугментировать если < N изображений

    # Доля unknown-сэмплов
    unknown_ratio: float = 0.2

    # Доля hard unknown-сэмплов (с высоким retrieval score).
    # Самые сложные случаи: все кандидаты визуально похожи на запрос
    # (retrieval_score >= hard_unknown_min_score), но правильного ответа нет.
    # Имитирует продакшен-сценарий: retrieval уверен, но ошибается.
    # Установите 0.0 чтобы отключить (обратная совместимость).
    hard_unknown_ratio: float = 0.08  # Было 0.15 в v2
    # Минимальный retrieval score для hard unknown кандидатов.
    # Кандидаты ниже порога отфильтровываются.
    hard_unknown_min_score: float = 0.88  # Было 0.85 — только самые сложные

    # Стратегия агрегации оценок
    score_aggregation_mode: str = "weighted_top2"  # "max" или "weighted_top2"
    weighted_top2_alpha: float = 0.8  # Вес первой оценки

    # Выбор изображения кандидата
    candidate_image_selection_mode: str = "top1"  # "top1" или "sample_topk"
    candidate_image_sample_topk: int = 3  # Используется при mode="sample_topk"

    # Генерация unknown-сэмплов
    unknown_generation_strategy: str = "outside_topk"
    unknown_exclude_topk: int = 25  # Исключаем top-k из unknown кандидатов

    # Классификация сложности кандидатов
    hardness_classification_mode: str = "score_based"
    hard_threshold: float = 0.85  # score >= 0.85 → hard
    semi_hard_threshold: float = 0.75  # 0.75 <= score < 0.85 → semi-hard
    # score < 0.75 → easy

    # Перцентильная калибровка порогов сложности под шкалу энкодера.
    # Абсолютные пороги (0.85/0.75/0.88) привязаны к SigLIP; у другого энкодера
    # (DINOv2) косинусы на иной шкале — те же пороги ломают hardness-разметку
    # (почти всё становится "easy"). При True пороги hard_threshold /
    # semi_hard_threshold / hard_unknown_min_score ПЕРЕСЧИТЫВАЮТСЯ по перцентилям
    # фактического распределения retrieval_score перед генерацией сэмплов.
    calibrate_thresholds_by_percentile: bool = True
    calibration_num_queries: int = 500  # сколько query просэмплировать
    hard_percentile: float = 0.70  # верхние 30% score → hard
    semi_hard_percentile: float = 0.40  # граница semi-hard
    hard_unknown_percentile: float = 0.90  # верхние 10% → hard_unknown

    # Обработка текста
    evidence_max_length: int = 80

    # Зерно генератора случайных чисел (random seed)
    random_seed: int = 42

    def __post_init__(self):
        """Валидация конфигурации."""
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            raise ValueError(
                "Сумма train_ratio + val_ratio + test_ratio должна быть 1.0"
            )

        # Проверка пропорций для редких объектов
        if (
            abs(
                self.two_image_train_ratio
                + self.two_image_val_ratio
                + self.two_image_test_ratio
                - 1.0
            )
            > 1e-6
        ):
            raise ValueError("Сумма пропорций для 2-изображений должна быть 1.0")

        if (
            abs(
                self.three_image_train_ratio
                + self.three_image_val_ratio
                + self.three_image_test_ratio
                - 1.0
            )
            > 1e-6
        ):
            raise ValueError("Сумма пропорций для 3-изображений должна быть 1.0")

        # Проверка стратегии агрегации
        valid_agg_modes = ["max", "top2_mean", "weighted_top2"]
        if self.score_aggregation_mode not in valid_agg_modes:
            raise ValueError(
                f"score_aggregation_mode должен быть одним из {valid_agg_modes}"
            )
        if not (0 <= self.weighted_top2_alpha <= 1.0):
            raise ValueError("weighted_top2_alpha должен быть в диапазоне [0, 1]")

        # Проверка выбора изображения кандидата
        if self.candidate_image_sample_topk < 1:
            raise ValueError("candidate_image_sample_topk должен быть >= 1")

        # Проверка стратегии генерации unknown
        valid_strategies = ["outside_topk", "manual_removal"]
        if self.unknown_generation_strategy not in valid_strategies:
            raise ValueError(
                f"unknown_generation_strategy должен быть одним из {valid_strategies}"
            )
        if self.unknown_exclude_topk < 1:
            raise ValueError("unknown_exclude_topk должен быть >= 1")

        # Проверка классификации сложности
        valid_hardness_modes = ["score_based", "rank_based"]
        if self.hardness_classification_mode not in valid_hardness_modes:
            raise ValueError(
                f"hardness_classification_mode должен быть одним из {valid_hardness_modes}"
            )
        if not (0 <= self.semi_hard_threshold <= self.hard_threshold <= 1.0):
            raise ValueError(
                "Пороги должны удовлетворять: 0 <= semi_hard <= hard <= 1.0"
            )

        # Перцентили калибровки: semi <= hard <= hard_unknown (hard_unknown —
        # самый строгий), все в (0, 1).
        if self.calibrate_thresholds_by_percentile and not (
            0
            < self.semi_hard_percentile
            <= self.hard_percentile
            <= self.hard_unknown_percentile
            < 1.0
        ):
            raise ValueError(
                "Перцентили должны удовлетворять: "
                "0 < semi_hard_percentile <= hard_percentile "
                "<= hard_unknown_percentile < 1.0"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)


# ======================
# ОБРАБОТКА ТЕКСТА
# ======================
class TextProcessor:
    """Обрабатывает текст описаний достопримечательностей."""

    @staticmethod
    def get_landmark_summary_caption(row: pd.Series) -> str:
        """Возвращает landmark_summary_caption из строки."""
        caption = row.get("landmark_summary_caption", "")
        if caption and isinstance(caption, str) and caption.strip():
            return str(caption)
        return str(row.get("name", ""))

    @staticmethod
    def get_image_and_caption_for_candidate(
        row: pd.Series, valid_images: list[str]
    ) -> tuple[str, str, str]:
        """Возвращает случайное изображение и caption для landmark-кандидата.

        Возвращает:
            Кортеж (candidate_image, caption, caption_landmark)
        """
        if not valid_images:
            return "", str(row.get("name", "")), str(row.get("name", ""))

        candidate_image = random.choice(valid_images)
        caption_landmark = TextProcessor.get_landmark_summary_caption(row)

        # Получаем caption для выбранного изображения
        valid_images_data = row.get("valid_images", [])
        caption = str(row.get("name", ""))  # Запасное значение

        for img_data in valid_images_data:
            if isinstance(img_data, dict):
                img_path = img_data.get("path", "")
                if img_path == candidate_image:
                    img_caption = img_data.get("caption", "")
                    if (
                        img_caption
                        and isinstance(img_caption, str)
                        and img_caption.strip()
                    ):
                        caption = str(img_caption)
                    break

        return candidate_image, caption, caption_landmark

    @staticmethod
    def make_evidence(description: str, max_length: int = 80) -> str:
        """Создаёт строку evidence из описания."""
        return description[:max_length]


# ======================
# НАЗНАЧЕНИЕ РОЛЕЙ ИЗОБРАЖЕНИЙ
# ======================
@dataclass
class LandmarkImageSplit:
    """
    Определяет распределение изображений для одного объекта.

    Основная структура данных для корректного closed-set retrieval.
    Каждому изображению назначается роль:
    - gallery: используется в FAISS-индексе для retrieval
    - query_train/val/test: используется как запрос в соответствующей выборке

    Ключевой принцип: gallery и query НЕ пересекаются.
    """

    landmark_id: str
    landmark_name: str
    total_images: int

    # Назначение ролей изображениям
    gallery_images: list[str] = field(default_factory=list)
    query_train_images: list[str] = field(default_factory=list)
    query_val_images: list[str] = field(default_factory=list)
    query_test_images: list[str] = field(default_factory=list)

    # Флаги возможностей
    retrieval_only: bool = False  # True если только 1 изображение
    supports_contrastive: bool = False  # True если >= 3 изображений
    supports_reranking: bool = True  # True если >= 2 изображений

    # Метаданные (сохранены из оригинала)
    confidence: float = 0.0
    mean_conf: float = 0.0
    max_conf: float = 0.0
    guide_description: str = ""
    row_idx: int = -1

    def get_all_query_images(self) -> list[str]:
        """Возвращает все query-изображения по всем выборкам."""
        return self.query_train_images + self.query_val_images + self.query_test_images

    def validate(self) -> None:
        """Проверяет отсутствие пересечения между gallery и query изображениями."""
        gallery_set = set(self.gallery_images)
        query_set = set(self.get_all_query_images())

        overlap = gallery_set & query_set
        if overlap:
            raise ValueError(
                f"Landmark {self.landmark_id}: "
                f"Gallery and query images overlap: {overlap}"
            )

        # Проверка общего числа изображений
        assigned = (
            len(self.gallery_images)
            + len(self.query_train_images)
            + len(self.query_val_images)
            + len(self.query_test_images)
        )
        if assigned != self.total_images:
            logger.warning(
                f"Landmark {self.landmark_id}: "
                f"Assigned {assigned} images but total is {self.total_images}"
            )


# ======================
# РАЗБИЕНИЕ ДАННЫХ
# ======================
class ImageRoleSplitter:
    """
    Распределяет изображения каждого объекта по ролям gallery/query.

    Адаптивная стратегия по числу изображений:
    - 1 изображение: только gallery (retrieval_only=True)
    - 2 изображения: вероятностное распределение train/val/test (80/10/10)
    - 3 изображения: вероятностное распределение train/val/test (70/15/15)
    - 4+ изображений: N-3 gallery + 1 train + 1 val + 1 test
    """

    def __init__(self, config: DatasetConfig, max_gallery_per_landmark: int = 5):
        self.config = config
        self.max_gallery_per_landmark = max_gallery_per_landmark

        # Статистика
        self.stats = {
            "total_landmarks": 0,
            "retrieval_only": 0,  # 1 изображение
            "two_image_train": 0,  # 2 изображения -> train
            "two_image_val": 0,  # 2 изображения -> val
            "two_image_test": 0,  # 2 изображения -> test
            "three_image_train": 0,  # 3 изображения -> train
            "three_image_val": 0,  # 3 изображения -> val
            "three_image_test": 0,  # 3 изображения -> test
            "full_split": 0,  # 4+ изображений
            "total_gallery_images": 0,
            "total_query_train": 0,
            "total_query_val": 0,
            "total_query_test": 0,
        }

    def _split_images_for_landmark(
        self, images: list[str], landmark_id: str
    ) -> dict[str, list[str]]:
        """
        Распределяет изображения одного объекта по адаптивной стратегии.

        Возвращает словарь с ключами: gallery, query_train, query_val, query_test
        """
        num_images = len(images)

        # Перемешиваем (воспроизводимо через seed)
        shuffled = images.copy()
        random.shuffle(shuffled)

        if num_images == 1:
            # Только 1 изображение — используем только для gallery
            logger.debug(f"Landmark {landmark_id}: 1 image -> gallery only")
            return {
                "gallery": shuffled,
                "query_train": [],
                "query_val": [],
                "query_test": [],
            }

        elif num_images == 2:
            # Вероятностное распределение для объектов с 2 изображениями:
            # 80% → gallery + train, 10% → gallery + val, 10% → gallery + test
            rand_val = random.random()

            if rand_val < self.config.two_image_train_ratio:
                # Train выборка
                split_type = "train"
                result = {
                    "gallery": [shuffled[0]],
                    "query_train": [shuffled[1]],
                    "query_val": [],
                    "query_test": [],
                }
            elif rand_val < (
                self.config.two_image_train_ratio + self.config.two_image_val_ratio
            ):
                # Val выборка
                split_type = "val"
                result = {
                    "gallery": [shuffled[0]],
                    "query_train": [],
                    "query_val": [shuffled[1]],
                    "query_test": [],
                }
            else:
                # Test выборка
                split_type = "test"
                result = {
                    "gallery": [shuffled[0]],
                    "query_train": [],
                    "query_val": [],
                    "query_test": [shuffled[1]],
                }

            logger.debug(
                f"Landmark {landmark_id}: 2 images -> 1 gallery + 1 {split_type}"
            )
            return result

        elif num_images == 3:
            # Вероятностное распределение для объектов с 3 изображениями:
            # 70% → 2 gallery + train, 15% → 2 gallery + val, 15% → 2 gallery + test
            rand_val = random.random()

            if rand_val < self.config.three_image_train_ratio:
                # Train выборка
                split_type = "train"
                result = {
                    "gallery": shuffled[:2],
                    "query_train": [shuffled[2]],
                    "query_val": [],
                    "query_test": [],
                }
            elif rand_val < (
                self.config.three_image_train_ratio + self.config.three_image_val_ratio
            ):
                # Val выборка
                split_type = "val"
                result = {
                    "gallery": shuffled[:2],
                    "query_train": [],
                    "query_val": [shuffled[2]],
                    "query_test": [],
                }
            else:
                # Test выборка
                split_type = "test"
                result = {
                    "gallery": shuffled[:2],
                    "query_train": [],
                    "query_val": [],
                    "query_test": [shuffled[2]],
                }

            logger.debug(
                f"Landmark {landmark_id}: 3 images -> 2 gallery + 1 {split_type}"
            )
            return result

        else:
            # Обычный случай: 4+ изображений
            # Резервируем по 1 для train, val, test; остальное — gallery

            # Сначала резервируем query-изображения
            query_test = [shuffled[-1]]
            query_val = [shuffled[-2]]
            query_train = [shuffled[-3]]

            # Оставшиеся изображения для gallery
            gallery_candidates = shuffled[:-3]

            # Ограничиваем размер gallery чтобы популярные объекты не доминировали
            if len(gallery_candidates) > self.max_gallery_per_landmark:
                gallery = gallery_candidates[: self.max_gallery_per_landmark]
                logger.debug(
                    f"Landmark {landmark_id}: "
                    f"Capped gallery from {len(gallery_candidates)} "
                    f"to {self.max_gallery_per_landmark}"
                )
            else:
                gallery = gallery_candidates

            logger.debug(
                f"Landmark {landmark_id}: {num_images} images -> "
                f"{len(gallery)} gallery + 1 train + 1 val + 1 test"
            )

            return {
                "gallery": gallery,
                "query_train": query_train,
                "query_val": query_val,
                "query_test": query_test,
            }

    def split_all_landmarks(self, df: pd.DataFrame) -> list[LandmarkImageSplit]:
        """
        Распределяет изображения для всех объектов в датафрейме.

        Args:
            df: DataFrame с колонками: landmark_id, images, name и др.

        Returns:
            Список LandmarkImageSplit, по одному на объект
        """
        logger.info("Начало распределения ролей изображений...")

        splits = []

        for idx, row in tqdm(
            df.iterrows(), total=len(df), desc="Splitting landmark images"
        ):
            landmark_id = str(row.get("landmark_id", ""))
            name = str(row.get("name", ""))
            images = row.get("images", [])

            if not images:
                logger.warning(f"Landmark {landmark_id} has no images, skipping")
                continue

            # Применяем адаптивную стратегию распределения
            split_result = self._split_images_for_landmark(images, landmark_id)

            # Создаём объект LandmarkImageSplit
            split = LandmarkImageSplit(
                landmark_id=landmark_id,
                landmark_name=name,
                total_images=len(images),
                gallery_images=split_result["gallery"],
                query_train_images=split_result["query_train"],
                query_val_images=split_result["query_val"],
                query_test_images=split_result["query_test"],
                retrieval_only=(len(images) == 1),
                supports_contrastive=(len(images) >= 3),
                supports_reranking=(len(images) >= 2),
                confidence=row.get("confidence", 0.0),
                mean_conf=row.get("mean_conf", 0.0),
                max_conf=row.get("max_conf", 0.0),
                guide_description=row.get("guide_description", ""),
                row_idx=idx,
            )

            # Валидация
            split.validate()

            splits.append(split)

            # Обновляем статистику
            self.stats["total_landmarks"] += 1
            self.stats["total_gallery_images"] += len(split.gallery_images)
            self.stats["total_query_train"] += len(split.query_train_images)
            self.stats["total_query_val"] += len(split.query_val_images)
            self.stats["total_query_test"] += len(split.query_test_images)

            # Обновляем статистику на основе фактического распределения
            if len(images) == 1:
                self.stats["retrieval_only"] += 1
            elif len(images) == 2:
                if split.query_train_images:
                    self.stats["two_image_train"] += 1
                elif split.query_val_images:
                    self.stats["two_image_val"] += 1
                elif split.query_test_images:
                    self.stats["two_image_test"] += 1
            elif len(images) == 3:
                if split.query_train_images:
                    self.stats["three_image_train"] += 1
                elif split.query_val_images:
                    self.stats["three_image_val"] += 1
                elif split.query_test_images:
                    self.stats["three_image_test"] += 1
            else:
                self.stats["full_split"] += 1

        # Логируем статистику
        self._log_statistics()

        return splits

    def _log_statistics(self) -> None:
        """Логирует подробную статистику распределения."""
        logger.info("=" * 60)
        logger.info("СТАТИСТИКА РАСПРЕДЕЛЕНИЯ РОЛЕЙ ИЗОБРАЖЕНИЙ")
        logger.info("=" * 60)
        logger.info(f"Всего объектов: {self.stats['total_landmarks']}")
        logger.info(f"  - Только retrieval (1 фото): {self.stats['retrieval_only']}")

        # Разбивка для объектов с 2 изображениями
        two_img_total = (
            self.stats["two_image_train"]
            + self.stats["two_image_val"]
            + self.stats["two_image_test"]
        )
        if two_img_total > 0:
            logger.info(f"  - 2 images: {two_img_total}")
            logger.info(
                f"    * Train: {self.stats['two_image_train']} "
                f"({100 * self.stats['two_image_train'] / two_img_total:.1f}%)"
            )
            logger.info(
                f"    * Val: {self.stats['two_image_val']} "
                f"({100 * self.stats['two_image_val'] / two_img_total:.1f}%)"
            )
            logger.info(
                f"    * Test: {self.stats['two_image_test']} "
                f"({100 * self.stats['two_image_test'] / two_img_total:.1f}%)"
            )

        # Разбивка для объектов с 3 изображениями
        three_img_total = (
            self.stats["three_image_train"]
            + self.stats["three_image_val"]
            + self.stats["three_image_test"]
        )
        if three_img_total > 0:
            logger.info(f"  - 3 images: {three_img_total}")
            logger.info(
                f"    * Train: {self.stats['three_image_train']} "
                f"({100 * self.stats['three_image_train'] / three_img_total:.1f}%)"
            )
            logger.info(
                f"    * Val: {self.stats['three_image_val']} "
                f"({100 * self.stats['three_image_val'] / three_img_total:.1f}%)"
            )
            logger.info(
                f"    * Test: {self.stats['three_image_test']} "
                f"({100 * self.stats['three_image_test'] / three_img_total:.1f}%)"
            )

        logger.info(f"  - Full split (4+ imgs): {self.stats['full_split']}")
        logger.info("")
        logger.info(f"Gallery-изображений: {self.stats['total_gallery_images']}")
        logger.info(f"Query-изображений (train): {self.stats['total_query_train']}")
        logger.info(f"Query-изображений (val): {self.stats['total_query_val']}")
        logger.info(f"Query-изображений (test): {self.stats['total_query_test']}")
        logger.info("=" * 60)


# ======================
# ПОСТРОИТЕЛЬ GALLERY-ИНДЕКСА
# ======================
# Примечание: GalleryImageMetadata импортируется из src.rag.landmark_retriever


class GalleryIndexBuilder:
    """
    Строит FAISS-индекс на уровне изображений только из gallery-изображений.

    Ключевые отличия от старого подхода:
    - СТАРО: один embedding на объект (mean pooling)
    - НОВО: один embedding на каждое gallery-изображение

    - СТАРО: индекс содержит все изображения
    - НОВО: индекс содержит ТОЛЬКО gallery-изображения

    - СТАРО: retrieval возвращает объекты
    - НОВО: retrieval возвращает изображения, затем агрегирует по landmark_id

    Это соответствует продакшен-поведению, где:
    1. Query-изображение → SigLIP embedding
    2. FAISS-поиск → top-k gallery-изображений
    3. Группировка по landmark_id → max similarity на объект
    4. Возврат top объектов
    """

    def __init__(
        self,
        index_builder: IndexBuilder,
        batch_size: int = 32,
        gallery_augmentations: int = 4,
        gallery_augmentation_threshold: int = 4,
    ):
        """
        Аргументы:
            index_builder: IndexBuilder из src.rag.indexing_v2
            batch_size: размер батча для кодирования изображений
            gallery_augmentations: N аугментированных версий на gallery-изображение
                (0 = отключено). Применяется только для объектов, у которых
                изображений меньше, чем gallery_augmentation_threshold.
            gallery_augmentation_threshold: аугментировать только если у объекта
                меньше gallery-изображений, чем это значение.
        """
        self.index_builder = index_builder
        self.batch_size = batch_size
        self.gallery_augmentations = gallery_augmentations
        self.gallery_augmentation_threshold = gallery_augmentation_threshold
        self.stats = {
            "total_landmarks": 0,
            "total_gallery_images": 0,
            "total_augmented_images": 0,
            "landmarks_with_1_image": 0,
            "landmarks_with_2_images": 0,
            "landmarks_with_3_images": 0,
            "landmarks_with_4plus_images": 0,
        }

    @staticmethod
    def encode_single_image(image_path: Path, encoder) -> np.ndarray | None:
        """
        Кодирует одно изображение с корректным управлением ресурсами.

        Аргументы:
            image_path: путь к файлу изображения
            encoder: экземпляр encoder с методом encode_batch

        Возвращает:
            Embedding изображения или None, если кодирование не удалось
        """
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                img = img.convert("RGB")
                results = encoder.encode_batch([img])

                if results and results[0] is not None:
                    embedding, _, _ = results[0]
                    return embedding
            return None
        except Exception as e:
            logger.error(f"Failed to encode {image_path}: {e}")
            return None

    def encode_images_batch(self, image_paths: list[Path]) -> list[np.ndarray | None]:
        """
        Кодирует несколько изображений батчами с корректным управлением ресурсами.

        Аргументы:
            image_paths: список путей к изображениям для кодирования

        Возвращает:
            Список embeddings (None для изображений с ошибкой)
        """
        from PIL import Image

        all_embeddings = []

        # Добавляем progress bar для кодирования
        num_batches = (len(image_paths) + self.batch_size - 1) // self.batch_size
        pbar = tqdm(total=len(image_paths), desc="Encoding gallery images", unit="img")

        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            batch_images = []
            failed_indices = set()

            # Загружаем изображения батча с корректным управлением ресурсами
            for idx, img_path in enumerate(batch_paths):
                try:
                    with Image.open(img_path) as raw_img:
                        batch_images.append(raw_img.convert("RGB"))
                except Exception as e:
                    logger.debug(f"Failed to load {img_path}: {e}")
                    failed_indices.add(idx)
                    batch_images.append(None)

            # Кодируем батч (только валидные изображения)
            valid_images = [img for img in batch_images if img is not None]

            if valid_images:
                try:
                    results = self.index_builder.encoder.encode_batch(valid_images)

                    # Сопоставляем результаты обратно с исходными позициями в батче
                    result_idx = 0
                    for idx in range(len(batch_paths)):
                        if idx in failed_indices:
                            all_embeddings.append(None)
                        else:
                            if (
                                results
                                and result_idx < len(results)
                                and results[result_idx] is not None
                            ):
                                embedding, _, _ = results[result_idx]
                                all_embeddings.append(embedding)
                            else:
                                all_embeddings.append(None)
                            result_idx += 1

                except Exception as e:
                    logger.error(f"Batch encoding failed: {e}")
                    # Добавляем None для всех изображений неудавшегося батча
                    all_embeddings.extend([None] * len(batch_paths))
            else:
                # Все изображения в батче не удалось загрузить
                all_embeddings.extend([None] * len(batch_paths))

            # Закрываем загруженные изображения во избежание утечки памяти
            for img in batch_images:
                if img is not None:
                    try:
                        img.close()
                    except Exception:
                        pass

            # Обновляем progress bar
            pbar.update(len(batch_paths))

        pbar.close()
        return all_embeddings

    def build_from_splits(
        self, splits: list[LandmarkImageSplit], df: pd.DataFrame, image_base_dir: Path
    ) -> tuple[np.ndarray, list[GalleryImageMetadata], faiss.Index]:
        """
        Строит FAISS-индекс из gallery-изображений в splits.

        Аргументы:
            splits: список объектов LandmarkImageSplit
            df: исходный датафрейм с данными объектов
            image_base_dir: базовая директория для изображений

        Возвращает:
            embeddings: массив (N, D) embeddings gallery-изображений
            metadata: список GalleryImageMetadata, по одному на gallery-изображение
            index: FAISS-индекс, содержащий gallery embeddings
        """
        logger.info("Building gallery index from image splits...")

        # Собираем все gallery-изображения с метаданными
        gallery_data = []
        image_id = 0

        for split in splits:
            if not split.gallery_images:
                continue

            # Получаем строку из датафрейма
            row = df.iloc[split.row_idx]

            # Получаем caption_landmark для этого объекта
            caption_landmark = TextProcessor.get_landmark_summary_caption(row)

            # Строим отображение path изображения → caption из valid_images
            valid_images = row.get("valid_images", [])
            image_caption_map = {}
            for img_data in valid_images:
                if isinstance(img_data, dict):
                    img_path = img_data.get("path", "")
                    img_caption = img_data.get("caption", "")
                    if img_path:
                        image_caption_map[img_path] = (
                            img_caption if img_caption else row.get("name", "")
                        )

            for img_path in split.gallery_images:
                # Получаем конкретный caption для этого изображения
                caption = image_caption_map.get(img_path, row.get("name", ""))

                gallery_data.append(
                    {
                        "image_id": image_id,
                        "image_path": img_path,
                        "landmark_id": split.landmark_id,
                        "landmark_name": split.landmark_name,
                        "caption": caption,
                        "caption_landmark": caption_landmark,
                        "confidence": split.confidence,
                        "mean_conf": split.mean_conf,
                        "max_conf": split.max_conf,
                        "guide_description": split.guide_description,
                        "row_idx": split.row_idx,
                        "num_gallery_images": len(split.gallery_images),
                    }
                )
                image_id += 1

            # Обновляем статистику
            self.stats["total_landmarks"] += 1
            self.stats["total_gallery_images"] += len(split.gallery_images)

            num_imgs = len(split.gallery_images)
            if num_imgs == 1:
                self.stats["landmarks_with_1_image"] += 1
            elif num_imgs == 2:
                self.stats["landmarks_with_2_images"] += 1
            elif num_imgs == 3:
                self.stats["landmarks_with_3_images"] += 1
            else:
                self.stats["landmarks_with_4plus_images"] += 1

        logger.info(
            f"Collected {len(gallery_data)} gallery images "
            f"from {self.stats['total_landmarks']} landmarks"
        )

        # Кодируем gallery-изображения батчами
        logger.info(f"Encoding gallery images in batches of {self.batch_size}...")
        if self.gallery_augmentations > 0:
            logger.info(
                f"Gallery augmentation enabled: "
                f"{self.gallery_augmentations} augments per image "
                f"(threshold: < {self.gallery_augmentation_threshold} images)"
            )

        embeddings_list = []
        metadata_list = []

        # Подготавливаем пути и проверяем существование
        valid_items = []
        image_paths = []

        for item in gallery_data:
            img_path = image_base_dir / item["image_path"]
            if img_path.exists():
                valid_items.append(item)
                image_paths.append(img_path)
            else:
                logger.warning(f"Image not found: {img_path}")

        logger.info(f"Found {len(image_paths)} valid images to encode")

        # Кодируем исходные gallery-изображения батчами
        embeddings = self.encode_images_batch(image_paths)
        logger.info("Gallery image encoding complete. Processing results...")

        augment_fn = self.index_builder.encoder.augment_fn
        do_augment = self.gallery_augmentations > 0 and augment_fn is not None

        if do_augment:
            aug_count = sum(
                1
                for item in valid_items
                if item["num_gallery_images"] < self.gallery_augmentation_threshold
            )
            logger.info(
                f"Augmentation: {aug_count} images will be augmented "
                f"({self.gallery_augmentations}x each = "
                f"{aug_count * self.gallery_augmentations} extra images)"
            )
        else:
            logger.info("Augmentation disabled, skipping.")

        from PIL import Image as PILImage

        # Потоковая аугментация: обрабатываем по одному изображению за раз,
        # накапливаем в небольшие батчи, кодируем, затем отбрасываем.
        # Это позволяет не загружать все аугментированные изображения в RAM сразу.
        aug_stream_images: list = []
        aug_stream_metas: list = []

        def _flush_aug_batch() -> None:
            """Кодирует и сбрасывает текущий батч аугментации."""
            if not aug_stream_images:
                return
            try:
                results = self.index_builder.encoder.encode_batch(aug_stream_images)
                for result, aug_meta in zip(results, aug_stream_metas):
                    if result is not None:
                        aug_emb, _, _ = result
                        embeddings_list.append(aug_emb)
                        metadata_list.append(aug_meta)
                        self.stats["total_augmented_images"] += 1
            except Exception as e:
                logger.error(f"Augmented batch encoding failed: {e}")
            finally:
                for img in aug_stream_images:
                    try:
                        img.close()
                    except Exception:
                        pass
                aug_stream_images.clear()
                aug_stream_metas.clear()

        # Подсчитываем элементы, требующие аугментации, для progress bar
        items_to_augment = [
            (item, emb)
            for item, emb in zip(valid_items, embeddings)
            if emb is not None
            and do_augment
            and item["num_gallery_images"] < self.gallery_augmentation_threshold
        ]
        aug_pbar = (
            tqdm(
                total=len(items_to_augment),
                desc="Augmenting gallery images",
                unit="img",
            )
            if items_to_augment
            else None
        )

        for item, embedding in zip(valid_items, embeddings):
            if embedding is None:
                logger.warning(f"Failed to encode: {item['image_path']}")
                continue

            embeddings_list.append(embedding)

            # Создаём метаданные
            meta = GalleryImageMetadata(
                image_id=item["image_id"],
                image_path=item["image_path"],
                landmark_id=item["landmark_id"],
                landmark_name=item["landmark_name"],
                caption=item["caption"],
                caption_landmark=item["caption_landmark"],
                confidence=item["confidence"],
                mean_conf=item["mean_conf"],
                max_conf=item["max_conf"],
                guide_description=item["guide_description"],
                row_idx=item["row_idx"],
                num_gallery_images=item["num_gallery_images"],
            )
            metadata_list.append(meta)

            # Потоковая аугментация: генерируем и сразу батчим
            should_augment = (
                do_augment
                and item["num_gallery_images"] < self.gallery_augmentation_threshold
            )

            if should_augment:
                img_path = image_base_dir / item["image_path"]
                try:
                    with PILImage.open(img_path) as orig_img:
                        orig_img = orig_img.convert("RGB")
                        for _ in range(self.gallery_augmentations):
                            try:
                                aug_img = augment_fn(orig_img.copy())
                                aug_stream_images.append(aug_img)
                                aug_stream_metas.append(meta)
                            except Exception as e:
                                logger.debug(f"Augmentation failed: {e}")

                            # Сбрасываем, когда батч заполнен
                            if len(aug_stream_images) >= self.batch_size:
                                _flush_aug_batch()
                except Exception as e:
                    logger.debug(f"Augmentation skipped for {item['image_path']}: {e}")
                if aug_pbar is not None:
                    aug_pbar.update(1)

        if aug_pbar is not None:
            aug_pbar.close()

        # Сбрасываем оставшиеся аугментированные изображения
        _flush_aug_batch()

        if self.stats["total_augmented_images"] > 0:
            logger.info(
                f"Added {self.stats['total_augmented_images']} augmented gallery images"
            )

        if not embeddings_list:
            raise ValueError("No gallery images were successfully encoded")

        # Складываем embeddings в стек
        embeddings = np.vstack(embeddings_list)
        logger.info(f"Encoded {len(embeddings)} gallery images")

        # Строим FAISS-индекс
        logger.info("Building FAISS index...")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Скалярное произведение (cosine similarity)

        # Нормализуем embeddings для cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        logger.info(f"FAISS index built with {index.ntotal} vectors")

        # Логируем статистику
        self._log_statistics()

        return embeddings, metadata_list, index

    def _log_statistics(self) -> None:
        """Логирует статистику gallery-индекса."""
        logger.info("=" * 60)
        logger.info("СТАТИСТИКА GALLERY-ИНДЕКСА")
        if self.stats.get("total_augmented_images", 0) > 0:
            logger.info(
                f"Augmented images added: {self.stats['total_augmented_images']}"
            )
        logger.info("=" * 60)
        logger.info(f"Всего объектов: {self.stats['total_landmarks']}")
        logger.info(f"Всего gallery-изображений: {self.stats['total_gallery_images']}")
        logger.info(f"  - 1 gallery-фото: {self.stats['landmarks_with_1_image']}")
        logger.info(f"  - 2 gallery images: {self.stats['landmarks_with_2_images']}")
        logger.info(f"  - 3 gallery images: {self.stats['landmarks_with_3_images']}")
        logger.info(
            f"  - 4+ gallery images: {self.stats['landmarks_with_4plus_images']}"
        )
        logger.info(
            f"Average gallery images per landmark: "
            f"{self.stats['total_gallery_images'] / max(1, self.stats['total_landmarks']):.2f}"
        )
        logger.info("=" * 60)

    # Агрегация через aggregate_scores из src.rag.landmark_retriever

    @staticmethod
    def search_and_aggregate(
        query_embedding: np.ndarray,
        gallery_index: faiss.Index,
        gallery_metadata: list[GalleryImageMetadata],
        k: int = 50,
        aggregation_mode: str = "weighted_top2",
        aggregation_alpha: float = 0.7,
    ) -> list[tuple[str, float, list[tuple[float, GalleryImageMetadata]]]]:
        """
        Ищет по gallery-индексу и агрегирует результаты по landmark_id.

        Имитирует продакшен-retrieval:
        1. FAISS возвращает top-k gallery-изображений
        2. Группировка по landmark_id
        3. Агрегация оценок на объект по настраиваемой стратегии
        4. Возврат top объектов с их gallery-изображениями И оценками

        Аргументы:
            query_embedding: query embedding размерности (D,)
            gallery_index: FAISS-индекс gallery-изображений
            gallery_metadata: метаданные каждого gallery-изображения
            k: число изображений для retrieval
            aggregation_mode: стратегия агрегации оценок
            aggregation_alpha: параметр alpha для weighted_top2

        Возвращает:
            Список кортежей (landmark_id, aggregated_score, scores_and_metas),
            где scores_and_metas — это List[Tuple[float, GalleryImageMetadata]],
            отсортированных по aggregated_score по убыванию
        """
        # Нормализуем query
        query_norm = query_embedding.copy()
        faiss.normalize_L2(query_norm.reshape(1, -1))

        # Поиск через FAISS
        distances, indices = gallery_index.search(
            query_norm.reshape(1, -1), min(k, gallery_index.ntotal)
        )

        # Группировка по landmark_id
        landmark_scores: dict[str, list[tuple[float, GalleryImageMetadata]]] = {}

        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(gallery_metadata):
                continue

            meta = gallery_metadata[idx]
            lid = meta.landmark_id

            if lid not in landmark_scores:
                landmark_scores[lid] = []

            landmark_scores[lid].append((float(dist), meta))

        # Агрегируем оценки на объект
        landmark_results = []
        for lid, scores_and_metas in landmark_scores.items():
            # Извлекаем оценки
            scores = [score for score, _ in scores_and_metas]

            # Агрегируем оценки через aggregate_scores из landmark_retriever
            aggregated_score = aggregate_scores(
                scores, mode=aggregation_mode, alpha=aggregation_alpha
            )

            # КРИТИЧНО: сохраняем оценки вместе с метаданными для последующего выбора
            # Возвращаем scores_and_metas, а не только gallery_images
            landmark_results.append((lid, aggregated_score, scores_and_metas))

        # Сортируем по aggregated_score по убыванию
        landmark_results.sort(key=lambda x: x[1], reverse=True)

        return landmark_results


# ======================
# ГЕНЕРАТОР СЭМПЛОВ НА ОСНОВЕ RETRIEVAL (НОВЫЙ)
# ======================
class RetrievalBasedSampleGenerator:
    """
    Генерирует обучающие сэмплы, используя естественный retrieval.

    Ключевые отличия от старого SampleGenerator:
    - СТАРО: positive-кандидат вставляется вручную
    - НОВО: positive должен приходить из retrieval естественным образом

    - СТАРО: query-изображение может совпадать с positive-изображением
    - НОВО: query-изображение должно отличаться от всех изображений кандидатов

    - СТАРО: target.name для идентификации
    - НОВО: target_idx для multiple-choice классификации

    Это соответствует продакшену, где:
    1. Пользователь загружает query-изображение
    2. Система извлекает кандидатов через FAISS
    3. Reranker выбирает лучшее совпадение среди кандидатов
    """

    def __init__(
        self,
        config: DatasetConfig,
        df: pd.DataFrame,
        gallery_index: faiss.Index,
        gallery_metadata: list[GalleryImageMetadata],
        landmark_splits: list[LandmarkImageSplit],
        index_builder: IndexBuilder,
    ):
        self.config = config
        self.df = df
        self.gallery_index = gallery_index
        self.gallery_metadata = gallery_metadata
        self.landmark_splits = landmark_splits
        self.index_builder = index_builder

        # Строим таблицы поиска
        self.lid_to_split = {split.landmark_id: split for split in landmark_splits}
        self.lid_to_row = {
            split.landmark_id: split.row_idx for split in landmark_splits
        }

        # Статистика
        self.stats = {
            "total_generated": 0,
            "positive_in_candidates": 0,
            "none_of_the_above": 0,
            "failed_retrieval": 0,
            "query_in_candidates": 0,  # Должно быть 0!
        }

    def _encode_query_image(self, image_path: Path) -> np.ndarray | None:
        """Кодирует одно query-изображение общим методом кодирования."""
        return GalleryIndexBuilder.encode_single_image(
            image_path, self.index_builder.encoder
        )

    def calibrate_score_thresholds(self, image_base_dir: Path) -> None:
        """Пересчитывает пороги сложности по перцентилям retrieval_score энкодера.

        Абсолютные пороги (hard/semi/hard_unknown) привязаны к шкале конкретного
        энкодера: у SigLIP медиана score ~0.83, у DINOv2 ~0.57, поэтому одни и те
        же 0.85/0.75/0.88 у DINOv2 метят почти все негативы как "easy" и почти не
        генерируют hard_unknown. Метод сэмплирует query-изображения, собирает
        агрегированные retrieval_score кандидатов и заменяет пороги на перцентили
        фактического распределения — датасет строится корректно под любой энкодер.

        Вызывать ДО generate_samples_from_splits.
        """
        cfg = self.config
        if not cfg.calibrate_thresholds_by_percentile:
            logger.info(
                "Калибровка порогов отключена — используются абсолютные значения"
            )
            return

        # Сэмплируем query-изображения train-сплита (воспроизводимо через seed).
        query_paths: list[str] = []
        for split in self.landmark_splits:
            query_paths.extend(split.query_train_images)
        random.shuffle(query_paths)
        query_paths = query_paths[: cfg.calibration_num_queries]

        scores: list[float] = []
        for qp in tqdm(query_paths, desc="Calibrating thresholds"):
            emb = self._encode_query_image(image_base_dir / qp)
            if emb is None:
                continue
            results = GalleryIndexBuilder.search_and_aggregate(
                query_embedding=emb,
                gallery_index=self.gallery_index,
                gallery_metadata=self.gallery_metadata,
                k=cfg.faiss_k,
                aggregation_mode=cfg.score_aggregation_mode,
                aggregation_alpha=cfg.weighted_top2_alpha,
            )
            scores.extend(float(agg) for _lid, agg, _imgs in results)

        if not scores:
            logger.warning(
                "Калибровка порогов: не собрано ни одного score — "
                "оставляю абсолютные пороги"
            )
            return

        scores.sort()

        def _pct(p: float) -> float:
            return scores[min(len(scores) - 1, int(p * len(scores)))]

        old = (cfg.hard_threshold, cfg.semi_hard_threshold, cfg.hard_unknown_min_score)
        cfg.hard_threshold = _pct(cfg.hard_percentile)
        cfg.semi_hard_threshold = _pct(cfg.semi_hard_percentile)
        cfg.hard_unknown_min_score = _pct(cfg.hard_unknown_percentile)

        logger.info(
            f"Калибровка порогов по {len(scores)} score (медиана {_pct(0.5):.3f}):"
        )
        logger.info(
            f"  hard:         {old[0]:.3f} → {cfg.hard_threshold:.3f} "
            f"(p{100 * cfg.hard_percentile:.0f})"
        )
        logger.info(
            f"  semi_hard:    {old[1]:.3f} → {cfg.semi_hard_threshold:.3f} "
            f"(p{100 * cfg.semi_hard_percentile:.0f})"
        )
        logger.info(
            f"  hard_unknown: {old[2]:.3f} → {cfg.hard_unknown_min_score:.3f} "
            f"(p{100 * cfg.hard_unknown_percentile:.0f})"
        )

    def _retrieve_candidates(
        self,
        query_embedding: np.ndarray,
        query_landmark_id: str,
        query_image_path: str,
        k: int = 50,
        for_unknown: bool = False,
        for_hard_unknown: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Извлекает кандидатов через FAISS и агрегирует по объекту.

        Аргументы:
            query_embedding: query embedding
            query_landmark_id: истинный ID объекта (ground truth)
            query_image_path: путь к query-изображению
            k: число изображений для retrieval
            for_unknown: если True, исключить top-k для реалистичных unknown-сэмплов
            for_hard_unknown: если True, оставить только кандидатов с
                retrieval_score >= config.hard_unknown_min_score и
                исключить правильный объект. Имитирует продакшен-сценарий,
                где retrieval уверен, но объекта нет в БД.

        Возвращает:
            candidates: список словарей кандидатов
            target_idx: индекс правильного ответа или -1, если его нет среди кандидатов
        """
        # Ищем по gallery-индексу с настроенной агрегацией
        results = GalleryIndexBuilder.search_and_aggregate(
            query_embedding=query_embedding,
            gallery_index=self.gallery_index,
            gallery_metadata=self.gallery_metadata,
            k=k,
            aggregation_mode=self.config.score_aggregation_mode,
            aggregation_alpha=self.config.weighted_top2_alpha,
        )

        if not results:
            return [], -1

        # Для unknown-сэмплов: пропускаем top-k, чтобы получить реалистичные distractor'ы
        start_rank = 0
        if for_unknown:
            start_rank = self.config.unknown_exclude_topk
            # Убеждаемся, что результатов достаточно
            if start_rank >= len(results):
                start_rank = max(0, len(results) - self.config.max_candidates)

        # Для hard_unknown: фильтруем только высокооценённых кандидатов и
        # исключаем правильный объект. Это создаёт максимально сложный
        # unknown-сценарий: retrieval очень уверен, но объекта
        # действительно нет в базе данных.
        if for_hard_unknown:
            results = [
                (lid, score, imgs)
                for lid, score, imgs in results
                if lid != query_landmark_id
                and score >= self.config.hard_unknown_min_score
            ]
            if not results:
                return [], -1

        # Строим кандидатов из результатов retrieval
        candidates = []
        target_idx = -1

        for rank, (lid, max_score, gallery_imgs) in enumerate(results):
            # Пропускаем top-k для unknown-сэмплов
            if rank < start_rank:
                continue
            # Выбираем одно gallery-изображение для этого объекта
            # gallery_imgs — это List[Tuple[float, GalleryImageMetadata]]

            # Сортируем по индивидуальным оценкам по убыванию
            scores_and_metas_sorted = sorted(
                gallery_imgs, key=lambda x: x[0], reverse=True
            )

            # Сначала отфильтровываем query-изображение (КРИТИЧНО)
            valid_gallery = [
                (score, img_meta)
                for score, img_meta in scores_and_metas_sorted
                if img_meta.image_path != query_image_path
            ]

            if not valid_gallery:
                # Все gallery-изображения совпадают с query (не должно происходить)
                logger.warning(
                    f"Query image {query_image_path} found in all gallery "
                    f"images for landmark {lid}"
                )
                self.stats["query_in_candidates"] += 1
                continue

            # Сэмплируем из top-k валидных изображений
            topk = min(self.config.candidate_image_sample_topk, len(valid_gallery))
            candidate_pool = valid_gallery[:topk]

            # Случайно выбираем одно из пула (детерминированно через seed)
            selected_score, selected_img_meta = random.choice(candidate_pool)

            # Определяем тип кандидата на основе настроенного режима
            if self.config.hardness_classification_mode == "score_based":
                # Классификация на основе score (стабильна при разной плотности)
                if max_score >= self.config.hard_threshold:
                    cand_type = "hard"
                elif max_score >= self.config.semi_hard_threshold:
                    cand_type = "semi_hard"
                else:
                    cand_type = "easy"
            else:  # rank_based (устаревший)
                # Классификация на основе rank (исходное поведение)
                if rank < 5:
                    cand_type = "hard"
                elif rank < 15:
                    cand_type = "semi_hard"
                else:
                    cand_type = "easy"

            # Создаём кандидата
            candidate = {
                "name": selected_img_meta.landmark_name,
                "landmark_id": lid,
                "image": selected_img_meta.image_path,
                "caption": selected_img_meta.caption,
                "caption_landmark": selected_img_meta.caption_landmark,
                "retrieval_score": float(max_score),  # Агрегированная оценка объекта
                "image_score": float(selected_score),  # Оценка отдельного изображения
                "retrieval_rank": rank,
                "candidate_type": cand_type,
            }

            candidates.append(candidate)

            # Проверяем, является ли это target
            if lid == query_landmark_id:
                target_idx = len(candidates) - 1

            # Останавливаемся, когда кандидатов достаточно
            if len(candidates) >= self.config.max_candidates:
                break

        return candidates, target_idx

    def generate_sample(
        self,
        query_image_path: str,
        query_landmark_id: str,
        image_base_dir: Path,
        force_unknown: bool = False,
        force_hard_unknown: bool = False,
    ) -> dict[str, Any] | None:
        """
        Генерирует один обучающий сэмпл.

        Аргументы:
            query_image_path: путь к query-изображению
            query_landmark_id: истинный ID объекта (ground truth)
            image_base_dir: базовая директория для изображений
            force_unknown: если True, создать none-of-the-above сэмпл
                по стратегии outside_topk (низкие retrieval-оценки).
            force_hard_unknown: если True, создать сложный none-of-the-above
                сэмпл, используя только кандидатов с retrieval_score >=
                config.hard_unknown_min_score. Имитирует продакшен-
                сценарий, где retrieval уверен, но объекта нет
                в базе данных.

        Возвращает:
            Словарь сэмпла или None, если генерация не удалась
        """
        # Кодируем query-изображение
        full_path = image_base_dir / query_image_path
        query_embedding = self._encode_query_image(full_path)

        if query_embedding is None:
            self.stats["failed_retrieval"] += 1
            return None

        # Извлекаем кандидатов
        candidates, target_idx = self._retrieve_candidates(
            query_embedding=query_embedding,
            query_landmark_id=query_landmark_id,
            query_image_path=query_image_path,
            k=self.config.faiss_k,
        )

        if not candidates:
            self.stats["failed_retrieval"] += 1
            return None

        # Решаем, должен ли это быть positive или unknown сэмпл
        if force_hard_unknown:
            # Hard unknown: повторно извлекаем, оставляя только высокооценённых
            # кандидатов и исключая правильный объект.
            candidates, target_idx = self._retrieve_candidates(
                query_embedding=query_embedding,
                query_landmark_id=query_landmark_id,
                query_image_path=query_image_path,
                k=self.config.faiss_k,
                for_hard_unknown=True,
            )

            if not candidates:
                # Недостаточно высокооценённых negatives — пропускаем этот сэмпл
                self.stats["failed_retrieval"] += 1
                return None

            # target_idx должен быть -1 (правильный объект был исключён)
            target_idx = -1

            if len(candidates) > self.config.max_candidates:
                candidates = candidates[: self.config.max_candidates]

            self.stats["none_of_the_above"] += 1

        elif force_unknown:
            # Генерируем реалистичный unknown-сэмпл
            if self.config.unknown_generation_strategy == "outside_topk":
                # Повторно извлекаем с for_unknown=True, чтобы получить
                # реалистичные distractor'ы (низкие retrieval-оценки)
                candidates, target_idx = self._retrieve_candidates(
                    query_embedding=query_embedding,
                    query_landmark_id=query_landmark_id,
                    query_image_path=query_image_path,
                    k=self.config.faiss_k,
                    for_unknown=True,
                )

                if not candidates:
                    self.stats["failed_retrieval"] += 1
                    return None

                # target_idx должен быть -1 (positive не входит в top-k)
                # Но проверяем и принудительно исправляем при необходимости
                if target_idx != -1:
                    # Positive случайно попал в исключаемый диапазон, удаляем его
                    candidates.pop(target_idx)
                    target_idx = -1

                # Убеждаемся, что имеем max_candidates
                if len(candidates) > self.config.max_candidates:
                    candidates = candidates[: self.config.max_candidates]

            else:  # manual_removal (устаревший)
                # Старое поведение: удаляем target, если он среди кандидатов
                if target_idx != -1:
                    candidates.pop(target_idx)
                    target_idx = -1

                # Убеждаемся, что имеем max_candidates
                if len(candidates) > self.config.max_candidates:
                    candidates = candidates[: self.config.max_candidates]

            self.stats["none_of_the_above"] += 1

        elif target_idx == -1:
            # Positive естественным образом отсутствует среди кандидатов (редко)
            # Трактуем как unknown-сэмпл
            if len(candidates) > self.config.max_candidates:
                candidates = candidates[: self.config.max_candidates]
            self.stats["none_of_the_above"] += 1

        else:
            # Positive-сэмпл
            # Перемешиваем кандидатов, чтобы рандомизировать позицию target
            # Но отслеживаем target
            target_candidate = candidates[target_idx]
            target_landmark_id = target_candidate.get("landmark_id")
            random.shuffle(candidates)

            # Находим новую позицию target (оптимизировано через landmark_id)
            target_idx = -1
            for idx, cand in enumerate(candidates):
                if cand.get("landmark_id") == target_landmark_id:
                    target_idx = idx
                    break

            # Убеждаемся, что имеем ровно max_candidates
            if len(candidates) > self.config.max_candidates:
                # Если target был бы удалён, сохраняем его
                if target_idx >= self.config.max_candidates:
                    # Меняем target местами с последним оставляемым кандидатом
                    candidates[self.config.max_candidates - 1] = target_candidate
                    target_idx = self.config.max_candidates - 1

                candidates = candidates[: self.config.max_candidates]

            self.stats["positive_in_candidates"] += 1

        # Получаем метаданные объекта
        split = self.lid_to_split.get(query_landmark_id)
        if not split:
            return None

        row = self.df.iloc[split.row_idx]

        # Строим сэмпл
        sample = {
            "query_image": query_image_path,
            "candidates": candidates,
            "target_idx": target_idx,
            "meta": {
                "landmark_id": query_landmark_id,
                "landmark_name": split.landmark_name,
                "num_candidates": len(candidates),
                "positive_in_candidates": (target_idx != -1),
                "num_images": split.total_images,
                "confidence": split.confidence,
                "mean_conf": split.mean_conf,
                "max_conf": split.max_conf,
                "guide_description": row.get("guide_description", ""),
                "wikidata_id": row.get("wikidata_id", ""),
                "coordinates": row.get("coordinates", {}),
                "country_ru": row.get("country_ru", ""),
                "country_en": row.get("country_en", ""),
                "city_ru": row.get("city_ru", ""),
                "city_en": row.get("city_en", ""),
                "landmark_type": row.get("landmark_type", {}),
                "year_built": row.get("year_built", ""),
                "architectural_style_ru": row.get("architectural_style_ru", []),
                "architectural_style_en": row.get("architectural_style_en", []),
                "heritage_status_ru": row.get("heritage_status_ru", ""),
                "heritage_status_en": row.get("heritage_status_en", ""),
                "wikipedia_url_ru": row.get("wikipedia_url_ru", ""),
                "wikipedia_url_en": row.get("wikipedia_url_en", ""),
                "website": row.get("website", ""),
                "wikidata_description_ru": row.get("wikidata_description_ru", ""),
                "wikidata_description_en": row.get("wikidata_description_en", ""),
            },
        }

        self.stats["total_generated"] += 1
        return sample

    def generate_samples_from_splits(
        self,
        split_type: str,
        image_base_dir: Path,
        unknown_ratio: float = 0.3,
        hard_unknown_ratio: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Генерирует сэмплы для конкретной выборки (train/val/test).

        Аргументы:
            split_type: "train", "val" или "test"
            image_base_dir: базовая директория для изображений
            unknown_ratio: доля сэмплов, которые должны быть
                none-of-the-above (distractor'ы с низким retrieval score).
            hard_unknown_ratio: доля сэмплов, которые должны быть
                сложными none-of-the-above (distractor'ы с высоким retrieval
                score, score >= config.hard_unknown_min_score). Они учат
                модель отклонять даже визуально похожих кандидатов.
                Оценивается раньше unknown_ratio — вероятности
                независимы (сэмпл может быть только одного типа).

        Возвращает:
            Список сэмплов
        """
        logger.info(f"Generating {split_type} samples...")
        logger.info(
            f"  unknown_ratio={unknown_ratio:.2f}, "
            f"hard_unknown_ratio={hard_unknown_ratio:.2f}"
        )

        samples = []

        for split in tqdm(self.landmark_splits, desc=f"Generating {split_type}"):
            # Получаем query-изображения для этой выборки
            if split_type == "train":
                query_images = split.query_train_images
            elif split_type == "val":
                query_images = split.query_val_images
            elif split_type == "test":
                query_images = split.query_test_images
            else:
                raise ValueError(f"Invalid split_type: {split_type}")

            if not query_images:
                continue

            for query_img in query_images:
                # Определяем тип сэмпла через независимые случайные розыгрыши.
                # Сначала проверяется hard_unknown; если не сработал,
                # проверяется обычный unknown.
                rand_val = random.random()
                if rand_val < hard_unknown_ratio:
                    force_hard_unknown = True
                    force_unknown = False
                elif rand_val < hard_unknown_ratio + unknown_ratio:
                    force_hard_unknown = False
                    force_unknown = True
                else:
                    force_hard_unknown = False
                    force_unknown = False

                sample = self.generate_sample(
                    query_image_path=query_img,
                    query_landmark_id=split.landmark_id,
                    image_base_dir=image_base_dir,
                    force_unknown=force_unknown,
                    force_hard_unknown=force_hard_unknown,
                )

                if sample:
                    samples.append(sample)

        logger.info(f"Generated {len(samples)} {split_type} samples")
        self._log_statistics()

        return samples

    def _log_statistics(self) -> None:
        """Логирует статистику генерации."""
        logger.info("=" * 60)
        logger.info("СТАТИСТИКА ГЕНЕРАЦИИ СЭМПЛОВ")
        logger.info("=" * 60)
        logger.info(f"Всего сгенерировано: {self.stats['total_generated']}")
        logger.info(f"Позитив в кандидатах: {self.stats['positive_in_candidates']}")
        logger.info(f"None-of-the-above: {self.stats['none_of_the_above']}")
        logger.info(f"Ошибок retrieval: {self.stats['failed_retrieval']}")
        logger.info(f"Query в кандидатах (ОШИБКА): {self.stats['query_in_candidates']}")
        logger.info("=" * 60)


# ======================
# ЗАПИСЬ ФАЙЛОВ
# ======================
class FileWriter:
    """Отвечает за запись сэмплов в JSON-файлы."""

    @staticmethod
    def save_json(path: Path, data: Any) -> None:
        """Сохраняет данные в JSON-файл."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Логируем подходящее сообщение в зависимости от типа данных
            if isinstance(data, list):
                logger.info(f"Saved {len(data)} samples to {path}")
            else:
                logger.info(f"Saved data to {path}")
        except Exception as e:
            logger.error(f"Failed to save file {path}: {e}")
            raise


# ======================
# ГЕНЕРАТОР CONTRASTIVE-СЭМПЛОВ (НОВЫЙ)
# ======================
class ImageLevelContrastiveGenerator:
    """
    Генерирует сэмплы contrastive-обучения с корректными ограничениями на уровне изображений.

    Ключевые требования:
    - anchor = query-изображение (из query_images)
    - positive = gallery-изображение того же объекта (из gallery_images)
    - negative = gallery-изображение другого объекта
    - Только для объектов с >= 3 изображениями (supports_contrastive=True)
    - НЕТ утечки query/gallery

    Это гарантирует:
    - Anchor != positive (разные изображения)
    - Positive берётся из gallery (не из query-набора)
    - Negative берётся из gallery (не из query-набора)
    """

    def __init__(self, landmark_splits: list[LandmarkImageSplit]):
        self.landmark_splits = landmark_splits

        # Строим таблицу поиска
        self.lid_to_split = {split.landmark_id: split for split in landmark_splits}

        # Статистика
        self.stats = {
            "total_generated": 0,
            "skipped_no_support": 0,
            "skipped_no_gallery": 0,
            "skipped_no_negative": 0,
        }

    def generate_from_samples(
        self, samples: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Генерирует contrastive-сэмплы из reranking-сэмплов.

        Аргументы:
            samples: список сэмплов из RetrievalBasedSampleGenerator

        Возвращает:
            Список contrastive-триплетов
        """
        contrastive_samples = []

        for sample in samples:
            # Получаем информацию об объекте
            landmark_id = sample["meta"]["landmark_id"]
            anchor_image = sample["query_image"]

            # Получаем информацию о split
            split = self.lid_to_split.get(landmark_id)
            if not split:
                continue

            # Проверяем, поддерживает ли объект contrastive-обучение
            if not split.supports_contrastive:
                self.stats["skipped_no_support"] += 1
                continue

            # Получаем positive из gallery-изображений
            # КРИТИЧНО: positive должен быть из gallery, а не из query
            if not split.gallery_images:
                self.stats["skipped_no_gallery"] += 1
                continue

            positive_image = random.choice(split.gallery_images)

            # Валидация: anchor != positive
            if anchor_image == positive_image:
                logger.warning(
                    f"Anchor equals positive for {landmark_id}: {anchor_image}"
                )
                continue

            # Получаем negative из кандидатов
            # Предпочитаем кандидатов из других объектов
            negative_candidates = [
                c
                for c in sample["candidates"]
                if c.get("landmark_id") != landmark_id and c.get("image")
            ]

            if not negative_candidates:
                self.stats["skipped_no_negative"] += 1
                continue

            # Выбираем negative (предпочитаем hard negatives)
            negative = random.choice(negative_candidates)
            negative_image = negative["image"]

            # Создаём contrastive-сэмпл
            contrastive_sample = {
                "anchor": anchor_image,
                "positive": positive_image,
                "negative": negative_image,
                "landmark_id": landmark_id,
                "landmark_name": split.landmark_name,
                "negative_landmark_id": negative.get("landmark_id", ""),
                "negative_landmark_name": negative.get("name", ""),
                "negative_type": negative.get("candidate_type", "unknown"),
                "negative_retrieval_score": negative.get("retrieval_score", 0.0),
            }

            contrastive_samples.append(contrastive_sample)
            self.stats["total_generated"] += 1

        self._log_statistics()
        return contrastive_samples

    def _log_statistics(self) -> None:
        """Логирует статистику генерации."""
        logger.info("=" * 60)
        logger.info("СТАТИСТИКА ГЕНЕРАЦИИ CONTRASTIVE-СЭМПЛОВ")
        logger.info("=" * 60)
        logger.info(f"Всего сгенерировано: {self.stats['total_generated']}")
        logger.info(f"Пропущено (нет поддержки): {self.stats['skipped_no_support']}")
        logger.info(f"Пропущено (нет gallery): {self.stats['skipped_no_gallery']}")
        logger.info(f"Пропущено (нет негатива): {self.stats['skipped_no_negative']}")
        logger.info("=" * 60)


# ======================
# ОЦЕНКА RETRIEVAL
# ======================
@dataclass
class RetrievalMetrics:
    """Контейнер для метрик оценки retrieval."""

    # Глобальные метрики
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    map_score: float = 0.0  # Mean Average Precision
    ndcg_at_10: float = 0.0  # Normalized Discounted Cumulative Gain
    positive_retrieval_rate: float = 0.0
    retrieval_ceiling: float = 0.0  # Макс. возможный recall (positive в top-k)

    # Счётчики сэмплов
    total_queries: int = 0
    queries_with_positive: int = 0

    # Распределение по сложности
    hard_negatives: int = 0
    semi_hard_negatives: int = 0
    easy_negatives: int = 0

    # Метрики по категориям (по числу изображений)
    metrics_by_image_count: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь для логирования."""
        return {
            "recall@1": self.recall_at_1,
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "mrr": self.mrr,
            "map": self.map_score,
            "ndcg@10": self.ndcg_at_10,
            "positive_retrieval_rate": self.positive_retrieval_rate,
            "retrieval_ceiling": self.retrieval_ceiling,
            "total_queries": self.total_queries,
            "queries_with_positive": self.queries_with_positive,
            "hardness_distribution": {
                "hard": self.hard_negatives,
                "semi_hard": self.semi_hard_negatives,
                "easy": self.easy_negatives,
            },
            "by_image_count": self.metrics_by_image_count,
        }


class RetrievalEvaluator:
    """Оценивает качество retrieval с подробными метриками."""

    def __init__(self, landmark_splits: list[LandmarkImageSplit]):
        self.landmark_splits = landmark_splits
        self.lid_to_image_count = {
            split.landmark_id: split.total_images for split in landmark_splits
        }

    def _get_image_count_category(self, num_images: int) -> str:
        """Возвращает метку категории по числу изображений."""
        if num_images == 1:
            return "1_image"
        elif num_images == 2:
            return "2_images"
        elif num_images == 3:
            return "3_images"
        else:
            return "4plus_images"

    @staticmethod
    def _compute_average_precision(target_idx: int, num_candidates: int) -> float:
        """Вычисляет Average Precision для одного запроса."""
        if target_idx == -1:
            return 0.0
        rank = target_idx + 1
        return 1.0 / rank

    @staticmethod
    def _compute_dcg(relevances: list[float], k: int = 10) -> float:
        """Вычисляет Discounted Cumulative Gain."""
        dcg = 0.0
        for i, rel in enumerate(relevances[:k], start=1):
            dcg += rel / np.log2(i + 1)
        return dcg

    @staticmethod
    def _compute_ndcg(target_idx: int, num_candidates: int, k: int = 10) -> float:
        """Вычисляет Normalized Discounted Cumulative Gain."""
        if target_idx == -1:
            return 0.0

        relevances = [0.0] * min(num_candidates, k)
        if target_idx < k:
            relevances[target_idx] = 1.0

        dcg = RetrievalEvaluator._compute_dcg(relevances, k)
        ideal_relevances = [1.0] + [0.0] * (k - 1)
        idcg = RetrievalEvaluator._compute_dcg(ideal_relevances, k)

        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_samples(
        self, samples: list[dict[str, Any]], split_name: str = "unknown"
    ) -> RetrievalMetrics:
        """Оценивает качество retrieval на сгенерированных сэмплах."""
        logger.info(f"Evaluating retrieval for {split_name} split...")

        metrics = RetrievalMetrics()
        category_stats = {}

        for sample in samples:
            query_lid = sample["meta"]["landmark_id"]
            target_idx = sample["target_idx"]
            candidates = sample["candidates"]
            num_images = sample["meta"]["num_images"]

            category = self._get_image_count_category(num_images)
            if category not in category_stats:
                category_stats[category] = {
                    "total": 0,
                    "recall@1": 0,
                    "recall@5": 0,
                    "recall@10": 0,
                    "mrr_sum": 0.0,
                    "map_sum": 0.0,
                    "ndcg_sum": 0.0,
                    "positive_retrieved": 0,
                }

            metrics.total_queries += 1
            category_stats[category]["total"] += 1

            # Подсчитываем распределение negatives по сложности
            for cand in candidates:
                cand_type = cand.get("candidate_type", "unknown")
                if cand_type == "hard":
                    metrics.hard_negatives += 1
                elif cand_type == "semi_hard":
                    metrics.semi_hard_negatives += 1
                elif cand_type == "easy":
                    metrics.easy_negatives += 1

            if target_idx == -1:
                continue

            metrics.queries_with_positive += 1
            category_stats[category]["positive_retrieved"] += 1

            rank = target_idx + 1

            if rank <= 1:
                metrics.recall_at_1 += 1
                category_stats[category]["recall@1"] += 1

            if rank <= 5:
                metrics.recall_at_5 += 1
                category_stats[category]["recall@5"] += 1

            if rank <= 10:
                metrics.recall_at_10 += 1
                category_stats[category]["recall@10"] += 1

            reciprocal_rank = 1.0 / rank
            metrics.mrr += reciprocal_rank
            category_stats[category]["mrr_sum"] += reciprocal_rank

            ap = self._compute_average_precision(target_idx, len(candidates))
            metrics.map_score += ap
            category_stats[category]["map_sum"] += ap

            ndcg = self._compute_ndcg(target_idx, len(candidates), k=10)
            metrics.ndcg_at_10 += ndcg
            category_stats[category]["ndcg_sum"] += ndcg

        if metrics.total_queries > 0:
            metrics.recall_at_1 /= metrics.total_queries
            metrics.recall_at_5 /= metrics.total_queries
            metrics.recall_at_10 /= metrics.total_queries
            metrics.mrr /= metrics.total_queries
            metrics.map_score /= metrics.total_queries
            metrics.ndcg_at_10 /= metrics.total_queries
            metrics.positive_retrieval_rate = (
                metrics.queries_with_positive / metrics.total_queries
            )
            metrics.retrieval_ceiling = (
                metrics.queries_with_positive / metrics.total_queries
            )

        for category, stats in category_stats.items():
            total = stats["total"]
            if total > 0:
                metrics.metrics_by_image_count[category] = {
                    "total_queries": total,
                    "recall@1": stats["recall@1"] / total,
                    "recall@5": stats["recall@5"] / total,
                    "recall@10": stats["recall@10"] / total,
                    "mrr": stats["mrr_sum"] / total,
                    "map": stats["map_sum"] / total,
                    "ndcg@10": stats["ndcg_sum"] / total,
                    "positive_retrieval_rate": (stats["positive_retrieved"] / total),
                }

        self._log_metrics(metrics, split_name)
        return metrics

    def _log_metrics(self, metrics: RetrievalMetrics, split_name: str) -> None:
        """Логирует подробные метрики."""
        logger.info("=" * 70)
        logger.info(f"RETRIEVAL EVALUATION: {split_name.upper()}")
        logger.info("=" * 70)

        logger.info("Global Metrics:")
        logger.info(f"  Total queries: {metrics.total_queries}")
        logger.info(
            f"  Positive retrieval rate: "
            f"{metrics.positive_retrieval_rate:.3f} "
            f"({metrics.queries_with_positive}/{metrics.total_queries})"
        )
        logger.info(f"  Recall@1:  {metrics.recall_at_1:.3f}")
        logger.info(f"  Recall@5:  {metrics.recall_at_5:.3f}")
        logger.info(f"  Recall@10: {metrics.recall_at_10:.3f}")
        logger.info(f"  MRR:       {metrics.mrr:.3f}")
        logger.info(f"  MAP:       {metrics.map_score:.3f}")
        logger.info(f"  NDCG@10:   {metrics.ndcg_at_10:.3f}")
        logger.info(f"  Retrieval Ceiling: {metrics.retrieval_ceiling:.3f}")
        logger.info("")

        total_negatives = (
            metrics.hard_negatives
            + metrics.semi_hard_negatives
            + metrics.easy_negatives
        )
        if total_negatives > 0:
            logger.info("Negative Hardness Distribution:")
            logger.info(
                f"  Hard:      {metrics.hard_negatives} "
                f"({100 * metrics.hard_negatives / total_negatives:.1f}%)"
            )
            logger.info(
                f"  Semi-hard: {metrics.semi_hard_negatives} "
                f"({100 * metrics.semi_hard_negatives / total_negatives:.1f}%)"
            )
            logger.info(
                f"  Easy:      {metrics.easy_negatives} "
                f"({100 * metrics.easy_negatives / total_negatives:.1f}%)"
            )
            logger.info("")

        if metrics.metrics_by_image_count:
            logger.info("Metrics by Landmark Image Count:")
            categories = sorted(metrics.metrics_by_image_count.keys())

            for category in categories:
                cat_metrics = metrics.metrics_by_image_count[category]
                logger.info(f"  {category}:")
                logger.info(f"    Queries: {cat_metrics['total_queries']}")
                logger.info(
                    f"    Positive rate: {cat_metrics['positive_retrieval_rate']:.3f}"
                )
                logger.info(f"    Recall@1:  {cat_metrics['recall@1']:.3f}")
                logger.info(f"    Recall@5:  {cat_metrics['recall@5']:.3f}")
                logger.info(f"    Recall@10: {cat_metrics['recall@10']:.3f}")
                logger.info(f"    MRR:       {cat_metrics['mrr']:.3f}")
                logger.info(f"    MAP:       {cat_metrics['map']:.3f}")
                logger.info(f"    NDCG@10:   {cat_metrics['ndcg@10']:.3f}")
                logger.info("")

        logger.info("=" * 70)

    def compare_splits(
        self,
        train_metrics: RetrievalMetrics,
        val_metrics: RetrievalMetrics,
        test_metrics: RetrievalMetrics,
    ) -> None:
        """Логирует сравнение метрик по всем выборкам."""
        logger.info("=" * 70)
        logger.info("RETRIEVAL METRICS COMPARISON")
        logger.info("=" * 70)

        logger.info(f"{'Metric':<25} {'Train':>12} {'Val':>12} {'Test':>12}")
        logger.info("-" * 70)

        metrics_to_compare = [
            ("Recall@1", "recall_at_1"),
            ("Recall@5", "recall_at_5"),
            ("Recall@10", "recall_at_10"),
            ("MRR", "mrr"),
            ("MAP", "map_score"),
            ("NDCG@10", "ndcg_at_10"),
            ("Positive Rate", "positive_retrieval_rate"),
        ]

        for label, attr in metrics_to_compare:
            train_val = getattr(train_metrics, attr)
            val_val = getattr(val_metrics, attr)
            test_val = getattr(test_metrics, attr)

            logger.info(
                f"{label:<25} {train_val:>12.3f} {val_val:>12.3f} {test_val:>12.3f}"
            )

        logger.info("-" * 70)
        logger.info(
            f"{'Total Queries':<25} "
            f"{train_metrics.total_queries:>12} "
            f"{val_metrics.total_queries:>12} "
            f"{test_metrics.total_queries:>12}"
        )
        logger.info("=" * 70)


# ======================
# ВАЛИДАЦИЯ
# ======================
class DatasetValidator:
    """
    Валидирует датасет для продакшен-корректного closed-set retrieval.

    Проверки:
    - Нет пересечения gallery-query
    - Нет пересечения query-candidate
    - Согласованность target_idx
    - Валидность contrastive-триплетов
    """

    @staticmethod
    def validate_splits(splits: list[LandmarkImageSplit]) -> bool:
        """Валидирует распределение ролей изображений."""
        logger.info("Validating image role splits...")

        errors = []
        for split in splits:
            try:
                split.validate()
            except ValueError as e:
                errors.append(str(e))

        if errors:
            logger.error(f"Split validation failed: {len(errors)} errors")
            for err in errors[:10]:  # Показываем первые 10
                logger.error(f"  - {err}")
            return False

        logger.info("✓ All splits valid (no gallery-query overlap)")
        return True

    @staticmethod
    def validate_samples(samples: list[dict[str, Any]]) -> bool:
        """Валидирует сгенерированные сэмплы."""
        logger.info(f"Validating {len(samples)} samples...")

        errors = []
        query_in_candidates = 0

        for i, sample in enumerate(samples):
            query_img = sample.get("query_image")
            candidates = sample.get("candidates", [])
            target_idx = sample.get("target_idx", -1)

            # Проверяем, что query не входит в кандидатов
            for cand in candidates:
                if cand.get("image") == query_img:
                    query_in_candidates += 1
                    errors.append(f"Sample {i}: query image in candidates: {query_img}")

            # Проверяем корректность target_idx
            if target_idx != -1:
                if target_idx < 0 or target_idx >= len(candidates):
                    errors.append(
                        f"Sample {i}: invalid target_idx {target_idx} "
                        f"for {len(candidates)} candidates"
                    )

        if errors:
            logger.error(f"Sample validation failed: {len(errors)} errors")
            for err in errors[:10]:
                logger.error(f"  - {err}")
            return False

        if query_in_candidates > 0:
            logger.error(
                f"✗ CRITICAL: {query_in_candidates} samples have "
                "query image in candidates!"
            )
            return False

        logger.info("✓ All samples valid (no query-candidate overlap)")
        return True

    @staticmethod
    def validate_contrastive(contrastive_samples: list[dict[str, Any]]) -> bool:
        """Валидирует contrastive-сэмплы."""
        logger.info(f"Validating {len(contrastive_samples)} contrastive...")

        errors = []

        for i, sample in enumerate(contrastive_samples):
            anchor = sample.get("anchor")
            positive = sample.get("positive")
            negative = sample.get("negative")

            # Проверяем anchor != positive
            if anchor == positive:
                errors.append(f"Contrastive {i}: anchor equals positive: {anchor}")

            # Проверяем anchor != negative
            if anchor == negative:
                errors.append(f"Contrastive {i}: anchor equals negative: {anchor}")

            # Проверяем positive != negative
            if positive == negative:
                errors.append(f"Contrastive {i}: positive equals negative: {positive}")

        if errors:
            logger.error(f"Contrastive validation failed: {len(errors)} errors")
            for err in errors[:10]:
                logger.error(f"  - {err}")
            return False

        logger.info("✓ All contrastive samples valid")
        return True

    @staticmethod
    def log_dataset_summary(
        splits: list[LandmarkImageSplit],
        train_samples: list[dict],
        val_samples: list[dict],
        test_samples: list[dict],
        train_contrastive: list[dict],
        val_contrastive: list[dict],
        test_contrastive: list[dict],
    ) -> None:
        """Логирует полную сводку по датасету."""
        logger.info("=" * 70)
        logger.info("FINAL DATASET SUMMARY")
        logger.info("=" * 70)

        # Сводка по splits
        total_gallery = sum(len(s.gallery_images) for s in splits)
        total_train_q = sum(len(s.query_train_images) for s in splits)
        total_val_q = sum(len(s.query_val_images) for s in splits)
        total_test_q = sum(len(s.query_test_images) for s in splits)

        logger.info(f"Landmarks: {len(splits)}")
        logger.info(f"Gallery images: {total_gallery}")
        logger.info(
            f"Query images: train={total_train_q}, "
            f"val={total_val_q}, test={total_test_q}"
        )
        logger.info("")

        # Сводка по сэмплам
        logger.info("Reranking samples:")
        logger.info(f"  Train: {len(train_samples)}")
        logger.info(f"  Val: {len(val_samples)}")
        logger.info(f"  Test: {len(test_samples)}")
        logger.info("")

        # Сводка по contrastive
        logger.info("Contrastive samples:")
        logger.info(f"  Train: {len(train_contrastive)}")
        logger.info(f"  Val: {len(val_contrastive)}")
        logger.info(f"  Test: {len(test_contrastive)}")
        logger.info("")

        # Распределение target
        train_positive = sum(1 for s in train_samples if s["target_idx"] != -1)
        train_unknown = len(train_samples) - train_positive

        logger.info("Train target distribution:")
        logger.info(
            f"  Positive: {train_positive} "
            f"({100 * train_positive / len(train_samples):.1f}%)"
        )
        logger.info(
            f"  Unknown: {train_unknown} "
            f"({100 * train_unknown / len(train_samples):.1f}%)"
        )

        logger.info("=" * 70)

    @staticmethod
    def evaluate_retrieval_quality(
        train_samples: list[dict],
        val_samples: list[dict],
        test_samples: list[dict],
        landmark_splits: list[LandmarkImageSplit],
    ) -> tuple[RetrievalMetrics, RetrievalMetrics, RetrievalMetrics]:
        """
        Оценивает качество retrieval по всем выборкам.

        Аргументы:
            train_samples: обучающие сэмплы
            val_samples: валидационные сэмплы
            test_samples: тестовые сэмплы
            landmark_splits: splits объектов для метаданных

        Возвращает:
            Кортеж (train_metrics, val_metrics, test_metrics)
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info("EVALUATING RETRIEVAL QUALITY")
        logger.info("=" * 70)

        evaluator = RetrievalEvaluator(landmark_splits)

        # Оцениваем каждую выборку
        train_metrics = evaluator.evaluate_samples(train_samples, "train")
        val_metrics = evaluator.evaluate_samples(val_samples, "val")
        test_metrics = evaluator.evaluate_samples(test_samples, "test")

        # Сравниваем по всем выборкам
        logger.info("")
        evaluator.compare_splits(train_metrics, val_metrics, test_metrics)

        return train_metrics, val_metrics, test_metrics


# ======================
# ОСНОВНОЙ PIPELINE
# ======================
def main():
    """
    НОВЫЙ ПРОДАКШЕН-КОРРЕКТНЫЙ PIPELINE

    Ключевые изменения по сравнению со старым pipeline:
    1. Разбиение на уровне изображений (не на уровне объектов)
    2. FAISS-индекс только из gallery
    3. Кандидаты на основе естественного retrieval
    4. Формат target_idx (не target.name)
    5. Корректные contrastive-ограничения

    Имитируемое продакшен-поведение:
    Фото пользователя → SigLIP → FAISS retrieval → Gallery-изображения
    → Группировка по объекту → Max similarity → Top-K объектов
    → Qwen2-VL reranker → Multiple-choice → Предсказание
    """
    # Инициализируем конфигурации
    dataset_config = DatasetConfig()

    # Определяем имя модели: явное переопределение или дефолт по типу
    _default_models = {
        "siglip": "google/siglip-base-patch16-224",
        "dinov2": "facebook/dinov2-base",
    }
    _model_name = (
        dataset_config.embedder_model
        if dataset_config.embedder_model
        else _default_models.get(dataset_config.embedder_type, "")
    )
    if not _model_name:
        raise ValueError(
            f"Unknown embedder_type '{dataset_config.embedder_type}'. "
            "Supported: 'siglip', 'dinov2'."
        )

    index_config = IndexConfig(
        model_name=_model_name,
        embedder_type=dataset_config.embedder_type,
        batch_size=32,
        max_images_per_landmark=10,
    )
    logger.info(f"Embedder: {dataset_config.embedder_type} ({index_config.model_name})")

    # Задаём random seed для воспроизводимости
    random.seed(dataset_config.random_seed)
    np.random.seed(dataset_config.random_seed)
    torch.manual_seed(dataset_config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(dataset_config.random_seed)

    # На macOS задаём один поток для предотвращения segfault
    if _platform.system() == "Darwin":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        logger.info("Set PyTorch to single-threaded mode for macOS")

    logger.info("=" * 70)
    logger.info("PRODUCTION-CORRECT DATASET GENERATION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Random seed: {dataset_config.random_seed}")
    logger.info("")

    try:
        # ============================================================
        # ШАГ 1: Загрузка и подготовка данных
        # ============================================================
        data_path = Path(dataset_config.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Input file not found: {data_path}")

        logger.info("Loading landmark data...")
        with open(dataset_config.data_path, encoding="utf-8") as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        # Извлекаем пути к изображениям
        def extract_paths(valid_images):
            if isinstance(valid_images, list):
                return [
                    img.get("path") if isinstance(img, dict) else img
                    for img in valid_images
                ]
            return []

        df["images"] = df["valid_images"].apply(extract_paths)

        # Объединяем поля с именами
        if "name_ru" in df.columns:
            df["name"] = df["name_ru"].fillna(df.get("name_en", ""))
        elif "name_en" in df.columns:
            df["name"] = df["name_en"]

        # Оставляем объекты, у которых есть изображения
        df = df[df["images"].map(len) > 0].reset_index(drop=True)
        logger.info(f"Loaded {len(df)} landmarks with images")

        # ============================================================
        # ШАГ 2: Распределение изображений по ролям gallery/query
        # ============================================================
        logger.info("")
        logger.info("STEP 2: Image role assignment")
        logger.info("-" * 70)

        splitter = ImageRoleSplitter(
            config=dataset_config,
            max_gallery_per_landmark=dataset_config.max_gallery_per_landmark,
        )
        splits = splitter.split_all_landmarks(df)

        # Валидация splits
        if not DatasetValidator.validate_splits(splits):
            raise ValueError("Split validation failed!")

        # ============================================================
        # ШАГ 3: Построение или загрузка gallery FAISS-индекса
        # ============================================================
        logger.info("")
        logger.info("STEP 3: Gallery index preparation")
        logger.info("-" * 70)

        # Проверяем, что директория с изображениями существует
        image_base_dir = dataset_config.image_base_dir
        if not image_base_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_base_dir}")
        logger.info(f"Using image directory: {image_base_dir}")

        # Определяем пути обучающего индекса (НЕ продакшен-индекса).
        # Включаем тип embedder в имя файла, чтобы индексы siglip и dinov2
        # могли сосуществовать в одной выходной директории.
        _emb_suffix = dataset_config.embedder_type  # "siglip" или "dinov2"
        training_index_path = (
            dataset_config.output_dir / f"training_gallery_index_{_emb_suffix}.faiss"
        )
        training_metadata_path = (
            dataset_config.output_dir / f"training_gallery_metadata_{_emb_suffix}.json"
        )
        training_embeddings_path = (
            dataset_config.output_dir / f"training_gallery_embeddings_{_emb_suffix}.npy"
        )

        # Проверяем, можем ли переиспользовать существующий обучающий индекс
        _index_exists = training_index_path.exists()
        _meta_exists = training_metadata_path.exists()
        _emb_exists = training_embeddings_path.exists()

        if dataset_config.reuse_training_index:
            if dataset_config.force_rebuild_index:
                logger.info("Index reuse disabled: force_rebuild_index=True")
            elif not _index_exists:
                logger.info(
                    f"Index reuse skipped: index file not found: {training_index_path}"
                )
            elif not _meta_exists:
                logger.info(
                    f"Index reuse skipped: metadata file not found: "
                    f"{training_metadata_path}"
                )
            elif not _emb_exists:
                logger.info(
                    f"Index reuse skipped: embeddings file not found: "
                    f"{training_embeddings_path}"
                )

        can_reuse = (
            dataset_config.reuse_training_index
            and not dataset_config.force_rebuild_index
            and _index_exists
            and _meta_exists
            and _emb_exists
        )

        # Инициализируем index builder (нужен и для построения, и для загрузки)
        index_builder = IndexBuilder(index_config)

        if can_reuse:
            # ============================================================
            # ШАГ 3A: Загрузка существующего обучающего индекса
            # ============================================================
            logger.info("Loading existing training gallery index...")
            logger.info(f"  Index: {training_index_path}")
            logger.info(f"  Metadata: {training_metadata_path}")
            logger.info(f"  Embeddings: {training_embeddings_path}")

            try:
                # Загружаем FAISS-индекс
                gallery_index = faiss.read_index(str(training_index_path))
                logger.info(f"Loaded FAISS index: {gallery_index.ntotal} images")

                # Загружаем метаданные
                with open(training_metadata_path, encoding="utf-8") as f:
                    metadata_dicts = json.load(f)
                gallery_metadata = [GalleryImageMetadata(**m) for m in metadata_dicts]
                logger.info(f"Loaded metadata: {len(gallery_metadata)} entries")

                # Загружаем embeddings
                embeddings = np.load(training_embeddings_path)
                logger.info(f"Loaded embeddings: {embeddings.shape}")

                # Проверяем согласованность (явное OR — цепочка != ненадёжна)
                n_faiss = gallery_index.ntotal
                n_meta = len(gallery_metadata)
                n_emb = len(embeddings)
                if n_faiss != n_meta or n_faiss != n_emb:
                    raise ValueError(
                        f"Inconsistent index sizes: "
                        f"FAISS={n_faiss}, "
                        f"metadata={n_meta}, "
                        f"embeddings={n_emb}"
                    )

                logger.info("✓ Successfully loaded training index")
                logger.info("  (Use force_rebuild_index=True to rebuild)")

            except Exception as e:
                import traceback

                logger.warning(f"Failed to load training index: {e}")
                logger.warning(traceback.format_exc())
                logger.info("Falling back to building new index...")
                can_reuse = False

        if not can_reuse:
            # ============================================================
            # ШАГ 3B: Построение нового gallery-индекса
            # ============================================================
            logger.info("Building new training gallery index...")

            try:
                # Строим gallery-индекс с батчевым кодированием + аугментацией
                gallery_builder = GalleryIndexBuilder(
                    index_builder,
                    batch_size=dataset_config.encoding_batch_size,
                    gallery_augmentations=(dataset_config.gallery_augmentations),
                    gallery_augmentation_threshold=(
                        dataset_config.gallery_augmentation_threshold
                    ),
                )
                embeddings, gallery_metadata, gallery_index = (
                    gallery_builder.build_from_splits(
                        splits=splits, df=df, image_base_dir=image_base_dir
                    )
                )

                logger.info(f"Gallery index built: {gallery_index.ntotal} images")

            except Exception as e:
                logger.error(f"Failed to build gallery index: {e}")
                raise

        # ============================================================
        # ШАГ 4: Генерация reranking-сэмплов
        # ============================================================
        logger.info("")
        logger.info("STEP 4: Generating reranking samples")
        logger.info("-" * 70)

        sample_generator = RetrievalBasedSampleGenerator(
            config=dataset_config,
            df=df,
            gallery_index=gallery_index,
            gallery_metadata=gallery_metadata,
            landmark_splits=splits,
            index_builder=index_builder,
        )

        # Калибруем пороги сложности под шкалу энкодера (SigLIP/DINOv2/…) ДО
        # генерации: иначе абсолютные 0.85/0.75/0.88 ломают hardness-разметку
        # на энкодере с другим распределением score.
        sample_generator.calibrate_score_thresholds(image_base_dir)

        # Генерируем для каждой выборки.
        # hard_unknown_ratio передаётся из config — train/val/test все получают
        # hard unknown сэмплы, чтобы обучать и оценивать способность модели
        # отклонять уверенные false positives (retrieval_score >= threshold).
        train_samples = sample_generator.generate_samples_from_splits(
            split_type="train",
            image_base_dir=image_base_dir,
            unknown_ratio=dataset_config.unknown_ratio,
            hard_unknown_ratio=dataset_config.hard_unknown_ratio,
        )

        val_samples = sample_generator.generate_samples_from_splits(
            split_type="val",
            image_base_dir=image_base_dir,
            unknown_ratio=dataset_config.unknown_ratio,
            hard_unknown_ratio=dataset_config.hard_unknown_ratio,
        )

        test_samples = sample_generator.generate_samples_from_splits(
            split_type="test",
            image_base_dir=image_base_dir,
            unknown_ratio=dataset_config.unknown_ratio,
            hard_unknown_ratio=dataset_config.hard_unknown_ratio,
        )

        # Валидация сэмплов
        logger.info("")
        if not DatasetValidator.validate_samples(train_samples):
            raise ValueError("Train sample validation failed!")
        if not DatasetValidator.validate_samples(val_samples):
            raise ValueError("Val sample validation failed!")
        if not DatasetValidator.validate_samples(test_samples):
            raise ValueError("Test sample validation failed!")

        # Оцениваем качество retrieval
        train_metrics, val_metrics, test_metrics = (
            DatasetValidator.evaluate_retrieval_quality(
                train_samples=train_samples,
                val_samples=val_samples,
                test_samples=test_samples,
                landmark_splits=splits,
            )
        )

        # ============================================================
        # ШАГ 5: Генерация contrastive-сэмплов
        # ============================================================
        logger.info("")
        logger.info("STEP 5: Generating contrastive samples")
        logger.info("-" * 70)

        contrastive_gen = ImageLevelContrastiveGenerator(splits)

        train_contrastive = contrastive_gen.generate_from_samples(train_samples)
        val_contrastive = contrastive_gen.generate_from_samples(val_samples)
        test_contrastive = contrastive_gen.generate_from_samples(test_samples)

        # Валидация contrastive
        logger.info("")
        if not DatasetValidator.validate_contrastive(train_contrastive):
            raise ValueError("Train contrastive validation failed!")
        if not DatasetValidator.validate_contrastive(val_contrastive):
            raise ValueError("Val contrastive validation failed!")
        if not DatasetValidator.validate_contrastive(test_contrastive):
            raise ValueError("Test contrastive validation failed!")

        # ============================================================
        # ШАГ 6: Сохранение датасетов
        # ============================================================
        logger.info("")
        logger.info("STEP 6: Saving datasets")
        logger.info("-" * 70)

        FileWriter.save_json(dataset_config.output_dir / "train.json", train_samples)
        FileWriter.save_json(dataset_config.output_dir / "val.json", val_samples)
        FileWriter.save_json(dataset_config.output_dir / "test.json", test_samples)

        FileWriter.save_json(
            dataset_config.output_dir / "train_contrastive.json", train_contrastive
        )
        FileWriter.save_json(
            dataset_config.output_dir / "val_contrastive.json", val_contrastive
        )
        FileWriter.save_json(
            dataset_config.output_dir / "test_contrastive.json", test_contrastive
        )

        # ============================================================
        # ШАГ 6.5: Сохранение обучающего gallery-индекса для повторного использования
        # ============================================================
        if not can_reuse:  # Сохраняем только если только что построили
            logger.info("")
            logger.info("STEP 6.5: Saving training gallery index")
            logger.info("-" * 70)
            logger.info("NOTE: This is a TRAINING index (gallery images only)")
            logger.info(
                "      For production, use LandmarkRetriever.build_index_from_landmarks()"
            )

            try:
                # Сохраняем FAISS-индекс
                logger.info(f"Saving FAISS index to {training_index_path}")
                faiss.write_index(gallery_index, str(training_index_path))

                # Сохраняем метаданные gallery
                logger.info(f"Saving metadata to {training_metadata_path}")
                metadata_dicts = [meta.to_dict() for meta in gallery_metadata]
                FileWriter.save_json(training_metadata_path, metadata_dicts)

                # Сохраняем embeddings (для анализа и проверок согласованности)
                logger.info(f"Saving embeddings to {training_embeddings_path}")
                np.save(training_embeddings_path, embeddings)

                logger.info(f"✓ Training index saved: {gallery_index.ntotal} images")
                logger.info(
                    f"  Index size: {training_index_path.stat().st_size / 1024 / 1024:.2f} MB"
                )
                logger.info(
                    f"  Embeddings size: {training_embeddings_path.stat().st_size / 1024 / 1024:.2f} MB"
                )
                logger.info("")
                logger.info("To reuse this index in future runs:")
                logger.info("  - Set reuse_training_index=True (default)")
                logger.info("  - Set force_rebuild_index=False (default)")
                logger.info("To force rebuild:")
                logger.info("  - Set force_rebuild_index=True")

            except Exception as e:
                logger.warning(f"Failed to save training index: {e}")
                logger.warning("Continuing without saving index...")

        # ============================================================
        # ШАГ 7: Сохранение метрик оценки
        # ============================================================
        logger.info("")
        logger.info("STEP 7: Saving evaluation metrics")
        logger.info("-" * 70)

        # Сохраняем метрики в JSON
        metrics_data = {
            "train": train_metrics.to_dict(),
            "val": val_metrics.to_dict(),
            "test": test_metrics.to_dict(),
        }

        FileWriter.save_json(
            dataset_config.output_dir / "retrieval_metrics.json", metrics_data
        )

        # ============================================================
        # ИТОГОВАЯ СВОДКА
        # ============================================================
        logger.info("")
        DatasetValidator.log_dataset_summary(
            splits=splits,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            train_contrastive=train_contrastive,
            val_contrastive=val_contrastive,
            test_contrastive=test_contrastive,
        )

        logger.info("")
        logger.info("✓ Dataset generation completed successfully!")
        logger.info(f"✓ Output directory: {dataset_config.output_dir}")
        logger.info("✓ All validation checks passed")
        logger.info("")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

    finally:
        # Освобождаем ресурсы, чтобы избежать предупреждений об утечке семафоров
        try:
            # Очищаем CUDA-кэш, если доступен
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Принудительно запускаем сборку мусора
            import gc

            gc.collect()

            logger.info("Cleanup completed")
        except Exception as cleanup_error:
            logger.warning(f"Cleanup warning: {cleanup_error}")


if __name__ == "__main__":
    main()

    # index_config = IndexConfig(
    #     embedder_type="siglip", batch_size=32, max_images_per_landmark=10
    # )
    # build_index_from_landmarks(
    #     landmarks_json_path="data/processed/landmarks_with_guide_descriptions_filtred.json",
    #     image_base_dir="images",
    #     output_dir="data/index/siglip",
    #     index_config=index_config,
    # )
    # index_config = IndexConfig(
    #     embedder_type="dinov2",
    #     batch_size=32,
    #     max_images_per_landmark=10)
    # build_index_from_landmarks(
    #     landmarks_json_path="data/processed/landmarks_with_guide_descriptions_filtred.json",
    #     image_base_dir="images",
    #     output_dir="data/index/dinov2",
    #     index_config=index_config
    # )
