# src/rag/indexing_v2.py
"""
Построитель FAISS-индекса для AI Tour Guide.

Поддерживает два бэкенда:
  - SigLIP  (google/siglip-so400m-patch14-384 или аналог)
  - DINOv2  (facebook/dinov2-base или аналог)

Для DINOv2 классификация exterior/interior не выполняется —
все изображения кодируются напрямую с равными весами.

Использование:
    from src.rag.indexing_v2 import IndexBuilder, IndexConfig

    # SigLIP (по умолчанию)
    config = IndexConfig(model_name="google/siglip-so400m-patch14-384")
    builder = IndexBuilder(config)

    embeddings, metadata, index = builder.build_from_json(
        "data/processed/landmarks.json"
    )
    builder.save(embeddings, metadata, index)
"""

# Предотвращение segfault на macOS: ограничиваем потоки до
# импорта нативных библиотек (numpy, torch, faiss).
import os as _os
import platform as _platform

_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

if _platform.system() == "Darwin":
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
    _os.environ.setdefault("MKL_NUM_THREADS", "1")
    _os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    _os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    _os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    _os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoProcessor, SiglipModel

logger = logging.getLogger(__name__)


# Конфигурация


@dataclass
class IndexConfig:
    """Конфигурация для построения FAISS-индекса."""

    # Название модели (пустая строка — автовыбор по embedder_type)
    model_name: str = ""
    # Тип энкодера: "siglip" или "dinov2"
    embedder_type: str = "siglip"
    # Устройство (всегда CPU для избежания segfault на macOS)
    device: str = "cpu"

    # Пороги классификации (только для SigLIP)
    object_threshold: float = 0.4
    exterior_weight: float = 1.0
    interior_weight: float = 0.5

    # Расчёт уверенности
    confidence_max_weight: float = 0.6
    confidence_mean_weight: float = 0.4
    confidence_quantity_scale: float = 2.0

    # Обработка
    batch_size: int = 32
    max_images_per_landmark: int = 10

    # Аугментация
    use_augmentation: bool = True
    augmentation_threshold: int = 3
    augmentations_per_image: int = 3

    # Пути для сохранения
    index_path: Path = Path("data/models/faiss_index.bin")
    metadata_path: Path = Path("data/models/faiss_metadata.pkl")
    embeddings_path: Path = Path("data/models/embeddings.npy")

    def __post_init__(self):
        """Автовыбор модели и создание директорий."""
        _defaults = {
            "siglip": "google/siglip-base-patch16-224",
            "dinov2": "facebook/dinov2-base",
        }
        if not self.model_name:
            self.model_name = _defaults.get(
                self.embedder_type.lower(),
                "google/siglip-base-patch16-224",
            )
        for path in [self.index_path, self.metadata_path, self.embeddings_path]:
            path.parent.mkdir(parents=True, exist_ok=True)


# Базовый энкодер


class _BaseEncoder:
    """Общая логика энкодеров: аугментация, слияние эмбеддингов, очистка."""

    @staticmethod
    def _build_augment_fn(use_augmentation: bool):
        """Пайплайн аугментаций для обучающих gallery-изображений (или None)."""
        if not use_augmentation:
            return None
        return transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=5, translate=(0.02, 0.02)),
            ]
        )

    @staticmethod
    def fuse_embeddings(
        embeddings: list[np.ndarray],
        weights: list[float] | None = None,
    ) -> np.ndarray:
        """Объединяет несколько эмбеддингов с опциональными весами."""
        if not embeddings:
            raise ValueError("Список эмбеддингов пуст")

        if weights is None:
            fused = np.mean(embeddings, axis=0)
        else:
            if len(embeddings) != len(weights):
                raise ValueError("Длины embeddings и weights не совпадают")
            fused = np.average(embeddings, axis=0, weights=weights)

        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm

        return np.array(fused)

    def __del__(self):
        try:
            if hasattr(self, "model"):
                del self.model
            if hasattr(self, "processor"):
                del self.processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# SigLIP энкодер


class SigLIPEncoder(_BaseEncoder):
    """
    Кодирует изображения через SigLIP с классификацией exterior/interior.

    Изображения, классифицированные как объекты (не здания),
    фильтруются и возвращают None.
    """

    # Текстовые промпты для классификации типа изображения
    TEXT_PROMPTS = [
        "a landmark building exterior",
        "architecture monument",
        "indoor room interior",
        "inside a building",
        "object or artifact",
    ]

    def __init__(self, config: IndexConfig):
        self.config = config
        self.device = config.device

        logger.info(f"Загрузка SigLIP модели: {config.model_name}")
        try:
            if _platform.system() == "Darwin":
                logger.info("macOS — используем CPU с отключёнными потоками")
                self.device = "cpu"

            self.processor = AutoProcessor.from_pretrained(
                config.model_name, local_files_only=False
            )
            self.model = SiglipModel.from_pretrained(
                config.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
            )
            self.model = self.model.to(self.device)
            self.model.eval()

            if _platform.system() == "Darwin":
                torch.set_num_threads(1)

            logger.info(f"SigLIP загружен на {self.device}")

            # Кэшируем текстовые эмбеддинги один раз
            self._cached_text_embeds, _ = self._encode_text_prompts()
            logger.info("Текстовые промпты закэшированы")

        except Exception as e:
            logger.error(f"Ошибка загрузки SigLIP: {e}")
            raise

        self.augment_fn = self._build_augment_fn(config.use_augmentation)

    def _encode_text_prompts(self) -> tuple[torch.Tensor, Any]:
        """Кодирует текстовые промпты для классификации."""
        text_inputs = self.processor(
            text=self.TEXT_PROMPTS,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            text_embeds = self.model.get_text_features(**text_inputs)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        return text_embeds, text_inputs

    def encode_batch(
        self, images: list[Image.Image]
    ) -> list[tuple[np.ndarray, str, float] | None]:
        """
        Кодирует батч изображений с классификацией exterior/interior.

        Returns:
            Список (embedding, label, confidence) или None для объектов.
        """
        if not images:
            return []

        try:
            image_inputs = self.processor(
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                img_embs = self.model.get_image_features(**image_inputs)
                img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)

                logits = img_embs @ self._cached_text_embeds.T
                probs = logits.softmax(dim=-1).cpu().numpy()
                img_embs_np = img_embs.cpu().numpy()

            results: list[tuple[np.ndarray, str, float] | None] = []
            for i in range(len(images)):
                exterior_prob = probs[i][0:2].mean()
                interior_prob = probs[i][2:4].mean()
                object_prob = probs[i][4]

                # Фильтруем объекты (не здания)
                if object_prob > self.config.object_threshold:
                    results.append(None)
                    continue

                if exterior_prob > interior_prob:
                    label = "exterior"
                    confidence = float(exterior_prob)
                else:
                    label = "interior"
                    confidence = float(interior_prob)

                results.append((np.array(img_embs_np[i]), label, confidence))

            return results

        except Exception as e:
            logger.error(f"Ошибка кодирования SigLIP батча: {e}")
            return [None] * len(images)


# DINOv2 энкодер


class DINOv2Encoder(_BaseEncoder):
    """
    Кодирует изображения через DINOv2.

    Классификация exterior/interior не выполняется —
    все изображения кодируются напрямую с confidence=1.0.
    """

    def __init__(self, config: IndexConfig):
        self.config = config
        self.device = config.device

        logger.info(f"Загрузка DINOv2 модели: {config.model_name}")
        try:
            from transformers import AutoImageProcessor, AutoModel

            if _platform.system() == "Darwin":
                logger.info("macOS — используем CPU с отключёнными потоками")
                self.device = "cpu"

            self.processor = AutoImageProcessor.from_pretrained(
                config.model_name, local_files_only=False
            )
            self.model = AutoModel.from_pretrained(
                config.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
            )
            self.model = self.model.to(self.device)
            self.model.eval()

            if _platform.system() == "Darwin":
                torch.set_num_threads(1)

            logger.info(f"DINOv2 загружен на {self.device}")

        except Exception as e:
            logger.error(f"Ошибка загрузки DINOv2: {e}")
            raise

        self.augment_fn = self._build_augment_fn(config.use_augmentation)

    def encode_batch(
        self, images: list[Image.Image]
    ) -> list[tuple[np.ndarray, str, float] | None]:
        """
        Кодирует батч изображений без классификации.

        Returns:
            Список (embedding, "unknown", 1.0) для каждого изображения.
        """
        if not images:
            return []

        try:
            inputs = self.processor(
                images=images,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Используем [CLS] токен как эмбеддинг изображения
                cls_embs = outputs.last_hidden_state[:, 0, :]
                cls_embs = cls_embs / cls_embs.norm(dim=-1, keepdim=True)
                cls_embs_np: np.ndarray = cls_embs.cpu().numpy()

            return [(cls_embs_np[i], "unknown", 1.0) for i in range(len(images))]

        except Exception as e:
            logger.error(f"Ошибка кодирования DINOv2 батча: {e}")
            return [None] * len(images)


# Фабрика энкодеров


def build_encoder(
    config: IndexConfig,
) -> SigLIPEncoder | DINOv2Encoder:
    """
    Создаёт энкодер на основе config.embedder_type.

    Поддерживаемые значения: "siglip", "dinov2".
    """
    etype = config.embedder_type.lower()
    if etype == "siglip":
        return SigLIPEncoder(config)
    if etype == "dinov2":
        return DINOv2Encoder(config)
    raise ValueError(
        f"Неизвестный embedder_type '{config.embedder_type}'. "
        "Поддерживаются: 'siglip', 'dinov2'."
    )


# Построитель индекса


class IndexBuilder:
    """Строит FAISS-индекс из данных о достопримечательностях."""

    def __init__(self, config: IndexConfig | None = None):
        self.config = config or IndexConfig()
        self.encoder = build_encoder(self.config)

    def cleanup(self):
        """Явно освобождает ресурсы."""
        try:
            if hasattr(self, "encoder"):
                del self.encoder
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            if logger is not None:
                logger.warning(f"Предупреждение при очистке: {e}")

    def __del__(self):
        self.cleanup()

    def _load_and_process_images(
        self,
        image_paths: list[str],
        base_dir: Path | None = None,
    ) -> tuple[list[np.ndarray], list[float], list[str]]:
        """
        Загружает и кодирует изображения для одной достопримечательности.

        Returns:
            (embeddings, weights, valid_paths)
        """
        images_to_process: list[Image.Image] = []
        valid_indices: list[int] = []

        for i, img_path in enumerate(
            image_paths[: self.config.max_images_per_landmark]
        ):
            try:
                full_path = (base_dir / img_path) if base_dir else Path(img_path)
                img = Image.open(full_path).convert("RGB")
                images_to_process.append(img)
                valid_indices.append(i)
            except Exception as e:
                logger.debug(f"Ошибка загрузки {img_path}: {e}")
                continue

        if not images_to_process:
            return [], [], []

        # Аугментация для landmark с малым числом изображений
        original_count = len(images_to_process)
        if (
            self.config.use_augmentation
            and self.encoder.augment_fn is not None
            and original_count < self.config.augmentation_threshold
        ):
            for img in images_to_process[:original_count]:
                for _ in range(self.config.augmentations_per_image):
                    try:
                        aug_img = self.encoder.augment_fn(img.copy())
                        images_to_process.append(aug_img)
                        valid_indices.append(valid_indices[0])
                    except Exception as e:
                        logger.debug(f"Ошибка аугментации: {e}")

        embeddings: list[np.ndarray] = []
        weights: list[float] = []
        valid_paths: list[str] = []

        for i in range(0, len(images_to_process), self.config.batch_size):
            batch = images_to_process[i : i + self.config.batch_size]
            batch_indices = valid_indices[i : i + self.config.batch_size]

            results = self.encoder.encode_batch(batch)

            for j, result in enumerate(results):
                if result is None:
                    continue

                emb, label, confidence = result
                embeddings.append(emb)
                valid_paths.append(image_paths[batch_indices[j]])

                # SigLIP: exterior весит больше interior
                # DINOv2: label == "unknown", confidence == 1.0
                if label == "exterior":
                    weight = confidence * self.config.exterior_weight
                elif label == "interior":
                    weight = confidence * self.config.interior_weight
                else:
                    weight = confidence

                weights.append(weight)

            for img in batch:
                img.close()

        return embeddings, weights, valid_paths

    def _calculate_confidence(self, weights: list[float]) -> dict[str, float]:
        """Рассчитывает метрики уверенности из весов изображений."""
        if not weights:
            return {"confidence": 0.0, "max_conf": 0.0, "mean_conf": 0.0}

        mean_w = float(np.mean(weights))
        max_w = float(max(weights))
        num_images = len(weights)

        confidence = (
            self.config.confidence_max_weight * max_w
            + self.config.confidence_mean_weight
            * mean_w
            * min(
                1.0,
                np.log(1 + num_images) / self.config.confidence_quantity_scale,
            )
        )
        confidence = min(confidence, 1.0)

        return {
            "confidence": confidence,
            "max_conf": max_w,
            "mean_conf": mean_w,
        }

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        image_base_dir: Path | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]], faiss.Index]:
        """
        Строит индекс из pandas DataFrame.

        Ожидаемые колонки: landmark_id, name (или name_en), images.
        """
        embeddings: list[np.ndarray] = []
        metadata: list[dict[str, Any]] = []

        logger.info(f"Кодирование {len(df)} достопримечательностей...")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Кодирование"):
            landmark_id = str(row["landmark_id"])
            name = row.get("name", row.get("name_en", ""))
            image_paths = row.get("images", [])

            if not image_paths:
                logger.warning(f"Нет изображений для {landmark_id}")
                continue

            img_embs, img_weights, valid_paths = self._load_and_process_images(
                image_paths, image_base_dir
            )

            if not img_embs:
                logger.warning(f"Нет валидных изображений для {landmark_id}")
                continue

            fused = self.encoder.fuse_embeddings(img_embs, img_weights)
            conf_metrics = self._calculate_confidence(img_weights)

            meta_entry: dict[str, Any] = {
                "lid": landmark_id,
                "name": name,
                "row_idx": idx,
                "num_images": len(img_embs),
                "valid_images": valid_paths,
                "image_weights": img_weights,
                "embedder_type": self.config.embedder_type,
                **conf_metrics,
            }

            optional_fields = [
                "guide_description",
                "name_ru",
                "name_en",
                "name_de",
                "wikidata_id",
                "wikidata_description_en",
                "wikidata_description_ru",
                "coordinates",
                "country_ru",
                "country_en",
                "city_ru",
                "city_en",
                "landmark_type",
                "wikipedia_url_en",
                "wikipedia_url_ru",
                "wikipedia_summary_en",
                "wikipedia_summary_ru",
                "landmark_summary_caption",
            ]

            for f_name in optional_fields:
                if f_name in row and pd.notna(row[f_name]):
                    meta_entry[f_name] = row[f_name]

            embeddings.append(fused)
            metadata.append(meta_entry)

        if not embeddings:
            raise ValueError("Не удалось сгенерировать ни одного эмбеддинга")

        embeddings_array = np.array(embeddings, dtype=np.float32)
        dim = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings_array)

        logger.info(f"Индекс построен: {index.ntotal} векторов (dim={dim})")

        return embeddings_array, metadata, index

    def build_from_json(
        self,
        json_path: str | Path,
        image_base_dir: Path | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]], faiss.Index]:
        """
        Строит индекс из JSON-файла.

        Ожидаемый формат: список словарей с landmark_id, valid_images и т.д.
        """
        logger.info(f"Загрузка данных из {json_path}")

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        def extract_paths(valid_images):
            if isinstance(valid_images, list):
                return [
                    img.get("path") if isinstance(img, dict) else img
                    for img in valid_images
                ]
            return []

        df["images"] = df["valid_images"].apply(extract_paths)

        if "name_ru" in df.columns:
            df["name"] = df["name_ru"].fillna(df.get("name_en", ""))
        elif "name_en" in df.columns:
            df["name"] = df["name_en"]

        df = df[df["images"].map(len) > 0].reset_index(drop=True)
        logger.info(f"Загружено {len(df)} достопримечательностей с изображениями")

        return self.build_from_dataframe(df, image_base_dir)

    def save(
        self,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
        index: faiss.Index,
    ) -> None:
        """Сохраняет индекс, эмбеддинги и метаданные на диск."""
        faiss.write_index(index, str(self.config.index_path))
        logger.info(f"FAISS индекс сохранён: {self.config.index_path}")

        np.save(self.config.embeddings_path, embeddings)
        logger.info(f"Эмбеддинги сохранены: {self.config.embeddings_path}")

        with open(self.config.metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        logger.info(f"Метаданные сохранены: {self.config.metadata_path}")


# Точка входа


def main():
    """Построение индекса из командной строки."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    data_path = "data/processed/landmarks_with_guide_descriptions.json"
    output_dir = Path("data/models")
    image_dir = Path("images")

    config = IndexConfig(
        model_name="google/siglip-base-patch16-224",
        embedder_type="siglip",
        batch_size=32,
        max_images_per_landmark=10,
        index_path=output_dir / "faiss_index.bin",
        metadata_path=output_dir / "faiss_metadata.pkl",
        embeddings_path=output_dir / "embeddings.npy",
    )

    logger.info("Построение индекса...")
    logger.info(f"Данные: {data_path}")
    logger.info(f"Выходная директория: {output_dir}")

    try:
        builder = IndexBuilder(config)
        image_base = Path(image_dir) if image_dir else None
        embeddings, metadata, index = builder.build_from_json(data_path, image_base)
        builder.save(embeddings, metadata, index)

        logger.info("Индекс успешно построен!")
        logger.info(f"Размер индекса: {index.ntotal} векторов")
        logger.info(f"Размерность: {embeddings.shape[1]}")

    except Exception as e:
        logger.error(f"Ошибка построения индекса: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
