# rag/retriever.py
"""
RAG Retriever для загрузки и поиска по CLIP + FAISS индексу.
С поддержкой Qwen/Qwen3-VL-Reranker-2B для мультимодального reranking.
"""

import pickle
import numpy as np
import faiss
from PIL import Image
from typing import List, Dict, Optional
from transformers import CLIPProcessor, CLIPModel
import torch
from src.rag.multimodal_reranker import MultimodalCrossEncoderReranker


class RAGRetriever:
    """
    Загружает CLIP + FAISS индекс и выполняет поиск по изображениям.
    """

    def __init__(
        self,
        index_path: str,
        facts_db_path: str,
        clip_model: str = "openai/clip-vit-large-patch14",
        top_k: int = 15,
        use_multimodal_reranker: bool = False,
        image_dir: Optional[str] = None,
        reranker_config: Optional[dict] = None,
    ):
        """
        Args:
            index_path: Путь к clip_index (без расширения)
            facts_db_path: Путь к facts_db.pkl
            clip_model: Модель CLIP для кодирования запросов
            top_k: Количество результатов для поиска
            use_multimodal_reranker: Включить ли мультимодальный reranking
            image_dir: Базовая директория для изображений
            reranker_config: Конфигурация для reranker
        """
        self.top_k = top_k
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_dir = image_dir

        # Загрузка FAISS индекса
        print(f"📂 Загрузка FAISS индекса из {index_path}...")
        self.index = faiss.read_index(f"{index_path}.faiss")

        with open(f"{index_path}.meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self.lid_list = meta["lid_list"]

        # Используем модель из метаданных индекса
        self.model_name = meta.get("model_name", clip_model)
        print(f"   Модель из индекса: {self.model_name}")
        print(f"   Размерность индекса: {self.index.d}")

        # Загрузка базы фактов
        print(f"📂 Загрузка базы фактов из {facts_db_path}...")
        with open(facts_db_path, "rb") as f:
            self.facts_db = pickle.load(f)

        # Загрузка CLIP для кодирования запросов
        print(f"🤖 Загрузка CLIP модели: {self.model_name}...")
        self.clip_model = CLIPModel.from_pretrained(
            self.model_name
        ).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(self.model_name)

        # Проверка размерности
        test_emb = self.encode_image(Image.new("RGB", (448, 448), color="red"))
        print(f"   Размерность эмбеддинга: {test_emb.shape[1]}")
        if test_emb.shape[1] != self.index.d:
            raise ValueError(
                f"Несовпадение размерностей: "
                f"эмбеддинг={test_emb.shape[1]}, индекс={self.index.d}"
            )

        # Инициализация мультимодального reranker (Qwen)
        self.use_multimodal_reranker = use_multimodal_reranker
        if use_multimodal_reranker:
            cfg = reranker_config or {}
            reranker_name = cfg.get("model_name", "qwen_base")
            self.reranker: Optional[MultimodalCrossEncoderReranker] = (
                MultimodalCrossEncoderReranker(
                    model_name=reranker_name,
                    device=cfg.get("device", self.device),
                    model_path=cfg.get("model_path"),
                    revision=cfg.get("revision"),
                    reranker_batch_size=cfg.get("reranker_batch_size", 4),
                )
            )
            print(
                f"✅ Мультимодальный reranker инициализирован: {reranker_name}"
            )
        else:
            self.reranker = None

        print(
            f"✅ RAGRetriever готов: {len(self.lid_list)} landmarks, "
            f"top_k={top_k}"
        )

    # ------------------------------------------------------------------
    # Приватные вспомогательные методы
    # ------------------------------------------------------------------

    def _encode_inputs(self, inputs) -> "torch.Tensor":
        """
        Кодирует входные тензоры через CLIP и возвращает нормализованный
        проекционный вектор (dim=768 для clip-vit-large-patch14).

        Использует get_image_features() — в transformers 4.57.1 возвращает
        прямой тензор 768 (проекция через visual_projection).
        """
        feats = self.clip_model.get_image_features(**inputs)
        return feats / feats.norm(dim=-1, keepdim=True)

    def _build_results(
        self,
        scores: np.ndarray,
        indices: np.ndarray,
    ) -> List[Dict]:
        """
        Формирует список результатов из выхода FAISS.

        Args:
            scores: Массив scores формы (1, k).
            indices: Массив индексов формы (1, k).

        Returns:
            Список словарей с полями landmark_id, name, description,
            score, image_path.
        """
        results = []
        for j, i in enumerate(indices[0]):
            if i < 0 or i >= len(self.lid_list):
                continue
            lid = self.lid_list[i]
            fact = self.facts_db.get(lid, {})
            results.append({
                "landmark_id": str(lid),
                "name": fact.get(
                    "name_ru", fact.get("name_en", "Неизвестно")
                ),
                "description": fact.get("ground_truth", ""),
                "score": float(scores[0][j]),
                "image_path": fact.get("image_path", ""),
            })
        return results

    # ------------------------------------------------------------------
    # Публичные методы
    # ------------------------------------------------------------------

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Кодирует изображение через CLIP.
        Возвращает нормализованный эмбеддинг формы (1, D).
        """
        inputs = self.clip_processor(
            images=image, return_tensors="pt"
        ).to(self.clip_model.device)

        with torch.no_grad():
            feats = self._encode_inputs(inputs)

        return feats.cpu().numpy().astype("float32")

    def encode_images_batch(
        self, images: List[Image.Image]
    ) -> List[np.ndarray]:
        """
        Батчевое кодирование изображений.
        Возвращает список нормализованных эмбеддингов формы (1, D).
        """
        inputs = self.clip_processor(
            images=images, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            feats = self._encode_inputs(inputs)

        arr = feats.cpu().numpy().astype("float32")
        return [arr[i: i + 1] for i in range(len(arr))]

    def search_by_embedding(
        self, query_emb: np.ndarray, top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Поиск по готовому эмбеддингу формы (1, D).

        Эмбеддинг должен быть нормализован (L2 = 1) для корректных
        cosine-scores при использовании IndexFlatIP.
        """
        k = top_k or self.top_k
        k = min(k, len(self.lid_list))

        # Нормализуем на случай, если вызывающий код не сделал этого
        norm = np.linalg.norm(query_emb, axis=-1, keepdims=True)
        if norm.any():
            query_emb = query_emb / norm

        scores, indices = self.index.search(query_emb, k)
        return self._build_results(scores, indices)

    def search(
        self, image: Image.Image, top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Поиск похожих landmarks по изображению.
        Изображение не модифицируется.
        """
        k = top_k or self.top_k
        k = min(k, len(self.lid_list))

        # Копируем, чтобы не мутировать оригинал
        img = image.copy()
        img.thumbnail((448, 448))

        inputs = self.clip_processor(
            images=img, return_tensors="pt"
        ).to(self.clip_model.device)

        with torch.no_grad():
            query_emb = self._encode_inputs(inputs)

        query_emb = query_emb.cpu().numpy().astype("float32")

        # Убедимся, что shape = (1, D)
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        elif query_emb.shape[0] != 1:
            query_emb = query_emb[0:1].reshape(1, -1)

        scores, indices = self.index.search(query_emb, k)
        return self._build_results(scores, indices)

    def search_with_multimodal_reranking(
        self,
        image: Image.Image,
        top_k: int = 15,
        initial_k: int = 30,
        query_text: Optional[str] = None,
    ) -> List[Dict]:
        """
        Поиск с мультимодальным reranking (Qwen VL).
        """
        # 1. Первоначальный retrieval с запасом
        retrieved = self.search(image, top_k=initial_k)

        # 2. Мультимодальный reranking
        if self.reranker:
            return self.reranker.rerank(
                image=image,
                retrieved=retrieved,
                top_k=top_k,
                field="description",
                include_name=True,
                image_dir=self.image_dir,
                query_prefix=query_text,
            )
        return retrieved[:top_k]

    def get_facts(self, landmark_id: str) -> Dict:
        """Получает факты по landmark_id."""
        return self.facts_db.get(landmark_id, {})

    def format_context(
        self, results: List[Dict], max_desc_len: int = 200
    ) -> str:
        """
        Форматирует результаты поиска в текст для промпта.
        """
        lines = []
        for i, r in enumerate(results, 1):
            desc = r["description"]
            if len(desc) > max_desc_len:
                desc = desc[:max_desc_len] + "..."
            lines.append(f"{i}. {r['name']}: {desc}")
        return "\n".join(lines)
