# rag/multimodal_reranker.py

from sentence_transformers import CrossEncoder
from PIL import Image
from typing import List, Dict, Union, Optional
import torch
import os


class MultimodalCrossEncoderReranker:
    """
    Reranking с использованием Qwen/Qwen3-VL-Reranker-2B.
    """

    MODEL_CONFIGS = {
        "qwen_base": {
            "path": "Qwen/Qwen3-VL-Reranker-2B",
            "revision": "refs/pr/11",
        },
    
    }

    def __init__(
        self,
        model_name: str = "qwen_base",
        device: Optional[str] = None,
        model_path: Optional[str] = None,
        revision: Optional[str] = None,
        reranker_batch_size: int = 4,
    ):
        if model_path:
            self.model_path = model_path
            self.revision = revision
        elif model_name in self.MODEL_CONFIGS:
            cfg = self.MODEL_CONFIGS[model_name]
            self.model_path = cfg["path"]
            self.revision = cfg.get("revision")
        else:
            self.model_path = model_name
            self.revision = revision

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(
            f"🔧 Загрузка мультимодального Cross-Encoder: {self.model_path}"
        )
        if self.revision:
            print(f"   Revision: {self.revision}")
        print(f"   Устройство: {device}")

        # Загрузка модели через sentence_transformers
        load_kwargs: dict = {"device": device}
        if self.revision:
            load_kwargs["revision"] = self.revision

        self.model = CrossEncoder(self.model_path, **load_kwargs)
        self.reranker_batch_size = reranker_batch_size
        print("✅ Мультимодальный Cross-Encoder загружен")

    def _prepare_document(
        self,
        name: str,
        description: str,
        image_path: Optional[str] = None,
        include_name: bool = True,
    ) -> Union[str, Dict[str, str]]:
        """
        Формирует документ для передачи в CrossEncoder.

        Если доступен image_path, возвращает dict с ключами
        "text" и "image". Иначе возвращает текстовую строку.

        Args:
            name: Название достопримечательности.
            description: Текстовое описание.
            image_path: Путь к изображению кандидата (опционально).
            include_name: Включать ли название в текст документа.

        Returns:
            Строка или dict {"text": ..., "image": ...}.
        """
        if include_name and name:
            text = f"{name}: {description}"
        else:
            text = description
        if image_path and os.path.exists(image_path):
            return {"text": text, "image": image_path}
        return text

    def rerank(
        self,
        image: Image.Image,
        retrieved: List[Dict],
        top_k: int = 10,
        field: str = "description",
        include_name: bool = True,
        image_dir: Optional[str] = None,
        query_prefix: Optional[str] = None,
    ) -> List[Dict]:
        """
        Переранжирует список кандидатов с помощью Cross-Encoder.

        Query формируется как мультимодальная пара: изображение +
        текстовый префикс. Qwen3-VL-Reranker-2B принимает query
        в виде dict {"image": PIL.Image, "text": str}.

        Args:
            image: Query-изображение (фото достопримечательности).
            retrieved: Список кандидатов из первичного retrieval.
                Не модифицируется.
            top_k: Количество результатов после reranking.
            field: Поле словаря кандидата, используемое как описание.
            include_name: Включать ли поле "name" в текст документа.
            image_dir: Базовая директория для изображений кандидатов.
            query_prefix: Текстовый префикс запроса.

        Returns:
            Новый список из top_k кандидатов, отсортированных по
            rerank_score (убывание). Оригинальный retrieved не изменяется.
        """
        if not retrieved:
            return []

        # Query — входное изображение достопримечательности.
        # Qwen3-VL-Reranker-2B принимает PIL.Image как query
        # и ранжирует текстовые описания кандидатов по релевантности.
        query = image

        documents = []
        for r in retrieved:
            image_path = None
            if image_dir and r.get("image_path"):
                candidate_path = os.path.join(image_dir, r["image_path"])
                if os.path.exists(candidate_path):
                    image_path = candidate_path
            elif r.get("image_path") and os.path.exists(r["image_path"]):
                image_path = r["image_path"]

            doc = self._prepare_document(
                name=r.get("name", ""),
                description=r.get(field, ""),
                image_path=image_path,
                include_name=include_name,
            )
            documents.append(doc)

        rankings = self.model.rank(
            query,
            documents,
            return_documents=False,
            batch_size=self.reranker_batch_size,
        )
        score_map = {rank["corpus_id"]: rank["score"] for rank in rankings}

        # Создаём копии словарей, не мутируя оригинальный список
        scored = [
            {**r, "rerank_score": float(score_map.get(i, float("-inf")))}
            for i, r in enumerate(retrieved)
        ]
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]
