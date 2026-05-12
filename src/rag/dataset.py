# rag/dataset.py
"""
RAG Dataset для обучения с retrieval.
"""

import os
import json
import hashlib
import pickle
from PIL import Image
from typing import Dict, List, Any, Optional
from torch.utils.data import Dataset
from torchvision import transforms
from src.rag.retriever import RAGRetriever


class RAGMultimodalDataset(Dataset):
    """
    Dataset с RAG retrieval для обучения Qwen2-VL.

    Для каждого примера:
    1. Загружает изображение
    2. Применяет аугментации (опционально)
    3. Делает retrieval топ-k похожих landmarks
    4. Формирует промпт с retrieved контекстом
    5. Возвращает текст для обучения

    Processor НЕ вызывается здесь — только в collate_fn на всём батче,
    что необходимо для корректного формирования pixel_values / image_grid_thw
    в Qwen2-VL (packed vision attention).
    """

    def __init__(
        self,
        data_json: str,
        image_dir: str,
        processor,
        retriever: RAGRetriever,
        augment: bool = False,
        max_rag_facts: int = 3,
        prompt_template: Optional[str] = None,
    ):
        """
        Args:
            data_json: Путь к JSON с данными (train.json / val.json)
            image_dir: Папка с изображениями
            processor: Qwen2-VL processor
            retriever: RAGRetriever для поиска
            augment: Применять ли аугментации
            max_rag_facts: Макс. количество retrieved фактов
            prompt_template: Шаблон промпта (опционально)

        Note:
            RAG retrieval выполняется один раз при инициализации датасета
            (в основном процессе) и кэшируется в self._cached_contexts.
            Это необходимо, так как RAGRetriever использует CUDA (CLIP),
            которая не может быть переинициализирована в форкнутых
            DataLoader worker-процессах (RuntimeError: Cannot re-initialize
            CUDA in forked subprocess).
        """
        with open(data_json, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.image_dir = image_dir
        self.processor = processor
        self.retriever = retriever
        self.augment = augment
        self.max_rag_facts = max_rag_facts

        # Аугментации: ColorJitter + RandomHorizontalFlip + RandomRotation.
        # RandomRotation безопасен для достопримечательностей при малых углах (±10°).
        self.augment_fn = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.1
            ),
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(p=0.5),
        ]) if augment else None

        # Промпт по умолчанию.
        # Фигурные скобки JSON экранируются двойными {{ }},
        # чтобы str.format(context=...) не воспринимал их как плейсхолдеры.
        self.prompt_template = prompt_template or (
            "Ты — профессиональный гид. Вот справочная информация:\n\n"
            "{context}\n\n"
            "Опиши достопримечательность на фото в формате JSON:\n"
            '{{"name": "...", "description": "..."}}'
        )

        # Персистентный кэш RAG-контекстов на диске.
        # Ключ кэша: хэш от пути к данным + top_k + модели retriever.
        # При повторном запуске кэш загружается мгновенно (~секунды вместо минут).
        cache_key = hashlib.md5(
            f"{data_json}|{max_rag_facts}|{retriever.model_name}".encode()
        ).hexdigest()[:12]
        cache_dir = os.path.join(os.path.dirname(data_json), ".rag_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"contexts_{cache_key}.pkl")

        if os.path.exists(cache_file):
            print(f"📂 Загрузка RAG-контекстов из кэша: {cache_file}")
            with open(cache_file, "rb") as f:
                self._cached_contexts: List[str] = pickle.load(f)
            print(f"✅ Загружено {len(self._cached_contexts)} контекстов из кэша.")
        else:
            # Вычисляем кэш батчевым CLIP-кодированием в основном процессе
            cache_batch_size = 64
            n = len(self.data)
            print(f"🔍 Кэширование RAG-контекстов для {n} примеров "
                  f"(batch={cache_batch_size})...")
            self._cached_contexts = [""] * n

            for batch_start in range(0, n, cache_batch_size):
                batch_end = min(batch_start + cache_batch_size, n)
                batch_items = self.data[batch_start:batch_end]
                batch_images: List[Image.Image] = []
                for item in batch_items:
                    img_path = os.path.join(
                        self.image_dir, item.get("image_path", "")
                    )
                    try:
                        img = Image.open(img_path).convert("RGB")
                        img.thumbnail((448, 448))
                    except Exception:
                        img = Image.new("RGB", (448, 448), color="black")
                    batch_images.append(img)

                batch_embs = self.retriever.encode_images_batch(batch_images)
                for j, emb in enumerate(batch_embs):
                    results = self.retriever.search_by_embedding(
                        emb, top_k=self.max_rag_facts
                    )
                    self._cached_contexts[batch_start + j] = (
                        self.retriever.format_context(results)
                    )

                if (batch_start // cache_batch_size) % 10 == 0:
                    print(f"  {batch_end}/{n}...")

            with open(cache_file, "wb") as f:
                pickle.dump(self._cached_contexts, f)
            print(f"✅ RAG-контексты закэшированы и сохранены: {cache_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        Возвращает сырые данные (PIL-изображение + текст) без вызова processor.
        Processor вызывается в collate_fn на всём батче — это единственный
        корректный способ получить правильные pixel_values / image_grid_thw
        для batch_size > 1 в Qwen2-VL.

        Передаём отдельно prompt_only (только user-часть без ответа assistant)
        для маскировки промпта в labels — loss считается только на ответе.

        RAG-контекст берётся из кэша (self._cached_contexts), вычисленного
        при инициализации в основном процессе — без CUDA в воркерах.
        """
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item.get("image_path", ""))

        # Загрузка изображения
        try:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((448, 448))
            if self.augment and self.augment_fn:
                image = self.augment_fn(image)
        except Exception as e:
            print(f"⚠️ Error loading {image_path}: {e}")
            image = Image.new("RGB", (448, 448), color="black")

        # RAG-контекст из кэша (без CUDA-вызовов в воркере)
        context_text = self._cached_contexts[idx]

        # Формирование промпта
        prompt_text = self.prompt_template.format(context=context_text)

        # Target ответ
        name = item.get("name_ru", item.get("name_en", ""))
        description = item.get("ground_truth", "").strip()
        target = json.dumps(
            {"name": name, "description": description},
            ensure_ascii=False
        )

        # Chat template для Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ]
            },
            {
                "role": "assistant",
                "content": target
            }
        ]

        # Полный диалог (user + assistant) — для обучения
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # Только user-часть (промпт без ответа) — для маскировки labels
        prompt_only = self.processor.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True
        )

        return {
            "image": image,
            "text": text,
            "prompt_only": prompt_only,
            "true_name": name,
            "true_desc": description,
            "image_path": item.get("image_path", ""),
            "landmark_id": item.get("landmark_id", ""),
        }




def compute_conditional_metrics(results: List[Dict]) -> Dict[str, float]:
    """Метрики при условии, что правильный landmark был в retrieved."""
    correct_retrieval = [r for r in results if r.get("true_in_top_k", False)]
    
    if not correct_retrieval:
        return {
            "name_accuracy_given_retrieval": 0.0,
            "retrieval_success_rate": 0.0,
        }
    
    matches = sum(
        1 for r in correct_retrieval
        if r["pred_name"].lower().strip() == r["true_name"].lower().strip()
    )
    
    return {
        "name_accuracy_given_retrieval": round(matches / len(correct_retrieval), 4),
        "retrieval_success_rate": round(len(correct_retrieval) / len(results), 4),
    }


def compute_gap_analysis(metrics: Dict[str, float]) -> Dict[str, float]:
    """Gap между retrieval и generation."""
    recall_15 = metrics.get("recall15", 0)
    name_acc = metrics.get("name_accuracy", 0)
    
    return {
        "gap_recall15_vs_accuracy": round(recall_15 - name_acc, 4),
        "utilization_rate": round(name_acc / recall_15 if recall_15 > 0 else 0, 4),
    }