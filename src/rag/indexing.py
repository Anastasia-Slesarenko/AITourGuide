"""
Построение CLIP + FAISS индекса для RAG-системы гида по достопримечательностям.

Кодирует изображения достопримечательностей через CLIP, усредняет эмбеддинги
по каждому landmark_id и строит FAISS индекс для быстрого поиска.
"""

import os
import pickle
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
from collections import defaultdict

import numpy as np
import torch
import faiss
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms


class CLIPIndexBuilder:
    """
    Построитель CLIP + FAISS индекса для image-based retrieval.
    """
    
    EXPECTED_DIMS = {
        "openai/clip-vit-base-patch32": 512,
        "openai/clip-vit-large-patch14": 768,
        "openai/clip-vit-base-patch16": 512,
    }
    
    def __init__(
        self, 
        model_name: str = "openai/clip-vit-large-patch14",
        seed: int = 42
    ):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        print(f"🤖 Загрузка CLIP модели: {model_name}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        self.embedding_dim = self.EXPECTED_DIMS.get(model_name, 512)
        print(f"✅ CLIP загружен: device={self.device}, dim={self.embedding_dim}")
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Кодирует одно изображение в нормализованный эмбеддинг"""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().astype("float32")
    
    def encode_images_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Кодирует список изображений одним батчем"""
        if not images:
            return np.zeros((0, self.embedding_dim), dtype="float32")
        
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().astype("float32")
    
    def build_index(
        self,
        facts_db: Dict[str, Dict[str, Any]],
        image_dir: str,
        max_images_per_landmark: int = 5,  # 🔴 Увеличено с 5 до 10
        use_batch_encoding: bool = True,
    ) -> Tuple[faiss.Index, List[str]]:
        """Строит FAISS индекс с аугментациями для редких landmarks"""
        print(f"\n🔍 Построение CLIP индекса...")
        print(f"   • Landmarks: {len(facts_db)}")
        print(f"   • Image dir: {image_dir}")
        print(f"   • Max images per landmark: {max_images_per_landmark}")
        
        embeddings: List[np.ndarray] = []
        lid_list: List[str] = []
        failed_images = 0
        processed_landmarks = 0
        total_images = 0
        total_augmented = 0
        
        # Сортируем для детерминированного порядка
        sorted_items = sorted(facts_db.items())
        
        # 🔴 Аугментации вынесены наружу (один раз на весь скрипт)
        from torchvision import transforms
        augment_fn = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=5, translate=(0.02, 0.02)),
        ])
        
        for lid, data in tqdm(sorted_items, desc="Encoding landmarks", unit="landmark"):
            img_paths = data.get("image_paths", [])[:max_images_per_landmark]
            total_images += len(img_paths)
            
            valid_images: List[Image.Image] = []
            
            # 🔴 Шаг 1: Загрузка всех доступных фото
            for img_path in img_paths:
                try:
                    full_path = os.path.join(image_dir, img_path)
                    
                    if not os.path.exists(full_path):
                        failed_images += 1
                        continue
                    
                    image = Image.open(full_path).convert("RGB")
                    image.thumbnail((448, 448))
                    valid_images.append(image)
                    
                except Exception as e:
                    failed_images += 1
                    continue
            
            # 🔴 Шаг 2: Аугментации для landmarks с <3 фото
            original_count = len(valid_images)

            if original_count < 3 and original_count > 0:    
                for img in valid_images[:original_count]:
                    for _ in range(3):
                        try:
                            aug_img = augment_fn(img.copy())
                            valid_images.append(aug_img)
                            total_augmented += 1
                        except:
                            continue

            if not valid_images:
                print(f"⚠️  Landmark {lid}: все {len(img_paths)} фото не загружены. Пропускаем.")
                continue
            
            # 🔴 Шаг 3: Кодирование
            if use_batch_encoding and len(valid_images) > 1:
                img_embs = self.encode_images_batch(valid_images)
            else:
                img_embs = np.vstack([self.encode_image(img) for img in valid_images])
            
            # 🔴 Шаг 4: Усреднение + нормализация
            avg_emb = np.mean(img_embs, axis=0, keepdims=True)
            avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)
            
            embeddings.append(avg_emb)
            lid_list.append(lid)
            processed_landmarks += 1
            
            # Логирование для первых 5 landmarks
            if processed_landmarks <= 5 and total_augmented > 0:
                print(f"   Landmark {lid}: {original_count} фото → {len(valid_images)} с аугментациями")
        
        if not embeddings:
            raise ValueError("❌ Не удалось закодировать ни одного изображения!")
        
        # Стек всех эмбеддингов
        embeddings_array = np.vstack(embeddings).astype("float32")
        
        # Валидация размерности
        if embeddings_array.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Несоответствие размерности: ожидалось {self.embedding_dim}, "
                f"получено {embeddings_array.shape[1]}"
            )
        
        # Статистика по нормализации
        norms = np.linalg.norm(embeddings_array, axis=1)
        print(f"\n📊 Embedding stats: mean_norm={np.mean(norms):.4f}, "
              f"std={np.std(norms):.4f}, min={np.min(norms):.4f}, max={np.max(norms):.4f}")
        
        if abs(np.mean(norms) - 1.0) > 0.01:
            print("⚠️  Эмбеддинги могут быть некорректно нормализованы!")
        
        # Построение FAISS индекса
        print(f"\n📊 Создание FAISS индекса...")
        print(f"   • Dimension: {embeddings_array.shape[1]}")
        print(f"   • Landmarks: {len(lid_list)}")
        
        index = faiss.IndexFlatIP(embeddings_array.shape[1])
        index.add(embeddings_array)
        
        # Итоговая статистика
        print(f"\n✅ Индекс построен:")
        print(f"   • Landmarks в индексе: {processed_landmarks}/{len(facts_db)}")
        print(f"   • Обработано фото: {total_images - failed_images}/{total_images}")
        print(f"   • Пропущено фото: {failed_images}")
        print(f"   • Добавлено аугментаций: {total_augmented}")
        print(f"   • Размер индекса: {index.ntotal} векторов × {embeddings_array.shape[1]} dim")
        
        return index, lid_list
    
    def save(self, index: faiss.Index, lid_list: List[str], output_path: str) -> None:
        """Сохраняет индекс и метаданные"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(index, output_path + ".faiss")
        
        with open(output_path + ".meta.pkl", "wb") as f:
            pickle.dump({"lid_list": lid_list, "model_name": self.model_name}, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"\n💾 Индекс сохранён:")
        print(f"   • {output_path}.faiss")
        print(f"   • {output_path}.meta.pkl")
    
    @classmethod
    def load_index(cls, index_path: str, model_name: str = None) -> Tuple[faiss.Index, List[str], str]:
        """Загружает сохранённый индекс"""
        print(f"📂 Загрузка индекса из {index_path}...")
        
        index = faiss.read_index(f"{index_path}.faiss")
        
        with open(f"{index_path}.meta.pkl", "rb") as f:
            meta = pickle.load(f)
        
        loaded_model = meta.get("model_name", model_name or "openai/clip-vit-large-patch14")
        print(f"✅ Индекс загружен: {index.ntotal} векторов, модель={loaded_model}")
        
        return index, meta["lid_list"], loaded_model
    
def build_facts_db_from_json(train_json_path: str, output_path: str) -> Dict[str, Dict]:
    """
    Строит базу фактов из train.json.
    
    Args:
        train_json_path: Путь к train.json
        output_path: Путь для сохранения facts_db.pkl
        
    Returns:
        Dict {landmark_id: {name_ru, ground_truth, image_paths}}
    """
    print(f"\n📦 Построение базы фактов из {train_json_path}...")
    
    if not os.path.exists(train_json_path):
        raise FileNotFoundError(f"train.json не найден: {train_json_path}")
    
    with open(train_json_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # Группировка по landmark_id
    facts_db = defaultdict(lambda: {"image_paths": [], "name_ru": "", "ground_truth": "", "name_en": ""})
    
    name_conflicts = 0
    desc_conflicts = 0
    
    for item in train_data:
        lid = str(item.get("landmark_id", ""))
        if not lid:
            continue
        
        # Добавляем путь к изображению
        img_path = item.get("image_path", "")
        if img_path and img_path not in facts_db[lid]["image_paths"]:
            facts_db[lid]["image_paths"].append(img_path)
        
        # Берём первое непустое название
        name_ru = item.get("name_ru", "").strip()
        if name_ru and not facts_db[lid]["name_ru"]:
            facts_db[lid]["name_ru"] = name_ru
        elif name_ru and facts_db[lid]["name_ru"] != name_ru:
            name_conflicts += 1
        
        name_en = item.get("name_en", "").strip()
        if name_en and not facts_db[lid]["name_en"]:
            facts_db[lid]["name_en"] = name_en
        
        # Берём первое непустое описание
        description = item.get("ground_truth", "").strip()
        if description and not facts_db[lid]["ground_truth"]:
            facts_db[lid]["ground_truth"] = description
        elif description and facts_db[lid]["ground_truth"] != description:
            desc_conflicts += 1
    
    # Конвертируем в обычный dict
    facts_db = dict(facts_db)
    
    # Статистика
    total_images = sum(len(v["image_paths"]) for v in facts_db.values())
    print(f"✅ База фактов построена:")
    print(f"   • Landmarks: {len(facts_db)}")
    print(f"   • Всего фото: {total_images}")
    print(f"   • Среднее фото на landmark: {total_images/len(facts_db):.1f}")
    if name_conflicts > 0:
        print(f"⚠️  Конфликтов названий: {name_conflicts}")
    if desc_conflicts > 0:
        print(f"⚠️  Конфликтов описаний: {desc_conflicts}")
    
    # Сохранение
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(facts_db, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"💾 Сохранено: {output_path}")
    
    return facts_db


def main():
    # ========================================
    # НАСТРОЙКИ (меняйте здесь)
    # ========================================
    FACTS_DB_PATH = "/home/jupyter/s3/ai-tour-guide/facts_db.pkl"
    TRAIN_JSON_PATH = "/home/jupyter/s3/ai-tour-guide/train_russian_landmarks.json"  # ← для авто-построения
    IMAGE_DIR = "/home/jupyter/s3/ai-tour-guide/landmarks"
    OUTPUT_PATH = "/home/jupyter/s3/ai-tour-guide/landmarks/clip_index"
    CLIP_MODEL = "openai/clip-vit-large-patch14"
    MAX_IMAGES_PER_LANDMARK = 5
    USE_BATCH_ENCODING = True
    SEED = 42
    # ========================================
    
    print("="*70)
    print("🚀 Построение CLIP + FAISS индекса для RAG-гида")
    print("="*70)
    print(f"📂 Facts DB: {FACTS_DB_PATH}")
    print(f"📄 Train JSON: {TRAIN_JSON_PATH}")
    print(f"🖼️  Image Dir: {IMAGE_DIR}")
    print(f"💾 Output: {OUTPUT_PATH}")
    print(f"🤖 CLIP Model: {CLIP_MODEL}")
    print(f"📸 Max images per landmark: {MAX_IMAGES_PER_LANDMARK}")
    print(f"🌱 Seed: {SEED}")
    print("="*70)
    
    # Проверка существования файлов
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ Ошибка: папка с изображениями не найдена: {IMAGE_DIR}")
        sys.exit(1)
    
    # ========================================
    # ШАГ 1: Загрузка или построение facts_db
    # ========================================
    
    if os.path.exists(FACTS_DB_PATH):
        print(f"\n📂 Загрузка базы фактов из {FACTS_DB_PATH}...")
        with open(FACTS_DB_PATH, "rb") as f:
            facts_db = pickle.load(f)
        print(f"✅ Загружено {len(facts_db)} landmarks")
    else:
        print(f"\n⚠️  Facts DB не найден: {FACTS_DB_PATH}")
        print(f"🔨 Построение базы фактов из {TRAIN_JSON_PATH}...")
        
        if not os.path.exists(TRAIN_JSON_PATH):
            print(f"❌ Ошибка: train.json не найден: {TRAIN_JSON_PATH}")
            print(f"   Сначала создайте {FACTS_DB_PATH} или положите {TRAIN_JSON_PATH}")
            sys.exit(1)
        
        facts_db = build_facts_db_from_json(TRAIN_JSON_PATH, FACTS_DB_PATH)
    
    # ========================================
    # ШАГ 2: Построение CLIP индекса
    # ========================================
    
    print("\n🔍 Построение CLIP индекса...")
    builder = CLIPIndexBuilder(model_name=CLIP_MODEL, seed=SEED)
    
    index, lid_list = builder.build_index(
        facts_db=facts_db,
        image_dir=IMAGE_DIR,
        max_images_per_landmark=MAX_IMAGES_PER_LANDMARK,
        use_batch_encoding=USE_BATCH_ENCODING,
    )
    
    # ========================================
    # ШАГ 3: Сохранение индекса
    # ========================================
    
    builder.save(index, lid_list, OUTPUT_PATH)
    
    # ========================================
    # ШАГ 4: Быстрый тест поиска
    # ========================================
    
    print(f"\n🧪 Быстрый тест поиска...")
    if len(lid_list) > 0:
        test_query = np.random.randn(1, builder.embedding_dim).astype("float32")
        test_query = test_query / np.linalg.norm(test_query)
        
        scores, indices = index.search(test_query, k=min(3, len(lid_list)))
        print(f"✅ Тест поиска: top-3 scores={scores[0].tolist()}")
    
    # ========================================
    # ФИНАЛ
    # ========================================
    
    print("\n" + "="*70)
    print("🎉 Построение индекса завершено успешно!")
    print("="*70)
    print(f"\n📁 Файлы:")
    print(f"   • {FACTS_DB_PATH}")
    print(f"   • {OUTPUT_PATH}.faiss")
    print(f"   • {OUTPUT_PATH}.meta.pkl")
    print(f"\n✅ Индекс готов к использованию в RAG-пайплайне!")




if __name__ == "__main__":
    main()