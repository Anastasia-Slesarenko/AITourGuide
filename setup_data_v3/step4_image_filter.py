#!/usr/bin/env python3
"""
Шаг 4: Фильтрация изображений достопримечательностей.
Использует локальную Qwen2-VL модель для проверки соответствия изображений описаниям.

Требования:
    - NVIDIA GPU с минимум 8GB VRAM (рекомендуется T4 24GB)
    - Результат работы step3_text_filter.py

Установка зависимостей:
    pip install transformers torch qwen-vl-utils accelerate pillow tqdm

Выходной формат:
    - Сохраняет все поля из text_filtered_landmarks.json
    - Заменяет поле 'image_path' на 'valid_images' (список валидных изображений с описаниями)
    - Добавляет поле 'not_valid_images' (список невалидных изображений)
    - Каждое изображение в 'valid_images' содержит:
        * path: путь к изображению
        * caption: автоматически сгенерированное описание изображения
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from tqdm import tqdm
from dataclasses import dataclass
import torch
from PIL import Image
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    CLIPProcessor,
    CLIPModel
)
from qwen_vl_utils import process_vision_info

# ======================
# ЛОГИРОВАНИЕ
# ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ======================
# КОНФИГУРАЦИЯ
# ======================
@dataclass
class Config:
    """Конфигурация для фильтрации изображений."""
    # Пути к файлам
    input_path: str = "setup_data_v3/data/text_filtered_landmarks.json"
    output_path: str = "setup_data_v3/data/clean_landmarks.json"
    quality_log_path: str = "setup_data_v3/data/cache/quality_log.jsonl"
    
    # Директория с изображениями
    images_dir: str = "setup_data_v3/data/images"
    
    # Параметры фильтрации
    min_images_per_landmark: int = 1
    
    # Кэш и прогресс
    image_cache_path: str = "setup_data_v3/data/cache/image_checks.jsonl"
    progress_path: str = "setup_data_v3/data/cache/image_processed_landmarks.txt"
    
    # CLIP pre-filter
    clip_model_name: str = "openai/clip-vit-base-patch32"
    use_clip_prefilter: bool = True
    clip_threshold: float = 0.22  # Минимальное сходство для прохождения CLIP фильтра
    
    # Модель
    vlm_model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    max_pixels: int = 512 * 512
    
    # Интервал сброса кэша
    cache_flush_interval: int = 50
    
    def __post_init__(self):
        # Проверка существования входного файла
        if not Path(self.input_path).exists():
            raise FileNotFoundError(
                f"Входной файл не найден: {self.input_path}\n"
                f"Сначала запустите step1_text_filter.py"
            )
        
        # Создание директорий
        for path in [self.image_cache_path, self.progress_path]:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)


# ======================
# МЕТРИКИ
# ======================
class Metrics:
    """Отслеживание метрик обработки."""
    def __init__(self):
        self.image_checks = 0
        self.image_cache_hits = 0
        self.objects_processed = 0
        self.objects_passed = 0
        self.total_images_checked = 0
        self.total_images_valid = 0
        self.clip_filtered = 0
        self.confidence_high = 0  # ДА
        self.confidence_medium = 0  # СКОРЕЕ ДА
        self.confidence_low = 0  # НЕТ
        self.rejection_reasons = {
            "clip_filter": 0,
            "interior": 0,
            "not_building": 0,
            "low_quality": 0,
            "other": 0
        }
        self.start_time = time.time()
    
    def report(self):
        """Генерация отчета по метрикам."""
        elapsed = time.time() - self.start_time
        total = self.image_cache_hits + self.image_checks
        
        return {
            "elapsed_time_sec": round(elapsed, 2),
            "objects_processed": self.objects_processed,
            "objects_passed": self.objects_passed,
            "image_checks": self.image_checks,
            "image_cache_hits": self.image_cache_hits,
            "cache_hit_rate": round(
                self.image_cache_hits / max(1, total) * 100, 2
            ),
            "total_images_checked": self.total_images_checked,
            "total_images_valid": self.total_images_valid,
            "clip_filtered": self.clip_filtered,
            "confidence_distribution": {
                "high": self.confidence_high,
                "medium": self.confidence_medium,
                "low": self.confidence_low
            },
            "rejection_reasons": self.rejection_reasons,
            "images_per_sec": round(
                self.total_images_checked / max(1, elapsed), 2
            ),
        }


# ======================
# КЭШИРОВАНИЕ
# ======================
class ImageCache:
    """Кэш результатов проверки изображений."""
    def __init__(self, path: Path, flush_interval: int = 50):
        self.path = path
        self.flush_interval = flush_interval
        self.data: Dict[str, bool] = {}
        self.pending_writes: List[Dict] = []
        
        if self.path.exists():
            with open(self.path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self.data[rec['image_path']] = rec['is_valid']
        logger.info(
            f"Загружено {len(self.data)} записей из кэша изображений"
        )
    
    def get(self, image_path: str) -> Optional[bool]:
        return self.data.get(image_path)
    
    def set(self, image_path: str, is_valid: bool):
        self.data[image_path] = is_valid
        self.pending_writes.append({
            'image_path': image_path,
            'is_valid': is_valid
        })
        
        if len(self.pending_writes) >= self.flush_interval:
            self.flush()
    
    def flush(self):
        """Принудительная запись на диск."""
        if self.pending_writes:
            with open(self.path, 'a', encoding='utf-8') as f:
                for record in self.pending_writes:
                    f.write(
                        json.dumps(record, ensure_ascii=False) + '\n'
                    )
            self.pending_writes.clear()


class ProgressTracker:
    """Отслеживание обработанных объектов."""
    def __init__(self, path: Path):
        self.path = path
        self.processed: Set[str] = set()
        self.pending_writes: List[str] = []
        
        if self.path.exists():
            with open(self.path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.processed.add(line)
    
    def mark_processed(self, obj_id: str):
        if obj_id not in self.processed:
            self.processed.add(obj_id)
            self.pending_writes.append(obj_id)
    
    def is_processed(self, obj_id: str) -> bool:
        return obj_id in self.processed
    
    def flush(self):
        """Принудительная запись на диск."""
        if self.pending_writes:
            with open(self.path, 'a', encoding='utf-8') as f:
                for obj_id in self.pending_writes:
                    f.write(obj_id + '\n')
            self.pending_writes.clear()


# ======================
# CLIP PRE-FILTER
# ======================
class CLIPPreFilter:
    """CLIP-based pre-filter для быстрой фильтрации изображений."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Загрузка CLIP модели: {config.clip_model_name}")
        self.model = CLIPModel.from_pretrained(config.clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(config.clip_model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Промпты для классификации
        self.prompts = [
            "a photo of a landmark building exterior",
            "a photo of architecture monument",
            "a photo of interior room",
            "a photo of a person",
            "a photo of food",
            "a photo of a map or text document"
        ]
        
        # Предварительно закодируем промпты
        with torch.no_grad():
            text_inputs = self.processor(
                text=self.prompts,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            text_outputs = self.model.get_text_features(**text_inputs)
            self.text_features = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
        
        logger.info(f"CLIP модель загружена на: {self.device}")
    
    def check_image(self, image_path: str) -> Tuple[bool, str, float]:
        """
        Быстрая проверка изображения с помощью CLIP.
        
        Returns:
            (is_valid, reason, similarity)
        """
        try:
            image = Image.open(image_path).convert("RGB")
            
            with torch.no_grad():
                image_inputs = self.processor(
                    images=image,
                    return_tensors="pt"
                ).to(self.device)
                image_outputs = self.model.get_image_features(**image_inputs)
                image_features = image_outputs / image_outputs.norm(dim=-1, keepdim=True)
                
                # Вычисляем сходство с каждым промптом
                similarities = (image_features @ self.text_features.T).squeeze(0)
                probs = similarities.softmax(dim=0).cpu().numpy()
            
            # Анализ результатов
            exterior_prob = float(probs[0:2].sum())  # landmark + architecture
            interior_prob = float(probs[2])
            person_prob = float(probs[3])
            food_prob = float(probs[4])
            map_text_prob = float(probs[5])
            
            # Определяем причину отклонения
            if interior_prob > 0.4:
                return False, "interior", interior_prob
            elif person_prob > 0.3:
                return False, "person", person_prob
            elif food_prob > 0.3:
                return False, "food", food_prob
            elif map_text_prob > 0.3:
                return False, "map_or_text", map_text_prob
            elif exterior_prob < self.config.clip_threshold:
                return False, "not_building", exterior_prob
            
            return True, "passed", exterior_prob
            
        except Exception as e:
            logger.error(f"CLIP ошибка для {image_path}: {e}")
            return False, "error", 0.0
        finally:
            if 'image' in locals():
                image.close()


# ======================
# ФИЛЬТР ИЗОБРАЖЕНИЙ
# ======================
class Qwen2VLImageFilter:
    """Проверка изображений с использованием Qwen2-VL."""
    def __init__(self, config: Config, metrics: Metrics):
        self.config = config
        self.metrics = metrics
        self.cache = ImageCache(
            Path(config.image_cache_path),
            config.cache_flush_interval
        )
        
        logger.info(f"Загрузка Qwen2-VL: {config.vlm_model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.device == "cpu":
            logger.warning(
                "GPU не обнаружен! Обработка на CPU будет очень медленной."
            )
        
        # Загрузка модели
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.vlm_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(
            config.vlm_model_name,
            max_pixels=config.max_pixels
        )
        
        logger.info(f"Модель загружена на: {self.device}")
    
    def check_image(
        self,
        image_path: str,
        expected_name: str,
        expected_description: str
    ) -> Tuple[bool, str]:
        """
        Проверка соответствия изображения описанию.
        
        Returns:
            (is_valid, confidence_level)
            confidence_level: "high" (ДА), "medium" (СКОРЕЕ ДА), "low" (НЕТ)
        """
        cached = self.cache.get(image_path)
        if cached is not None:
            self.metrics.image_cache_hits += 1
            # Кэш возвращает только bool, считаем как high confidence
            return cached, "high" if cached else "low"
        
        try:
            # Формирование промпта
            prompt = f"""Перед тобой изображение и описание достопримечательности.

Название: {expected_name}
Описание: {expected_description}

Задача:
Определи, соответствует ли изображение описанию.

Критерии оценки:
✓ Подходит если:
  - это архитектурный объект (здание, мост, башня и т.д.)
  - визуально похоже на описание (форма, стиль, элементы)
  - внешний вид здания/сооружения

✗ НЕ подходит если:
  - это интерьер
  - это человек / карта / текст / еда
  - это вообще не архитектура

Ответь ОДНИМ из вариантов:
- ДА (уверен, что это именно эта достопримечательность)
- СКОРЕЕ ДА (похоже, но есть сомнения)
- НЕТ (не подходит)
"""
            
            # Подготовка сообщений
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # Применение шаблона чата
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Обработка
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # Генерация
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.1,
                    do_sample=False
                )
            
            # Декодирование
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in
                zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip().lower()
            
            # Определение confidence level
            if "скорее да" in output_text or "вероятно" in output_text:
                is_valid = True
                confidence = "medium"
                self.metrics.confidence_medium += 1
            elif output_text.startswith('да') or "определенно" in output_text:
                is_valid = True
                confidence = "high"
                self.metrics.confidence_high += 1
            else:
                is_valid = False
                confidence = "low"
                self.metrics.confidence_low += 1
            
            self.metrics.image_checks += 1
            self.cache.set(image_path, is_valid)
            
            return is_valid, confidence
            
        except Exception as e:
            logger.error(f"Ошибка проверки {image_path}: {e}")
            self.cache.set(image_path, False)
            self.metrics.confidence_low += 1
            return False, "low"
    
    def generate_caption(
        self,
        image_path: str,
        expected_name: str
    ) -> str:
        """Генерация описания для изображения."""
        try:
            # Формирование промпта для генерации описания
            prompt = """Опиши изображение одной фразой.

Сосредоточься только на визуальных деталях:
- форма
- архитектура
- материалы
- особенности

Не используй название объекта.

Формат:
[тип объекта], [форма], [ключевая особенность]
"""
            
            # Подготовка сообщений
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # Применение шаблона чата
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Обработка
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # Генерация
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True
                )
            
            # Декодирование
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in
                zip(inputs.input_ids, generated_ids)
            ]
            caption = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            return caption if caption else f"Изображение {expected_name}"
            
        except Exception as e:
            logger.error(f"Ошибка генерации описания для {image_path}: {e}")
            return f"Изображение {expected_name}"


# ======================
# ОСНОВНОЙ КЛАСС
# ======================
class ImageFilterProcessor:
    """Процессор фильтрации изображений."""
    def __init__(self, config: Config):
        self.config = config
        self.progress = ProgressTracker(Path(config.progress_path))
        self.metrics = Metrics()
    
    def _generate_obj_id(self, item: Dict) -> str:
        """Генерация консистентного ID объекта."""
        if "id" in item and item["id"]:
            return str(item["id"])
        
        name = item.get("name_ru") or item.get("name_en") or ""
        desc = (
            item.get("wikipedia_summary_ru") or
            item.get("wikipedia_summary_en") or
            ""
        )
        
        return f"{name}::{desc[:50]}"
    
    def process_one(
        self,
        item: Dict,
        image_filter: Qwen2VLImageFilter,
        clip_filter: Optional[CLIPPreFilter] = None
    ) -> Optional[Dict]:
        """Обработка одного объекта."""
        obj_id = self._generate_obj_id(item)
        
        if self.progress.is_processed(obj_id):
            return None
        
        name = item.get("name_ru") or item.get("name_en")
        description = (
            item.get("wikipedia_summary_ru") or
            item.get("wikipedia_summary_en")
        )
        
        if not name or not description:
            self.progress.mark_processed(obj_id)
            return None
        
        # Фильтрация изображений
        image_names = item.get("image_path", [])
        if not image_names:
            logger.info(f"Нет изображений для: {name}")
            self.progress.mark_processed(obj_id)
            return None
        
        valid_images = []
        not_valid_images = []
        
        for img_name in image_names:
            # Формируем полный путь к изображению
            full_img_path = str(Path(self.config.images_dir) / img_name)
            
            if not Path(full_img_path).exists():
                logger.warning(f"Изображение не найдено: {full_img_path}")
                not_valid_images.append({
                    "path": img_name,
                    "reason": "file_not_found"
                })
                continue
            
            self.metrics.total_images_checked += 1
            
            # CLIP pre-filter
            if clip_filter and self.config.use_clip_prefilter:
                clip_valid, clip_reason, clip_sim = clip_filter.check_image(full_img_path)
                if not clip_valid:
                    not_valid_images.append({
                        "path": img_name,
                        "reason": clip_reason,
                        "clip_similarity": float(clip_sim)
                    })
                    self.metrics.clip_filtered += 1
                    self.metrics.rejection_reasons[clip_reason] = \
                        self.metrics.rejection_reasons.get(clip_reason, 0) + 1
                    continue
            
            # VLM проверка
            is_valid, confidence = image_filter.check_image(full_img_path, name, description)
            
            if is_valid:
                # Генерация описания для валидного изображения
                caption = image_filter.generate_caption(full_img_path, name)
                valid_images.append({
                    "path": img_name,  # Сохраняем только название файла
                    "caption": caption,
                    "confidence": confidence
                })
                self.metrics.total_images_valid += 1
            else:
                not_valid_images.append({
                    "path": img_name,  # Сохраняем только название файла
                    "reason": "vlm_rejected",
                    "confidence": confidence
                })
                self.metrics.rejection_reasons["other"] += 1
        
        if len(valid_images) < self.config.min_images_per_landmark:
            logger.info(
                f"Недостаточно валидных изображений для {name}: "
                f"{len(valid_images)}/{self.config.min_images_per_landmark}"
            )
            self.progress.mark_processed(obj_id)
            return None
        
        self.progress.mark_processed(obj_id)
        self.metrics.objects_passed += 1
        
        # Создаем копию всех полей из исходного объекта
        result = item.copy()
        
        # Удаляем старое поле image_path
        if "image_path" in result:
            del result["image_path"]
        
        # Добавляем новые поля
        result["valid_images"] = valid_images
        result["not_valid_images"] = not_valid_images
        
        return result
    
    def run(self):
        """Запуск процесса фильтрации."""
        # Загрузка данных
        with open(self.config.input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(
            f"Загружено {len(data)} объектов из {self.config.input_path}"
        )
        
        # Фильтрация уже обработанных
        to_process = [
            item for item in data
            if not self.progress.is_processed(
                self._generate_obj_id(item)
            )
        ]
        logger.info(
            f"Объектов к обработке: {len(to_process)} "
            f"(уже обработано: {len(data) - len(to_process)})"
        )
        
        if not to_process:
            logger.info("Все объекты уже обработаны")
            return
        
        # Инициализация фильтров
        clip_filter = None
        if self.config.use_clip_prefilter:
            clip_filter = CLIPPreFilter(self.config)
        
        image_filter = Qwen2VLImageFilter(self.config, self.metrics)
        
        filtered_data = []
        quality_log = []
        
        # Обработка с прогресс-баром
        with tqdm(
            total=len(to_process),
            desc="Фильтрация изображений"
        ) as pbar:
            for item in to_process:
                result = self.process_one(item, image_filter, clip_filter)
                
                if result is not None:
                    filtered_data.append(result)
                    
                    # Логирование качества
                    quality_entry = {
                        "landmark_id": result.get("landmark_id"),
                        "name": result.get("name_en") or result.get("name_ru"),
                        "valid_count": len(result.get("valid_images", [])),
                        "invalid_count": len(result.get("not_valid_images", [])),
                        "confidence_distribution": {
                            "high": sum(1 for img in result.get("valid_images", [])
                                       if img.get("confidence") == "high"),
                            "medium": sum(1 for img in result.get("valid_images", [])
                                         if img.get("confidence") == "medium"),
                        },
                        "rejection_reasons": {}
                    }
                    
                    # Подсчет причин отклонения
                    for invalid in result.get("not_valid_images", []):
                        reason = invalid.get("reason", "unknown")
                        quality_entry["rejection_reasons"][reason] = \
                            quality_entry["rejection_reasons"].get(reason, 0) + 1
                    
                    quality_log.append(quality_entry)
                
                self.metrics.objects_processed += 1
                pbar.update(1)
                
                # Периодический сброс кэшей
                if self.metrics.objects_processed % 10 == 0:
                    image_filter.cache.flush()
                    self.progress.flush()
                
                # Сохранение промежуточных результатов
                if self.metrics.objects_processed % 50 == 0 and filtered_data:
                    with open(
                        self.config.output_path,
                        'w',
                        encoding='utf-8'
                    ) as f:
                        json.dump(
                            filtered_data,
                            f,
                            ensure_ascii=False,
                            indent=2
                        )
                    
                    # Сохранение лога качества
                    with open(
                        self.config.quality_log_path,
                        'w',
                        encoding='utf-8'
                    ) as f:
                        for entry in quality_log:
                            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # Финальный сброс
        image_filter.cache.flush()
        self.progress.flush()
        
        # Финальное сохранение
        with open(self.config.output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        
        # Финальное сохранение лога качества
        with open(self.config.quality_log_path, 'w', encoding='utf-8') as f:
            for entry in quality_log:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # Отчет по метрикам
        metrics_report = self.metrics.report()
        logger.info("=" * 60)
        logger.info("ШАГ 4: ФИЛЬТРАЦИЯ ИЗОБРАЖЕНИЙ ЗАВЕРШЕНА")
        logger.info("=" * 60)
        logger.info(
            f"Всего обработано объектов: "
            f"{metrics_report['objects_processed']}"
        )
        logger.info(
            f"Прошло фильтрацию: {metrics_report['objects_passed']}"
        )
        logger.info(
            f"Проверено изображений: "
            f"{metrics_report['total_images_checked']}"
        )
        logger.info(
            f"Валидных изображений: "
            f"{metrics_report['total_images_valid']}"
        )
        logger.info(
            f"CLIP отфильтровано: {metrics_report['clip_filtered']}"
        )
        logger.info("Распределение confidence:")
        logger.info(
            f"  - Высокая (ДА): {metrics_report['confidence_distribution']['high']}"
        )
        logger.info(
            f"  - Средняя (СКОРЕЕ ДА): {metrics_report['confidence_distribution']['medium']}"
        )
        logger.info(
            f"  - Низкая (НЕТ): {metrics_report['confidence_distribution']['low']}"
        )
        logger.info("Причины отклонения:")
        for reason, count in metrics_report['rejection_reasons'].items():
            if count > 0:
                logger.info(f"  - {reason}: {count}")
        logger.info(
            f"Время выполнения: {metrics_report['elapsed_time_sec']}с"
        )
        logger.info(
            f"Скорость: {metrics_report['images_per_sec']} изобр/сек"
        )
        logger.info(
            f"Cache hit rate: {metrics_report['cache_hit_rate']}%"
        )
        logger.info(
            f"Результат сохранен в: {self.config.output_path}"
        )
        logger.info(
            f"Лог качества сохранен в: {self.config.quality_log_path}"
        )
        logger.info("=" * 60)


# ======================
# ТОЧКА ВХОДА
# ======================
def main():
    """Главная точка входа."""
    try:
        config = Config()
        processor = ImageFilterProcessor(config)
        processor.run()
    except KeyboardInterrupt:
        logger.info("Прервано пользователем, сохранение прогресса...")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
