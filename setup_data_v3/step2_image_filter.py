#!/usr/bin/env python3
"""
Шаг 2: Фильтрация изображений достопримечательностей.
Использует локальную Qwen2-VL модель для проверки соответствия изображений описаниям.

Требования:
    - NVIDIA GPU с минимум 8GB VRAM (рекомендуется T4 24GB)
    - Результат работы step1_text_filter.py

Установка зависимостей:
    pip install transformers torch qwen-vl-utils accelerate pillow tqdm

Запуск:
    python step2_image_filter.py
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from tqdm import tqdm
from dataclasses import dataclass
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
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
    
    # Параметры фильтрации
    min_images_per_landmark: int = 2
    
    # Кэш и прогресс
    image_cache_path: str = "setup_data_v3/cache/image_checks.jsonl"
    progress_path: str = "setup_data_v3/cache/image_processed_landmarks.txt"
    
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
    ) -> bool:
        """Проверка соответствия изображения описанию."""
        cached = self.cache.get(image_path)
        if cached is not None:
            self.metrics.image_cache_hits += 1
            return cached
        
        try:
            # Формирование промпта
            prompt = f"""Проверь соответствие изображения описанию.

Ожидаемый объект: {expected_name}
Описание: {expected_description}

Ответь ТОЛЬКО "ДА" если на изображении эта достопримечательность.
Ответь ТОЛЬКО "НЕТ" если это другой объект, человек, еда, карта, текст или животное.

Твой ответ:"""
            
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
                    max_new_tokens=10,
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
            
            # Проверка ответа
            is_valid = output_text.startswith('да')
            
            self.metrics.image_checks += 1
            self.cache.set(image_path, is_valid)
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Ошибка проверки {image_path}: {e}")
            self.cache.set(image_path, False)
            return False


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
        image_filter: Qwen2VLImageFilter
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
        image_paths = item.get("image_path", [])
        if not image_paths:
            logger.info(f"Нет изображений для: {name}")
            self.progress.mark_processed(obj_id)
            return None
        
        valid_images = []
        for img_path in image_paths:
            if not Path(img_path).exists():
                logger.warning(f"Изображение не найдено: {img_path}")
                continue
            
            self.metrics.total_images_checked += 1
            is_valid = image_filter.check_image(img_path, name, description)
            
            if is_valid:
                valid_images.append({
                    "path": img_path,
                    "caption": ""
                })
                self.metrics.total_images_valid += 1
        
        if len(valid_images) < self.config.min_images_per_landmark:
            logger.info(
                f"Недостаточно валидных изображений для {name}: "
                f"{len(valid_images)}/{self.config.min_images_per_landmark}"
            )
            self.progress.mark_processed(obj_id)
            return None
        
        self.progress.mark_processed(obj_id)
        self.metrics.objects_passed += 1
        
        return {
            "name": name,
            "description": description,
            "images": valid_images
        }
    
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
        
        # Инициализация фильтра
        image_filter = Qwen2VLImageFilter(self.config, self.metrics)
        
        filtered_data = []
        
        # Обработка с прогресс-баром
        with tqdm(
            total=len(to_process),
            desc="Фильтрация изображений"
        ) as pbar:
            for item in to_process:
                result = self.process_one(item, image_filter)
                
                if result is not None:
                    filtered_data.append(result)
                
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
        
        # Финальный сброс
        image_filter.cache.flush()
        self.progress.flush()
        
        # Финальное сохранение
        with open(self.config.output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        
        # Отчет по метрикам
        metrics_report = self.metrics.report()
        logger.info("=" * 60)
        logger.info("ШАГ 2: ФИЛЬТРАЦИЯ ИЗОБРАЖЕНИЙ ЗАВЕРШЕНА")
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
