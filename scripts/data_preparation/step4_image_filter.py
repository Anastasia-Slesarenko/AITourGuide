#!/usr/bin/env python3
"""
Шаг 4: Фильтрация изображений достопримечательностей.
Использует локальную Qwen2-VL модель для проверки соответствия изображений описаниям.

Требования:
    - NVIDIA GPU с минимум 8GB VRAM (рекомендуется T4 24GB)
    - Результат работы step3_text_filter.py

Установка зависимостей:
    pip install transformers torch qwen-vl-utils accelerate pillow tqdm

Архитектура:
    1. CLIP pre-filter - быстрая батчевая фильтрация нерелевантных изображений
    2. Qwen2-VL - детальная проверка и генерация описаний (single-pass)
    3. Retry логика - автоматические повторы при OOM ошибках
    4. Кэширование - результаты сохраняются для возобновления работы

Выходной формат:
    - Сохраняет все поля из text_filtered_landmarks.json
    - Заменяет поле 'image_path' на 'valid_images' (список валидных изображений с описаниями)
    - Добавляет поле 'not_valid_images' (список невалидных изображений)
    - Добавляет поле 'landmark_summary_caption' (сводное каноническое описание достопримечательности)
    
    Структура данных (2 уровня):
    1. Уровень изображений - каждое изображение в 'valid_images' содержит:
        * path: путь к изображению
        * caption: автоматически сгенерированное описание изображения
        * confidence: уровень уверенности (high/medium)
    
    2. Уровень достопримечательности:
        * landmark_summary_caption: сводное каноническое визуальное описание,
          объединяющее информацию из всех изображений
"""

import json
import logging
import time
import gc
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
from tqdm import tqdm
from dataclasses import dataclass
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
from PIL import Image
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    CLIPProcessor,
    CLIPModel,
    BitsAndBytesConfig
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
# КОНСТАНТЫ
# ======================
GPU_CACHE_CLEAR_INTERVAL = 50  # Интервал очистки GPU кэша
MAX_SUMMARY_CAPTIONS = 3  # Максимальное количество captions для сводного описания

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
    caption_cache_path: str = "setup_data_v3/data/cache/image_captions.jsonl"
    progress_path: str = "setup_data_v3/data/cache/image_processed_landmarks.txt"
    
    # CLIP pre-filter
    clip_model_name: str = "openai/clip-vit-base-patch32"
    use_clip_prefilter: bool = True
    clip_threshold: float = 0.22  # Минимальное сходство для прохождения CLIP фильтра
    clip_interior_threshold: float = 0.4  # Порог для определения интерьера
    clip_person_threshold: float = 0.3  # Порог для определения людей
    clip_food_threshold: float = 0.3  # Порог для определения еды
    clip_map_text_threshold: float = 0.3  # Порог для определения карт/текста
    
    # Модель
    vlm_model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    max_pixels: int = 256 * 256  # Уменьшено для скорости (было 384x384)
    max_new_tokens: int = 80  # Уменьшено - достаточно для VALID/CONFIDENCE/CAPTION
    
    # Batching
    batch_size: int = 16  # Увеличено для скорости (было 8)
    clip_batch_size: int = 32  # Количество изображений для одновременной обработки CLIP (не используется)
    
    # Интервал сброса кэша
    cache_flush_interval: int = 50
    progress_flush_interval: int = 10  # Интервал сброса прогресса и промежуточного сохранения
    
    # Retry параметры
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        # Проверка существования входного файла
        if not Path(self.input_path).exists():
            raise FileNotFoundError(
                f"Входной файл не найден: {self.input_path}\n"
                f"Сначала запустите step3_text_filter.py"
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
        self.rejection_reasons = defaultdict(int)
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
    """Кэш результатов проверки изображений с confidence."""
    def __init__(self, path: Path, flush_interval: int = 50):
        self.path = path
        self.flush_interval = flush_interval
        self.data: Dict[str, Tuple[bool, str]] = {}  # image_path -> (is_valid, confidence)
        self.pending_writes: List[Dict] = []
        
        if self.path.exists():
            with open(self.path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    confidence = rec.get('confidence', 'low')
                    self.data[rec['image_path']] = (rec['is_valid'], confidence)
        logger.info(
            f"Загружено {len(self.data)} записей из кэша изображений"
        )
    
    def get(self, image_path: str) -> Optional[Tuple[bool, str]]:
        return self.data.get(image_path)
    
    def set(self, image_path: str, is_valid: bool, confidence: str = "low"):
        self.data[image_path] = (is_valid, confidence)
        self.pending_writes.append({
            'image_path': image_path,
            'is_valid': is_valid,
            'confidence': confidence
        })
        
        if len(self.pending_writes) >= self.flush_interval:
            self.flush()
    
    def flush(self):
        """Принудительная запись на диск."""
        if self.pending_writes:
            with open(self.path, 'a', encoding='utf-8') as f:
                for record in self.pending_writes:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            self.pending_writes.clear()


class CaptionCache:
    """Кэш сгенерированных captions для изображений."""
    def __init__(self, path: Path, flush_interval: int = 50):
        self.path = path
        self.flush_interval = flush_interval
        self.data: Dict[str, str] = {}
        self.pending_writes: List[Dict] = []
        
        if self.path.exists():
            with open(self.path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self.data[rec['image_path']] = rec['caption']
        logger.info(f"Загружено {len(self.data)} captions из кэша")
    
    def get(self, image_path: str) -> Optional[str]:
        return self.data.get(image_path)
    
    def set(self, image_path: str, caption: str):
        self.data[image_path] = caption
        self.pending_writes.append({
            'image_path': image_path,
            'caption': caption
        })
        
        if len(self.pending_writes) >= self.flush_interval:
            self.flush()
    
    def flush(self):
        """Принудительная запись на диск."""
        if self.pending_writes:
            with open(self.path, 'a', encoding='utf-8') as f:
                for record in self.pending_writes:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
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
        
        # Используем float16 на GPU для экономии памяти
        if self.device == "cuda":
            self.model = self.model.to(self.device, dtype=torch.float16)
        else:
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
    
    def check_images_batch(self, image_paths: List[str]) -> List[Tuple[bool, str, float]]:
        """
        Батчевая проверка изображений с помощью CLIP для ускорения.
        
        Returns:
            List[(is_valid, reason, similarity)] для каждого изображения
        """
        results: List[Tuple[bool, str, float]] = []
        images = []
        valid_indices = []
        
        # Загружаем все изображения
        for idx, image_path in enumerate(image_paths):
            try:
                img = Image.open(image_path).convert("RGB")
                images.append(img)
                valid_indices.append(idx)
            except Exception as e:
                logger.error(f"CLIP ошибка загрузки {image_path}: {e}")
                results.append((False, "clip_error", 0.0))
        
        if not images:
            return results if results else [(False, "clip_error", 0.0) for _ in image_paths]
        
        try:
            with torch.no_grad():
                # Батчевая обработка всех изображений
                image_inputs = self.processor(
                    images=images,
                    return_tensors="pt"
                ).to(self.device)
                image_outputs = self.model.get_image_features(**image_inputs)
                image_features = image_outputs / image_outputs.norm(dim=-1, keepdim=True)
                
                # Вычисляем сходство для всех изображений
                similarities = (image_features @ self.text_features.T)
                probs = similarities.softmax(dim=1).cpu().numpy()
            
            # Анализ результатов для каждого изображения
            batch_results = []
            for prob_row in probs:
                exterior_prob = float(prob_row[0:2].sum())
                interior_prob = float(prob_row[2])
                person_prob = float(prob_row[3])
                food_prob = float(prob_row[4])
                map_text_prob = float(prob_row[5])
                
                # Определяем причину отклонения
                if interior_prob > self.config.clip_interior_threshold:
                    batch_results.append((False, "interior", interior_prob))
                elif person_prob > self.config.clip_person_threshold:
                    batch_results.append((False, "person", person_prob))
                elif food_prob > self.config.clip_food_threshold:
                    batch_results.append((False, "food", food_prob))
                elif map_text_prob > self.config.clip_map_text_threshold:
                    batch_results.append((False, "map_or_text", map_text_prob))
                elif exterior_prob < self.config.clip_threshold:
                    batch_results.append((False, "not_building", exterior_prob))
                else:
                    batch_results.append((True, "passed", exterior_prob))
            
            # Собираем финальные результаты в правильном порядке
            final_results: List[Tuple[bool, str, float]] = []
            batch_idx = 0
            error_idx = 0
            
            for idx in range(len(image_paths)):
                if idx in valid_indices:
                    final_results.append(batch_results[batch_idx])
                    batch_idx += 1
                else:
                    # Это изображение с ошибкой загрузки
                    if error_idx < len(results):
                        final_results.append(results[error_idx])
                        error_idx += 1
                    else:
                        final_results.append((False, "clip_error", 0.0))
            
            return final_results
            
        except Exception as e:
            logger.error(f"CLIP батчевая ошибка: {e}")
            # Fallback на последовательную обработку
            return [self.check_image(path) for path in image_paths]
        finally:
            for img in images:
                img.close()
    
    def check_image(self, image_path: str) -> Tuple[bool, str, float]:
        """
        Проверка одного изображения (fallback).
        
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
            exterior_prob = float(probs[0:2].sum())
            interior_prob = float(probs[2])
            person_prob = float(probs[3])
            food_prob = float(probs[4])
            map_text_prob = float(probs[5])
            
            # Определяем причину отклонения
            if interior_prob > self.config.clip_interior_threshold:
                return False, "interior", interior_prob
            elif person_prob > self.config.clip_person_threshold:
                return False, "person", person_prob
            elif food_prob > self.config.clip_food_threshold:
                return False, "food", food_prob
            elif map_text_prob > self.config.clip_map_text_threshold:
                return False, "map_or_text", map_text_prob
            elif exterior_prob < self.config.clip_threshold:
                return False, "not_building", exterior_prob
            
            return True, "passed", exterior_prob
            
        except Exception as e:
            logger.error(f"CLIP ошибка для {image_path}: {e}")
            return False, "clip_error", 0.0
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
        self.caption_cache = CaptionCache(
            Path(config.caption_cache_path),
            config.cache_flush_interval
        )
        
        logger.info(f"Загрузка Qwen2-VL: {config.vlm_model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.device == "cpu":
            logger.warning(
                "GPU не обнаружен! Обработка на CPU будет очень медленной."
            )
        
        # Загрузка модели
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.vlm_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="sdpa"
        )

        self.processor = AutoProcessor.from_pretrained(
            config.vlm_model_name,
            max_pixels=config.max_pixels,
            use_fast=True
        )

        
        logger.info(f"Модель загружена на: {self.device}")
    
    def _parse_vlm_response(self, output_text: str, expected_name: str) -> Tuple[bool, str, str]:
        """
        Парсинг структурированного ответа VLM.
        
        Returns:
            (is_valid, confidence_level, caption)
        """
        is_valid = False
        confidence = "low"
        caption = f"Изображение {expected_name}"
        
        for line in output_text.strip().split('\n'):
            line = line.strip()
            if line.startswith('VALID:'):
                is_valid = 'yes' in line.lower()
            elif line.startswith('CONFIDENCE:'):
                conf_text = line.split(':', 1)[1].strip().lower()
                confidence = "high" if 'high' in conf_text else ("medium" if 'medium' in conf_text else "low")
            elif line.startswith('CAPTION:'):
                caption_text = line.split(':', 1)[1].strip()
                if caption_text and is_valid:
                    caption = caption_text
        
        return is_valid, confidence, caption
    
    def check_and_caption_images_batch(
        self,
        image_paths: List[str],
        expected_name: str,
        expected_description: str
    ) -> List[Tuple[bool, str, str]]:
        """
        Qwen single-pass: VALID + CONFIDENCE + CAPTION за один проход.
        
        Returns:
            List[(is_valid, confidence_level, caption)] для каждого изображения
        """
        results = []
        to_process = []
        to_process_indices = []
        
        # Проверяем кэш
        for idx, image_path in enumerate(image_paths):
            cached_result = self.cache.get(image_path)
            cached_caption = self.caption_cache.get(image_path)
            
            if cached_result is not None and cached_caption is not None:
                self.metrics.image_cache_hits += 1
                cached_valid, cached_confidence = cached_result
                results.append((cached_valid, cached_confidence, cached_caption))
            else:
                results.append((False, "low", f"Изображение {expected_name}"))  # Placeholder
                to_process.append(image_path)
                to_process_indices.append(idx)
        
        if not to_process:
            return results
        
        try:
            # Single-pass промпт: валидация + описание
            messages_batch = []
            for image_path in to_process:
                prompt = f"""Изображение должно показывать: {expected_name}

Описание: {expected_description}

Ответь строго в формате:
VALID: yes/no
CONFIDENCE: high/medium/low
CAPTION: [краткое визуальное описание 1-2 предложения, ТОЛЬКО если VALID=yes]

yes = архитектурный объект снаружи, соответствует описанию
no = интерьер, человек, карта, текст, еда, или НЕ соответствует

CAPTION должен описывать ТОЛЬКО видимые детали: тип, форму, элементы, материал, цвет, стиль."""
                
                messages_batch.append([{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }])
            
            # Обработка батча
            texts = []
            all_image_inputs = []
            
            for messages in messages_batch:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                texts.append(text)
                image_inputs, _ = process_vision_info(messages)
                all_image_inputs.extend(image_inputs)
            
            inputs = self.processor(
                text=texts,
                images=all_image_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # Генерация с autocast
            with torch.no_grad(), torch.cuda.amp.autocast():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in
                zip(inputs.input_ids, generated_ids)
            ]
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            # Парсинг результатов
            for idx, (image_path, output_text) in enumerate(zip(to_process, output_texts)):
                is_valid, confidence, caption = self._parse_vlm_response(output_text, expected_name)
                
                # Метрики
                if confidence == "high":
                    self.metrics.confidence_high += 1
                elif confidence == "medium":
                    self.metrics.confidence_medium += 1
                else:
                    self.metrics.confidence_low += 1
                
                self.metrics.image_checks += 1
                
                # Сохраняем в кэш
                self.cache.set(image_path, is_valid, confidence)
                self.caption_cache.set(image_path, caption)
                
                result_idx = to_process_indices[idx]
                results[result_idx] = (is_valid, confidence, caption)
            
            # Очистка памяти от memory leak
            del all_image_inputs, texts, messages_batch, inputs, generated_ids
            gc.collect()
            
            return results
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"OOM ошибка при батчевой обработке: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            # Fallback на последовательную обработку с кэшированием
            fallback_results = []
            for image_path in to_process:
                result = self._check_single_image_with_retry(image_path, expected_name, expected_description)
                fallback_results.append(result)
            
            # Обновляем results с fallback результатами
            for idx, result in zip(to_process_indices, fallback_results):
                results[idx] = result
            
            return results
        except Exception as e:
            logger.error(f"Ошибка батчевой обработки: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            # Fallback на последовательную обработку с кэшированием
            fallback_results = []
            for image_path in to_process:
                result = self._check_single_image_with_retry(image_path, expected_name, expected_description)
                fallback_results.append(result)
            
            # Обновляем results с fallback результатами
            for idx, result in zip(to_process_indices, fallback_results):
                results[idx] = result
            
            return results
        finally:
            if self.metrics.image_checks % GPU_CACHE_CLEAR_INTERVAL == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
    def _check_single_image_with_retry(
        self,
        image_path: str,
        expected_name: str,
        expected_description: str
    ) -> Tuple[bool, str, str]:
        """
        Проверка одного изображения с retry логикой.
        
        Returns:
            (is_valid, confidence_level, caption)
        """
        # Проверяем кэш
        cached_result = self.cache.get(image_path)
        cached_caption = self.caption_cache.get(image_path)
        
        if cached_result is not None and cached_caption is not None:
            self.metrics.image_cache_hits += 1
            cached_valid, cached_confidence = cached_result
            return cached_valid, cached_confidence, cached_caption
        
        # Retry логика
        for attempt in range(self.config.max_retries):
            try:
                return self._check_single_image(image_path, expected_name, expected_description)
            except torch.cuda.OutOfMemoryError as e:
                logger.warning(f"OOM на попытке {attempt + 1}/{self.config.max_retries} для {image_path}")
                torch.cuda.empty_cache()
                gc.collect()
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error(f"OOM после всех попыток для {image_path}")
                    fallback_caption = f"Изображение {expected_name}"
                    # Кэшируем неудачу чтобы не повторять
                    self.cache.set(image_path, False, "low")
                    self.caption_cache.set(image_path, fallback_caption)
                    return False, "low", fallback_caption
            except Exception as e:
                logger.warning(f"Ошибка на попытке {attempt + 1}/{self.config.max_retries} для {image_path}: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error(f"Ошибка после всех попыток для {image_path}")
                    fallback_caption = f"Изображение {expected_name}"
                    self.cache.set(image_path, False, "low")
                    self.caption_cache.set(image_path, fallback_caption)
                    return False, "low", fallback_caption
        
        # Не должно достигаться, но на всякий случай
        fallback_caption = f"Изображение {expected_name}"
        return False, "low", fallback_caption
    
    def _check_single_image(
        self,
        image_path: str,
        expected_name: str,
        expected_description: str
    ) -> Tuple[bool, str, str]:
        """
        Проверка одного изображения (внутренний метод).
        
        Returns:
            (is_valid, confidence_level, caption)
        """
        try:
            # Используем тот же промпт что и в батче для консистентности
            prompt = f"""Изображение должно показывать: {expected_name}

Описание: {expected_description}

Ответь строго в формате:
VALID: yes/no
CONFIDENCE: high/medium/low
CAPTION: [краткое визуальное описание 1-2 предложения, ТОЛЬКО если VALID=yes]

yes = архитектурный объект снаружи, соответствует описанию
no = интерьер, человек, карта, текст, еда, или НЕ соответствует

CAPTION должен описывать ТОЛЬКО видимые детали: тип, форму, элементы, материал, цвет, стиль."""
            
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
            
            # Генерация (без sampling для детерминированности)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
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
            )[0].strip()
            
            # Парсинг через общий метод
            is_valid, confidence, caption = self._parse_vlm_response(output_text, expected_name)
            
            # Обновляем метрики
            if confidence == "high":
                self.metrics.confidence_high += 1
            elif confidence == "medium":
                self.metrics.confidence_medium += 1
            else:
                self.metrics.confidence_low += 1
            
            self.metrics.image_checks += 1
            
            # Сохраняем в кэш
            self.cache.set(image_path, is_valid, confidence)
            self.caption_cache.set(image_path, caption)
            
            return is_valid, confidence, caption
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"OOM ошибка для {image_path}: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            # Пробрасываем исключение для retry логики
            raise
        except Exception as e:
            logger.error(f"Ошибка проверки {image_path}: {e}")
            # Пробрасываем исключение для retry логики
            raise
        finally:
            # Очистка памяти
            if 'inputs' in locals():
                del inputs
            if 'generated_ids' in locals():
                del generated_ids
            # Периодическая очистка GPU кэша
            if self.metrics.image_checks % GPU_CACHE_CLEAR_INTERVAL == 0:
                torch.cuda.empty_cache()
                gc.collect()
    


# ======================
# ОСНОВНОЙ КЛАСС
# ======================
class ImageFilterProcessor:
    """Процессор фильтрации изображений."""
    def __init__(self, config: Config):
        self.config = config
        self.progress = ProgressTracker(Path(config.progress_path))
        self.metrics = Metrics()
    
    def _generate_landmark_summary(
        self,
        valid_images: List[Dict],
        landmark_name: str
    ) -> str:
        """
        Генерация сводного канонического описания достопримечательности.
        Использует простую текстовую агрегацию captions.
        """
        try:
            all_captions = [str(img.get("caption", "")) for img in valid_images if img.get("caption")]
            
            if not all_captions:
                return f"Визуальное описание {landmark_name}"
            
            if len(all_captions) == 1:
                return all_captions[0]
            
            # Убираем дубликаты
            unique_captions = []
            seen = set()
            for caption in all_captions:
                caption_lower = caption.lower().strip()
                if caption_lower not in seen:
                    seen.add(caption_lower)
                    unique_captions.append(caption)
            
            if len(unique_captions) == 1:
                return unique_captions[0]
            
            # Простое объединение - берем N самых длинных и информативных
            sorted_captions = sorted(unique_captions, key=len, reverse=True)[:MAX_SUMMARY_CAPTIONS]
            return " ".join(sorted_captions)
            
        except Exception as e:
            logger.error(f"Ошибка генерации сводного описания для {landmark_name}: {e}")
            return f"Визуальное описание {landmark_name}"
    
    def _generate_obj_id(self, item: Dict[str, Any]) -> str:
        """Генерация консистентного ID объекта."""
        if "id" in item and item["id"]:
            return str(item["id"])
        
        name = str(item.get("name_ru") or item.get("name_en") or "")
        desc = str(
            item.get("wikipedia_summary_ru") or
            item.get("wikipedia_summary_en") or
            ""
        )
        
        return f"{name}::{desc[:50]}"
    
    def process_one(
        self,
        item: Dict[str, Any],
        image_filter: Qwen2VLImageFilter,
        clip_filter: Optional[CLIPPreFilter] = None
    ) -> Optional[Dict[str, Any]]:
        """Обработка одного объекта."""
        obj_id = self._generate_obj_id(item)
        
        if self.progress.is_processed(obj_id):
            return None
        
        name = str(item.get("name_ru") or item.get("name_en") or "")
        description = str(
            item.get("wikipedia_summary_ru") or
            item.get("wikipedia_summary_en") or
            ""
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
        
        # Предварительная фильтрация - проверка существования файлов
        existing_images = []
        existing_image_names = []
        
        for img_name in image_names:
            full_img_path = str(Path(self.config.images_dir) / img_name)
            
            if not Path(full_img_path).exists():
                logger.warning(f"Изображение не найдено: {full_img_path}")
                not_valid_images.append({
                    "path": img_name,
                    "reason": "file_not_found"
                })
                continue
            
            self.metrics.total_images_checked += 1
            existing_images.append(full_img_path)
            existing_image_names.append(img_name)
        
        # Батчевая CLIP фильтрация всех существующих изображений
        images_to_check = []
        image_name_mapping = []
        
        if clip_filter and self.config.use_clip_prefilter and existing_images:
            # Батчевая проверка через CLIP
            clip_results = clip_filter.check_images_batch(existing_images)
            
            for img_name, full_img_path, (clip_valid, clip_reason, clip_sim) in zip(
                existing_image_names, existing_images, clip_results
            ):
                if not clip_valid:
                    not_valid_images.append({
                        "path": img_name,
                        "reason": clip_reason,
                        "clip_similarity": float(clip_sim)
                    })
                    self.metrics.clip_filtered += 1
                    self.metrics.rejection_reasons[clip_reason] += 1
                else:
                    # Прошло CLIP фильтр - добавляем для VLM проверки
                    images_to_check.append(full_img_path)
                    image_name_mapping.append(img_name)
        else:
            # Без CLIP фильтра - все изображения идут на VLM проверку
            images_to_check = existing_images
            image_name_mapping = existing_image_names
        
        # Батчевая VLM проверка
        if images_to_check:
            # Обрабатываем батчами по batch_size
            for i in range(0, len(images_to_check), self.config.batch_size):
                batch_paths = images_to_check[i:i + self.config.batch_size]
                batch_names = image_name_mapping[i:i + self.config.batch_size]
                
                # Батчевая проверка
                batch_results = image_filter.check_and_caption_images_batch(
                    batch_paths, name, description
                )
                
                # Обработка результатов батча
                for img_name, (is_valid, confidence, caption) in zip(batch_names, batch_results):
                    if is_valid:
                        valid_images.append({
                            "path": img_name,
                            "caption": caption,
                            "confidence": confidence
                        })
                        self.metrics.total_images_valid += 1
                    else:
                        not_valid_images.append({
                            "path": img_name,
                            "reason": "vlm_rejected",
                            "confidence": confidence
                        })
                        self.metrics.rejection_reasons["vlm_rejected"] += 1
        
        if len(valid_images) < self.config.min_images_per_landmark:
            logger.info(
                f"Недостаточно валидных изображений для {name}: "
                f"{len(valid_images)}/{self.config.min_images_per_landmark}"
            )
            self.progress.mark_processed(obj_id)
            return None
        
        # Генерация сводного канонического описания достопримечательности
        # Pure text aggregation без использования VLM
        landmark_summary = self._generate_landmark_summary(
            valid_images,
            name
        )
        
        self.progress.mark_processed(obj_id)
        self.metrics.objects_passed += 1
        
        # Создаем копию всех полей из исходного объекта
        result = item.copy()
        
        # Удаляем старое поле image_path
        if "image_path" in result:
            del result["image_path"]
        
        # Добавляем новые поля
        # Уровень 1: Индивидуальные captions для каждого изображения
        result["valid_images"] = valid_images
        result["not_valid_images"] = not_valid_images
        
        # Уровень 2: Сводное каноническое описание достопримечательности
        result["landmark_summary_caption"] = landmark_summary
        
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
        
        # Загружаем существующие результаты если есть
        filtered_data = []
        if Path(self.config.output_path).exists():
            try:
                with open(self.config.output_path, 'r', encoding='utf-8') as f:
                    filtered_data = json.load(f)
                logger.info(f"Загружено {len(filtered_data)} ранее обработанных объектов")
            except Exception as e:
                logger.warning(f"Не удалось загрузить существующие результаты: {e}")
                filtered_data = []
        
        quality_log = []
        newly_processed = []  # Новые объекты с последнего сохранения
        
        # Обработка с прогресс-баром
        with tqdm(
            total=len(to_process),
            desc="Фильтрация изображений"
        ) as pbar:
            for item in to_process:
                result = self.process_one(item, image_filter, clip_filter)
                
                if result is not None:
                    filtered_data.append(result)
                    newly_processed.append(result)
                    
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
                        "rejection_reasons": defaultdict(int)
                    }
                    
                    # Подсчет причин отклонения
                    for invalid in result.get("not_valid_images", []):
                        reason = invalid.get("reason", "unknown")
                        quality_entry["rejection_reasons"][reason] += 1
                    
                    quality_log.append(quality_entry)
                
                self.metrics.objects_processed += 1
                pbar.update(1)
                
                # Периодический сброс кэшей (используем конфигурируемый интервал)
                if self.metrics.objects_processed % self.config.progress_flush_interval == 0:
                    image_filter.cache.flush()
                    image_filter.caption_cache.flush()
                    self.progress.flush()
                
                # Сохранение промежуточных результатов - чаще для предотвращения потери данных
                if self.metrics.objects_processed % self.config.progress_flush_interval == 0 and newly_processed:
                    # Сохраняем ВСЕ данные (старые + новые) в JSON
                    with open(self.config.output_path, 'w', encoding='utf-8') as f:
                        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
                    
                    logger.info(f"Промежуточное сохранение: {len(filtered_data)} объектов (новых: {len(newly_processed)})")
                    
                    # Дополнительное сохранение лога качества (append mode для сохранения истории)
                    with open(
                        self.config.quality_log_path,
                        'a',
                        encoding='utf-8'
                    ) as f:
                        for entry in quality_log[-len(newly_processed):]:
                            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    
                    # Очищаем список новых объектов после сохранения
                    newly_processed.clear()
        
        # Финальный сброс
        image_filter.cache.flush()
        image_filter.caption_cache.flush()
        self.progress.flush()
        
        # Финальное сохранение в JSON
        with open(self.config.output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Финальное сохранение: {len(filtered_data)} объектов")
        
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
