#!/usr/bin/env python3
"""
Шаг 4b: Генерация captions для валидных изображений.
Использует Qwen2-VL для создания визуальных описаний.

Этап 2: Генерация captions для уже валидированных изображений
Входные данные: validated_images.json из step4a

Преимущества:
    - Настоящий cross-object batching (изображения из разных объектов в одном батче)
    - Можно перезапустить без повторной валидации
    - Более эффективное использование GPU
"""

import json
import logging
import time
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm
from dataclasses import dataclass
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
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
GPU_CACHE_CLEAR_INTERVAL = 50
MAX_SUMMARY_CAPTIONS = 3

# ======================
# КОНФИГУРАЦИЯ
# ======================
@dataclass
class Config:
    """Конфигурация для генерации captions."""
    # Пути к файлам
    input_path: str = "setup_data_v3/data/validated_images.json"
    output_path: str = "setup_data_v3/data/clean_landmarks.json"
    checkpoint_path: str = "setup_data_v3/data/cache/caption_checkpoint.json"
    
    # Директория с изображениями
    images_dir: str = "setup_data_v3/data/images"
    
    # Кэш
    caption_cache_path: str = "setup_data_v3/data/cache/image_captions.jsonl"
    
    # Модель
    vlm_model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    max_pixels: int = 256 * 256
    max_new_tokens: int = 100  # Для генерации описаний
    
    # Cross-object batching
    batch_size: int = 32  # Большой батч из разных объектов
    
    # Интервалы
    cache_flush_interval: int = 100
    checkpoint_interval: int = 100  # Сохранять checkpoint каждые N изображений
    
    # Фильтрация
    min_images_per_landmark: int = 1
    
    def __post_init__(self):
        if not Path(self.input_path).exists():
            raise FileNotFoundError(f"Входной файл не найден: {self.input_path}")
        
        Path(self.caption_cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)


# ======================
# КЭШИРОВАНИЕ
# ======================
class CaptionCache:
    """Кэш сгенерированных captions."""
    def __init__(self, path: Path):
        self.path = path
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
    
    def flush(self):
        if self.pending_writes:
            with open(self.path, 'a', encoding='utf-8') as f:
                for record in self.pending_writes:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            self.pending_writes.clear()


# ======================
# CAPTION GENERATOR
# ======================
class Qwen2VLCaptionGenerator:
    """Генерация captions с использованием Qwen2-VL."""
    def __init__(self, config: Config):
        self.config = config
        self.cache = CaptionCache(Path(config.caption_cache_path))
        
        logger.info(f"Загрузка Qwen2-VL: {config.vlm_model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.device == "cpu":
            logger.warning("GPU не обнаружен! Обработка на CPU будет очень медленной.")
        
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
    
    def generate_captions_batch(
        self,
        batch_data: List[Tuple[str, str]]  # [(image_path, landmark_name), ...]
    ) -> List[str]:
        """
        Батчевая генерация captions для изображений из разных объектов.
        
        Returns:
            List[caption] для каждого изображения
        """
        results = []
        to_process = []
        to_process_indices = []
        
        # Проверяем кэш
        for idx, (image_path, landmark_name) in enumerate(batch_data):
            cached = self.cache.get(image_path)
            if cached is not None:
                results.append(cached)
            else:
                results.append(f"Изображение {landmark_name}")  # Placeholder
                to_process.append(batch_data[idx])
                to_process_indices.append(idx)
        
        if not to_process:
            return results
        
        try:
            # Подготовка промптов для батча
            messages_batch = []
            for image_path, landmark_name in to_process:
                prompt = f"""Создай КОНКРЕТНОЕ визуальное описание достопримечательности.

CAPTION должен:
- описывать ТОЛЬКО видимые детали
- содержать отличительные архитектурные признаки
- упоминать уникальные элементы объекта
- быть полезным для отличия объекта от других зданий

Старайся замечать:
- форму здания
- количество и тип башен
- купола
- шпили
- арки
- колонны
- необычные элементы фасада
- статуи
- часы
- цвет материалов
- геометрию
- симметрию
- декоративные детали
- современный или исторический стиль

Не придумывай факты.
Не упоминай историю, местоположение или эмоции.

Описание должно быть:
- конкретным
- визуально точным
- информативным
- 1-3 предложения
"""
                
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
            
            # Генерация
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
            
            # Сохранение результатов
            for idx, (image_path, landmark_name) in enumerate(to_process):
                caption = output_texts[idx].strip()
                if not caption:
                    caption = f"Изображение {landmark_name}"
                
                # Сохраняем в кэш
                self.cache.set(image_path, caption)
                
                result_idx = to_process_indices[idx]
                results[result_idx] = caption
            
            # Очистка памяти
            del all_image_inputs, texts, messages_batch, inputs, generated_ids
            gc.collect()
            
            return results
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"OOM ошибка при генерации captions: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            return [f"Изображение {name}" for _, name in batch_data]
        except Exception as e:
            logger.error(f"Ошибка генерации captions: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            return [f"Изображение {name}" for _, name in batch_data]


# ======================
# ГЕНЕРАЦИЯ СВОДНОГО ОПИСАНИЯ
# ======================
def generate_landmark_summary(valid_images: List[Dict], landmark_name: str) -> str:
    """Генерация сводного канонического описания достопримечательности."""
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


# ======================
# CHECKPOINT УПРАВЛЕНИЕ
# ======================
class CheckpointManager:
    """Управление checkpoint'ами для возобновления обработки."""
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.processed_indices = set()
        
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.processed_indices = set(data.get('processed_indices', []))
                logger.info(f"Загружен checkpoint: {len(self.processed_indices)} изображений уже обработано")
            except Exception as e:
                logger.warning(f"Не удалось загрузить checkpoint: {e}")
                self.processed_indices = set()
    
    def is_processed(self, batch_start_idx: int, batch_size: int) -> bool:
        """Проверяет, обработан ли весь батч."""
        batch_indices = set(range(batch_start_idx, batch_start_idx + batch_size))
        return batch_indices.issubset(self.processed_indices)
    
    def mark_processed(self, batch_start_idx: int, batch_size: int):
        """Отмечает батч как обработанный."""
        for idx in range(batch_start_idx, batch_start_idx + batch_size):
            self.processed_indices.add(idx)
    
    def save(self):
        """Сохраняет checkpoint."""
        try:
            with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'processed_indices': sorted(list(self.processed_indices)),
                    'timestamp': time.time()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения checkpoint: {e}")
    
    def clear(self):
        """Удаляет checkpoint после завершения."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()


# ======================
# ОСНОВНОЙ ПРОЦЕССОР
# ======================
def main():
    """Главная точка входа."""
    config = Config()
    
    # Загрузка данных
    with open(config.input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Загружено {len(data)} объектов с валидными изображениями")
    
    # Подготовка батчей изображений (cross-object)
    image_batch_data = []  # [(image_path, landmark_name, obj_idx, img_idx), ...]
    
    for obj_idx, item in enumerate(data):
        name = str(item.get("name_en") or item.get("name_ru") or "")
        valid_images = item.get("valid_images", [])
        
        for img_idx, img_info in enumerate(valid_images):
            img_name = img_info.get("path", "")
            full_img_path = str(Path(config.images_dir) / img_name)
            if Path(full_img_path).exists():
                image_batch_data.append((full_img_path, name, obj_idx, img_idx))
    
    logger.info(f"Всего изображений для генерации captions: {len(image_batch_data)}")
    
    # Инициализация генератора и checkpoint manager
    generator = Qwen2VLCaptionGenerator(config)
    checkpoint_manager = CheckpointManager(Path(config.checkpoint_path))
    
    # Результаты генерации
    caption_results = {}  # (obj_idx, img_idx) -> caption
    
    # Загружаем уже обработанные captions из кэша
    for full_img_path, name, obj_idx, img_idx in image_batch_data:
        cached_caption = generator.cache.get(full_img_path)
        if cached_caption is not None:
            caption_results[(obj_idx, img_idx)] = cached_caption
    
    logger.info(f"Из кэша загружено {len(caption_results)} captions")
    
    # Обработка батчами
    processed_count = len([idx for idx in range(len(image_batch_data)) if idx in checkpoint_manager.processed_indices])
    
    with tqdm(total=len(image_batch_data), initial=processed_count, desc="Генерация captions") as pbar:
        for batch_idx in range(0, len(image_batch_data), config.batch_size):
            batch = image_batch_data[batch_idx:batch_idx + config.batch_size]
            actual_batch_size = len(batch)
            
            # Пропускаем уже обработанные батчи
            if checkpoint_manager.is_processed(batch_idx, actual_batch_size):
                pbar.update(actual_batch_size)
                continue
            
            # Генерация captions
            batch_input = [(item[0], item[1]) for item in batch]
            captions = generator.generate_captions_batch(batch_input)
            
            for (_, _, obj_idx, img_idx), caption in zip(batch, captions):
                caption_results[(obj_idx, img_idx)] = caption
            
            # Отмечаем батч как обработанный
            checkpoint_manager.mark_processed(batch_idx, actual_batch_size)
            
            pbar.update(actual_batch_size)
            
            # Сохраняем checkpoint после каждого батча
            generator.cache.flush()
            checkpoint_manager.save()
            
            # Периодическое логирование
            processed_total = batch_idx + actual_batch_size
            if processed_total % config.checkpoint_interval == 0:
                logger.info(f"Обработано: {processed_total}/{len(image_batch_data)} изображений")
    
    # Финальный сброс кэша
    generator.cache.flush()
    checkpoint_manager.save()
    
    # Формирование финальных данных
    output_data = []
    for obj_idx, item in enumerate(data):
        valid_images = item.get("valid_images", [])
        
        # Добавляем captions к валидным изображениям
        for img_idx, img_info in enumerate(valid_images):
            caption = caption_results.get((obj_idx, img_idx))
            if caption is not None:
                img_info["caption"] = caption
        
        # Фильтрация по минимальному количеству изображений
        if len(valid_images) >= config.min_images_per_landmark:
            # Генерация сводного описания
            name = str(item.get("name_en") or item.get("name_ru") or "")
            landmark_summary = generate_landmark_summary(valid_images, name)
            
            result_item = item.copy()
            result_item["valid_images"] = valid_images
            result_item["landmark_summary_caption"] = landmark_summary
            output_data.append(result_item)
    
    # Сохранение
    with open(config.output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # Удаляем checkpoint после успешного завершения
    checkpoint_manager.clear()
    logger.info("Checkpoint удален после успешного завершения")
    
    logger.info("=" * 60)
    logger.info("ШАГ 4b: ГЕНЕРАЦИЯ CAPTIONS ЗАВЕРШЕНА")
    logger.info("=" * 60)
    logger.info(f"Входных объектов: {len(data)}")
    logger.info(f"Выходных объектов: {len(output_data)}")
    logger.info(f"Результат сохранен в: {config.output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
