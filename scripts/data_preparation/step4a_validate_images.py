#!/usr/bin/env python3
"""
Шаг 4a: Валидация изображений достопримечательностей.
Использует CLIP + Qwen2-VL для проверки соответствия изображений описаниям.

Этап 1: Только валидация (VALID: yes/no, CONFIDENCE: high/medium/low)
Этап 2: Генерация captions будет в step4b_generate_captions.py

Преимущества разделения:
    - Настоящий cross-object batching (изображения из разных объектов в одном батче)
    - Можно перезапустить генерацию captions без повторной валидации
    - Более эффективное использование GPU
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
GPU_CACHE_CLEAR_INTERVAL = 50

# ======================
# КОНФИГУРАЦИЯ
# ======================
@dataclass
class Config:
    """Конфигурация для валидации изображений."""
    # Пути к файлам
    input_path: str = "setup_data_v3/data/text_filtered_landmarks.json"
    output_path: str = "setup_data_v3/data/validated_images.json"
    
    # Директория с изображениями
    images_dir: str = "setup_data_v3/data/images"
    
    # Кэш
    validation_cache_path: str = "setup_data_v3/data/cache/image_validation.jsonl"
    
    # CLIP pre-filter
    clip_model_name: str = "openai/clip-vit-base-patch32"
    use_clip_prefilter: bool = True
    clip_threshold: float = 0.22
    clip_interior_threshold: float = 0.4
    clip_person_threshold: float = 0.3
    clip_food_threshold: float = 0.3
    clip_map_text_threshold: float = 0.3
    
    # Модель
    vlm_model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    max_pixels: int = 256 * 256
    max_new_tokens: int = 50  # Только для VALID/CONFIDENCE
    
    # Батчинг по разным объектам (cross-object batching)
    batch_size: int = 32  # Большой батч из разных объектов

    # Интервалы
    cache_flush_interval: int = 100
    save_interval: int = 500  # Сохранение каждые 500 изображений

    # Повторные попытки
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        if not Path(self.input_path).exists():
            raise FileNotFoundError(f"Входной файл не найден: {self.input_path}")
        
        Path(self.validation_cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)


# ======================
# КЭШИРОВАНИЕ
# ======================
class ValidationCache:
    """Кэш результатов валидации изображений."""
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, Tuple[bool, str]] = {}  # image_path -> (is_valid, confidence)
        self.pending_writes: List[Dict] = []
        
        if self.path.exists():
            with open(self.path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self.data[rec['image_path']] = (rec['is_valid'], rec.get('confidence', 'low'))
        logger.info(f"Загружено {len(self.data)} записей из кэша валидации")
    
    def get(self, image_path: str) -> Optional[Tuple[bool, str]]:
        return self.data.get(image_path)
    
    def set(self, image_path: str, is_valid: bool, confidence: str):
        self.data[image_path] = (is_valid, confidence)
        self.pending_writes.append({
            'image_path': image_path,
            'is_valid': is_valid,
            'confidence': confidence
        })
    
    def flush(self):
        if self.pending_writes:
            with open(self.path, 'a', encoding='utf-8') as f:
                for record in self.pending_writes:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            self.pending_writes.clear()


# ======================
# CLIP PRE-FILTER
# ======================
class CLIPPreFilter:
    """CLIP-based pre-filter для быстрой фильтрации."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Загрузка CLIP модели: {config.clip_model_name}")
        self.model = CLIPModel.from_pretrained(config.clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(config.clip_model_name)
        
        if self.device == "cuda":
            self.model = self.model.to(self.device, dtype=torch.float16)
        else:
            self.model = self.model.to(self.device)
        self.model.eval()
        
        self.prompts = [
            "a photo of a landmark building exterior",
            "a photo of architecture monument",
            "a photo of interior room",
            "a photo of a person",
            "a photo of food",
            "a photo of a map or text document"
        ]
        
        with torch.no_grad():
            text_inputs = self.processor(text=self.prompts, return_tensors="pt", padding=True).to(self.device)
            text_outputs = self.model.get_text_features(**text_inputs)
            self.text_features = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
        
        logger.info(f"CLIP модель загружена на: {self.device}")
    
    def check_images_batch(self, image_paths: List[str]) -> List[Tuple[bool, str]]:
        """Батчевая проверка изображений."""
        results = []
        images = []
        valid_indices = []
        
        for idx, image_path in enumerate(image_paths):
            try:
                img = Image.open(image_path).convert("RGB")
                images.append(img)
                valid_indices.append(idx)
            except Exception as e:
                logger.error(f"CLIP ошибка загрузки {image_path}: {e}")
                results.append((False, "clip_error"))
        
        if not images:
            return results if results else [(False, "clip_error") for _ in image_paths]
        
        try:
            with torch.no_grad():
                image_inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                image_outputs = self.model.get_image_features(**image_inputs)
                image_features = image_outputs / image_outputs.norm(dim=-1, keepdim=True)
                
                similarities = (image_features @ self.text_features.T)
                probs = similarities.softmax(dim=1).cpu().numpy()
            
            batch_results = []
            for prob_row in probs:
                exterior_prob = float(prob_row[0:2].sum())
                interior_prob = float(prob_row[2])
                person_prob = float(prob_row[3])
                food_prob = float(prob_row[4])
                map_text_prob = float(prob_row[5])
                
                if interior_prob > self.config.clip_interior_threshold:
                    batch_results.append((False, "interior"))
                elif person_prob > self.config.clip_person_threshold:
                    batch_results.append((False, "person"))
                elif food_prob > self.config.clip_food_threshold:
                    batch_results.append((False, "food"))
                elif map_text_prob > self.config.clip_map_text_threshold:
                    batch_results.append((False, "map_or_text"))
                elif exterior_prob < self.config.clip_threshold:
                    batch_results.append((False, "not_building"))
                else:
                    batch_results.append((True, "passed"))
            
            final_results = []
            batch_idx = 0
            error_idx = 0
            
            for idx in range(len(image_paths)):
                if idx in valid_indices:
                    final_results.append(batch_results[batch_idx])
                    batch_idx += 1
                else:
                    if error_idx < len(results):
                        final_results.append(results[error_idx])
                        error_idx += 1
                    else:
                        final_results.append((False, "clip_error"))
            
            return final_results
            
        except Exception as e:
            logger.error(f"CLIP батчевая ошибка: {e}")
            return [(False, "clip_error") for _ in image_paths]
        finally:
            for img in images:
                img.close()


# ======================
# VLM VALIDATOR
# ======================
class Qwen2VLValidator:
    """Валидация изображений с использованием Qwen2-VL."""
    def __init__(self, config: Config):
        self.config = config
        self.cache = ValidationCache(Path(config.validation_cache_path))
        
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
    
    def validate_images_batch(
        self,
        batch_data: List[Tuple[str, str, str]]  # [(image_path, landmark_name, description), ...]
    ) -> List[Tuple[bool, str]]:
        """
        Батчевая валидация изображений из разных объектов.
        
        Returns:
            List[(is_valid, confidence)] для каждого изображения
        """
        results = []
        to_process = []
        to_process_indices = []
        
        # Проверяем кэш
        for idx, (image_path, _, _) in enumerate(batch_data):
            cached = self.cache.get(image_path)
            if cached is not None:
                results.append(cached)
            else:
                results.append((False, "low"))  # Placeholder
                to_process.append(batch_data[idx])
                to_process_indices.append(idx)
        
        if not to_process:
            return results
        
        try:
            # Подготовка промптов для батча
            messages_batch = []
            for image_path, name, description in to_process:
                prompt = f"""Is this image showing: {name}?

Description: {description}

Answer in format:
VALID: yes/no
CONFIDENCE: high/medium/low

yes = architectural object exterior, matches description
no = interior, person, map, text, food, or does NOT match"""
                
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
            
            # Парсинг результатов
            for idx, (image_path, _, _) in enumerate(to_process):
                output_text = output_texts[idx]
                is_valid = False
                confidence = "low"
                
                for line in output_text.strip().split('\n'):
                    line = line.strip()
                    if line.startswith('VALID:'):
                        is_valid = 'yes' in line.lower()
                    elif line.startswith('CONFIDENCE:'):
                        conf_text = line.split(':', 1)[1].strip().lower()
                        confidence = "high" if 'high' in conf_text else ("medium" if 'medium' in conf_text else "low")
                
                # Сохраняем в кэш
                self.cache.set(image_path, is_valid, confidence)
                
                result_idx = to_process_indices[idx]
                results[result_idx] = (is_valid, confidence)
            
            # Очистка памяти
            del all_image_inputs, texts, messages_batch, inputs, generated_ids
            gc.collect()
            
            return results
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"OOM ошибка при батчевой валидации: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            return [(False, "low") for _ in batch_data]
        except Exception as e:
            logger.error(f"Ошибка батчевой валидации: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            return [(False, "low") for _ in batch_data]


# ======================
# ОСНОВНОЙ ПРОЦЕССОР
# ======================
def main():
    """Главная точка входа."""
    config = Config()
    
    # Загрузка данных
    with open(config.input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Загружено {len(data)} объектов")
    
    # Подготовка батчей изображений (cross-object)
    image_batch_data = []  # [(image_path, landmark_name, description, obj_idx, img_idx), ...]
    
    for obj_idx, item in enumerate(data):
        name = str(item.get("name_en") or item.get("name_ru") or "")
        description = str(item.get("wikipedia_summary_en") or item.get("wikipedia_summary_ru") or "")
        
        if not name or not description:
            continue
        
        image_names = item.get("image_path", [])
        for img_idx, img_name in enumerate(image_names):
            full_img_path = str(Path(config.images_dir) / img_name)
            if Path(full_img_path).exists():
                image_batch_data.append((full_img_path, name, description, obj_idx, img_idx))
    
    logger.info(f"Всего изображений для валидации: {len(image_batch_data)}")
    
    # Инициализация фильтров
    clip_filter = CLIPPreFilter(config) if config.use_clip_prefilter else None
    validator = Qwen2VLValidator(config)
    
    # Результаты валидации
    validation_results = {}  # (obj_idx, img_idx) -> (is_valid, confidence, reason)
    
    # Обработка батчами
    total_batches = (len(image_batch_data) + config.batch_size - 1) // config.batch_size
    
    with tqdm(total=len(image_batch_data), desc="Валидация изображений") as pbar:
        for batch_idx in range(0, len(image_batch_data), config.batch_size):
            batch = image_batch_data[batch_idx:batch_idx + config.batch_size]
            
            # CLIP pre-filter
            if clip_filter:
                clip_paths = [item[0] for item in batch]
                clip_results = clip_filter.check_images_batch(clip_paths)
                
                # Фильтруем через CLIP
                vlm_batch = []
                for (img_path, name, desc, obj_idx, img_idx), (clip_valid, clip_reason) in zip(batch, clip_results):
                    if not clip_valid:
                        validation_results[(obj_idx, img_idx)] = (False, "low", clip_reason)
                    else:
                        vlm_batch.append((img_path, name, desc, obj_idx, img_idx))
            else:
                vlm_batch = batch
            
            # VLM валидация
            if vlm_batch:
                vlm_input = [(item[0], item[1], item[2]) for item in vlm_batch]
                vlm_results = validator.validate_images_batch(vlm_input)
                
                for (_, _, _, obj_idx, img_idx), (is_valid, confidence) in zip(vlm_batch, vlm_results):
                    reason = "vlm_rejected" if not is_valid else "valid"
                    validation_results[(obj_idx, img_idx)] = (is_valid, confidence, reason)
            
            pbar.update(len(batch))
            
            # Периодическое сохранение
            if (batch_idx + config.batch_size) % config.save_interval == 0:
                validator.cache.flush()
                logger.info(f"Обработано {batch_idx + len(batch)}/{len(image_batch_data)} изображений")
    
    # Финальный сброс кэша
    validator.cache.flush()
    
    # Сохранение результатов
    output_data = []
    for obj_idx, item in enumerate(data):
        valid_images = []
        invalid_images = []
        
        image_names = item.get("image_path", [])
        for img_idx, img_name in enumerate(image_names):
            result = validation_results.get((obj_idx, img_idx))
            if result:
                is_valid, confidence, reason = result
                if is_valid:
                    valid_images.append({
                        "path": img_name,
                        "confidence": confidence
                    })
                else:
                    invalid_images.append({
                        "path": img_name,
                        "reason": reason,
                        "confidence": confidence
                    })
        
        if valid_images:
            result_item = item.copy()
            if "image_path" in result_item:
                del result_item["image_path"]
            result_item["valid_images"] = valid_images
            result_item["invalid_images"] = invalid_images
            output_data.append(result_item)
    
    # Сохранение
    with open(config.output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info("=" * 60)
    logger.info("ШАГ 4a: ВАЛИДАЦИЯ ИЗОБРАЖЕНИЙ ЗАВЕРШЕНА")
    logger.info("=" * 60)
    logger.info(f"Всего объектов: {len(data)}")
    logger.info(f"Объектов с валидными изображениями: {len(output_data)}")
    logger.info(f"Результат сохранен в: {config.output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
