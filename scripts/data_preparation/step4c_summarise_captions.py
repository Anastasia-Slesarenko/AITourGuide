"""
Шаг 4c: Суммаризация landmark_summary_caption для достопримечательности.
Использует Qwen2-7B-Instruct (LLM) для создания общего описания достопримечательности.

Этап 3: Генерация сводных описаний на основе captions
Входные данные: clean_landmarks.json из step4b

Преимущества:
    - Использует только LLM (не VLM)
    - Объединяет множественные captions в одно каноническое описание
    - Работает с уже сгенерированными captions
"""

import json
import logging
import time
import gc
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from dataclasses import dataclass
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ЛОГИРОВАНИЕ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# КОНСТАНТЫ
MAX_SUMMARY_CAPTIONS = 3

# КОНФИГУРАЦИЯ
@dataclass
class Config:
    """Конфигурация для генерации сводных описаний."""

    # Пути к файлам
    input_path: str = "/home/jupyter/s3/ai-tour-guide/clean_landmarks.json"
    output_path: str = "/home/jupyter/s3/ai-tour-guide/summarise_caption_landmarks.json"
    
    # Модель
    vlm_model_name: str = "Qwen/Qwen2-7B-Instruct"
    max_new_tokens: int = 80  # Для генерации описаний
    min_new_tokens: int = 40
    
    # Batching
    batch_size: int = 8  # Уменьшенный батч для экономии памяти
    
    def __post_init__(self):
        if not Path(self.input_path).exists():
            raise FileNotFoundError(f"Входной файл не найден: {self.input_path}")
        
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)


# LLM SUMMARIZER
class Qwen2Summarizer:
    """Суммаризация captions с использованием Qwen2-7B-Instruct."""
    def __init__(self, config: Config):
        self.config = config
        
        logger.info(f"Загрузка Qwen2-7B-Instruct: {config.vlm_model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.device == "cpu":
            logger.warning("GPU не обнаружен! Обработка на CPU будет очень медленной.")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            config.vlm_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.vlm_model_name,
            use_fast=True,
            padding_side='left'
        )
        
        # Устанавливаем pad_token если его нет
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Модель загружена на: {self.device}")
    
    def summarize_captions_batch(
        self,
        batch_data: List[str]  # [captions_text, ...]
    ) -> List[str]:
        """
        Батчевая суммаризация captions.
        
        Returns:
            List[summary] для каждого набора captions
        """
        if not batch_data:
            return []
        
        try:
            # Подготовка промптов для батча
            messages_batch = []
            for captions_text in batch_data:
                prompt = f"""Merge multiple captions describing the same landmark into one concise visual description.

Keep:
- distinctive architectural features
- geometry
- facade structure
- towers, domes, arches, columns
- roof shapes
- spatial relationships
- unique decorative elements

Remove:
- repetition
- vague phrases
- scene descriptions
- atmosphere
- assumptions
- historical interpretation
- building function

Rules:
- describe only visible details
- do not invent information
- do not use:
  "appears to be",
  "likely",
  "suggests",
  "beautiful",
  "historic",
  "modern aesthetic"

Write 2-4 concise sentences.

Captions:
{captions_text}

Canonical description:
"""
                
                messages_batch.append([{
                    "role": "user",
                    "content": prompt,
                }])
            
            # Обработка батча
            texts = []
            for messages in messages_batch:
                text = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                texts.append(text)
            
            inputs = self.tokenizer(
                texts,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=1024  # Уменьшаем max_length для экономии памяти
            )
            inputs = inputs.to(self.device)
            
            # Генерация
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    min_new_tokens=self.config.min_new_tokens,
                    use_cache=True,
                    repetition_penalty=1.12,
                    do_sample=False
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in
                zip(inputs.input_ids, generated_ids)
            ]
            output_texts = self.tokenizer.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            # Очистка памяти
            del texts, messages_batch, inputs, generated_ids, generated_ids_trimmed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return [text.strip() for text in output_texts]
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"OOM ошибка при суммаризации: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            return ["" for _ in batch_data]
        except Exception as e:
            logger.error(f"Ошибка суммаризации: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            return ["" for _ in batch_data]


# ПОДГОТОВКА ДАННЫХ
def prepare_captions_text(valid_images: List[Dict]) -> str:
    """Подготовка текста из captions для суммаризации."""
    all_captions = [str(img.get("caption", "")) for img in valid_images if img.get("caption")]
    
    if not all_captions:
        return ""
    
    # Убираем дубликаты
    unique_captions = []
    seen = set()
    for caption in all_captions:
        caption_lower = caption.lower().strip()
        if caption_lower and caption_lower not in seen:
            seen.add(caption_lower)
            unique_captions.append(caption)
    
    # Объединяем captions
    return "\n".join(f"- {cap}" for cap in unique_captions)


# ОСНОВНОЙ ПРОЦЕССОР
def main():
    """Главная точка входа."""
    config = Config()
    
    # Загрузка данных
    with open(config.input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Загружено {len(data)} объектов с captions")
    
    # Подготовка батчей для суммаризации
    batch_data = []  # [(obj_idx, captions_text, landmark_name), ...]
    
    for obj_idx, item in enumerate(data):
        name = str(item.get("name_en") or item.get("name_ru") or "")
        valid_images = item.get("valid_images", [])
        
        if valid_images:
            captions_text = prepare_captions_text(valid_images)
            if captions_text:
                batch_data.append((obj_idx, captions_text, name))
    
    logger.info(f"Всего объектов для суммаризации: {len(batch_data)}")
    
    # Инициализация суммаризатора
    summarizer = Qwen2Summarizer(config)
    
    # Результаты суммаризации
    summary_results = {}  # obj_idx -> summary
    
    # Обработка батчами
    with tqdm(total=len(batch_data), desc="Суммаризация captions") as pbar:
        for batch_idx in range(0, len(batch_data), config.batch_size):
            batch = batch_data[batch_idx:batch_idx + config.batch_size]
            
            # Подготовка входных данных для батча
            batch_captions = [item[1] for item in batch]
            
            # Суммаризация
            summaries = summarizer.summarize_captions_batch(batch_captions)
            
            # Сохранение результатов
            for (obj_idx, _, landmark_name), summary in zip(batch, summaries):
                if summary:
                    summary_results[obj_idx] = summary
                else:
                    # Fallback: берем первые N captions
                    valid_images = data[obj_idx].get("valid_images", [])
                    all_captions = [str(img.get("caption", "")) for img in valid_images if img.get("caption")]
                    if all_captions:
                        sorted_captions = sorted(set(all_captions), key=len, reverse=True)[:MAX_SUMMARY_CAPTIONS]
                        summary_results[obj_idx] = " ".join(sorted_captions)
                    else:
                        summary_results[obj_idx] = f"Визуальное описание {landmark_name}"
            
            pbar.update(len(batch))
            
            # Очистка памяти после каждого батча
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    # Формирование финальных данных
    output_data = []
    for obj_idx, item in enumerate(data):
        result_item = item.copy()
        
        # Добавляем сводное описание
        if obj_idx in summary_results:
            result_item["landmark_summary_caption"] = summary_results[obj_idx]
        else:
            # Fallback для объектов без captions
            name = str(item.get("name_en") or item.get("name_ru") or "")
            result_item["landmark_summary_caption"] = f"Визуальное описание {name}"
        
        output_data.append(result_item)
    
    # Сохранение результатов в output_path
    with open(config.output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info("=" * 60)
    logger.info("ШАГ 4c: СУММАРИЗАЦИЯ CAPTIONS ЗАВЕРШЕНА")
    logger.info("=" * 60)
    logger.info(f"Входных объектов: {len(data)}")
    logger.info(f"Обработано объектов: {len(summary_results)}")
    logger.info(f"Результат сохранен в: {config.output_path}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
