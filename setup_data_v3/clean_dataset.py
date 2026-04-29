#!/usr/bin/env python3
"""
Очистка датасета достопримечательностей.
Использует Yandex GPT для текстовой фильтрации и локальную Qwen2-VL модель для проверки изображений.

Требует переменные окружения:
    YC_API_KEY     — API ключ для доступа к Yandex AI Studio
    YC_FOLDER_ID   — идентификатор каталога в Yandex Cloud

Установка зависимостей:
    pip install aiohttp pillow tqdm python-dotenv transformers torch qwen-vl-utils

Запуск:
    python clean_dataset.py
"""

import asyncio
import aiohttp
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv
import os
from dataclasses import dataclass, field
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

load_dotenv()

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
    """Конфигурация для очистки датасета."""
    # Пути к файлам
    data_path: str = "setup_data_v3/landmarks_data_wiki.json"
    output_path: str = "setup_data_v3/data/clean_landmarks.json"
    
    # Параметры фильтрации
    min_images_per_landmark: int = 2
    
    # Кэш и прогресс
    text_cache_path: str = "setup_data_v3/cache/text_classification.json"
    image_cache_path: str = "setup_data_v3/cache/image_checks.jsonl"
    progress_path: str = "setup_data_v3/cache/processed_landmarks.txt"
    
    # API Yandex (только для текста)
    yc_api_key: str = field(
        default_factory=lambda: os.getenv("YC_API_KEY", "")
    )
    yc_folder_id: str = field(
        default_factory=lambda: os.getenv("YC_FOLDER_ID", "")
    )
    
    # Модели
    text_model_uri: str = field(init=False)
    vlm_model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    
    # Настройки параллелизма
    max_concurrent_text: int = 10
    rate_limit_per_sec: int = 10
    
    # Настройки retry
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Интервал сброса кэша
    cache_flush_interval: int = 50
    
    # Обработка изображений
    max_pixels: int = 512 * 512  # для Qwen2-VL
    
    def __post_init__(self):
        if not self.yc_api_key or not self.yc_folder_id:
            raise RuntimeError(
                "Переменные окружения YC_API_KEY и YC_FOLDER_ID "
                "должны быть установлены"
            )
        
        self.text_model_uri = f"gpt://{self.yc_folder_id}/yandexgpt-lite"
        
        # Проверка существования входного файла
        if not Path(self.data_path).exists():
            raise FileNotFoundError(
                f"Входной файл не найден: {self.data_path}"
            )
        
        # Создание директорий для кэша
        for path in [
            self.text_cache_path,
            self.image_cache_path,
            self.progress_path
        ]:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Создание выходной директории
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)


# ======================
# МЕТРИКИ
# ======================
class Metrics:
    """Отслеживание метрик обработки."""
    def __init__(self):
        self.text_api_calls = 0
        self.text_api_errors = 0
        self.text_cache_hits = 0
        self.image_checks = 0
        self.image_cache_hits = 0
        self.objects_processed = 0
        self.objects_passed = 0
        self.start_time = time.time()
    
    def report(self):
        """Генерация отчета по метрикам."""
        elapsed = time.time() - self.start_time
        text_total = self.text_cache_hits + self.text_api_calls
        image_total = self.image_cache_hits + self.image_checks
        
        return {
            "elapsed_time_sec": round(elapsed, 2),
            "objects_processed": self.objects_processed,
            "objects_passed": self.objects_passed,
            "text_api_calls": self.text_api_calls,
            "text_api_errors": self.text_api_errors,
            "text_cache_hits": self.text_cache_hits,
            "text_cache_hit_rate": round(
                self.text_cache_hits / max(1, text_total) * 100, 2
            ),
            "image_checks": self.image_checks,
            "image_cache_hits": self.image_cache_hits,
            "image_cache_hit_rate": round(
                self.image_cache_hits / max(1, image_total) * 100, 2
            ),
        }


# ======================
# КЭШИРОВАНИЕ
# ======================
class TextCache:
    """Кэш результатов текстовой классификации с батчевой записью."""
    def __init__(self, path: Path, flush_interval: int = 50):
        self.path = path
        self.flush_interval = flush_interval
        self.data: Dict[str, bool] = {}
        self.pending_writes = 0
        
        if self.path.exists():
            with open(self.path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        logger.info(
            f"Загружено {len(self.data)} записей из текстового кэша"
        )
    
    def get(self, name: str, description: str) -> Optional[bool]:
        key = f"{name}|{description}"
        return self.data.get(key)
    
    def set(self, name: str, description: str, value: bool):
        key = f"{name}|{description}"
        self.data[key] = value
        self.pending_writes += 1
        
        if self.pending_writes >= self.flush_interval:
            self.flush()
    
    def flush(self):
        """Принудительная запись на диск."""
        if self.pending_writes > 0:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            self.pending_writes = 0


class ImageCache:
    """Кэш результатов проверки изображений с батчевой записью."""
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
# RATE LIMITER
# ======================
class RateLimiter:
    """Ограничитель частоты запросов."""
    def __init__(self, rate_per_sec: int):
        self.rate = rate_per_sec
        self.interval = 1.0 / rate_per_sec
        self.last_time = 0
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        async with self.lock:
            now = time.time()
            wait = self.last_time + self.interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self.last_time = time.time()


# ======================
# ТЕКСТОВЫЙ ФИЛЬТР (Yandex GPT)
# ======================
class YandexTextFilter:
    """Текстовая классификация с использованием Yandex GPT."""
    def __init__(
        self,
        config: Config,
        session: aiohttp.ClientSession,
        metrics: Metrics
    ):
        self.config = config
        self.session = session
        self.metrics = metrics
        self.cache = TextCache(
            Path(config.text_cache_path),
            config.cache_flush_interval
        )
    
    async def is_landmark(self, name: str, description: str) -> bool:
        """Проверка, является ли объект достопримечательностью."""
        cached = self.cache.get(name, description)
        if cached is not None:
            self.metrics.text_cache_hits += 1
            return cached
        
        prompt = f"""Определи, является ли объект достопримечательностью.

Достопримечательность: здание, сооружение, памятник, мост, башня, храм, музей, парк, статуя, фонтан, историческое место.

НЕ подходит: территории, люди, абстракции, бытовые объекты, еда, животные.

Название: {name}
Описание: {description}

Ответ: только "Да" или "Нет"."""
        
        body = {
            "modelUri": self.config.text_model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": 0,
                "maxTokens": 10
            },
            "messages": [{"role": "user", "text": prompt}]
        }
        
        headers = {
            "Authorization": f"Api-Key {self.config.yc_api_key}",
            "Content-Type": "application/json"
        }
        
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        
        for attempt in range(self.config.max_retries):
            try:
                self.metrics.text_api_calls += 1
                async with self.session.post(
                    url, json=body, headers=headers
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.warning(
                            f"Ошибка Text API (попытка {attempt + 1}/"
                            f"{self.config.max_retries}): "
                            f"{resp.status} - {text}"
                        )
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(
                                self.config.retry_delay * (2 ** attempt)
                            )
                            continue
                        self.metrics.text_api_errors += 1
                        return False
                    
                    data = await resp.json()
                    answer = data['result']['alternatives'][0][
                        'message'
                    ]['text'].strip().lower()
                    is_valid = answer.startswith('да')
                    self.cache.set(name, description, is_valid)
                    return is_valid
            except Exception as e:
                logger.warning(
                    f"Исключение Text API (попытка {attempt + 1}/"
                    f"{self.config.max_retries}): {e}"
                )
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(
                        self.config.retry_delay * (2 ** attempt)
                    )
                    continue
                self.metrics.text_api_errors += 1
                return False
        
        return False


# ======================
# ФИЛЬТР ИЗОБРАЖЕНИЙ (Qwen2-VL)
# ======================
class Qwen2VLImageFilter:
    """Проверка изображений с использованием локальной Qwen2-VL модели."""
    def __init__(self, config: Config, metrics: Metrics):
        self.config = config
        self.metrics = metrics
        self.cache = ImageCache(
            Path(config.image_cache_path),
            config.cache_flush_interval
        )
        
        logger.info(f"Загрузка Qwen2-VL модели: {config.vlm_model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Загрузка модели с оптимизациями для T4
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.vlm_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(
            config.vlm_model_name,
            max_pixels=config.max_pixels
        )
        
        logger.info(f"Qwen2-VL модель загружена на устройство: {self.device}")
    
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
            # Формирование промпта для VLM
            prompt = f"""Проверь, соответствует ли изображение описанию достопримечательности.

Ожидаемый объект: {expected_name}
Описание: {expected_description}

Ответь ТОЛЬКО "ДА" если на изображении показана эта достопримечательность (или очень похожая).
Ответь ТОЛЬКО "НЕТ" если это:
- Другой объект
- Человек или группа людей
- Еда или напитки
- Карта или схема
- Текстовый документ
- Животное

Твой ответ (одно слово):"""
            
            # Подготовка сообщений для модели
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # Применение шаблона чата
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Обработка изображения и текста
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # Генерация ответа
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False
                )
            
            # Декодирование ответа
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
            
            logger.debug(
                f"Изображение {image_path}: {output_text} -> {is_valid}"
            )
            
            return is_valid
            
        except Exception as e:
            logger.error(
                f"Ошибка проверки изображения {image_path}: {e}"
            )
            self.cache.set(image_path, False)
            return False


# ======================
# ОСНОВНОЙ КЛАСС ОЧИСТКИ
# ======================
class DatasetCleaner:
    """Основной оркестратор очистки датасета."""
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
    
    async def process_one(
        self,
        item: Dict,
        text_filter: YandexTextFilter,
        image_filter: Qwen2VLImageFilter,
        rate_limiter: RateLimiter
    ) -> Optional[Dict]:
        """Обработка одного объекта достопримечательности."""
        obj_id = self._generate_obj_id(item)
        
        if self.progress.is_processed(obj_id):
            return None
        
        name = item.get("name_ru") or item.get("name_en")
        description = (
            item.get("wikipedia_summary_ru") or
            item.get("wikipedia_summary_en")
        )
        
        if not name or not description:
            logger.info(
                f"Пропуск объекта без имени/описания: {obj_id}"
            )
            self.progress.mark_processed(obj_id)
            return None
        
        # Текстовая фильтрация
        await rate_limiter.acquire()
        if not await text_filter.is_landmark(name, description):
            logger.info(f"Текстовый фильтр отклонил: {name}")
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
            
            # Qwen2-VL работает локально, rate limiter не нужен
            is_valid = image_filter.check_image(
                img_path, name, description
            )
            if is_valid:
                valid_images.append({
                    "path": img_path,
                    "caption": ""
                })
        
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
    
    async def run(self):
        """Запуск процесса очистки."""
        # Загрузка данных
        with open(self.config.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(
            f"Загружено {len(data)} объектов из {self.config.data_path}"
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
        
        # Настройка
        connector = aiohttp.TCPConnector(
            limit=self.config.max_concurrent_text
        )
        rate_limiter = RateLimiter(self.config.rate_limit_per_sec)
        
        # Инициализация Qwen2-VL фильтра (локально, вне async контекста)
        image_filter = Qwen2VLImageFilter(self.config, self.metrics)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            text_filter = YandexTextFilter(
                self.config, session, self.metrics
            )
            
            clean_data = []
            
            # Обработка с прогресс-баром
            with tqdm(
                total=len(to_process),
                desc="Обработка достопримечательностей"
            ) as pbar:
                # Обработка батчами для контроля памяти
                batch_size = 100
                for i in range(0, len(to_process), batch_size):
                    batch = to_process[i:i + batch_size]
                    
                    tasks = []
                    for item in batch:
                        task = self.process_one(
                            item,
                            text_filter,
                            image_filter,
                            rate_limiter
                        )
                        tasks.append(task)
                    
                    results = await asyncio.gather(
                        *tasks, return_exceptions=True
                    )
                    
                    for result in results:
                        if isinstance(result, Exception):
                            logger.error(
                                f"Ошибка обработки: {result}"
                            )
                            self.metrics.objects_processed += 1
                        elif result is not None:
                            clean_data.append(result)
                            self.metrics.objects_processed += 1
                        else:
                            self.metrics.objects_processed += 1
                        
                        pbar.update(1)
                    
                    # Периодический сброс кэшей
                    text_filter.cache.flush()
                    image_filter.cache.flush()
                    self.progress.flush()
                    
                    # Сохранение промежуточных результатов
                    if clean_data:
                        with open(
                            self.config.output_path,
                            'w',
                            encoding='utf-8'
                        ) as f:
                            json.dump(
                                clean_data,
                                f,
                                ensure_ascii=False,
                                indent=2
                            )
            
            # Финальный сброс
            text_filter.cache.flush()
            image_filter.cache.flush()
            self.progress.flush()
            
            # Финальное сохранение
            with open(
                self.config.output_path, 'w', encoding='utf-8'
            ) as f:
                json.dump(clean_data, f, ensure_ascii=False, indent=2)
            
            # Отчет по метрикам
            metrics_report = self.metrics.report()
            logger.info("=" * 60)
            logger.info("ОБРАБОТКА ЗАВЕРШЕНА")
            logger.info("=" * 60)
            logger.info(
                f"Всего обработано объектов: "
                f"{metrics_report['objects_processed']}"
            )
            logger.info(
                f"Объектов прошло фильтрацию: "
                f"{metrics_report['objects_passed']}"
            )
            logger.info(
                f"Время выполнения: "
                f"{metrics_report['elapsed_time_sec']}с"
            )
            logger.info(
                f"Text API вызовов: {metrics_report['text_api_calls']} "
                f"(ошибок: {metrics_report['text_api_errors']})"
            )
            logger.info(
                f"Text cache hit rate: "
                f"{metrics_report['text_cache_hit_rate']}%"
            )
            logger.info(
                f"Image проверок: {metrics_report['image_checks']}"
            )
            logger.info(
                f"Image cache hit rate: "
                f"{metrics_report['image_cache_hit_rate']}%"
            )
            logger.info(
                f"Результат сохранен в: {self.config.output_path}"
            )
            logger.info("=" * 60)
            
            # Оценка стоимости
            text_cost = (
                metrics_report['text_api_calls'] * 140 / 1000 * 0.4
            )
            logger.info(
                f"Примерная стоимость Text API: {text_cost:.2f} ₽"
            )


# ======================
# ТОЧКА ВХОДА
# ======================
async def main():
    """Главная точка входа."""
    try:
        config = Config()
        cleaner = DatasetCleaner(config)
        await cleaner.run()
    except KeyboardInterrupt:
        logger.info("Прервано пользователем, сохранение прогресса...")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
