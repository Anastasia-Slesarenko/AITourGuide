#!/usr/bin/env python3
"""
Шаг 1: Текстовая фильтрация датасета достопримечательностей.
Использует Yandex GPT для классификации объектов.

Требует переменные окружения:
    YC_IAM_TOKEN или YC_API_KEY — токен для доступа к Yandex AI Studio
    YC_FOLDER_ID                 — идентификатор каталога в Yandex Cloud
"""

import asyncio
import aiohttp
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from tqdm import tqdm
from dotenv import load_dotenv
import os
from dataclasses import dataclass, field

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
    """Конфигурация для текстовой фильтрации."""
    # Пути к файлам
    data_path: str = "setup_data_v3/data/landmarks_data_wiki_with_img.json"
    output_path: str = "setup_data_v3/data/text_filtered_landmarks.json"
    
    # Кэш и прогресс
    text_cache_path: str = "setup_data_v3/data/cache/text_classification.json"
    progress_path: str = "setup_data_v3/data/cache/text_processed_landmarks.txt"
    
    # API Yandex
    yc_iam_token: str = field(
        default_factory=lambda: os.getenv("YC_IAM_TOKEN", "")
    )
    yc_api_key: str = field(
        default_factory=lambda: os.getenv("YC_API_KEY", "")
    )
    yc_folder_id: str = field(
        default_factory=lambda: os.getenv("YC_FOLDER_ID", "")
    )
    
    # Модель
    text_model_uri: str = field(init=False)
    
    # Настройки параллелизма
    max_concurrent: int = 10
    rate_limit_per_sec: int = 10
    
    # Настройки retry
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Интервал сброса кэша
    cache_flush_interval: int = 50
    
    def __post_init__(self):
        if not self.yc_folder_id:
            raise RuntimeError(
                "Переменная окружения YC_FOLDER_ID должна быть установлена"
            )
        
        if not self.yc_iam_token and not self.yc_api_key:
            raise RuntimeError(
                "Должна быть установлена переменная окружения "
                "YC_IAM_TOKEN или YC_API_KEY"
            )
        
        self.text_model_uri = f"gpt://{self.yc_folder_id}/yandexgpt-5-lite"
        
        # Логирование используемого метода аутентификации
        if self.yc_iam_token:
            logger.info("Используется IAM токен для аутентификации")
        else:
            logger.info("Используется API ключ для аутентификации")
        
        # Проверка существования входного файла
        if not Path(self.data_path).exists():
            raise FileNotFoundError(
                f"Входной файл не найден: {self.data_path}"
            )
        
        # Создание директорий
        for path in [self.text_cache_path, self.progress_path]:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)


# ======================
# МЕТРИКИ
# ======================
class Metrics:
    """Отслеживание метрик обработки."""
    def __init__(self):
        self.api_calls = 0
        self.api_errors = 0
        self.cache_hits = 0
        self.objects_processed = 0
        self.objects_passed = 0
        self.start_time = time.time()
    
    def report(self):
        """Генерация отчета по метрикам."""
        elapsed = time.time() - self.start_time
        total = self.cache_hits + self.api_calls
        
        return {
            "elapsed_time_sec": round(elapsed, 2),
            "objects_processed": self.objects_processed,
            "objects_passed": self.objects_passed,
            "api_calls": self.api_calls,
            "api_errors": self.api_errors,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(
                self.cache_hits / max(1, total) * 100, 2
            ),
        }


# ======================
# КЭШИРОВАНИЕ
# ======================
class TextCache:
    """Кэш результатов текстовой классификации."""
    def __init__(self, path: Path, flush_interval: int = 50):
        self.path = path
        self.flush_interval = flush_interval
        self.data: Dict[str, bool] = {}
        self.pending_writes = 0
        
        if self.path.exists():
            with open(self.path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        logger.info(
            f"Загружено {len(self.data)} записей из кэша"
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
# ТЕКСТОВЫЙ ФИЛЬТР
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
            self.metrics.cache_hits += 1
            return cached
        
        prompt = f"""Определи, является ли объект достопримечательностью.

Достопримечательность: здание, сооружение, памятник, мост, башня, храм, музей, статуя.

НЕ подходит: территории, люди, абстракции, бытовые объекты, еда, животные, парки.

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
        
        # Выбор метода аутентификации
        if self.config.yc_iam_token:
            headers = {
                "Authorization": f"Bearer {self.config.yc_iam_token}",
                "Content-Type": "application/json",
                "x-folder-id": self.config.yc_folder_id
            }
        else:
            headers = {
                "Authorization": f"Api-Key {self.config.yc_api_key}",
                "Content-Type": "application/json"
            }
        
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        
        for attempt in range(self.config.max_retries):
            try:
                self.metrics.api_calls += 1
                async with self.session.post(
                    url, json=body, headers=headers
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.warning(
                            f"Ошибка API (попытка {attempt + 1}/"
                            f"{self.config.max_retries}): "
                            f"{resp.status} - {text}"
                        )
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(
                                self.config.retry_delay * (2 ** attempt)
                            )
                            continue
                        self.metrics.api_errors += 1
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
                    f"Исключение API (попытка {attempt + 1}/"
                    f"{self.config.max_retries}): {e}"
                )
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(
                        self.config.retry_delay * (2 ** attempt)
                    )
                    continue
                self.metrics.api_errors += 1
                return False
        
        return False


# ======================
# ОСНОВНОЙ КЛАСС
# ======================
class TextFilterProcessor:
    """Процессор текстовой фильтрации."""
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
        rate_limiter: RateLimiter
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
            logger.info(
                f"Пропуск объекта без имени/описания: {obj_id}"
            )
            self.progress.mark_processed(obj_id)
            return None
        
        # Текстовая фильтрация
        await rate_limiter.acquire()
        if not await text_filter.is_landmark(name, description):
            logger.info(f"Фильтр отклонил: {name}")
            self.progress.mark_processed(obj_id)
            return None
        
        self.progress.mark_processed(obj_id)
        self.metrics.objects_passed += 1
        
        # Возвращаем объект со всеми данными
        return item
    
    async def run(self):
        """Запуск процесса фильтрации."""
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
        connector = aiohttp.TCPConnector(limit=self.config.max_concurrent)
        rate_limiter = RateLimiter(self.config.rate_limit_per_sec)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            text_filter = YandexTextFilter(
                self.config, session, self.metrics
            )
            
            filtered_data = []
            
            # Обработка с прогресс-баром
            with tqdm(
                total=len(to_process),
                desc="Текстовая фильтрация"
            ) as pbar:
                # Обработка батчами
                batch_size = 100
                for i in range(0, len(to_process), batch_size):
                    batch = to_process[i:i + batch_size]
                    
                    tasks = []
                    for item in batch:
                        task = self.process_one(
                            item,
                            text_filter,
                            rate_limiter
                        )
                        tasks.append(task)
                    
                    results = await asyncio.gather(
                        *tasks, return_exceptions=True
                    )
                    
                    for result in results:
                        if isinstance(result, Exception):
                            logger.error(f"Ошибка: {result}")
                            self.metrics.objects_processed += 1
                        elif result is not None:
                            filtered_data.append(result)
                            self.metrics.objects_processed += 1
                        else:
                            self.metrics.objects_processed += 1
                        
                        pbar.update(1)
                    
                    # Периодический сброс кэшей
                    text_filter.cache.flush()
                    self.progress.flush()
                    
                    # Сохранение промежуточных результатов
                    if filtered_data:
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
            text_filter.cache.flush()
            self.progress.flush()
            
            # Финальное сохранение
            with open(
                self.config.output_path, 'w', encoding='utf-8'
            ) as f:
                json.dump(filtered_data, f, ensure_ascii=False, indent=2)
            
            # Отчет по метрикам
            metrics_report = self.metrics.report()
            logger.info("=" * 60)
            logger.info("ШАГ 1: ТЕКСТОВАЯ ФИЛЬТРАЦИЯ ЗАВЕРШЕНА")
            logger.info("=" * 60)
            logger.info(
                f"Всего обработано: {metrics_report['objects_processed']}"
            )
            logger.info(
                f"Прошло фильтрацию: {metrics_report['objects_passed']}"
            )
            logger.info(
                f"Время выполнения: {metrics_report['elapsed_time_sec']}с"
            )
            logger.info(
                f"API вызовов: {metrics_report['api_calls']} "
                f"(ошибок: {metrics_report['api_errors']})"
            )
            logger.info(
                f"Cache hit rate: {metrics_report['cache_hit_rate']}%"
            )
            logger.info(
                f"Результат сохранен в: {self.config.output_path}"
            )
            logger.info("=" * 60)
            
            # Оценка стоимости
            cost = metrics_report['api_calls'] * 140 / 1000 * 0.4
            logger.info(f"Примерная стоимость: {cost:.2f} ₽")


# ======================
# ТОЧКА ВХОДА
# ======================
async def main():
    """Главная точка входа."""
    try:
        config = Config()
        processor = TextFilterProcessor(config)
        await processor.run()
    except KeyboardInterrupt:
        logger.info("Прервано пользователем, сохранение прогресса...")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
