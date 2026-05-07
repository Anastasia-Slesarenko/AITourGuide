#!/usr/bin/env python3
"""
Шаг 4.5: Генерация описаний в стиле гида для достопримечательностей.
Использует Yandex GPT для создания живых, увлекательных описаний на основе
информации из Википедии и метаданных.

Требования:
    - Результат работы step4_image_filter.py (clean_landmarks.json)
    - YC_IAM_TOKEN или YC_API_KEY — токен для доступа к Yandex AI Studio
    - YC_FOLDER_ID — идентификатор каталога в Yandex Cloud

Выходной формат:
    - Сохраняет все поля из clean_landmarks.json
    - Добавляет поле 'guide_description' — описание в стиле гида (2-3 абзаца)
"""

import asyncio
import aiohttp
import json
import logging
import time
import tempfile
import shutil
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
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
    """Конфигурация для генерации описаний."""
    # Пути к файлам
    input_path: str = "setup_data_v3/data/clean_landmarks.json"
    output_path: str = "setup_data_v3/data/landmarks_with_guide_descriptions.json"
    
    # Кэш (только для оптимизации API вызовов)
    cache_path: str = "setup_data_v3/data/cache/guide_descriptions.json"
    
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
    model_uri: str = field(init=False)
    use_pro_model: bool = False  # True для yandexgpt, False для yandexgpt-lite
    
    # Настройки параллелизма
    max_concurrent: int = 5
    
    # Настройки retry
    max_retries: int = 3
    retry_delay: float = 2.0
    
    # Интервал сохранения
    save_interval: int = 10
    
    # Параметры генерации
    temperature: float = 0.7
    max_tokens: int = 1500
    
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
        
        # Выбор модели
        model_name = "yandexgpt" if self.use_pro_model else "yandexgpt-lite"
        self.model_uri = f"gpt://{self.yc_folder_id}/{model_name}"
        
        # Логирование используемого метода аутентификации
        if self.yc_iam_token:
            logger.info("Используется IAM токен для аутентификации")
        else:
            logger.info("Используется API ключ для аутентификации")
        
        logger.info(f"Используется модель: {model_name}")
        
        # Проверка существования входного файла
        if not Path(self.input_path).exists():
            raise FileNotFoundError(
                f"Входной файл не найден: {self.input_path}\n"
                f"Сначала запустите step4_image_filter.py"
            )
        
        # Создание директорий
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
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
        self.objects_success = 0
        self.start_time = time.time()
    
    def report(self):
        """Генерация отчета по метрикам."""
        elapsed = time.time() - self.start_time
        total = self.cache_hits + self.api_calls
        
        return {
            "elapsed_time_sec": round(elapsed, 2),
            "objects_processed": self.objects_processed,
            "objects_success": self.objects_success,
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
class DescriptionCache:
    """Кэш сгенерированных описаний (только для оптимизации API)."""
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()
        
        if self.path.exists():
            with open(self.path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        logger.info(
            f"Загружено {len(self.data)} записей из кэша описаний"
        )
    
    async def get(self, landmark_id: str) -> Optional[Dict[str, Any]]:
        async with self.lock:
            return self.data.get(landmark_id)
    
    async def set(self, landmark_id: str, value: Dict[str, Any]):
        async with self.lock:
            self.data[landmark_id] = value
            # Сохраняем кэш сразу (атомарно)
            await self._save_unsafe()
    
    async def _save_unsafe(self):
        """Внутренний метод записи без блокировки (вызывается внутри lock)."""
        # Atomic write: записываем во временный файл, затем переименовываем
        temp_fd, temp_path = tempfile.mkstemp(
            dir=self.path.parent,
            prefix='.tmp_cache_',
            suffix='.json'
        )
        try:
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            # Atomic rename
            shutil.move(temp_path, self.path)
        except Exception as e:
            logger.error(f"Ошибка записи кэша: {e}")
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise


# ======================
# ГЕНЕРАТОР ОПИСАНИЙ
# ======================
class GuideDescriptionGenerator:
    """Генерация описаний в стиле гида с использованием Yandex GPT."""
    
    PROMPT_TEMPLATE =PROMPT_TEMPLATE = """Ты — профессиональный экскурсовод.

Твоя задача — написать описание достопримечательности для туриста.

=====================
ИСХОДНЫЕ ДАННЫЕ
=====================
Название: {name}
Город: {city}
Стиль: {style}
Описание: {wiki_summary}

=====================
ПРАВИЛА
=====================

1. Используй только факты из блока "Описание"
2. Не добавляй новые факты, даже если они кажутся очевидными
3. Если информации мало — пиши нейтрально и обобщённо
4. Пиши живо, но без художественных выдумок

ВАЖНО:
Перед добавлением любого факта мысленно проверь:
"Это прямо есть в тексте?"
Если нет — НЕ используй

=====================
ЗАДАНИЕ
=====================

Напиши описание:
- 120–160 слов
- 2 абзаца
- простой и понятный язык
- без списков
- без повторов
- без фраз типа "это достопримечательность"

Первое предложение должно начинаться с конкретной детали из текста (не с общих слов).

=====================
ФОРМАТ ОТВЕТА
=====================

Верни только текст описания.

Если не можешь соблюсти правила — верни строку:
ERROR
"""

    
    def __init__(
        self,
        config: Config,
        session: aiohttp.ClientSession,
        metrics: Metrics
    ):
        self.config = config
        self.session = session
        self.metrics = metrics
        self.cache = DescriptionCache(Path(config.cache_path))
    
    def _prepare_input_data(self, item: Dict[str, Any]) -> Dict[str, str]:
        """Подготовка данных для промпта."""
        name = item.get("name_ru") or item.get("name_en") or "Неизвестная достопримечательность"
        city = item.get("city_ru") or item.get("city_en") or "неизвестный город"
        style = item.get("architectural_style_ru") or item.get("architectural_style_en") or "различные стили"
        
        # Берем полное описание из Википедии
        wiki_summary = item.get("wikipedia_summary_ru") or item.get("wikipedia_summary_en") or ""
        
        # Ограничиваем длину для промпта (первые 500 символов)
        if len(wiki_summary) > 500:
            wiki_summary = wiki_summary[:500] + "..."
        
        return {
            "name": name,
            "city": city,
            "style": style,
            "wiki_summary": wiki_summary
        }
    
    def _parse_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Парсинг ответа модели - просто берем весь текст как описание."""
        # Убираем возможные маркеры и лишние пробелы
        description_text = text.strip()
        
        # Проверка на ERROR от модели
        if description_text.upper().startswith('ERROR'):
            logger.warning("Модель вернула ERROR")
            return None
        
        # Убираем маркеры списков если они есть
        description_text = re.sub(r'^[•→\-\*]\s*', '', description_text, flags=re.MULTILINE)
        
        # Валидация длины описания
        word_count = len(description_text.split())
        if word_count < 50:
            logger.error(f"Описание слишком короткое: {word_count} слов")
            return None
        if word_count > 300:
            logger.warning(f"Описание слишком длинное: {word_count} слов (обрезаем)")
            # Обрезаем до 300 слов
            words = description_text.split()
            description_text = ' '.join(words[:300])
        
        if not description_text:
            logger.error("Пустое описание")
            return None
        
        return {"guide_description": description_text}
    
    async def generate_description(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Генерация описания для одной достопримечательности."""
        landmark_id = str(item.get("landmark_id", ""))
        
        # Проверка кэша (async)
        cached = await self.cache.get(landmark_id)
        if cached is not None:
            self.metrics.cache_hits += 1
            return cached
        
        # Подготовка данных
        input_data = self._prepare_input_data(item)
        prompt = self.PROMPT_TEMPLATE.format(**input_data)
        
        body = {
            "modelUri": self.config.model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": self.config.temperature,
                "maxTokens": self.config.max_tokens
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
                    url, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.warning(
                            f"Ошибка API для {input_data['name']} "
                            f"(попытка {attempt + 1}/{self.config.max_retries}): "
                            f"{resp.status} - {text}"
                        )
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(
                                self.config.retry_delay * (2 ** attempt)
                            )
                            continue
                        self.metrics.api_errors += 1
                        return None
                    
                    data = await resp.json()
                    response_text = data['result']['alternatives'][0]['message']['text']
                    
                    # Парсинг ответа с валидацией
                    result = self._parse_response(response_text)
                    
                    if result is None:
                        logger.error(f"Не удалось распарсить ответ для {input_data['name']}")
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                            continue
                        self.metrics.api_errors += 1
                        return None
                    
                    # Сохранение в кэш (async)
                    await self.cache.set(landmark_id, result)
                    
                    return result
                    
            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout для {input_data['name']} "
                    f"(попытка {attempt + 1}/{self.config.max_retries})"
                )
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    continue
                self.metrics.api_errors += 1
                return None
            except Exception as e:
                logger.warning(
                    f"Исключение API для {input_data['name']} "
                    f"(попытка {attempt + 1}/{self.config.max_retries}): {e}"
                )
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    continue
                self.metrics.api_errors += 1
                return None
        
        return None


# ======================
# ОСНОВНОЙ КЛАСС
# ======================
class GuideDescriptionProcessor:
    """Процессор генерации описаний в стиле гида."""
    def __init__(self, config: Config):
        self.config = config
        self.metrics = Metrics()
    
    def _atomic_save(self, data: List[Dict[str, Any]], output_path: Path):
        """Атомарное сохранение данных."""
        # Записываем во временный файл
        temp_fd, temp_path = tempfile.mkstemp(
            dir=output_path.parent,
            prefix='.tmp_output_',
            suffix='.json'
        )
        try:
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            # Atomic rename
            shutil.move(temp_path, output_path)
        except Exception as e:
            logger.error(f"Ошибка атомарного сохранения: {e}")
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    
    async def run(self):
        """Запуск процесса генерации описаний."""
        # Загрузка данных
        with open(self.config.input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(
            f"Загружено {len(data)} объектов из {self.config.input_path}"
        )
        
        # Загрузка существующих результатов (единственный источник истины)
        existing_results: Dict[str, Dict[str, Any]] = {}
        if Path(self.config.output_path).exists():
            try:
                with open(self.config.output_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    for item in existing_data:
                        landmark_id = str(item.get("landmark_id", ""))
                        if landmark_id:
                            existing_results[landmark_id] = item
                logger.info(f"Загружено {len(existing_results)} существующих результатов")
            except Exception as e:
                logger.warning(f"Не удалось загрузить существующие результаты: {e}")
        
        # Фильтрация: обрабатываем только те, которых нет в existing_results
        to_process = [
            item for item in data
            if str(item.get("landmark_id", "")) not in existing_results
        ]
        logger.info(
            f"Объектов к обработке: {len(to_process)} "
            f"(уже обработано: {len(data) - len(to_process)})"
        )
        
        if not to_process:
            logger.info("Все объекты уже обработаны")
            logger.info(f"Результаты доступны в: {self.config.output_path}")
            return
        
        # Настройка
        connector = aiohttp.TCPConnector(limit=self.config.max_concurrent)
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def process_with_semaphore(item: Dict[str, Any], generator: GuideDescriptionGenerator):
            """Обработка одного объекта с контролем параллелизма."""
            async with semaphore:
                landmark_id = str(item.get("landmark_id", ""))
                if not landmark_id:
                    logger.warning("Объект без landmark_id, пропускаем")
                    return None
                
                # Генерация описания
                description_data = await generator.generate_description(item)
                
                if description_data is None:
                    name = item.get('name_ru') or item.get('name_en') or 'Неизвестная достопримечательность'
                    logger.warning(f"Не удалось сгенерировать описание для {name}")
                    
                    # ERROR FALLBACK: используем описание из Википедии
                    wiki_summary = item.get("wikipedia_summary_ru") or item.get("wikipedia_summary_en") or ""
                    if wiki_summary:
                        # Берем первые 2-3 предложения (примерно 150 слов)
                        sentences = wiki_summary.split('.')[:3]
                        fallback_description = '. '.join(sentences).strip()
                        if fallback_description and not fallback_description.endswith('.'):
                            fallback_description += '.'
                        
                        description_data = {"guide_description": fallback_description}
                        logger.info(f"Использован fallback для {name}")
                    else:
                        # Если даже Википедии нет, пропускаем
                        logger.error(f"Нет данных для fallback: {name}")
                        return None
                
                # Создаем копию объекта с новыми полями
                result = item.copy()
                result.update(description_data)
                
                return result
        
        async with aiohttp.ClientSession(connector=connector) as session:
            generator = GuideDescriptionGenerator(
                self.config, session, self.metrics
            )
            
            # Обработка с прогресс-баром
            with tqdm(
                total=len(to_process),
                desc="Генерация описаний"
            ) as pbar:
                # Создаем задачи для всех объектов
                tasks = [
                    process_with_semaphore(item, generator)
                    for item in to_process
                ]
                
                # Обрабатываем по мере завершения
                for coro in asyncio.as_completed(tasks):
                    try:
                        result = await coro
                        
                        if result is not None:
                            # Обновляем existing_results сразу
                            landmark_id = str(result.get("landmark_id", ""))
                            if landmark_id:
                                existing_results[landmark_id] = result
                                self.metrics.objects_success += 1
                        
                        self.metrics.objects_processed += 1
                        pbar.update(1)
                        
                        # Периодическое сохранение
                        if self.metrics.objects_processed % self.config.save_interval == 0:
                            all_results = list(existing_results.values())
                            self._atomic_save(all_results, Path(self.config.output_path))
                            logger.info(f"Промежуточное сохранение: {len(all_results)} объектов")
                        
                    except Exception as e:
                        logger.error(f"Ошибка обработки: {e}")
                        self.metrics.objects_processed += 1
                        pbar.update(1)
            
            # Финальное сохранение
            all_results = list(existing_results.values())
            self._atomic_save(all_results, Path(self.config.output_path))
            
            # Отчет по метрикам
            metrics_report = self.metrics.report()
            logger.info("=" * 60)
            logger.info("ШАГ 4.5: ГЕНЕРАЦИЯ ОПИСАНИЙ ГИДА ЗАВЕРШЕНА")
            logger.info("=" * 60)
            logger.info(
                f"Всего обработано: {metrics_report['objects_processed']}"
            )
            logger.info(
                f"Успешно сгенерировано: {metrics_report['objects_success']}"
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
            if self.config.use_pro_model:
                # YandexGPT: 1₽ за 1000 токенов промпта + 2₽ за 1000 токенов ответа
                # Примерно 800 токенов промпт + 500 токенов ответ
                cost = metrics_report['api_calls'] * (800 * 1.0 / 1000 + 500 * 2.0 / 1000)
            else:
                # YandexGPT Lite: 0.4₽ за 1000 токенов промпта + 0.8₽ за 1000 токенов ответа
                cost = metrics_report['api_calls'] * (800 * 0.4 / 1000 + 500 * 0.8 / 1000)
            
            logger.info(f"Примерная стоимость: {cost:.2f} ₽")


# ======================
# ТОЧКА ВХОДА
# ======================
async def main():
    """Главная точка входа."""
    try:
        config = Config()
        processor = GuideDescriptionProcessor(config)
        await processor.run()
    except KeyboardInterrupt:
        logger.info("Прервано пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
