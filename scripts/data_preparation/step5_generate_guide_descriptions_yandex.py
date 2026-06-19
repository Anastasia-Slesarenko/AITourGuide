#!/usr/bin/env python3
"""
Шаг 4.5: Генерация описаний в стиле гида для достопримечательностей.
Использует Yandex AliceAI LLM для создания живых, увлекательных описаний на основе
информации из Википедии и метаданных.

Требования:
    - Результат работы step4_image_filter.py (clean_landmarks.json)
    - aiohttp, asyncio, python-dotenv, tqdm
    - API ключ Yandex AI Studio (переменная окружения YC_API_KEY или YANDEX_API_KEY)
    - Идентификатор каталога Yandex Cloud (переменная окружения YC_FOLDER_ID или YANDEX_FOLDER_ID)

Выходной формат:
    - Сохраняет все поля из clean_landmarks.json
    - Добавляет поле 'guide_description' — описание в стиле гида (2-3 абзаца)
    
Особенности:
    - Асинхронная обработка с использованием aiohttp
    - Возможность остановки и продолжения с того же места
    - Автоматическое сохранение прогресса после каждого успешного запроса
    - Автоматическая загрузка переменных из .env файла
    - Прогресс-бар с детальной статистикой
"""

import json
import logging
import time
import tempfile
import shutil
import re
import os
import asyncio
import signal
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import aiohttp
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

# Загрузка переменных окружения из .env файла
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
    input_path: str = "/Users/anastasiya/Documents/AITourGuide/data/processed/summarise_caption_landmarks.json"
    output_path: str = "/Users/anastasiya/Documents/AITourGuide/data/processed/landmarks_with_guide_descriptions.json"
    
    # Yandex AI Studio API
    api_key: str = field(default_factory=lambda: os.getenv("YC_API_KEY", ""))
    folder_id: str = field(default_factory=lambda: os.getenv("YC_FOLDER_ID", ""))
    model_uri: str = field(init=False)
    api_url: str = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    
    # Параметры генерации
    max_tokens: int = 500
    temperature: float = 0.4
    
    # Асинхронная обработка
    concurrent_requests: int = 5  # Количество одновременных запросов (уменьшено для стабильности)
    
    # Rate limiting
    request_delay: float = 4.0  # Задержка между запросами в секундах (увеличено)
    retry_attempts: int = 5  # Количество попыток при ошибке (увеличено)
    retry_delay: float = 10.0  # Задержка между попытками (увеличено)
    
    def __post_init__(self):
        # Формируем model_uri
        if not self.folder_id:
            raise ValueError(
                "YC_FOLDER_ID не установлен. "
                "Установите переменную окружения YC_FOLDER_ID или YANDEX_FOLDER_ID в файле .env"
            )
        
        self.model_uri = f"gpt://{self.folder_id}/aliceai-llm"
        
        logger.info(f"Используется модель: {self.model_uri}")
        logger.info(f"API URL: {self.api_url}")
        
        # Проверка API ключа
        if not self.api_key:
            raise ValueError(
                "YC_API_KEY не установлен. "
                "Установите переменную окружения YC_API_KEY или YANDEX_API_KEY в файле .env"
            )
        
        # Проверка существования входного файла
        if not Path(self.input_path).exists():
            raise FileNotFoundError(
                f"Входной файл не найден: {self.input_path}\n"
                f"Сначала запустите step4_image_filter.py"
            )
        
        # Создание директорий
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)


# ======================
# МЕТРИКИ
# ======================
class Metrics:
    """Отслеживание метрик обработки."""
    def __init__(self):
        self.objects_processed = 0
        self.objects_success = 0
        self.objects_failed = 0
        self.objects_fallback = 0  # Количество fallback на Wikipedia
        self.generation_calls = 0
        self.api_errors = 0
        self.start_time = time.time()
    
    def report(self):
        """Генерация отчета по метрикам."""
        elapsed = time.time() - self.start_time
        
        return {
            "elapsed_time_sec": round(elapsed, 2),
            "objects_processed": self.objects_processed,
            "objects_success": self.objects_success,
            "objects_failed": self.objects_failed,
            "objects_fallback": self.objects_fallback,
            "generation_calls": self.generation_calls,
            "api_errors": self.api_errors,
        }


# ======================
# ГЕНЕРАТОР ОПИСАНИЙ
# ======================
class GuideDescriptionGenerator:
    """Генерация описаний в стиле гида с использованием Yandex AliceAI LLM."""
    
    def __init__(
        self,
        config: Config,
        metrics: Metrics
    ):
        self.config = config
        self.metrics = metrics
        self.semaphore = asyncio.Semaphore(config.concurrent_requests)
        self.last_request_time = 0
        self.request_lock = asyncio.Lock()
        
        # Заголовки для API запросов
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {config.api_key}",
            "Connection": "keep-alive"
        }
        
        # Настройки для ClientSession
        self.connector_settings = {
            "limit": 100,
            "limit_per_host": 10,
            "ttl_dns_cache": 300,
            "keepalive_timeout": 300,  # 5 минут keep-alive
            "force_close": False,
            "enable_cleanup_closed": True
        }
        
        logger.info("Генератор описаний инициализирован")
    
    async def _rate_limit(self):
        """Контроль частоты запросов к API."""
        async with self.request_lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.config.request_delay:
                sleep_time = self.config.request_delay - elapsed
                await asyncio.sleep(sleep_time)
            self.last_request_time = time.time()
    
    def _build_structured_summary(self, item: Dict[str, Any]) -> str:
        """
        Создает компактное структурированное summary для генерации аудиогида.
        """
        name = item.get("name_ru") or item.get("name_en") or ""
        location = ", ".join([
            x for x in [
                item.get("city_ru") or item.get("city_en"),
                item.get("country_ru") or item.get("country_en")
            ] if x
        ])

        year = item.get("year_built")
        
        # Тип достопримечательности
        landmark_type = ""
        if "landmark_type" in item and isinstance(item["landmark_type"], dict):
            landmark_type = item["landmark_type"].get("ru") or item["landmark_type"].get("en") or ""
        
        # Краткое описание из Wikidata
        wikidata_description = item.get("wikidata_description_ru") or item.get("wikidata_description_en") or ""

        wiki = item.get("wikipedia_summary_ru") or item.get("wikipedia_summary_en") or ""
        wiki = wiki.strip().replace("\n", " ")

        # Ограничиваем контекст
        wiki = wiki[:800]

        parts = []

        if name:
            parts.append(f"Название: {name}")

        if location:
            parts.append(f"Местоположение: {location}")

        if year:
            parts.append(f"Период постройки: {year}")

        if landmark_type:
            parts.append(f"Тип: {landmark_type}")
        
        if wikidata_description:
            parts.append(f"Описание: {wikidata_description}")

        if wiki:
            parts.append(f"Факты:\n{wiki}")

        return "\n".join(parts)
    
    def _prepare_prompt(self, item: Dict[str, Any]) -> str:
        """Подготовка промпта для генерации аудиогида."""
        structured_summary = self._build_structured_summary(item)
        prompt = f"""Ты — профессиональный экскурсовод и автор текстов для путеводителей.

Твоя задача — создать универсальный текст о достопримечательности, который одинаково хорошо:
- звучит как живая речь гида на месте
- и читается как текст для аудиогида или путеводителя

────────────────────────
 ГЛАВНАЯ ЦЕЛЬ
────────────────────────
Сделать информативный, живой и связный рассказ о месте без сухого энциклопедического стиля и без излишней театральности.

────────────────────────
 СТИЛЬ
────────────────────────
- Пиши ТОЛЬКО на русском языке
- Естественный, уверенный тон экскурсовода
- Живой, но не разговорный “наигранный” стиль
- Без пафоса и рекламных формулировок
- Без канцелярита и энциклопедических определений

────────────────────────
 ЗАПРЕЩЕНО
────────────────────────
- Пересказ Wikipedia или базы данных в лоб
- Списки фактов
- Фразы типа: "перед вами", "добро пожаловать", "здесь находится"
- Избыточная эмоциональность ("потрясающий", "великолепный", "уникальный")
- Выдумывание фактов

────────────────────────
 ПРИНЦИП ПЕРЕРАБОТКИ ИНФОРМАЦИИ
────────────────────────
Любые данные должны быть:
- объединены в связный рассказ
- переформулированы своими словами
- встроены в контекст истории места

────────────────────────
 СТРУКТУРА (ГИБКАЯ, НО ОБЯЗАТЕЛЬНАЯ ЛОГИКА)
────────────────────────
Текст должен содержать 3 смысловых блока:

1. Контекст и первое впечатление (1–2 предложения)
   — что это за место и в каком оно окружении

2. История и ключевые факты (2–4 предложения)
   — происхождение, развитие, важные события, особенности

3. Значение и восприятие сегодня (1–2 предложения)
   — роль места сейчас, чем оно интересно

ВАЖНО:
- это не заголовки и не разделы — это единый связный текст

────────────────────────
 ОГРАНИЧЕНИЯ
────────────────────────
- 60–120 слов
- максимум 2–3 ключевых факта на один текст
- каждое предложение = одна мысль
- без перегруженных перечислений

────────────────────────
 ДАННЫЕ
────────────────────────
{structured_summary}

────────────────────────
 НАЧИНАЙ СРАЗУ С ТЕКСТА:
"""



        return prompt
    
    def _parse_response(self, text: str) -> Optional[str]:
        """Парсинг и постобработка ответа модели."""
        # Базовая очистка
        description_text = text.strip()
        
        # Проверка на китайские символы (CJK Unified Ideographs)
        chinese_chars = len([c for c in description_text if '\u4e00' <= c <= '\u9fff'])
        if chinese_chars > 0:
            logger.warning(f"WARNING - Обнаружены китайские символы: {chinese_chars} символов")
            return None
        
        # Проверка на нерусский текст
        russian_chars = len([c for c in description_text if '\u0400' <= c <= '\u04FF'])
        total_chars = len([c for c in description_text if c.isalpha()])
        if total_chars > 0 and russian_chars / total_chars < 0.6:
            logger.warning(f"WARNING - Текст содержит слишком много нерусских символов: {russian_chars}/{total_chars}")
            return None
        
        # Проверка на запрещенные фразы и клише
        forbidden = [
            "добро пожаловать", 
            "аудио гид по", "я могу предложить",
            "请忽略", "根据提供", "以下是"
           
        ]
        for phrase in forbidden:
            if phrase.lower() in description_text.lower():
                logger.warning(f"WARNING - Найдена запрещенная фраза: '{phrase}'")
                return None
        
        # Убираем маркеры списков
        description_text = re.sub(r'^[•→\-\*]\s*', '', description_text, flags=re.MULTILINE)
        
        # Постобработка: нормализация пробелов
        description_text = re.sub(r"\s+", " ", description_text).strip()
        
        # Удаляем мусор в начале
        description_text = re.sub(r"^[\.\,\-\:]+", "", description_text).strip()
        
        # Удаляем оборванный хвост - обрезаем до последней точки
        last_dot = description_text.rfind(".")
        if last_dot != -1:
            description_text = description_text[:last_dot + 1]
        
        # Валидация длины описания
        word_count = len(description_text.split())
        if word_count < 30:
            logger.warning(f"Описание короткое: {word_count} слов, но принимаем")
        if word_count > 300:
            logger.warning(f"Описание слишком длинное: {word_count} слов (обрезаем)")
            words = description_text.split()
            description_text = ' '.join(words[:300])
            last_dot = description_text.rfind(".")
            if last_dot != -1:
                description_text = description_text[:last_dot + 1]
        
        if not description_text:
            logger.warning("WARNING - Пустое описание после постобработки")
            return None
        
        return description_text
    
    async def _call_api(self, session: aiohttp.ClientSession, prompt: str, attempt: int = 1) -> Optional[str]:
        """Вызов Yandex AI Studio API для генерации текста."""
        await self._rate_limit()
        
        payload = {
            "modelUri": self.config.model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": self.config.temperature,
                "maxTokens": str(self.config.max_tokens)
            },
            "messages": [
                {
                    "role": "system",
                    "text": "Ты талантливый автор аудиогидов, который умеет рассказывать истории живо и увлекательно. Твоя задача - заинтересовать туриста, а не просто перечислить факты. Пиши ТОЛЬКО на русском языке. Используй разговорный, но грамотный стиль. Никогда не переключайся на другие языки."
                },
                {
                    "role": "user",
                    "text": prompt
                }
            ]
        }
        
        try:
            async with self.semaphore:
                # Увеличенные таймауты: connect=30s, total=120s
                timeout = aiohttp.ClientTimeout(
                    total=120,
                    connect=30,
                    sock_connect=30,
                    sock_read=90
                )
                
                async with session.post(
                    self.config.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Извлекаем текст из ответа
                        if "result" in result and "alternatives" in result["result"]:
                            alternatives = result["result"]["alternatives"]
                            if alternatives and len(alternatives) > 0:
                                message = alternatives[0].get("message", {})
                                text = message.get("text", "")
                                return text
                        
                        logger.error(f"Неожиданный формат ответа API: {result}")
                        return None
                    else:
                        error_text = await response.text()
                        logger.error(f"API ошибка {response.status}: {error_text}")
                        self.metrics.api_errors += 1
                        
                        # Retry logic
                        if attempt < self.config.retry_attempts:
                            logger.info(f"Повторная попытка {attempt + 1}/{self.config.retry_attempts}")
                            await asyncio.sleep(self.config.retry_delay)
                            return await self._call_api(session, prompt, attempt + 1)
                        
                        return None
                        
        except asyncio.TimeoutError:
            logger.error(f"Timeout при запросе к API (попытка {attempt}/{self.config.retry_attempts})")
            self.metrics.api_errors += 1
            
            # Retry logic
            if attempt < self.config.retry_attempts:
                logger.info(f"Повторная попытка {attempt + 1}/{self.config.retry_attempts} через {self.config.retry_delay}с")
                await asyncio.sleep(self.config.retry_delay)
                return await self._call_api(session, prompt, attempt + 1)
            
            logger.error("Исчерпаны все попытки после timeout")
            return None
        except aiohttp.ServerDisconnectedError as e:
            logger.warning(f"Сервер отключился (попытка {attempt}/{self.config.retry_attempts}): {e}")
            self.metrics.api_errors += 1
            
            # Retry logic с увеличенной задержкой
            if attempt < self.config.retry_attempts:
                delay = self.config.retry_delay * (attempt + 1)  # Экспоненциальная задержка
                logger.info(f"Повторная попытка {attempt + 1}/{self.config.retry_attempts} через {delay}с")
                await asyncio.sleep(delay)
                return await self._call_api(session, prompt, attempt + 1)
            
            logger.error("Исчерпаны все попытки после ServerDisconnectedError")
            return None
        except aiohttp.ClientError as e:
            logger.warning(f"Ошибка клиента aiohttp (попытка {attempt}/{self.config.retry_attempts}): {type(e).__name__}: {e}")
            self.metrics.api_errors += 1
            
            # Retry logic
            if attempt < self.config.retry_attempts:
                logger.info(f"Повторная попытка {attempt + 1}/{self.config.retry_attempts} через {self.config.retry_delay}с")
                await asyncio.sleep(self.config.retry_delay)
                return await self._call_api(session, prompt, attempt + 1)
            
            logger.error(f"Исчерпаны все попытки после {type(e).__name__}")
            return None
        except (asyncio.CancelledError, KeyboardInterrupt):
            # Задача была отменена - не пробрасываем, просто возвращаем None
            logger.debug("Запрос к API был отменен")
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка при вызове API (попытка {attempt}/{self.config.retry_attempts}): {type(e).__name__}: {e}", exc_info=True)
            self.metrics.api_errors += 1
            
            # Retry logic только для неизвестных ошибок
            if attempt < self.config.retry_attempts:
                logger.info(f"Повторная попытка {attempt + 1}/{self.config.retry_attempts} через {self.config.retry_delay}с")
                await asyncio.sleep(self.config.retry_delay)
                return await self._call_api(session, prompt, attempt + 1)
            
            logger.error(f"Исчерпаны все попытки после {type(e).__name__}")
            return None
    
    async def generate_description(self, session: aiohttp.ClientSession, item: Dict[str, Any]) -> Optional[str]:
        """Генерация описания для одной достопримечательности."""
        # Подготовка промпта
        prompt = self._prepare_prompt(item)
        
        # Вызов API
        self.metrics.generation_calls += 1
        response_text = await self._call_api(session, prompt)
        
        if response_text is None:
            return None
        
        # Парсинг ответа
        result = self._parse_response(response_text)
        
        if result is None:
            name = item.get('name_ru') or item.get('name_en') or 'Неизвестная достопримечательность'
            logger.warning(f"WARNING - Не удалось сгенерировать описание для '{name}'")
            logger.debug(f"Ответ модели: {response_text[:500]}")
        
        return result


# ======================
# ОСНОВНОЙ КЛАСС
# ======================
class GuideDescriptionProcessor:
    """Процессор генерации описаний в стиле гида."""
    def __init__(self, config: Config):
        self.config = config
        self.metrics = Metrics()
        self.shutdown_event = asyncio.Event()
        self.save_lock = asyncio.Lock()
    
    async def _atomic_save(self, data: List[Dict[str, Any]], output_path: Path):
        """Атомарное сохранение данных с защитой от прерываний."""
        async with self.save_lock:
            temp_fd, temp_path = tempfile.mkstemp(
                dir=output_path.parent,
                prefix='.tmp_output_',
                suffix='.json'
            )
            try:
                # Выполняем блокирующую операцию записи в executor
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self._write_json_sync,
                    data,
                    temp_path
                )
                # Перемещение файла - быстрая операция
                shutil.move(temp_path, output_path)
            except Exception as e:
                logger.error(f"Ошибка атомарного сохранения: {e}")
                if Path(temp_path).exists():
                    try:
                        Path(temp_path).unlink()
                    except:
                        pass
                raise
    
    def _write_json_sync(self, data: List[Dict[str, Any]], path: str):
        """Синхронная запись JSON (для executor)."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    async def process_item(
        self,
        session: aiohttp.ClientSession,
        generator: GuideDescriptionGenerator,
        item: Dict[str, Any],
        existing_results: Dict[str, Dict[str, Any]]
    ) -> tuple[Optional[Dict[str, Any]], bool]:
        """Обработка одного элемента. Возвращает (результат, использован_fallback)."""
        # Проверяем, не было ли запроса на остановку
        if self.shutdown_event.is_set():
            return None, False
        
        landmark_id = str(item.get("landmark_id", ""))
        if not landmark_id:
            logger.warning("Объект без landmark_id, пропускаем")
            return None, False
        
        # Генерация описания
        description_text = await generator.generate_description(session, item)
        used_fallback = False
        
        if description_text is None:
            name = item.get('name_ru') or item.get('name_en') or 'Неизвестная достопримечательность'
            
            # ERROR FALLBACK: используем описание из Википедии
            wiki_summary = item.get("wikipedia_summary_ru") or item.get("wikipedia_summary_en") or ""
            if wiki_summary:
                sentences = wiki_summary.split('.')[:3]
                fallback_description = '. '.join(sentences).strip()
                if fallback_description and not fallback_description.endswith('.'):
                    fallback_description += '.'
                
                description_text = fallback_description
                used_fallback = True
                self.metrics.objects_fallback += 1
            else:
                logger.error(f"Нет данных для fallback: {name}")
                self.metrics.objects_failed += 1
                return None, False
        
        # Создаем копию объекта с новыми полями
        result = item.copy()
        result["guide_description"] = description_text
        
        # Обновляем existing_results сразу
        existing_results[landmark_id] = result
        self.metrics.objects_success += 1
        
        # Сохраняем после каждого успешного запроса
        all_results = list(existing_results.values())
        await self._atomic_save(all_results, Path(self.config.output_path))
        
        return result, used_fallback
    
    async def run(self):
        """Запуск процесса генерации описаний."""
        # Загрузка данных
        with open(self.config.input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(
            f"Загружено {len(data)} объектов из {self.config.input_path}"
        )
        
        # Загрузка существующих результатов
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
        
        # Инициализация генератора
        generator = GuideDescriptionGenerator(self.config, self.metrics)
        
        # Создаем connector с настройками keep-alive
        connector = aiohttp.TCPConnector(**generator.connector_settings)
        
        # Создаем timeout для всей сессии
        timeout = aiohttp.ClientTimeout(
            total=None,  # Без общего таймаута для сессии
            connect=30,
            sock_connect=30,
            sock_read=90
        )
        
        # Асинхронная обработка с прогресс-баром
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            connector_owner=True
        ) as session:
            tasks = []
            for item in to_process:
                task = asyncio.create_task(
                    self.process_item(session, generator, item, existing_results)
                )
                tasks.append(task)
            
            # Обработка с прогресс-баром
            with tqdm(
                total=len(tasks),
                desc="Генерация описаний",
                unit="объект",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            ) as pbar:
                try:
                    for coro in asyncio.as_completed(tasks):
                        try:
                            result, used_fallback = await coro
                            self.metrics.objects_processed += 1
                            
                            # Обновляем описание прогресс-бара с детальной статистикой
                            pbar.set_postfix({
                                'успешно': self.metrics.objects_success,
                                'fallback': self.metrics.objects_fallback,
                                'ошибок': self.metrics.objects_failed,
                                'API ошибок': self.metrics.api_errors
                            })
                            pbar.update(1)
                        
                        except asyncio.CancelledError as e:
                            # Отмена задачи - логируем и продолжаем
                            logger.warning(f"Задача отменена: {e}")
                            self.metrics.objects_failed += 1
                            self.metrics.objects_processed += 1
                            pbar.update(1)
                        
                        except asyncio.TimeoutError as e:
                            # Таймаут - это серьезнее, логируем подробнее
                            logger.error(f"Таймаут задачи: {e}")
                            self.metrics.objects_failed += 1
                            self.metrics.objects_processed += 1
                            pbar.update(1)
                            
                        except Exception as e:
                            logger.error(f"Ошибка обработки элемента: {type(e).__name__}: {e}", exc_info=True)
                            self.metrics.objects_failed += 1
                            self.metrics.objects_processed += 1
                            pbar.update(1)
                
                except KeyboardInterrupt:
                    # Получен Ctrl+C - устанавливаем флаг остановки
                    logger.info("\n\nПолучен сигнал прерывания (Ctrl+C)")
                    self.shutdown_event.set()
                    
                    # Отменяем все незавершенные задачи
                    logger.info("Отмена незавершенных задач...")
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    
                    # Ждем завершения всех задач с таймаутом
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*tasks, return_exceptions=True),
                            timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning("Таймаут при ожидании завершения задач")
                    
                    # Финальное сохранение
                    logger.info("Сохранение прогресса...")
                    all_results = list(existing_results.values())
                    await self._atomic_save(all_results, Path(self.config.output_path))
                    
                    raise  # Пробрасываем KeyboardInterrupt дальше
        
        # Финальное сохранение (если не было прерывания)
        all_results = list(existing_results.values())
        await self._atomic_save(all_results, Path(self.config.output_path))
        
        # Отчет по метрикам
        metrics_report = self.metrics.report()
        logger.info("=" * 70)
        logger.info("ШАГ 4.5: ГЕНЕРАЦИЯ ОПИСАНИЙ ГИДА ЗАВЕРШЕНА")
        logger.info("=" * 70)
        logger.info(
            f"Всего обработано: {metrics_report['objects_processed']}"
        )
        logger.info(
            f"✓ Успешно сгенерировано (AI): {metrics_report['objects_success'] - metrics_report['objects_fallback']}"
        )
        logger.info(
            f"⚠ Использован fallback (Wikipedia): {metrics_report['objects_fallback']}"
        )
        logger.info(
            f"✗ Ошибок (пропущено): {metrics_report['objects_failed']}"
        )
        logger.info("-" * 70)
        logger.info(
            f"Время выполнения: {metrics_report['elapsed_time_sec']}с"
        )
        logger.info(
            f"Вызовов API: {metrics_report['generation_calls']}"
        )
        logger.info(
            f"Ошибок API: {metrics_report['api_errors']}"
        )
        logger.info("-" * 70)
        
        # Статистика успешности
        total_success = metrics_report['objects_success']
        total_processed = metrics_report['objects_processed']
        if total_processed > 0:
            success_rate = (total_success / total_processed) * 100
            ai_rate = ((total_success - metrics_report['objects_fallback']) / total_processed) * 100
            logger.info(f"Общая успешность: {success_rate:.1f}%")
            logger.info(f"AI генерация: {ai_rate:.1f}%")
        
        logger.info("-" * 70)
        logger.info(
            f"Результат сохранен в: {self.config.output_path}"
        )
        logger.info("=" * 70)


# ======================
# ТОЧКА ВХОДА
# ======================
def main():
    """Главная точка входа."""
    # Настройка обработки сигналов
    loop = None
    
    def signal_handler(signum, frame):
        """Обработчик сигналов SIGINT и SIGTERM."""
        logger.info("\n\nПолучен сигнал завершения, останавливаем...")
        if loop and loop.is_running():
            # Останавливаем event loop
            for task in asyncio.all_tasks(loop):
                task.cancel()
    
    # Устанавливаем обработчики сигналов
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        config = Config()
        processor = GuideDescriptionProcessor(config)
        
        # Создаем и запускаем event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(processor.run())
        except KeyboardInterrupt:
            logger.info("\n\n⚠️  Прервано пользователем (Ctrl+C). Прогресс сохранен, можно продолжить позже.")
        finally:
            # Очистка
            try:
                # Отменяем все оставшиеся задачи
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                
                # Ждем завершения с таймаутом
                if pending:
                    loop.run_until_complete(
                        asyncio.wait_for(
                            asyncio.gather(*pending, return_exceptions=True),
                            timeout=3.0
                        )
                    )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            finally:
                loop.close()
        
    except asyncio.CancelledError:
        logger.info("\n\n⚠️  Задачи отменены. Прогресс сохранен, можно продолжить позже.")
        return
    except (aiohttp.ServerDisconnectedError, aiohttp.ClientError) as e:
        logger.error(f"\n\n❌ Ошибка соединения с API: {e}")
        logger.info("💾 Прогресс сохранен. Перезапустите скрипт для продолжения.")
        return
    except Exception as e:
        logger.error(f"\n\n❌ Критическая ошибка: {e}", exc_info=True)
        logger.info("💾 Прогресс сохранен. Проверьте ошибку и перезапустите скрипт.")
        raise

if __name__ == "__main__":
    main()
