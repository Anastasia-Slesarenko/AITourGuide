# services/ai_tour_guide.py
# -*- coding: utf-8 -*-
"""
AI Tour Guide — сервис для распознавания достопримечательностей.

Пайплайн:
  1. CLIP + FAISS Retrieval (top-10 кандидатов)
  2. VLM Генерация ответа (LoRA r=16 модель)
  3. Расчёт уверенности (улучшенная формула)
  4. Интернет-поиск при низкой уверенности (Yandex + Wikipedia)
  5. Переформулирование ответа в стиль гида
"""

import torch
import json
import os
import re
import time
import logging
import asyncio
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime
from dotenv import load_dotenv
from src.rag.retriever import RAGRetriever
from llama_cpp import Llama

# Импорты для интернет-поиска
from .yandex_search import YandexSearchService, WikipediaService
from .translator import YandexTranslator

logger = logging.getLogger(__name__)

load_dotenv()


class PredictionSource(Enum):
    """Источник предсказания."""
    RETRIEVAL = "retrieval"
    INTERNET = "internet"
    FALLBACK = "fallback"


@dataclass
class AITourGuideConfig:
    """Конфигурация для AITourGuide."""
    model_path: str
    index_path: str
    facts_db_path: str
    mmproj_path: str
    n_ctx: int = 32768
    device: str = "cuda"
    top_k_retrieval: int = 10
    confidence_threshold: float = 0.78
    max_new_tokens: int = 256
    enable_internet_search: bool = True
    
    # Параметры расчета confidence
    gap_multiplier: float = 10.0
    position_decay: float = 0.15
    confidence_weights: Dict[str, float] = field(default_factory=lambda: {
        "clip_score": 0.25,
        "gap": 0.20,
        "name_match": 0.35,
        "position": 0.20,
    })
    
    def to_dict(self) -> Dict:
        """Конвертация в словарь."""
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Метрики производительности сервиса."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_confidence: float = 0.0
    internet_search_rate: float = 0.0
    avg_retrieval_time: float = 0.0
    avg_generation_time: float = 0.0
    avg_total_time: float = 0.0
    last_updated: Optional[str] = None
    
    # Накопительные суммы для расчёта средних
    _sum_confidence: float = field(default=0.0, repr=False)
    _sum_retrieval_time: float = field(default=0.0, repr=False)
    _sum_generation_time: float = field(default=0.0, repr=False)
    _sum_total_time: float = field(default=0.0, repr=False)
    _internet_searches: int = field(default=0, repr=False)
    
    def update(self, result: Dict):
        """Обновление метрик на основе результата предсказания."""
        self.total_requests += 1
        
        if result.get("error"):
            self.failed_requests += 1
        else:
            self.successful_requests += 1
            
            # Обновление confidence
            conf = result.get("confidence", 0.0)
            self._sum_confidence += conf
            self.avg_confidence = (
                self._sum_confidence / self.successful_requests
            )
            
            # Обновление времени
            timing = result.get("timing", {})
            if "retrieval" in timing:
                self._sum_retrieval_time += timing["retrieval"]
                self.avg_retrieval_time = (
                    self._sum_retrieval_time / self.successful_requests
                )
            
            if "vlm_generation" in timing:
                self._sum_generation_time += timing["vlm_generation"]
                self.avg_generation_time = (
                    self._sum_generation_time / self.successful_requests
                )
            
            # Общее время
            total_time = sum(timing.values())
            self._sum_total_time += total_time
            self.avg_total_time = (
                self._sum_total_time / self.successful_requests
            )
            
            # Internet search rate
            if result.get("source") == PredictionSource.INTERNET.value:
                self._internet_searches += 1
            self.internet_search_rate = (
                self._internet_searches / self.successful_requests
            )
        
        self.last_updated = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Конвертация в словарь (без приватных полей)."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (
                self.successful_requests / self.total_requests
                if self.total_requests > 0 else 0.0
            ),
            "avg_confidence": round(self.avg_confidence, 4),
            "internet_search_rate": round(self.internet_search_rate, 4),
            "avg_retrieval_time": round(self.avg_retrieval_time, 3),
            "avg_generation_time": round(self.avg_generation_time, 3),
            "avg_total_time": round(self.avg_total_time, 3),
            "last_updated": self.last_updated,
        }


class AITourGuide:
    """
    Сервис распознавания достопримечательностей по изображениям.
    Использует VLM (Vision Language Model) с LoRA адаптером.
    """
    
    MAX_CONTEXT_LENGTH = 200
    DEFAULT_MAX_RESULTS = 5
    
    def __init__(self, config: Union[AITourGuideConfig, Dict, None] = None,
                 **kwargs):
        """
        Инициализация сервиса.
        
        Args:
            config: Конфигурация (AITourGuideConfig или dict) или None
            **kwargs: Параметры конфигурации (если config=None)
        
        Examples:
            # Вариант 1: через config объект
            config = AITourGuideConfig(
                model_path="path/to/model",
                index_path="path/to/index",
                facts_db_path="path/to/facts.pkl"
            )
            guide = AITourGuide(config)
            
            # Вариант 2: через kwargs (обратная совместимость)
            guide = AITourGuide(
                model_path="path/to/model",
                index_path="path/to/index",
                facts_db_path="path/to/facts.pkl"
            )
        """
        # Обработка конфигурации
        if config is None:
            # Создаём из kwargs (обратная совместимость)
            config = AITourGuideConfig(**kwargs)
        elif isinstance(config, dict):
            config = AITourGuideConfig(**config)
        
        self.config = config
        self.device = config.device
        self.confidence_threshold = config.confidence_threshold
        self.top_k_retrieval = config.top_k_retrieval
        
        # API ключи из конфигурации окружения
        self.yc_folder_id = os.getenv('YC_FOLDER_ID')
        self.yc_api_key = os.getenv('YC_API_KEY')
        
        # Инициализация переводчика
        self.translator = YandexTranslator(
            yc_folder_id=self.yc_folder_id,
            yc_api_key=self.yc_api_key
        )
        
        # Метрики производительности
        self.metrics = PerformanceMetrics()
        
        # Флаг готовности сервиса
        self._is_ready = False
        
        # 1. Загрузка retriever (CLIP + FAISS)
        logger.info("Загрузка RAGRetriever...")
        self.retriever = RAGRetriever(
            index_path=config.index_path,
            facts_db_path=config.facts_db_path,
            top_k=config.top_k_retrieval,
            use_multimodal_reranker=False,  # Отключено для скорости
        )
        
        # 2. Загрузка VLM (Vision Language Model с LoRA)
        logger.info("Загрузка VLM модели...")
       
        self.model = Llama(
            model_path=config.model_path,
            n_ctx=config.n_ctx,
            verbose=False,
            mmproj_path=config.mmproj_path,
        )
        
        # 3. Паттерн для парсинга JSON
        self.json_pattern = re.compile(r'\{.*\}', re.DOTALL)
        
        # Сервис готов
        self._is_ready = True
        
        logger.info("AITourGuide готов")
        logger.info(f"  Устройство: {config.device}")
        logger.info(f"  Порог уверенности: {config.confidence_threshold}")
        logger.info(f"  Top-K retrieval: {config.top_k_retrieval}")
        logger.info("  Модель: VLM с LoRA адаптером")
    
    # ========================================
    # HEALTH CHECK И МЕТРИКИ
    # ========================================
    
    def health_check(self) -> Dict[str, Union[bool, str, Dict]]:
        """
        Проверка состояния сервиса.
        
        Returns:
            Dict с информацией о состоянии сервиса
        """
        health = {
            "status": "healthy" if self._is_ready else "not_ready",
            "ready": self._is_ready,
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
        }
        
        # Проверка компонентов
        try:
            # Проверка retriever
            health["components"]["retriever"] = {
                "status": "ok" if hasattr(self, 'retriever') else "error",
                "index_size": (
                    len(self.retriever.facts_db)
                    if hasattr(self, 'retriever') else 0
                )
            }
            
            # Проверка модели
            health["components"]["model"] = {
                "status": "ok" if hasattr(self, 'model') else "error",
                "device": str(self.device),
            }
            
            # Проверка GPU
            if torch.cuda.is_available():
                health["components"]["gpu"] = {
                    "status": "ok",
                    "device_name": torch.cuda.get_device_name(0),
                    "memory_allocated": (
                        f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB"
                    ),
                    "memory_reserved": (
                        f"{torch.cuda.memory_reserved(0) / 1e9:.2f} GB"
                    ),
                }
            else:
                health["components"]["gpu"] = {
                    "status": "unavailable",
                    "message": "CUDA not available"
                }
            
        except Exception as e:
            health["status"] = "degraded"
            health["error"] = str(e)
        
        return health
    
    def get_metrics(self) -> Dict:
        """
        Получение метрик производительности.
        
        Returns:
            Dict с метриками
        """
        return self.metrics.to_dict()
    
    def reset_metrics(self):
        """Сброс метрик производительности."""
        self.metrics = PerformanceMetrics()
        logger.info("Метрики сброшены")
    
    # ========================================
    # УЛУЧШЕННАЯ ФОРМУЛА CONFIDENCE
    # ========================================
    
    def _calculate_confidence(
        self,
        retrieved_scores: List[float],
        retrieved_names: List[str],
        pred_name: str,
    ) -> float:
        """
        Улучшенная формула расчёта уверенности.
        
        Учитывает:
        1. CLIP score (top-1)
        2. Gap между 1-м и 2-м кандидатом
        3. Совпадение имени (exact/partial)
        4. Позиция совпадения в retrieved
        
        Args:
            retrieved_scores: Список CLIP scores из retrieval
            retrieved_names: Список названий кандидатов из retrieval
            pred_name: Предсказанное моделью название
        
        Returns:
            Confidence score в диапазоне [0, 1]
        """
        if not pred_name or not retrieved_scores:
            return 0.0
        
        # 1. CLIP score (уже в [0, 1])
        top_score = retrieved_scores[0]
        
        # 2. Gap между 1-м и 2-м
        if len(retrieved_scores) > 1:
            gap = retrieved_scores[0] - retrieved_scores[1]
            gap_conf = min(max(gap * self.config.gap_multiplier, 0.0), 1.0)
        else:
            gap_conf = 0.5
        
        # 3. Совпадение имени
        pred_lower = pred_name.lower().strip()
        
        if retrieved_names:
            names_clean = [n.lower().strip() for n in retrieved_names if n]
            
            exact_match = any(pred_lower == name for name in names_clean)
            partial_match = any(
                pred_lower in name or name in pred_lower 
                for name in names_clean
            )
            name_match_score = 1.0 if exact_match else (0.7 if partial_match else 0.0)
        else:
            name_match_score = 0.0
        
        # 4. Позиция совпадения
        position_score = 0.0
        if retrieved_names:
            names_clean = [n.lower().strip() for n in retrieved_names if n]
            for i, name in enumerate(names_clean[:5]):
                if name and (pred_lower == name or pred_lower in name):
                    position_score = 1.0 - (i * self.config.position_decay)
                    break
        
        # 5. Комбинация (веса подобраны по анализу)
        weights = self.config.confidence_weights
        confidence = (
            weights["clip_score"] * top_score +
            weights["gap"] * gap_conf +
            weights["name_match"] * name_match_score +
            weights["position"] * position_score
        )
        
        return float(min(max(confidence, 0.0), 1.0))
    
    # ========================================
    # ПАРСИНГ ОТВЕТА МОДЕЛИ (из train_rag_lora.py)
    # ========================================
    
    def _parse_json_response(self, text: str) -> Dict[str, str]:
        """
        Извлекает JSON из текста ответа.
        Возвращает dict с ключами 'name' и 'description'.
        
        Args:
            text: Текст для парсинга
        
        Returns:
            Dict с ключами 'name' и 'description'
        """
        text = text.strip()

        # Убираем маркеры кода
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()

        # Извлекаем JSON-объект
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            text = match.group(0)
        
        # Если JSON обрезан, пытаемся найти последнюю валидную закрывающую скобку
        # и обрезать до неё
        if text.count('{') > text.count('}'):
            # JSON не закрыт, обрезаем до последней валидной позиции
            last_quote = text.rfind('"')
            if last_quote > 0:
                text = text[:last_quote] + '"}'
        
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                return {"name": "", "description": ""}
            return {
                "name": str(data.get("name", "")).strip(),
                "description": str(data.get("description", "")).strip(),
            }
        except json.JSONDecodeError as e:
            # Fallback: извлекаем через regex (более гибкий паттерн)
            # Ищем name и description, учитывая возможные переносы строк
            name_match = re.search(
                r'"name"\s*:\s*"([^"]*(?:\\"[^"]*)*)"',
                text,
                re.DOTALL
            )
            desc_match = re.search(
                r'"description"\s*:\s*"((?:[^"\\]|\\.)*)"',
                text,
                re.DOTALL
            )
            if name_match and desc_match:
                logger.debug("JSON parse failed, used regex fallback")
                return {
                    "name": name_match.group(1).strip(),
                    "description": desc_match.group(1).strip(),
                }
            logger.warning(
                f"Failed to parse JSON response: {e}. "
                f"Full text: {text}"
            )
            return {"name": "", "description": ""}
    
    # ========================================
    # ПРОМПТ ДЛЯ VLM (из train_rag_lora.py)
    # ========================================
    
    def _make_rag_prompt(self, retrieved_context: str) -> str:
        """
        Создаёт промпт с retrieved контекстом для RAG.
        Логика идентична make_rag_prompt из train_rag_lora.py
        """
        return f"""Ты — профессиональный гид. Вот справочная информация о возможных достопримечательностях:

{retrieved_context}

ЗАДАЧА:
1. Определи, какая достопримечательность из списка на фотографии.
2. Верни описание в формате JSON, используя ТОЛЬКО факты из справочной информации выше.

Ответ должен быть строго в формате JSON:
{{
    "name": "Название на русском",
    "description": "Описание (3-5 предложений)"
}}

Не добавляй никакой текст до или после JSON!"""
    
    def _make_wiki_summary_prompt(self, retrieved_context: str) -> str:
        return f"""СПРАВОЧНАЯ ИНФОРМАЦИЯ (несколько вариантов):
{retrieved_context}

ЗАДАЧА:
1. Посмотри на изображение и определи, какая достопримечательность на фото
2. Выбери ОДИН самый подходящий вариант из справочной информации
3. Используй ТОЛЬКО описание выбранного варианта, НЕ смешивай информацию из разных вариантов
4. Верни JSON с названием и описанием ТОЛЬКО выбранного варианта

ВАЖНО: В описании должна быть информация ТОЛЬКО об одной достопримечательности!"""
    
    # ========================================
    # VLM ГЕНЕРАЦИЯ (изображение + текст)
    # ========================================
    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Конвертирует PIL Image в base64 строку.
        
        Args:
            image: PIL изображение
        
        Returns:
            Base64 строка изображения
        """
        with BytesIO() as buffered:
            image.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()
            return base64.b64encode(img_bytes).decode('utf-8')
    
    def prepare_vlm_messages(
        self,
        prompt: str,
        image: Union[Image.Image, str, bytes],
        reformulate: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Формирует сообщения в формате для Qwen2-VL с изображениями.
        Изображение кодируется в base64.
        
        Args:
            prompt: Текстовый промпт
            image: PIL Image, путь к изображению или bytes
        
        Returns:
            Список сообщений для модели
        """
        if reformulate:
            sys_prompt = """Ты — профессиональный русскоязычный гид.

КРИТИЧЕСКИ ВАЖНО: ЗАПРЕЩЕНО придумывать любые факты! Используй ТОЛЬКО информацию из справочного текста!

ЗАДАЧА:
1. Тебе дано изображение и справочная информация (может быть несколько пунктов).
2. Выбери ОДИН пункт, который соответствует изображению.
3. Перефразируй ТОЛЬКО факты из выбранного пункта в стиле гида.

АЛГОРИТМ:
1. Посмотри на изображение → определи, что это (собор, дворец, памятник и т.д.)
2. Найди в справке пункт с подходящим названием
3. Возьми ВСЕ факты ТОЛЬКО из этого пункта
4. Перефразируй их в стиле гида (начни с «Перед вами...»)

АБСОЛЮТНЫЕ ЗАПРЕТЫ:
- ЗАПРЕЩЕНО смешивать информацию из разных пунктов
- ЗАПРЕЩЕНО добавлять даты, если их нет в выбранном пункте
- ЗАПРЕЩЕНО добавлять имена, если их нет в выбранном пункте
- ЗАПРЕЩЕНО добавлять числа, если их нет в выбранном пункте
- ЗАПРЕЩЕНО использовать слова «самый», «крупнейший», «один из», если их нет в источнике
- ЗАПРЕЩЕНО придумывать архитектурные детали (арки, колонны, купола), если их нет в источнике
- ЗАПРЕЩЕНО придумывать исторические события
- ЗАПРЕЩЕНО добавлять любую информацию, которой нет в выбранном пункте

ПРАВИЛО: Если факта нет в справке — его НЕТ в ответе!

Формат ответа (только JSON):
{"name": "<название из выбранного пункта>", "description": "<перефразированные факты ТОЛЬКО из выбранного пункта>"}
"""
        else:
            sys_prompt = "Ты — профессиональный русскоязычный гид. Отвечай в формате JSON."

        system_msg = {
            "role": "system",
            "content": [{
                "type": "text",
                "text": sys_prompt
                }]
        }
        
        content_items = []
        
        # Добавляем изображение с base64 кодированием
        if image:
            if isinstance(image, Image.Image):
                # PIL Image - конвертируем в base64
                image_data = self._image_to_base64(image)
                
            elif isinstance(image, str):
                # Путь к файлу - загружаем и конвертируем
                pil_image = Image.open(image).convert("RGB")
                image_data = self._image_to_base64(pil_image)

            elif isinstance(image, bytes):
                # Bytes - конвертируем в base64 напрямую
                image_data = base64.b64encode(image).decode('utf-8')
                
            content_items.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
            })
        
        # Текст промпта
        content_items.append({
            "type": "text",
            "text": prompt
        })
        
        user_msg = {
            "role": "user",
            "content": content_items,
        }
    
        return [system_msg, user_msg]

    def _generate_with_vlm(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        reformulate: bool = False,
    ) -> str:
        """
        Генерация ответа через VLM (изображение + текст).
        
        Args:
            image: PIL изображение
            prompt: Текстовый промпт
            max_new_tokens: Максимум новых токенов
            temperature: Температура генерации
            reformulate: Использовать специальный промпт для переформулирования
        
        Returns:
            Сгенерированный текст
        """
        try:
            messages = self.prepare_vlm_messages(prompt, image, reformulate)
            # Для VLM нужны и изображение, и текст
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                stream=False,
            )
            
            # Извлекаем текст из ответа
            generated_text = response['choices'][0]['message']['content']
            
            return generated_text.strip()
            
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU out of memory during generation")
            torch.cuda.empty_cache()
            raise RuntimeError("GPU out of memory")
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise
    
    # ========================================
    # ПЕРЕФОРМУЛИРОВАНИЕ В СТИЛЬ ГИДА (с изображением!)
    # ========================================
    
    def _reformulate_as_tour_guide(
        self,
        image: Image.Image,
        names: List[str],
        summaries: List[str]
    ) -> Dict[str, str]:
        """
        Переводит Wikipedia описание на русский язык.
        ВРЕМЕННО: Переформулирование через VLM отключено из-за галлюцинаций.
        
        Args:
            image: PIL изображение достопримечательности (не используется)
            names: Названия достопримечательности
            summaries: Исходное описания из Wikipedia
        
        Returns:
            Dict с ключами 'name' и 'description'
        """
        # Переводим только первый (самый релевантный) результат
        logger.info("Translating Wikipedia content to Russian...")
        
        if not names or not summaries:
            logger.warning("No names or summaries provided")
            return {"name": "", "description": ""}
        
        # Переводим название
        translated_name = self.translator.translate(names[0])
        if not translated_name:
            translated_name = names[0]
        
        # Переводим описание
        translated_summary = self.translator.translate(summaries[0])
        if not translated_summary:
            translated_summary = summaries[0]
        
        logger.info(f"Translation successful: {translated_name[:50]}...")
        
        return {
            "name": translated_name,
            "description": translated_summary
        }
    
    # ========================================
    # ИНТЕРНЕТ-ПОИСК (YANDEX + WIKIPEDIA)
    # ========================================
    
    async def _search_internet(
        self,
        image: Union[Image.Image, str, bytes],
        retrieved_scores: List[float],
        retrieved_descs: List[str],
        retrieved_names: List[str],
        fallback_name: str,
        image_path: Union[str, bytes],
        timeout: float = 90.0,
    ) -> Dict:
        """
        Интернет-поиск при низкой уверенности с таймаутом.
        Использует Yandex Image Search + Wikipedia.
        
        Args:
            image: PIL изображение, путь к файлу или байты для Yandex поиска
            retrieved_scores: CLIP scores из retrieval
            retrieved_descs: Описания кандидатов из retrieval
            retrieved_names: Названия кандидатов из retrieval
            fallback_name: Название по умолчанию
            image_path: Путь к изображению для логирования
            timeout: Таймаут в секундах
        
        Returns:
            Dict с результатами поиска
        """
        
        result = {
            "found": False,
            "name": fallback_name,
            "description": "",
            "query": None,
            "confidence": 0.5,
        }
        
        # Проверка API ключей
        if not self.yc_folder_id or not self.yc_api_key:
            logger.warning("Yandex API keys not configured, skipping search")
            return result
        
        try:
            # Оборачиваем в таймаут
            async with asyncio.timeout(timeout):
                # 1. Yandex Image Search (блокирующий вызов в thread)
                landmark_names = await asyncio.to_thread(
                    self._yandex_search_sync,
                    image
                )
                
                if landmark_names:
                    result["query"] = list(landmark_names)
                    async with WikipediaService(
                        language="ru",
                    ) as wiki_service:
                        wiki_result = (
                            await wiki_service.get_landmark_info_async(
                                landmark_names
                            )
                        )
                    if any(wiki_result.values()):
                        # Фильтруем нерелевантные результаты
                        # 1. Исключаем названия фото и общие статьи
                        filtered_results = {
                            name: desc for name, desc in wiki_result.items()
                            if desc and not any(
                                x in name.lower()
                                for x in ['.jpg', 'panoramio', 'georama',
                                         'honeymoon', 'travel', 'lgbtq',
                                         'religious beliefs', 'religion in']
                            )
                        }
                        
                        # 2. Приоритизируем архитектурные объекты
                        priority_results = {
                            name: desc for name, desc in filtered_results.items()
                            if any(
                                x in name.lower()
                                for x in ['cathedral', 'church', 'temple',
                                         'mosque', 'synagogue', 'palace',
                                         'castle', 'fortress', 'tower',
                                         'monument', 'memorial', 'museum']
                            )
                        }
                        
                        # Используем приоритетные, если есть, иначе все отфильтрованные
                        if priority_results:
                            filtered_results = priority_results
                        elif not filtered_results:
                            filtered_results = wiki_result
                        
                        # Переформулирование в thread
                        pred_landmark = await asyncio.to_thread(
                            self._reformulate_as_tour_guide,
                            image,
                            list(filtered_results.keys()),
                            list(filtered_results.values()),
                        )
                        # Возвращаем только если есть описание
                        if pred_landmark.get("description"):
                            result["found"] = True
                            result["name"] = pred_landmark["name"]
                            result["description"] = pred_landmark["description"]
                            result["confidence"] = 0.8
                            return result
        
        except asyncio.TimeoutError:
            logger.warning(f"Internet search timeout after {timeout}s")
        except Exception as e:
            logger.error(f"Internet search error: {e}")
        
        # Fallback: отдаем top-1 retrieved name
        if retrieved_names and retrieved_names[0]:
            result["found"] = True
            result["name"] = retrieved_names[0]
            result["query"] = [retrieved_names[0]]
            result["description"] = retrieved_descs[0]
            result["confidence"] = 0.75
        
        return result
    
    def _yandex_search_sync(
        self,
        image_path: Union[str, bytes]
    ) -> Optional[set]:
        """
        Синхронный wrapper для Yandex поиска.
        
        Args:
            image_path: Путь к изображению
        
        Returns:
            Set названий достопримечательностей или None
        """
        # Проверка наличия API ключей
        if not self.yc_folder_id or not self.yc_api_key:
            logger.warning("Yandex API keys not configured")
            return None
            
        with YandexSearchService(
            yc_folder_id=self.yc_folder_id,
            yc_api_key=self.yc_api_key,
        ) as yandex_service:
            return yandex_service.search_by_image(
                image_path,
                num_results=self.DEFAULT_MAX_RESULTS
            )
    
    # ========================================
    # ОСНОВНОЙ МЕТОД ПРЕДСКАЗАНИЯ (VLM)
    # ========================================
    
    def _init_result(self) -> Dict:
        """Инициализация структуры результата."""
        return {
            "name": "",
            "description": "",
            "confidence": 0.0,
            "source": PredictionSource.RETRIEVAL.value,
            "retrieved_names": [],
            "retrieved_scores": [],
            "search_query": None,
            "error": None,
            "timing": {},
        }
    
    async def _validate_and_load_image(
        self,
        image_input: Union[str, Path, bytes],
        timing: Dict[str, float]
    ) -> Tuple[Union[Path, str, bytes], Image.Image]:
        """
        Валидация и загрузка изображения из пути или байтов.
        
        Args:
            image_input: Путь к изображению или байты
            timing: Словарь для записи времени выполнения
        
        Returns:
            Tuple из пути/идентификатора и загруженного изображения
        
        Raises:
            ValueError: Если данные невалидны
            FileNotFoundError: Если файл не найден
            RuntimeError: Если не удалось загрузить изображение
        """
        t0 = time.time()
        
        # Если байты - загружаем напрямую
        if isinstance(image_input, bytes):
            try:
                image = Image.open(BytesIO(image_input)).convert("RGB")
                timing["image_load"] = round(time.time() - t0, 3)
                return "bytes_image", image
            except Exception as e:
                raise RuntimeError(f"Failed to load image from bytes: {e}")
        
        # Если путь - валидируем и загружаем
        image_path = Path(image_input)
        
        # resolve() автоматически обрабатывает path traversal и симлинки
        try:
            image_path = image_path.resolve(strict=True)
        except (ValueError, OSError, RuntimeError) as e:
            raise ValueError(f"Invalid or inaccessible path: {e}")
        
        if not image_path.is_file():
            raise ValueError(f"Not a file: {image_path}")
        
        # Загрузка изображения
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {e}")
        
        timing["image_load"] = round(time.time() - t0, 3)
        return image_path, image
    
    async def _retrieve_candidates(
        self,
        image: Image.Image,
        timing: Dict[str, float]
    ) -> List[Dict]:
        """
        Поиск кандидатов через CLIP + FAISS.
        
        Args:
            image: PIL изображение
            timing: Словарь для записи времени выполнения
        
        Returns:
            Список кандидатов из retrieval
        
        Raises:
            RuntimeError: Если retrieval не вернул кандидатов
        """
        t0 = time.time()
        retrieved = await asyncio.to_thread(
            self.retriever.search,
            image,
            top_k=self.top_k_retrieval
        )
        timing["retrieval"] = round(time.time() - t0, 3)
        
        if not retrieved:
            raise RuntimeError("No candidates found in retrieval")
        
        return retrieved
    
    async def _generate_vlm_prediction(
        self,
        image: Image.Image,
        retrieved: List[Dict],
        timing: Dict[str, float]
    ) -> Dict[str, str]:
        """
        Генерация предсказания через VLM.
        
        Args:
            image: PIL изображение
            retrieved: Список кандидатов из retrieval
            timing: Словарь для записи времени выполнения
        
        Returns:
            Dict с ключами 'name' и 'description'
        """
        t0 = time.time()
        context_text = self.retriever.format_context(retrieved)
        prompt = self._make_rag_prompt(context_text)
        response = await asyncio.to_thread(
            self._generate_with_vlm,
            image,
            prompt,
            max_new_tokens=256
        )
        timing["vlm_generation"] = round(time.time() - t0, 3)
        
        return self._parse_json_response(response)
    
    async def _enhance_with_internet_search(
        self,
        image: Image.Image,
        image_path: Union[Path, str, bytes],
        retrieved: List[Dict],
        result: Dict,
        timing: Dict[str, float]
    ):
        """
        Улучшение результата через интернет-поиск.
        
        Args:
            image: PIL изображение
            image_path: Путь к изображению или "bytes_image" для байтов
            retrieved: Список кандидатов из retrieval
            result: Словарь результата для обновления
            timing: Словарь для записи времени выполнения
        """
        retrieved_scores = [r.get("score", 0.0) for r in retrieved]
        retrieved_names = [r.get("name", "") for r in retrieved]
        retrieved_descs = [r.get("description", "") for r in retrieved]
        
        # Для bytes используем само изображение, для путей - строку пути
        search_image_input = image if image_path == "bytes_image" else str(image_path)
        
        t0 = time.time()
        search_result = await self._search_internet(
            image=search_image_input,
            retrieved_scores=retrieved_scores,
            retrieved_descs=retrieved_descs,
            retrieved_names=retrieved_names,
            fallback_name=result["name"],
            image_path=str(image_path),
        )
        timing["internet_search"] = round(time.time() - t0, 3)
        
        if search_result.get("found"):
            result["source"] = PredictionSource.INTERNET.value
            result["name"] = search_result["name"]
            result["description"] = search_result["description"]
            result["search_query"] = search_result["query"]
            result["confidence"] = round(search_result["confidence"], 4)
    
    async def predict(
        self,
        image_input: Union[str, Path, bytes],
        use_internet_search: bool = True,
    ) -> Dict:
        """
        Предсказание для одного изображения через VLM.
        
        Args:
            image_input: Путь к изображению или байты изображения
            use_internet_search: Включить интернет-поиск при низкой уверенности
        
        Returns:
            {
                "name": "...",
                "description": "...",
                "confidence": 0.85,
                "source": "retrieval" | "internet",
                "retrieved_names": [...],
                "retrieved_scores": [...],
                "search_query": "...",
                "timing": {...},
                "error": None,
            }
        """
        timing: Dict[str, float] = {}
        result = self._init_result()
        
        # Генерируем correlation ID для трейсинга
        import uuid
        correlation_id = str(uuid.uuid4())[:8]
        
        # Определяем идентификатор для логирования
        if isinstance(image_input, bytes):
            input_id = f"bytes ({len(image_input)} bytes)"
        else:
            input_id = str(image_input)
        
        logger.info(f"[{correlation_id}] Starting prediction for {input_id}")
        
        try:
            # 1. Валидация и загрузка изображения
            try:
                image_path, image = await self._validate_and_load_image(
                    image_input, timing
                )
                logger.info(
                    f"[{correlation_id}] Image loaded: {image.size}"
                )
            except (ValueError, FileNotFoundError, RuntimeError) as e:
                result["error"] = str(e)
                result["timing"] = timing
                logger.error(f"[{correlation_id}] Validation error: {e}")
                return result
            
            # 2. Retrieval кандидатов
            try:
                retrieved = await self._retrieve_candidates(image, timing)
                retrieved_scores = [r.get("score", 0.0) for r in retrieved]
                retrieved_names = [r.get("name", "") for r in retrieved]
                result["retrieved_scores"] = retrieved_scores
                result["retrieved_names"] = retrieved_names
                logger.info(
                    f"[{correlation_id}] Retrieved {len(retrieved)} candidates, "
                    f"top score: {retrieved_scores[0]:.4f}"
                )
            except RuntimeError as e:
                result["error"] = str(e)
                result["timing"] = timing
                logger.error(f"[{correlation_id}] Retrieval error: {e}")
                return result
            
            # 3. VLM генерация предсказания
            try:
                parsed = await self._generate_vlm_prediction(
                    image, retrieved, timing
                )
                result["name"] = parsed["name"]
                result["description"] = parsed["description"]
                logger.info(
                    f"[{correlation_id}] VLM predicted: {parsed['name']}"
                )
            except Exception as e:
                result["error"] = f"VLM generation failed: {e}"
                result["timing"] = timing
                logger.error(f"[{correlation_id}] VLM error: {e}")
                return result
            
            # 4. Расчёт уверенности
            confidence = self._calculate_confidence(
                retrieved_scores=retrieved_scores,
                retrieved_names=retrieved_names,
                pred_name=parsed["name"],
            )
            result["confidence"] = round(confidence, 4)
            logger.info(
                f"[{correlation_id}] Confidence: {confidence:.4f}"
            )
            
            # 5. Интернет-поиск при низкой уверенности
            if use_internet_search and confidence < self.confidence_threshold:
                logger.info(
                    f"[{correlation_id}] Low confidence, "
                    f"triggering internet search"
                )
                await self._enhance_with_internet_search(
                    image, image_path, retrieved, result, timing
                )
                logger.info(
                    f"[{correlation_id}] After search - "
                    f"source: {result['source']}, "
                    f"confidence: {result['confidence']}"
                )
            
            result["timing"] = timing
            
            # Обновление метрик
            self.metrics.update(result)
            
            logger.info(
                f"[{correlation_id}] Prediction completed successfully, "
                f"total time: {sum(timing.values()):.3f}s"
            )
            
            return result
            
        except Exception as e:
            result["error"] = "Internal error occurred"
            result["timing"] = timing
            
            # Логируем полную ошибку внутренне
            import traceback
            error_details = traceback.format_exc()
            logger.error(
                f"[{correlation_id}] Unexpected error: {e}\n{error_details}"
            )
            
            # Обновление метрик (ошибка)
            self.metrics.update(result)
            
            return result
    
    # ========================================
    # ПАКЕТНАЯ ОБРАБОТКА
    # ========================================
    
    async def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        use_internet_search: bool = True,
    ) -> List[Dict]:
        """
        Пакетная обработка изображений.
        
        Note: Текущая реализация обрабатывает последовательно.
        Для production рекомендуется реализовать батчинг на GPU.
        
        Args:
            image_paths: Список путей к изображениям
            use_internet_search: Включить ли интернет-поиск
        
        Returns:
            Список результатов предсказаний
        """
        results = []
        total = len(image_paths)
        
        for i, path in enumerate(image_paths):
            logger.info(f"Прогресс: {i+1}/{total}")
            result = await self.predict(path, use_internet_search)
            results.append(result)
        
        return results
    
    # ========================================
    # CLEANUP
    # ========================================
    
    def cleanup(self):
        """Публичный метод для очистки ресурсов."""
        self._cleanup_resources()
    
    def _cleanup_resources(self):
        """Очистка ресурсов."""
        logger.info("Cleaning up resources...")
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'retriever'):
            del self.retriever
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ========================================
    # CONTEXT MANAGER SUPPORT
    # ========================================
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self._cleanup_resources()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        self._cleanup_resources()

if __name__ == "__main__":
    pass