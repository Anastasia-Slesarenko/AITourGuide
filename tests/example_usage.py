"""
Примеры использования AITourGuide
"""
import asyncio
import sys 
from pathlib import Path

# Добавляем корневую папку AIGuide в sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.ai_tour_guide import AITourGuide, AITourGuideConfig


# ============================================
# ПРИМЕР 1: Базовое использование
# ============================================

async def basic_example():
    """Простой пример распознавания одного изображения."""
    
    # Создание конфигурации
    config = AITourGuideConfig(
        model_path="models/qwen2-vl-2b-r16/model-q5_k_m.gguf",
        mmproj_path="models/qwen2-vl-2b-r16/mmproj-model-f16.gguf",
        index_path="rag/clip_index",
        facts_db_path="rag/facts_db.pkl",
        device="cuda",
        confidence_threshold=0.78,
        top_k_retrieval=10,
    )
    
    # Инициализация сервиса
    guide = AITourGuide(config)
    
    # Предсказание
    result = await guide.predict("images/475_3.jpg")
    
    print(f"Название: {result['name']}")
    print(f"Описание: {result['description']}")
    print(f"Уверенность: {result['confidence']:.2%}")
    print(f"Источник: {result['source']}")
    print(f"Время: {sum(result['timing'].values()):.2f}s")


# ============================================
# ПРИМЕР 2: Использование с context manager
# ============================================

async def context_manager_example():
    """Пример с автоматической очисткой ресурсов."""
    
    config = AITourGuideConfig(
        model_path="models/qwen2-vl-2b-r16/model-q5_k_m.gguf",
        mmproj_path="models/qwen2-vl-2b-r16/mmproj-model-f16.gguf",
        index_path="rag/clip_index",
        facts_db_path="rag/facts_db.pkl",
    )
    
    # Async context manager - автоматически очистит ресурсы
    async with AITourGuide(config) as guide:
        result = await guide.predict("images/475_3.jpg")
        print(f"Результат: {result['name']}")
        print(f"Описание: {result['description']}")
        print(f"Уверенность: {result['confidence']:.2%}")
        print(f"Источник: {result['source']}")
        print(f"Время: {sum(result['timing'].values()):.2f}s")
    
    # Ресурсы автоматически освобождены


# ============================================
# ПРИМЕР 3: Пакетная обработка
# ============================================

async def batch_example():
    """Обработка нескольких изображений."""
    
    config = AITourGuideConfig(
        model_path="models/qwen2-vl-2b-r16/model-q5_k_m.gguf",
        mmproj_path="models/qwen2-vl-2b-r16/mmproj-model-f16.gguf",
        index_path="rag/clip_index",
        facts_db_path="rag/facts_db.pkl",
    )
    
    guide = AITourGuide(config)
    
    # Список изображений
    image_paths = [
        "images/14_1.jpg", #Театр Делакорта
        "images/50_1.jpg", #Раймунд-театр
        "images/22_3.jpg", #Кастильо-де-Санта-Крус
    ]
    
    # Пакетная обработка
    results = await guide.predict_batch(image_paths)
    
    for i, result in enumerate(results):
        print(f"\n--- Изображение {i+1} ---")
        print(f"Название: {result['name']}")
        print(f"Описание: {result['description']}")
        print(f"Уверенность: {result['confidence']:.2%}")
        print(f"Источник: {result['source']}")
        print(f"Время: {sum(result['timing'].values()):.2f}s")


# ============================================
# ПРИМЕР 4: Настройка параметров confidence
# ============================================

async def custom_confidence_example():
    """Пример с кастомными параметрами расчета уверенности."""
    
    config = AITourGuideConfig(
        model_path="models/qwen2-vl-2b-r16/model-q5_k_m.gguf",
        mmproj_path="models/qwen2-vl-2b-r16/mmproj-model-f16.gguf",
        index_path="rag/clip_index",
        facts_db_path="rag/facts_db.pkl",
        
        # Кастомные параметры confidence
        confidence_threshold=0.85,  # Более строгий порог
        gap_multiplier=15.0,  # Больший вес для gap между кандидатами
        position_decay=0.10,  # Меньший штраф за позицию
        confidence_weights={
            "clip_score": 0.30,
            "gap": 0.25,
            "name_match": 0.45,
            "position": 0.15,
        }
    )
    
    guide = AITourGuide(config)
    result = await guide.predict("images/475_3.jpg")
    
    print(f"Уверенность: {result['confidence']:.2%}")
    print(f"Источник: {result['source']}")


# ============================================
# ПРИМЕР 5: Отключение интернет-поиска
# ============================================

async def no_internet_search_example():
    """Пример без использования интернет-поиска."""
    
    config = AITourGuideConfig(
        model_path="models/qwen2-vl-2b-r16/model-q5_k_m.gguf",
        mmproj_path="models/qwen2-vl-2b-r16/mmproj-model-f16.gguf",
        index_path="rag/clip_index",
        facts_db_path="rag/facts_db.pkl",
    )
    
    guide = AITourGuide(config)
    
    # Отключаем интернет-поиск
    result = await guide.predict(
        "images/962_5.jpg",
        use_internet_search=False
    )
    
    print(f"Результат (только RAG): {result['name']}")
    print(f"Описание: {result['description']}")
    print(f"Уверенность: {result['confidence']:.2%}")
    print(f"Время: {sum(result['timing'].values()):.2f}s")
    print(f"Источник: {result['source']}")  # Всегда будет "retrieval"


# ============================================
# ПРИМЕР 6: Мониторинг и метрики
# ============================================

async def metrics_example():
    """Пример работы с метриками производительности."""
    
    config = AITourGuideConfig(
        model_path="models/qwen2-vl-2b-r16/model-q5_k_m.gguf",
        mmproj_path="models/qwen2-vl-2b-r16/mmproj-model-f16.gguf",
        index_path="rag/clip_index",
        facts_db_path="rag/facts_db.pkl",
    )
    
    guide = AITourGuide(config)
    
    # Обработка нескольких изображений
    image_paths = [
        "images/14_1.jpg", #Театр Делакорта
        "images/50_1.jpg", #Раймунд-театр
        "images/22_3.jpg", #Кастильо-де-Санта-Крус
    ]
    for image_path in image_paths:
        await guide.predict(image_path)
    
    # Получение метрик
    metrics = guide.get_metrics()
    print("\n=== Метрики производительности ===")
    print(f"Всего запросов: {metrics['total_requests']}")
    print(f"Успешных: {metrics['successful_requests']}")
    print(f"Success rate: {metrics['success_rate']:.2%}")
    print(f"Средняя уверенность: {metrics['avg_confidence']:.2%}")
    print(f"Частота интернет-поиска: {metrics['internet_search_rate']:.2%}")
    print(f"Среднее время retrieval: {metrics['avg_retrieval_time']:.3f}s")
    print(f"Среднее время генерации: {metrics['avg_generation_time']:.3f}s")
    print(f"Среднее общее время: {metrics['avg_total_time']:.3f}s")
    
    # Сброс метрик
    guide.reset_metrics()


# ============================================
# ПРИМЕР 7: Health check
# ============================================

def health_check_example():
    """Проверка состояния сервиса."""
    
    config = AITourGuideConfig(
        model_path="models/qwen2-vl-2b-r16/model-q5_k_m.gguf",
        mmproj_path="models/qwen2-vl-2b-r16/mmproj-model-f16.gguf",
        index_path="rag/clip_index",
        facts_db_path="rag/facts_db.pkl",
    )
    
    guide = AITourGuide(config)
    
    # Проверка здоровья
    health = guide.health_check()
    
    print(f"Статус: {health['status']}")
    print(f"Готов: {health['ready']}")
    print(f"\nКомпоненты:")
    for component, info in health['components'].items():
        print(f"  {component}: {info['status']}")
    
    print(f"\nКонфигурация:")
    print(f"  Device: {health['config']['device']}")
    print(f"  Confidence threshold: {health['config']['confidence_threshold']}")


# ============================================
# ПРИМЕР 8: Обработка ошибок
# ============================================

async def error_handling_example():
    """Пример обработки ошибок."""
    
    config = AITourGuideConfig(
        model_path="models/qwen2-vl-2b-r16/model-q5_k_m.gguf",
        mmproj_path="models/qwen2-vl-2b-r16/mmproj-model-f16.gguf",
        index_path="rag/clip_index",
        facts_db_path="rag/facts_db.pkl",
    )
    
    guide = AITourGuide(config)
    
    # Попытка обработать несуществующий файл
    result = await guide.predict("nonexistent.jpg")
    
    if result['error']:
        print(f"Ошибка: {result['error']}")
        # Ошибка логируется внутренне, но не раскрывается пользователю
    else:
        print(f"Успех: {result['name']}")


# ============================================
# ПРИМЕР 9: Детальный анализ результата
# ============================================

async def detailed_result_example():
    """Пример детального анализа результата."""
    
    config = AITourGuideConfig(
        model_path="models/qwen2-vl-2b-r16/model-q5_k_m.gguf",
        mmproj_path="models/qwen2-vl-2b-r16/mmproj-model-f16.gguf",
        index_path="rag/clip_index",
        facts_db_path="rag/facts_db.pkl",
    )
    
    guide = AITourGuide(config)
    result = await guide.predict("images/962_5.jpg")
    
    print("=== Детальный результат ===\n")
    
    print(f"Название: {result['name']}")
    print(f"Описание: {result['description']}\n")
    
    print(f"Уверенность: {result['confidence']:.2%}")
    print(f"Источник: {result['source']}\n")
    
    print("Retrieved кандидаты:")
    for i, (name, score) in enumerate(
        zip(result['retrieved_names'][:5], result['retrieved_scores'][:5])
    ):
        print(f"  {i+1}. {name} (score: {score:.4f})")
    
    if result['search_query']:
        print(f"\nПоисковый запрос: {result['search_query']}")
    
    print("\nВремя выполнения:")
    for stage, time_sec in result['timing'].items():
        print(f"  {stage}: {time_sec:.3f}s")
    print(f"  TOTAL: {sum(result['timing'].values()):.3f}s")


# ============================================
# ПРИМЕР 10: Обратная совместимость (kwargs)
# ============================================

async def backward_compatibility_example():
    """Пример использования через kwargs (старый способ)."""
    
    # Старый способ - через kwargs (все еще работает)
    guide = AITourGuide(
        model_path="models/qwen2-vl-2b-r16/model-q5_k_m.gguf",
        mmproj_path="models/qwen2-vl-2b-r16/mmproj-model-f16.gguf",
        index_path="rag/clip_index",
        facts_db_path="rag/facts_db.pkl",
        device="cuda",
        confidence_threshold=0.70,
    )
    
    result = await guide.predict("images/962_5.jpg")
    print(f"Результат: {result['name']}")


# ============================================
# Запуск примеров
# ============================================

if __name__ == "__main__":
    print("Выберите пример для запуска:")
    print("1. Базовое использование")
    print("2. Context manager")
    print("3. Пакетная обработка")
    print("4. Кастомные параметры confidence")
    print("5. Без интернет-поиска")
    print("6. Метрики производительности")
    print("7. Health check")
    print("8. Обработка ошибок")
    print("9. Детальный анализ результата")
    print("10. Обратная совместимость")
    
    choice = input("\nВведите номер примера (1-10): ")
    
    examples = {
        "1": basic_example,
        "2": context_manager_example,
        "3": batch_example,
        "4": custom_confidence_example,
        "5": no_internet_search_example,
        "6": metrics_example,
        "7": lambda: health_check_example(),  # Синхронный
        "8": error_handling_example,
        "9": detailed_result_example,
        "10": backward_compatibility_example,
    }
    
    if choice in examples:
        example_func = examples[choice]
        if choice == "7":
            example_func()
        else:
            asyncio.run(example_func())
    else:
        print("Неверный выбор!")
