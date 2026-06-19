"""
Скрипт для загрузки Siglip модели локально.
После загрузки модель будет использоваться из локальной директории,
что ускорит инициализацию и позволит работать без интернета.
"""

from transformers import AutoProcessor, SiglipModel
import os

# Конфигурация
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"
LOCAL_MODEL_DIR = "data/models/siglip-base-patch16-224"


def download_siglip_model():
    """Загружает Siglip модель и процессор локально."""
    
    print(f"Загрузка Siglip модели: {SIGLIP_MODEL_NAME}")
    print(f"Сохранение в: {LOCAL_MODEL_DIR}")
    
    # Создаем директорию если не существует
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    
    # Загрузка модели
    print("Загрузка модели...")
    model = SiglipModel.from_pretrained(SIGLIP_MODEL_NAME)
    model.save_pretrained(LOCAL_MODEL_DIR)
    print(f"Модель сохранена в {LOCAL_MODEL_DIR}")
    
    # Загрузка процессора
    print("Загрузка процессора...")
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
    processor.save_pretrained(LOCAL_MODEL_DIR)
    print(f"Процессор сохранен в {LOCAL_MODEL_DIR}")
    
    # Проверка размера
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, dirnames, filenames in os.walk(LOCAL_MODEL_DIR)
        for filename in filenames
    )
    size_mb = total_size / (1024 * 1024)
    
    print(f"Загрузка завершена!")
    print(f"Размер модели: {size_mb:.2f} MB")

if __name__ == "__main__":
    try:
        download_siglip_model()
    except Exception as e:
        print(f"Ошибка при загрузке: {e}")
