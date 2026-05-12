"""
Скрипт для загрузки CLIP модели локально.
После загрузки модель будет использоваться из локальной директории,
что ускорит инициализацию и позволит работать без интернета.
"""

from transformers import CLIPModel, CLIPProcessor
import os

# Конфигурация
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
LOCAL_MODEL_DIR = "models/clip-vit-large-patch14"


def download_clip_model():
    """Загружает CLIP модель и процессор локально."""
    
    print(f"🔽 Загрузка CLIP модели: {CLIP_MODEL_NAME}")
    print(f"📁 Сохранение в: {LOCAL_MODEL_DIR}")
    
    # Создаем директорию если не существует
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    
    # Загрузка модели
    print("\n1️⃣ Загрузка модели...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    model.save_pretrained(LOCAL_MODEL_DIR)
    print(f"   ✅ Модель сохранена в {LOCAL_MODEL_DIR}")
    
    # Загрузка процессора
    print("\n2️⃣ Загрузка процессора...")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    processor.save_pretrained(LOCAL_MODEL_DIR)
    print(f"   ✅ Процессор сохранен в {LOCAL_MODEL_DIR}")
    
    # Проверка размера
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, dirnames, filenames in os.walk(LOCAL_MODEL_DIR)
        for filename in filenames
    )
    size_mb = total_size / (1024 * 1024)
    
    print(f"\n✅ Загрузка завершена!")
    print(f"📊 Размер модели: {size_mb:.2f} MB")
    print(f"\n💡 Теперь используйте в коде:")
    print(f'   clip_model="{LOCAL_MODEL_DIR}"')
    print(f"\nПример конфигурации:")
    print(f"""
config = AITourGuideConfig(
    model_path="models/qwen2-vl-2b-r16/model-q5_k_m.gguf",
    mmproj_path="models/qwen2-vl-2b-r16/mmproj-model-f16.gguf",
    index_path="rag/clip_index",
    facts_db_path="rag/facts_db.pkl",
)

# RAGRetriever автоматически использует локальную модель из метаданных индекса
# Или можно явно указать:
# retriever = RAGRetriever(
#     index_path="rag/clip_index",
#     facts_db_path="rag/facts_db.pkl",
#     clip_model="{LOCAL_MODEL_DIR}",
# )
""")


if __name__ == "__main__":
    try:
        download_clip_model()
    except Exception as e:
        print(f"\n❌ Ошибка при загрузке: {e}")
        print("\nПопробуйте:")
        print("1. Проверить интернет-соединение")
        print("2. Установить/обновить transformers: pip install -U transformers")
        print("3. Проверить доступ к HuggingFace Hub")
