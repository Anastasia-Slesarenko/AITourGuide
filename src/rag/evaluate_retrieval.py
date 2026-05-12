"""
Оценка качества retrieval: recall@1, recall@3, recall@5.
"""

import os
import json
from PIL import Image
from tqdm import tqdm
from typing import List, Optional

# Ваши модули
from src.rag.retriever import RAGRetriever

# Размер батча для CLIP-кодирования изображений
_CLIP_BATCH_SIZE = 32


def _load_images(
    val_data: list,
    image_dir: str,
) -> tuple:
    """
    Загружает все изображения из val_data.

    Returns:
        images: список PIL-изображений (None для неудачных загрузок)
        valid_mask: список bool — True если изображение загружено
    """
    images: List[Optional[Image.Image]] = []
    valid_mask: List[bool] = []

    for item in tqdm(val_data, desc="Loading images", unit="img"):
        image_path = os.path.join(image_dir, item.get("image_path", ""))
        try:
            img = Image.open(image_path).convert("RGB")
            images.append(img)
            valid_mask.append(True)
        except Exception as e:
            print(f"⚠️ Error loading {image_path}: {e}")
            images.append(None)
            valid_mask.append(False)

    return images, valid_mask


def evaluate_retrieval_with_reranking(
    val_json: str,
    image_dir: str,
    retriever: RAGRetriever,
    top_k_list: list = [1, 3, 5, 10, 15],
    max_samples: Optional[int] = None,
    use_reranking: bool = True,
    initial_k: int = 30,
    clip_batch_size: int = _CLIP_BATCH_SIZE,
) -> dict:
    """
    Оценивает качество retrieval с опциональным мультимодальным reranking.

    Ускорение: CLIP-кодирование выполняется батчами (clip_batch_size),
    FAISS-поиск и reranking — поштучно (быстро, на CPU/GPU).

    Args:
        val_json: Путь к JSON-файлу с валидационными данными.
        image_dir: Директория с изображениями.
        retriever: Инициализированный RAGRetriever.
        top_k_list: Список значений k для расчёта recall@k.
        max_samples: Максимальное число примеров (None = все).
        use_reranking: Использовать ли reranking.
            Требует retriever с use_multimodal_reranker=True.
        initial_k: Число кандидатов для первичного retrieval
            перед reranking.
        clip_batch_size: Размер батча для CLIP-кодирования.

    Returns:
        Словарь {recall@k: значение} для каждого k из top_k_list.
    """
    # Загрузка валидационных данных
    with open(val_json, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    if max_samples:
        val_data = val_data[:max_samples]

    total_samples = len(val_data)
    print(f"🔍 Оценка retrieval на {total_samples} примерах...")
    print(
        f"   Reranking: {'✅ Включен' if use_reranking else '❌ Выключен'}"
    )
    if use_reranking and not retriever.reranker:
        print(
            "⚠️  use_reranking=True, но retriever.reranker "
            "не инициализирован. Reranking не будет выполнен."
        )
    if use_reranking:
        print(f"   Initial k: {initial_k} → Final k: {max(top_k_list)}")
    print(f"   CLIP batch size: {clip_batch_size}")

    # Шаг 1: загрузка всех изображений
    images, valid_mask = _load_images(val_data, image_dir)

    # Шаг 2: батчевое CLIP-кодирование только валидных изображений
    valid_images = [img for img in images if img is not None]
    valid_indices = [i for i, ok in enumerate(valid_mask) if ok]

    print(
        f"✅ Загружено {len(valid_images)}/{total_samples} изображений. "
        f"Кодирование батчами по {clip_batch_size}..."
    )

    # Кодируем батчами и собираем эмбеддинги
    embeddings = []
    for batch_start in tqdm(
        range(0, len(valid_images), clip_batch_size),
        desc="CLIP encoding",
        unit="batch",
    ):
        batch = valid_images[batch_start: batch_start + clip_batch_size]
        batch_embs = retriever.encode_images_batch(batch)
        embeddings.extend(batch_embs)

    # Шаг 3: FAISS-поиск + опциональный reranking поштучно
    recall_counts = {k: 0 for k in top_k_list}

    for emb_idx, orig_idx in enumerate(
        tqdm(valid_indices, desc="Searching", unit="example")
    ):
        item = val_data[orig_idx]
        image = valid_images[emb_idx]
        emb = embeddings[emb_idx]

        if use_reranking:
            # Первичный retrieval по эмбеддингу
            candidates = retriever.search_by_embedding(
                emb, top_k=initial_k
            )
            # Reranking только по изображению (без текстового запроса)
            retrieved = retriever.reranker.rerank(
                image=image,
                retrieved=candidates,
                top_k=max(top_k_list),
            ) if retriever.reranker else candidates[:max(top_k_list)]
        else:
            retrieved = retriever.search_by_embedding(
                emb, top_k=max(top_k_list)
            )

        retrieved_ids = [r["landmark_id"] for r in retrieved]
        true_lid = str(item.get("landmark_id", ""))

        for k in top_k_list:
            if true_lid in retrieved_ids[:k]:
                recall_counts[k] += 1

    # Расчёт метрик: делим на total_samples
    # (все примеры, включая ошибки загрузки)
    metrics = {
        f"recall@{k}": round(recall_counts[k] / total_samples, 4)
        for k in top_k_list
    }

    return metrics


def compare_reranking(
    val_json: str,
    image_dir: str,
    retriever: RAGRetriever,
    top_k_list: list = [1, 3, 5, 10, 15],
    max_samples: Optional[int] = None,
    clip_batch_size: int = _CLIP_BATCH_SIZE,
) -> dict:
    """
    Сравнивает retrieval с reranking и без.

    Args:
        val_json: Путь к JSON-файлу с валидационными данными.
        image_dir: Директория с изображениями.
        retriever: Инициализированный RAGRetriever
            (должен иметь use_multimodal_reranker=True для reranking).
        top_k_list: Список значений k для расчёта recall@k.
        max_samples: Максимальное число примеров (None = все).
        clip_batch_size: Размер батча для CLIP-кодирования.

    Returns:
        Словарь с ключами without_reranking, with_reranking,
        improvements, avg_improvement.
    """
    print("="*70)
    print("📊 СРАВНЕНИЕ: Retrieval с reranking vs без reranking")
    print("="*70)

    # Без reranking
    print("\n🔍 Оценка БЕЗ reranking...")
    metrics_without = evaluate_retrieval_with_reranking(
        val_json=val_json,
        image_dir=image_dir,
        retriever=retriever,
        top_k_list=top_k_list,
        use_reranking=False,
        max_samples=max_samples,
        clip_batch_size=clip_batch_size,
    )

    # С reranking
    print("\n🔍 Оценка С reranking...")
    metrics_with = evaluate_retrieval_with_reranking(
        val_json=val_json,
        image_dir=image_dir,
        retriever=retriever,
        top_k_list=top_k_list,
        use_reranking=True,
        initial_k=30,
        max_samples=max_samples,
        clip_batch_size=clip_batch_size,
    )

    # Сравнение
    print("\n" + "="*70)
    print("📈 РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    print("="*70)
    print(f"{'Метрика':<15} {'Без CE':<15} {'С CE':<15} {'Улучшение':<15}")
    print("-"*70)

    improvements = {}
    for k in top_k_list:
        key = f"recall@{k}"
        without = metrics_without.get(key, 0)
        with_ce = metrics_with.get(key, 0)
        improvement = with_ce - without

        improvements[key] = improvement
        print(
            f"{key:<15} {without:<15.1%} {with_ce:<15.1%} {improvement:+.1%}"
        )

    print("="*70)

    # Итог
    avg_improvement = sum(improvements.values()) / len(improvements)
    print(f"\n🏆 Среднее улучшение: {avg_improvement:+.1%}")

    if avg_improvement > 0.05:
        print("✅ Reranking даёт значимое улучшение (>5%)!")
    elif avg_improvement > 0.02:
        print("🟡 Reranking даёт небольшое улучшение (2-5%)")
    else:
        print("⚠️ Reranking не даёт значимого улучшения (<2%)")

    print("="*70)

    return {
        "without_reranking": metrics_without,
        "with_reranking": metrics_with,
        "improvements": improvements,
        "avg_improvement": avg_improvement,
    }


if __name__ == "__main__":
    VAL_JSON = "/home/jupyter/s3/ai-tour-guide/val_russian_landmarks.json"
    IMAGE_DIR = "/home/jupyter/s3/ai-tour-guide/landmarks"
    INDEX_PATH = "/home/jupyter/s3/ai-tour-guide/landmarks/clip_index"
    FACTS_DB_PATH = "/home/jupyter/s3/ai-tour-guide/facts_db.pkl"

    print("="*70)
    print("🔍 Оценка retrieval с мультимодальным reranking")
    print("="*70)

    # Загрузка retriever (с мультимодальным reranker)
    print("\n📦 Загрузка RAGRetriever...")
    retriever = RAGRetriever(
        index_path=INDEX_PATH,
        facts_db_path=FACTS_DB_PATH,
        top_k=15,
        use_multimodal_reranker=True,
    )

    # Сравнение с reranking и без
    results = compare_reranking(
        val_json=VAL_JSON,
        image_dir=IMAGE_DIR,
        retriever=retriever,
        max_samples=200,
        clip_batch_size=32,
    )

    # Сохранение результатов
    output_path = (
        "/home/jupyter/s3/ai-tour-guide/retrieval_reranking_comparison.json"
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Результаты сохранены: {output_path}")
