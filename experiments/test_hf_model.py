"""
Тестирование Qwen2-VL reranker через Hugging Face transformers.
Результаты должны быть идентичны e2e_eval.py (P(yes) ≈ 0.95 для GT).
"""

import os
import json
import time
import math
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# ==========================================
# 1. Пути и константы
# ==========================================
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.rag.landmark_retriever import LandmarkRetriever

INDEX_PATH = "/Users/anastasiya/Documents/AITourGuide/data/index/siglip"
IMAGE_DIR = "/Users/anastasiya/Documents/AITourGuide/images"

# Путь к слитой модели (результат export_model_for_production.py)
# ВАЖНО: сначала запустите export_model_for_production.py!
MODEL_PATH = "/Users/anastasiya/Documents/AITourGuide/data/models/hf/qwen2-vl-2b-r16-merged"

VAL_DATASET = "/Users/anastasiya/Documents/AITourGuide/data/processed/dataset_v1/test.json"

# Порог уверенности (синхронизирован с e2e_eval.py и train.py)
RERANK_THRESHOLD = 0.5

# Параметры retrieval
RETRIEVAL_TOP_K = 3
RETRIEVAL_FAISS_K = 50


# ==========================================
# 2. Загрузка модели
# ==========================================
def load_model(
    model_path: str,
    verbose: bool = True,
) -> Tuple[AutoModelForVision2Seq, AutoProcessor]:
    """
    Загружает Qwen2-VL модель на CPU.
    
    Args:
        model_path: путь к слитой HF модели
        verbose: выводить логи загрузки
        
    Returns:
        (model, processor)
    """
    if verbose:
        print(f"\n📥 Загрузка модели: {model_path}")
    
    # Загрузка модели на CPU
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # На CPU используем float32 для стабильности
        device_map="cpu",
    )
    model.eval()  # Inference mode
    
    # Загрузка processor (tokenizer + image_processor)
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    if verbose:
        print(f"   ✅ Модель загружена (CPU, FP32)")
    
    return model, processor


# ==========================================
# 3. Подготовка промпта (идентично e2e_eval.py)
# ==========================================
def prepare_pairwise_messages(
    query_image_path: str,
    candidate_image_path: str,
    candidate_name: str,
    candidate_caption: str,
    caption_max_length: int = 300,
) -> List[Dict]:
    """
    Формирует сообщения для попарного сравнения.
    Идентично e2e_eval.py.
    
    ВАЖНО: для Qwen2-VL изображения передаются как {"type": "image", "image": PIL.Image}
    """
    # Загружаем изображения как PIL.Image
    query_img = Image.open(query_image_path).convert("RGB")
    cand_img = Image.open(candidate_image_path).convert("RGB")

    caption = candidate_caption[:caption_max_length]

    prompt_text = (
        "Question: Are these photos showing"
        f' the same landmark: "{candidate_name}"?\n'
        f"Candidate details: {caption}\n"
        "Answer only with Yes or No."
    )

    # Формат для Qwen2-VL: изображения вставляются в content как отдельные элементы
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": query_img},      # Первое изображение
                {"type": "text", "text": "Query Photo:"},
                {"type": "image", "image": cand_img},       # Второе изображение
                {"type": "text", "text": "Candidate Photo:"},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]


# ==========================================
# 4. Вычисление P(yes) через logits
# ==========================================

# Кеш ID токенов
_YES_TOKEN_ID: Optional[int] = None
_NO_TOKEN_ID: Optional[int] = None


def _get_yes_no_ids(processor) -> Tuple[int, int]:
    """
    Возвращает (yes_id, no_id) — ID токенов 'Yes' и 'No' в словаре.
    Кешируется глобально.
    """
    global _YES_TOKEN_ID, _NO_TOKEN_ID
    if _YES_TOKEN_ID is not None and _NO_TOKEN_ID is not None:
        return _YES_TOKEN_ID, _NO_TOKEN_ID

    tokenizer = processor.tokenizer

    # Пробуем с пробелом (' Yes', ' No') и без
    for with_space in [True, False]:
        prefix = " " if with_space else ""
        yes_ids = tokenizer.encode(f"{prefix}Yes", add_special_tokens=False)
        no_ids = tokenizer.encode(f"{prefix}No", add_special_tokens=False)

        if len(yes_ids) == 1 and len(no_ids) == 1:
            _YES_TOKEN_ID = yes_ids[0]
            _NO_TOKEN_ID = no_ids[0]
            print(
                f"   ℹ️  Token IDs: "
                f"'{prefix}Yes'={_YES_TOKEN_ID}, '{prefix}No'={_NO_TOKEN_ID}"
            )
            return _YES_TOKEN_ID, _NO_TOKEN_ID

    raise ValueError("Не удалось найти single-token Yes/No в словаре")


def get_yes_probability(
    model: AutoModelForVision2Seq,
    processor: AutoProcessor,
    messages: List[Dict],
) -> float:
    """
    Вычисляет P(yes) через прямой forward pass и извлечение logits.
    
    В отличие от GGUF подхода:
      - Не нужны monkey-patches
      - Не нужен logits_processor
      - Прямой доступ к logits[:, -1, :] как в e2e_eval.py
    """
    yes_id, no_id = _get_yes_no_ids(processor)

    # Применяем chat template (идентично e2e_eval.py)
    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Извлекаем изображения из messages для processor
    images = []
    for msg in messages:
        for part in msg.get("content", []):
            if isinstance(part, dict) and part.get("type") == "image":
                images.append(part["image"])

    # Подготовка inputs (processor сам обработает изображения и текст)
    inputs = processor(
        text=[text_prompt],
        images=images,
        padding=True,
        return_tensors="pt",
    )

    # Переносим на устройство модели (CPU)
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v
              for k, v in inputs.items()}

    # Forward pass — получаем logits
    with torch.no_grad():
        outputs = model(**inputs)

    # logits[:, -1, :] = логиты для следующего токена (после промпта)
    # Идентично e2e_eval.py: outputs.logits[:, -1, :]
    next_token_logits = outputs.logits[:, -1, :]

    logit_yes = next_token_logits[0, yes_id].item()
    logit_no = next_token_logits[0, no_id].item()

    # Бинарный softmax — идентично e2e_eval.py
    max_logit = max(logit_yes, logit_no)
    exp_yes = math.exp(logit_yes - max_logit)
    exp_no = math.exp(logit_no - max_logit)
    p_yes = exp_yes / (exp_yes + exp_no)

    return round(p_yes, 4)


# ==========================================
# 5. Ранжирование кандидатов
# ==========================================
def rank_candidates(
    model: AutoModelForVision2Seq,
    processor: AutoProcessor,
    query_image_path: str,
    candidates: List[Dict],
) -> List[Dict]:
    """Оценивает каждого кандидата и возвращает отсортированный список по P(yes)."""
    print(f"\n🚀 Начало попарного ранжирования ({len(candidates)} кандидатов)...")
    results = []

    for i, cand in enumerate(candidates):
        cand_id = cand.landmark_id
        cand_name = cand.landmark_name

        top_image = cand.get_top_image()
        if top_image is None:
            print(f"  [{i+1}/{len(candidates)}] ⚠️ Пропуск (нет gallery_images)")
            results.append({
                "rank": 0, "landmark_id": cand_id,
                "p_yes": 0.0, "status": "missing_gallery",
            })
            continue

        cand_caption = top_image.caption
        cand_img_path = os.path.join(IMAGE_DIR, top_image.image_path)

        print(f"  [{i+1}/{len(candidates)}] Оценка кандидата {cand_id}...", end=" ")

        if not Path(cand_img_path).exists():
            print("⚠️ Пропуск (нет изображения)")
            results.append({
                "rank": 0, "landmark_id": cand_id,
                "p_yes": 0.0, "status": "missing_image",
            })
            continue

        messages = prepare_pairwise_messages(
            query_image_path, cand_img_path, cand_name, cand_caption,
        )

        p_yes = get_yes_probability(model, processor, messages)
        results.append({
            "rank": 0,
            "landmark_id": cand_id,
            "p_yes": p_yes,
            "status": "success",
        })
        print(f"P(yes)={p_yes:.4f}")

    # Сортировка по убыванию P(yes)
    results.sort(key=lambda x: x["p_yes"], reverse=True)
    for rank, item in enumerate(results, start=1):
        item["rank"] = rank

    return results


# ==========================================
# 6. Главный тестовый сценарий
# ==========================================
def main_test(
    retrieved_candidates: List[Dict],
    query_image_path: str,
    true_lid: str,
):
    """Главная функция теста ранжирования."""
    print("=" * 70)
    print("🧪 ТЕСТ AI TOUR GUIDE: РАНЖИРОВАНИЕ ПО P(yes) (HF)")
    print("=" * 70)
    print(f"Query Image: {query_image_path}")
    print(f"True Landmark ID: {true_lid}")
    print(f"Model: {MODEL_PATH}")

    # 1. Загрузка модели
    print("\n🚀 Шаг 1: Загрузка модели...")
    t0_load = time.time()
    model, processor = load_model(MODEL_PATH, verbose=True)
    load_time = time.time() - t0_load
    print(f"   ⏱️  Время загрузки: {load_time:.1f} сек")

    # 2. Ранжирование
    print("\n🎯 Шаг 2: Оценка и ранжирование кандидатов...")
    t0 = time.time()
    ranked_results = rank_candidates(
        model, processor, query_image_path, retrieved_candidates,
    )
    infer_time = time.time() - t0

    # 3. Вывод результатов
    print("\n" + "=" * 70)
    print("📊 ИТОГОВЫЙ РЕЙТИНГ КАНДИДАТОВ")
    print("=" * 70)
    print(f"{'Ранг':<5} | {'Landmark ID':<20} | {'P(yes)':<10} | {'Статус'}")
    print("-" * 70)
    for res in ranked_results:
        marker = "✅ (GT)" if str(res["landmark_id"]) == str(true_lid) else ""
        print(
            f"{res['rank']:<5} | {res['landmark_id']:<20} | "
            f"{res['p_yes']:<10.4f} | {res['status']} {marker}"
        )
    print("=" * 70)
    print(f"⏱️  Инференс: {infer_time:.2f} сек")
    print(f"⏱️  Общее время: {load_time + infer_time:.2f} сек")

    # Unknown detection
    top_score = ranked_results[0]["p_yes"] if ranked_results else 0.0
    is_unknown = not ranked_results or top_score < RERANK_THRESHOLD
    if is_unknown:
        print(
            f"🔍 Unknown detection: ❓ UNKNOWN "
            f"(top P(yes)={top_score:.4f} < {RERANK_THRESHOLD})"
        )

    # Hit@1
    top_1_is_correct = (
        not is_unknown
        and str(ranked_results[0]["landmark_id"]) == str(true_lid)
    )
    print(f"🏆 Hit@1: {'✅ ДА' if top_1_is_correct else '❌ НЕТ'}")

    # 4. Сохранение лога
    log_path = Path(__file__).parent / "results" / "ranking_results_hf.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "query_image": query_image_path,
            "true_landmark_id": true_lid,
            "hit_at_1": top_1_is_correct,
            "is_unknown": is_unknown,
            "rerank_threshold": RERANK_THRESHOLD,
            "total_time_sec": round(load_time + infer_time, 2),
            "inference_time_sec": round(infer_time, 2),
            "model": MODEL_PATH,
            "results": ranked_results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Лог сохранён: {log_path}")
    print("\n✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")


# ==========================================
# 7. Точка входа
# ==========================================
QUERY_ITEM_IDX = 0

if __name__ == "__main__":
    with open(VAL_DATASET, mode="r", encoding="utf-8") as f:
        val_data = json.load(f)

    if QUERY_ITEM_IDX >= len(val_data):
        print(
            f"❌ QUERY_ITEM_IDX={QUERY_ITEM_IDX} "
            f"выходит за пределы датасета (размер: {len(val_data)})"
        )
        sys.exit(1)

    retriever = LandmarkRetriever.from_index_dir(INDEX_PATH)

    item = val_data[QUERY_ITEM_IDX]
    image_path = os.path.join(IMAGE_DIR, item.get("query_image", ""))

    retrieved = retriever.retrieve(
        Image.open(image_path).convert("RGB"),
        top_k=RETRIEVAL_TOP_K,
        faiss_k=RETRIEVAL_FAISS_K,
    )
    true_lid = item["candidates"][item["target_idx"]]["landmark_id"]

    try:
        main_test(retrieved, image_path, true_lid)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)