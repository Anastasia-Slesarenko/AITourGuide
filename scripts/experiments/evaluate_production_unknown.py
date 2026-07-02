# scripts/experiments/evaluate_production_unknown.py
"""
Оценивает none accuracy на production-like hard unknown сэмплах.

Что делает скрипт:
1. Загружает predictions JSON (scores = сырые P(Yes) от VLM)
2. Загружает production-like hard unknown датасет
3. Оценивает none accuracy при заданном пороге
4. Строит детализацию по бинам retrieval score
5. Сохраняет результаты в JSON

Запуск:
    python experiments/evaluate_production_unknown.py
"""
import json
import numpy as np
from pathlib import Path


# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================

# Путь к predictions JSON (результаты inference на val/test)
_PREDICTIONS_PATH = (
    "experiments/results/"
    "val_rerank_exp_r16_alpha32_lr2e-5_rerank_full_lora_448_lora_predictions.json"
)

# Путь к production-like hard unknown датасету
_HARD_UNKNOWN_PATH = (
    "/Users/anastasiya/Documents/AITourGuide/data/processed/dataset_v1/val_super_hard_unknown.json"
)

# Порог для unknown detection
# Подбирается на cal-split (см. find_th_and_recompute_metrics.py)
_THRESHOLD = 0.4226


# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# ОЦЕНКА
# ============================================================

def evaluate_production_unknown(
    predictions_path: str,
    hard_unknown_path: str,
    threshold: float
):
    print("="*70)
    print("EVALUATION ON PRODUCTION-LIKE HARD UNKNOWN")
    print("="*70)
    print(f"  Threshold: {threshold}")
    print(f"  Predictions: {predictions_path}")
    print(f"  Hard Unknown: {hard_unknown_path}")
    print("="*70)
    
    # Загружаем predictions
    print("\n1. Загрузка predictions...")
    all_predictions = load_json(predictions_path)
    print(f"   Загружено {len(all_predictions)} predictions")
    
    # Создаем маппинг query_image -> prediction
    pred_map = {p["query_image"]: p for p in all_predictions}
    
    # Загружаем hard unknown
    print("\n2. Загрузка hard unknown samples...")
    hard_unknown_samples = load_json(hard_unknown_path)
    print(f"   Загружено {len(hard_unknown_samples)} hard unknown samples")
    
    # Оцениваем
    print(f"\n3. Оценка с threshold={threshold}...")
    
    results = {
        "total": 0,
        "correctly_rejected": 0,
        "false_positive": 0,
        "by_retrieval_score_bin": {}  # Группировка по retrieval score
    }
    
    confidence_scores = []
    retrieval_scores = []
    
    for sample in hard_unknown_samples:
        query_image = sample["query_image"]
        candidates = sample["candidates"]
        
        if query_image not in pred_map:
            print(f"   ️  Warning: {query_image} not found in predictions")
            continue
        
        pred = pred_map[query_image]
        pred_scores = pred.get("scores", [])
        
        if not pred_scores:
            continue
        
        max_confidence = max(pred_scores)
        confidence_scores.append(max_confidence)
        
        # Собираем retrieval scores для анализа
        for cand in candidates:
            retrieval_scores.append(cand.get("retrieval_score", 0.0))
        
        # Проверяем, отвергнута ли моделью
        is_rejected = max_confidence < threshold
        
        if is_rejected:
            results["correctly_rejected"] += 1
        else:
            results["false_positive"] += 1
        
        results["total"] += 1
        
        # Биннинг по retrieval score
        avg_retrieval_score = np.mean([cand.get("retrieval_score", 0.0) for cand in candidates])
        bin_label = f"{int(avg_retrieval_score * 10) / 10:.1f}-{(int(avg_retrieval_score * 10) + 1) / 10:.1f}"
        
        if bin_label not in results["by_retrieval_score_bin"]:
            results["by_retrieval_score_bin"][bin_label] = {
                "total": 0,
                "correctly_rejected": 0
            }
        
        results["by_retrieval_score_bin"][bin_label]["total"] += 1
        if is_rejected:
            results["by_retrieval_score_bin"][bin_label]["correctly_rejected"] += 1
    
    # Вычисляем метрики
    none_accuracy = (
        results["correctly_rejected"] / results["total"]
        if results["total"] > 0 else 0.0
    )
    
    false_positive_rate = (
        results["false_positive"] / results["total"]
        if results["total"] > 0 else 0.0
    )
    
    # Вывод результатов
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"  Total hard unknown samples:     {results['total']}")
    print(f"  Correctly rejected:             {results['correctly_rejected']}")
    print(f"  False positives:                {results['false_positive']}")
    print(f"\n  📊 PRODUCTION NONE ACCURACY:    {none_accuracy:.3f}")
    print(f"  📊 FALSE POSITIVE RATE:         {false_positive_rate:.3f}")
    print("="*70)
    
    # Статистика confidence
    if confidence_scores:
        print(f"\n📈 CONFIDENCE SCORES DISTRIBUTION:")
        print(f"  Mean:    {np.mean(confidence_scores):.3f}")
        print(f"  Median:  {np.median(confidence_scores):.3f}")
        print(f"  Std:     {np.std(confidence_scores):.3f}")
        print(f"  Min:     {np.min(confidence_scores):.3f}")
        print(f"  Max:     {np.max(confidence_scores):.3f}")
        print(f"  >= {threshold:.2f}: {sum(1 for s in confidence_scores if s >= threshold) / len(confidence_scores) * 100:.1f}%")
    
    # Статистика retrieval scores
    if retrieval_scores:
        print(f"\n📈 RETRIEVAL SCORES DISTRIBUTION (hard negatives):")
        print(f"  Mean:    {np.mean(retrieval_scores):.3f}")
        print(f"  Median:  {np.median(retrieval_scores):.3f}")
        print(f"  >= 0.85: {sum(1 for s in retrieval_scores if s >= 0.85) / len(retrieval_scores) * 100:.1f}%")
        print(f"  >= 0.90: {sum(1 for s in retrieval_scores if s >= 0.90) / len(retrieval_scores) * 100:.1f}%")
    
    # Детализация по бинам retrieval score
    print(f"\n📊 NONE ACCURACY BY RETRIEVAL SCORE BIN:")
    print(f"  {'Bin':<15} {'Total':>8} {'Rejected':>10} {'Accuracy':>10}")
    print("  " + "-"*45)
    
    sorted_bins = sorted(results["by_retrieval_score_bin"].items(), key=lambda x: float(x[0].split("-")[0]))
    for bin_label, bin_stats in sorted_bins:
        acc = bin_stats["correctly_rejected"] / bin_stats["total"] if bin_stats["total"] > 0 else 0.0
        print(f"  {bin_label:<15} {bin_stats['total']:>8} {bin_stats['correctly_rejected']:>10} {acc:>10.3f}")
    
    # Сохраняем результаты
    output_path = Path(predictions_path).parent / "production_unknown_results.json"
    output_data = {
        "threshold": threshold,
        "hard_unknown_path": hard_unknown_path,
        "metrics": {
            "production_none_accuracy": none_accuracy,
            "false_positive_rate": false_positive_rate,
            "total_samples": results["total"],
            "correctly_rejected": results["correctly_rejected"],
            "false_positives": results["false_positive"]
        },
        "confidence_distribution": {
            "mean": float(np.mean(confidence_scores)) if confidence_scores else 0.0,
            "median": float(np.median(confidence_scores)) if confidence_scores else 0.0,
            "std": float(np.std(confidence_scores)) if confidence_scores else 0.0,
            "min": float(np.min(confidence_scores)) if confidence_scores else 0.0,
            "max": float(np.max(confidence_scores)) if confidence_scores else 0.0
        },
        "by_retrieval_score_bin": results["by_retrieval_score_bin"]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Результаты сохранены в: {output_path}")
    print("="*70)
    
    return output_data


# ============================================================
# ТОЧКА ВХОДА
# ============================================================

if __name__ == "__main__":
    evaluate_production_unknown(
        predictions_path=_PREDICTIONS_PATH,
        hard_unknown_path=_HARD_UNKNOWN_PATH,
        threshold=_THRESHOLD
    )
