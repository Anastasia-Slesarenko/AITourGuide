# -*- coding: utf-8 -*-
"""
Подбор порога отсечки known/unknown для retrieval baseline (без реранкера).

Зачем отдельный скрипт:
    Для zero-shot и LoRA порог подбирается в find_th_and_recompute_metrics.py по
    сигналу max(P(yes)) на val-предсказаниях из eval.py. У retrieval baseline
    реранкера нет — сигналом отсечки служит max(retrieval_score) по кандидатам.
    Эти скоры уже лежат в val.json (step6), VLM не нужен.

    Скрипт переиспользует ТЕ ЖЕ функции find_th (stratified_cal_eval_split,
    run_threshold_analysis): тот же cal/eval-сплит, тот же критерий F1-macro,
    та же held-out оценка. Значит порог baseline получен идентично zero-shot/LoRA
    и три столбца в итоговой таблице сопоставимы по методике подбора.

Как использовать результат:
    Полученный порог подставить в e2e_eval_v2.py как RETRIEVAL_THRESHOLD
    (USE_RERANKER=False) и посчитать e2e на test — как для остальных моделей.

Запуск (из каталога scripts/):
    python experiments/find_baseline_threshold.py \
        ../data/processed/dataset_v1/val.json
"""
from __future__ import annotations

import sys
from typing import Dict, List

from find_th_and_recompute_metrics import (
    load_eval_json,  # noqa: F401  (переиспользуем формат загрузки при желании)
    run_threshold_analysis,
    stratified_cal_eval_split,
)

# Стратегия отсечки: тот же критерий, что для zero-shot/LoRA.
_MODEL_NAME = "retrieval_baseline"
_K_VALUES = [1, 3, 5, 10]
_CAL_SPLIT_RATIO = 0.5   # как в find_th
_RANDOM_SEED = 42        # как в find_th


def build_baseline_predictions(val_samples: List[Dict]) -> List[Dict]:
    """Строит predictions в формате find_th из step6-сэмплов val.

    Сигнал отсечки baseline = retrieval_score кандидата (косинус, агрегированный).
    Формат совпадает с VLM-предсказаниями: список scores по кандидатам + target_idx,
    поэтому find_th обрабатывает их той же логикой.
    """
    predictions = []
    skipped = 0
    for s in val_samples:
        candidates = s.get("candidates", [])
        if not candidates:
            skipped += 1
            continue
        scores = [float(c["retrieval_score"]) for c in candidates]
        predictions.append({
            "scores": scores,
            "target_idx": s.get("target_idx", -1),
        })
    if skipped:
        print(f"⚠️  Пропущено {skipped} сэмплов без кандидатов")
    return predictions


def main() -> None:
    import json

    val_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/dataset_v1/val.json"

    print("=" * 60)
    print("ПОДБОР ПОРОГА для RETRIEVAL BASELINE (сигнал = retrieval_score)")
    print("=" * 60)
    print(f"  Val: {val_path}")

    with open(val_path, "r", encoding="utf-8") as f:
        val_samples = json.load(f)
    print(f"  Загружено {len(val_samples)} сэмплов")

    predictions = build_baseline_predictions(val_samples)
    print(f"  Построено {len(predictions)} baseline-предсказаний")

    # Тот же стратифицированный cal/eval-сплит, что для zero-shot/LoRA.
    cal_preds, eval_preds = stratified_cal_eval_split(
        predictions, cal_ratio=_CAL_SPLIT_RATIO, seed=_RANDOM_SEED
    )

    # Та же процедура: порог по F1-macro на cal-split, оценка на eval-split.
    results = run_threshold_analysis(
        cal_preds=cal_preds,
        eval_preds=eval_preds,
        model_name=_MODEL_NAME,
        k_values=_K_VALUES,
    )

    opt_t = results["opt"]["threshold"]
    opt_f1 = results["opt"]["none"]["f1_macro"]

    print("\n" + "=" * 60)
    print("ИТОГ ДЛЯ BASELINE")
    print("=" * 60)
    print(f"  Оптимальный порог (retrieval_score): {opt_t:.6f}")
    print(f"  F1-macro на held-out eval-split:     {opt_f1:.4f}")
    print(f"\n  → подставить в e2e_eval_v2.py: RETRIEVAL_THRESHOLD = {opt_t:.6f}")
    print(f"     (USE_RERANKER = False) и посчитать e2e на test.")
    print("=" * 60)


if __name__ == "__main__":
    main()
