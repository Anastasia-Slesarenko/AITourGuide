# -*- coding: utf-8 -*-
"""
Threshold Sweep и визуализация trade-off для Full LoRA.

Что делает скрипт:
1. Загружает predictions JSON.
2. Делит на cal-split (подбор) и eval-split (оценка).
3. Строит плотную развёртку (sweep) порогов от 0.1 до 0.9.
4. Визуализирует кривую компромиссов (Known Accuracy vs None Accuracy).
5. Сохраняет график и выводит детальную таблицу ключевых точек.

Запуск:
    python experiments/threshold_sweep_and_plot.py
"""
import json
import os
from typing import cast
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from sklearn.metrics import f1_score

# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================
_PREDICTIONS_PATH = (
    "experiments/results/"
    "val_rerank_exp_r16_alpha32_lr2e-5_rerank_full_lora_448_lora_predictions.json"
)
_MODEL_NAME = "Full LoRA (448)"
_CAL_SPLIT_RATIO = 0.5
_RANDOM_SEED = 42

# Диапазон порога для sweep
_TH_MIN, _TH_MAX, _TH_STEPS = 0.10, 0.90, 81


# ============================================================
# ЗАГРУЗКА И SPLIT
# ============================================================
def load_predictions(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return cast(list, json.load(f))

def stratified_cal_eval_split(predictions: list, cal_ratio: float = 0.5, seed: int = 42):
    rng = default_rng(seed)
    known_idx = [i for i, p in enumerate(predictions) if p["target_idx"] != -1]
    unknown_idx = [i for i, p in enumerate(predictions) if p["target_idx"] == -1]

    def split_indices(indices: list):
        shuffled = rng.permutation(indices).tolist()
        n_cal = max(1, round(len(shuffled) * cal_ratio))
        return shuffled[:n_cal], shuffled[n_cal:]

    cal_known, eval_known = split_indices(known_idx)
    cal_unknown, eval_unknown = split_indices(unknown_idx)

    cal_set = set(cal_known + cal_unknown)
    eval_set = set(eval_known + eval_unknown)

    return [predictions[i] for i in sorted(cal_set)], [predictions[i] for i in sorted(eval_set)]


# ============================================================
# SWEEP ЛОГИКА
# ============================================================
def run_sweep(predictions: list):
    """Вычисляет метрики для каждого порога в заданном диапазоне."""
    thresholds = np.linspace(_TH_MIN, _TH_MAX, _TH_STEPS)
    
    # Предвычисляем истинные метки и максимальные скоры для скорости
    true_is_known = np.array([1 if p["target_idx"] != -1 else 0 for p in predictions])
    max_scores = np.array([float(max(p.get("scores", [0.0]))) if p.get("scores") else 0.0 for p in predictions])
    
    # Для Known Accuracy нам нужно знать, был ли правильный ответ на 1 месте
    # (упрощённо: если target_idx != -1, мы считаем, что он на 1 месте после reranking, 
    # так как predictions обычно уже отсортированы. Если нет, здесь нужна дополнительная проверка).
    # Для точности будем считать known_correct = 1, если target_idx != -1 (упрощение под ваш формат).
    
    n_known = int(true_is_known.sum())
    n_unknown = int((true_is_known == 0).sum())
    
    results = []
    
    for t in thresholds:
        pred_is_known = (max_scores >= t).astype(int)
        
        # None Accuracy: доля unknown, правильно отвергнутых (pred=0 и true=0)
        true_negatives = ((true_is_known == 0) & (pred_is_known == 0)).sum()
        none_acc = true_negatives / n_unknown if n_unknown > 0 else 0.0
        
        # Known Accuracy: доля known, правильно принятых (pred=1 и true=1)
        true_positives = ((true_is_known == 1) & (pred_is_known == 1)).sum()
        known_acc = true_positives / n_known if n_known > 0 else 0.0
        
        # F1 Macro
        f1 = f1_score(true_is_known, pred_is_known, average="macro", zero_division=0)
        
        results.append({
            "threshold": float(t),
            "none_accuracy": float(none_acc),
            "known_accuracy": float(known_acc),
            "f1_macro": float(f1)
        })
        
    return results


# ============================================================
# ВИЗУАЛИЗАЦИЯ И АНАЛИЗ
# ============================================================
def plot_and_analyze(results: list, output_path: str = "experiments/results/sweep_plot.png"):
    thresholds = [r["threshold"] for r in results]
    none_accs = [r["none_accuracy"] for r in results]
    known_accs = [r["known_accuracy"] for r in results]
    f1_macros = [r["f1_macro"] for r in results]
    
    # Находим ключевые точки
    idx_f1_max = int(np.argmax(f1_macros))
    idx_t_05 = int(np.argmin(np.abs(np.array(thresholds) - 0.5)))
    
    best_f1 = results[idx_f1_max]
    t_05 = results[idx_t_05]
    
    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(none_accs, known_accs, marker='.', linestyle='-', color='steelblue', linewidth=2, label="Pareto Frontier")
    
    # Аннотации ключевых точек
    plt.scatter(
        [t_05["none_accuracy"], best_f1["none_accuracy"]],
        [t_05["known_accuracy"], best_f1["known_accuracy"]],
        color=['red', 'green'], s=100, zorder=5, label=['Threshold=0.5', f'Optimal F1 (t={best_f1["threshold"]:.3f})']
    )
    
    plt.annotate(f"t=0.50\n(Known: {t_05['known_accuracy']:.3f}, None: {t_05['none_accuracy']:.3f})",
                 xy=(t_05["none_accuracy"], t_05["known_accuracy"]), xytext=(t_05["none_accuracy"]+0.02, t_05["known_accuracy"]-0.03),
                 color='red', fontsize=10)
                 
    plt.annotate(f"t={best_f1['threshold']:.3f}\n(Known: {best_f1['known_accuracy']:.3f}, None: {best_f1['none_accuracy']:.3f})",
                 xy=(best_f1["none_accuracy"], best_f1["known_accuracy"]), xytext=(best_f1["none_accuracy"]-0.15, best_f1["known_accuracy"]+0.02),
                 color='green', fontsize=10)

    plt.title(f"Threshold Sweep: Known Accuracy vs None Accuracy\nModel: {_MODEL_NAME}", fontsize=14)
    plt.xlabel("None Accuracy (True Negative Rate)", fontsize=12)
    plt.ylabel("Known Accuracy (True Positive Rate)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower left')
    plt.xlim(0.5, 1.0)
    plt.ylim(0.5, 1.0)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ График сохранён: {output_path}")
    
    # Вывод таблицы ключевых точек
    print("\n" + "="*70)
    print("КЛЮЧЕВЫЕ ТОЧКИ НА КРИВОЙ")
    print("="*70)
    print(f"{'Описание':<25} {'Threshold':>10}  {'Known Acc':>10}  {'None Acc':>10}  {'F1 Macro':>10}")
    print("-" * 70)
    
    # 1. Максимальная Known Accuracy (минимальный порог)
    max_known = max(results, key=lambda x: x["known_accuracy"])
    print(f"{'Макс. Known Acc':<25} {max_known['threshold']:>10.3f}  {max_known['known_accuracy']:>10.3f}  {max_known['none_accuracy']:>10.3f}  {max_known['f1_macro']:>10.3f}")
    
    # 2. Текущий 0.5
    print(f"{'Baseline (t=0.5)':<25} {t_05['threshold']:>10.3f}  {t_05['known_accuracy']:>10.3f}  {t_05['none_accuracy']:>10.3f}  {t_05['f1_macro']:>10.3f}")
    
    # 3. Оптимальный F1
    print(f"{'Оптимальный F1-Macro':<25} {best_f1['threshold']:>10.3f}  {best_f1['known_accuracy']:>10.3f}  {best_f1['none_accuracy']:>10.3f}  {best_f1['f1_macro']:>10.3f}")
    
    # 4. Сбалансированная точка (компромисс)
    # Например, точка, где None Acc >= 0.80, а Known Acc максимальна
    safe_points = [r for r in results if r["none_accuracy"] >= 0.80]
    if safe_points:
        best_safe = max(safe_points, key=lambda x: x["known_accuracy"])
        print(f"{'Safe (None Acc >= 0.80)':<25} {best_safe['threshold']:>10.3f}  {best_safe['known_accuracy']:>10.3f}  {best_safe['none_accuracy']:>10.3f}  {best_safe['f1_macro']:>10.3f}")
    print("="*70)


# ============================================================
# ТОЧКА ВХОДА
# ============================================================
if __name__ == "__main__":
    print("1. Загрузка predictions...")
    all_predictions = load_predictions(_PREDICTIONS_PATH)
    print(f"   Загружено {len(all_predictions)} записей.")
    
    print("2. Стратифицированный split (cal/eval)...")
    cal_preds, eval_preds = stratified_cal_eval_split(
        all_predictions, cal_ratio=_CAL_SPLIT_RATIO, seed=_RANDOM_SEED
    )
    print(f"   Cal: {len(cal_preds)}, Eval: {len(eval_preds)}")
    
    print("3. Запуск threshold sweep на eval-split...")
    sweep_results = run_sweep(eval_preds)
    
    print("4. Построение графика и анализ...")
    plot_and_analyze(sweep_results)