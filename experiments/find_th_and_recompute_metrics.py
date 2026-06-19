# -*- coding: utf-8 -*-
"""
Выбор оптимального порога отсечки.

Что делает скрипт:
1. Загружает predictions JSON (scores = сырые P(Yes) от VLM)
2. Делит predictions на cal-split (подбор порога) и eval-split (оценка)
3. Подбирает оптимальный порог для none-of-the-above detection по F1-macro
   на cal-split (без data leakage)
4. Пересчитывает все метрики с optimal threshold на eval-split
   и добавляет их в исходный файл с метриками
5. Сравнивает: исходные (threshold=0.5) vs оптимальный порог
6. Выводит итоговую таблицу сравнения

Запуск:
    python experiments/find_th_and_recompute_metrics.py
"""
import json
from typing import cast
import numpy as np
from numpy.random import default_rng
from sklearn.metrics import (
    f1_score, roc_auc_score, roc_curve,
)


# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================
_PREDICTIONS_PATH = (
    "experiments/results/"
    "val_rerank_exp_r16_alpha32_lr2e-5_rerank_full_lora_448_dataset_v3_lora_predictions.json"
)

_EVAL_JSON_PATH = (
    "experiments/results/"
    "val_rerank_exp_r16_alpha32_lr2e-5_rerank_full_lora_448_dataset_v3.json"
)
_MODEL_NAME = "lora_vlm"
_K_VALUES = [1, 3, 5, 10]

# Доля данных для подбора порога (cal-split).
# Остаток идёт в eval-split для честной оценки метрик.
# Разбивка стратифицирована по known/unknown.
_CAL_SPLIT_RATIO = 0.5
_RANDOM_SEED = 42


# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================

def load_predictions(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return cast(list, json.load(f))


def load_eval_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return cast(dict, json.load(f))


# ============================================================
# CAL / EVAL SPLIT
# ============================================================

def stratified_cal_eval_split(
    predictions: list,
    cal_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[list, list]:
    """Стратифицированный split predictions на cal и eval части.

    Стратификация по known (target_idx != -1) / unknown (target_idx == -1),
    чтобы оба класса были представлены в обоих сплитах.

    Args:
        predictions: список prediction-словарей.
        cal_ratio: доля данных для cal-split (подбор порога).
        seed: seed для воспроизводимости.

    Returns:
        (cal_preds, eval_preds) — два непересекающихся списка.
    """
    if not 0.0 < cal_ratio < 1.0:
        raise ValueError(
            f"cal_ratio должен быть в (0, 1), получено: {cal_ratio}"
        )

    rng = default_rng(seed)

    known_idx = [
        i for i, p in enumerate(predictions) if p["target_idx"] != -1
    ]
    unknown_idx = [
        i for i, p in enumerate(predictions) if p["target_idx"] == -1
    ]

    def split_indices(indices: list) -> tuple[list, list]:
        shuffled = rng.permutation(indices).tolist()
        n_cal = max(1, round(len(shuffled) * cal_ratio))
        return shuffled[:n_cal], shuffled[n_cal:]

    cal_known, eval_known = split_indices(known_idx)
    cal_unknown, eval_unknown = split_indices(unknown_idx)

    cal_set = set(cal_known + cal_unknown)
    eval_set = set(eval_known + eval_unknown)

    cal_preds = [predictions[i] for i in sorted(cal_set)]
    eval_preds = [predictions[i] for i in sorted(eval_set)]

    return cal_preds, eval_preds


# ============================================================
# МЕТРИКИ
# ============================================================

def compute_hit_mrr(
    predictions: list, cal_map: dict, k_values: list
) -> dict:
    """Hit@K и MRR с (опционально) откалиброванными скорами."""
    hits = {k: 0 for k in k_values}
    mrr_sum = 0.0
    total_valid = 0

    for i, pred in enumerate(predictions):
        target_idx = pred["target_idx"]
        if target_idx == -1:
            continue
        scores = cal_map.get(i, pred["scores"])
        ranked = np.argsort(scores)[::-1]
        for k in k_values:
            if target_idx in ranked[:k]:
                hits[k] += 1
        pos = np.where(ranked == target_idx)[0]
        if len(pos) > 0:
            mrr_sum += 1.0 / (pos[0] + 1)
        total_valid += 1

    metrics = {}
    for k in k_values:
        metrics[f"hit_{k}"] = hits[k] / total_valid if total_valid > 0 else 0.0
    metrics["mrr"] = mrr_sum / total_valid if total_valid > 0 else 0.0
    metrics["total_valid"] = total_valid
    return metrics


def compute_none_metrics(
    predictions: list, cal_map: dict, threshold: float
) -> dict:
    """Метрики none-detection при заданном пороге."""
    true_labels = []
    pred_labels = []
    max_scores = []

    for i, pred in enumerate(predictions):
        scores = cal_map.get(i, pred["scores"])
        max_sc = float(max(scores)) if scores else 0.0
        max_scores.append(max_sc)
        is_known = 0 if pred["target_idx"] == -1 else 1
        true_labels.append(is_known)
        pred_labels.append(1 if max_sc >= threshold else 0)

    true_arr = np.array(true_labels)
    pred_arr = np.array(pred_labels)
    max_arr = np.array(max_scores)

    n_none = int((true_arr == 0).sum())
    n_known = int((true_arr == 1).sum())

    none_acc = float(
        ((true_arr == 0) & (pred_arr == 0)).sum() / n_none
    ) if n_none > 0 else 0.0
    known_acc = float(
        ((true_arr == 1) & (pred_arr == 1)).sum() / n_known
    ) if n_known > 0 else 0.0

    m = {
        "threshold": threshold,
        "none_accuracy": none_acc,
        "known_accuracy": known_acc,
    }

    if len(set(true_labels)) > 1:
        m["f1_known"] = float(
            f1_score(true_arr, pred_arr, pos_label=1, zero_division=0)
        )
        m["f1_unknown"] = float(
            f1_score(true_arr, pred_arr, pos_label=0, zero_division=0)
        )
        m["f1_macro"] = float(
            f1_score(true_arr, pred_arr, average="macro", zero_division=0)
        )
        # AUROC: pos_label=1 (known), т.е. score = P(known).
        # Если нужен AUROC для unknown как positive — инвертируйте метку.
        m["auroc"] = float(roc_auc_score(true_arr, max_arr))
        fpr_v, tpr_v, _ = roc_curve(true_arr, max_arr)
        # roc_curve возвращает точки в порядке убывания threshold →
        # TPR возрастает. Ищем первый индекс, где TPR >= 0.95.
        # np.interp ненадёжен при дублирующихся TPR (дискретные скоры).
        idx = int(np.searchsorted(tpr_v, 0.95))
        if idx >= len(fpr_v):
            idx = len(fpr_v) - 1
        m["fpr_at_95tpr"] = float(fpr_v[idx])
    else:
        m["f1_known"] = 0.0
        m["f1_unknown"] = 0.0
        m["f1_macro"] = 0.0
        m["auroc"] = 0.0
        m["fpr_at_95tpr"] = 0.0

    return m


def find_optimal_threshold(
    predictions: list,
    cal_map: dict,
    thresholds=None,
    metric: str = "f1_macro",
) -> tuple:
    """Перебирает пороги, возвращает (best_threshold, best_score, all_results).

    Если thresholds не задан, использует уникальные значения max(scores)
    по всем predictions — это даёт точный перебор без пропуска оптимума.
    """
    if thresholds is None:
        # Используем уникальные max-скоры как кандидаты порогов
        max_scores = []
        for i, pred in enumerate(predictions):
            scores = cal_map.get(i, pred["scores"])
            if scores:
                max_scores.append(float(max(scores)))
        thresholds = sorted(set(max_scores))
        # Добавляем граничные значения чуть ниже минимума и выше максимума
        if thresholds:
            thresholds = (
                [thresholds[0] - 1e-6]
                + thresholds
                + [thresholds[-1] + 1e-6]
            )

    best_t = 0.5
    best_score = -1.0
    all_results = []

    for t in thresholds:
        m = compute_none_metrics(predictions, cal_map, t)
        score = m.get(metric, 0.0)
        all_results.append(m)
        if score > best_score:
            best_score = score
            best_t = t

    return best_t, best_score, all_results


# ============================================================
# СВОДНАЯ ТАБЛИЦА
# ============================================================

def print_comparison_table(
    baseline: dict,
    opt: dict,
    model_name: str,
    k_values: list,
):
    """Печатает таблицу сравнения threshold=0.5 vs optimal threshold."""
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ: threshold=0.5  vs  optimal threshold")
    print("=" * 70)

    col_w = 20
    header = f"{'Метрика':<30}{'threshold=0.5':>{col_w}}{'opt threshold':>{col_w}}"
    print(header)
    print("-" * len(header))

    def row(label, v_base, v_opt):
        delta = v_opt - v_base
        sign = "+" if delta >= 0 else ""
        print(
            f"  {label:<28}"
            f"{v_base:>{col_w}.4f}"
            f"{v_opt:>{col_w}.4f}"
            f"  ({sign}{delta:.4f})"
        )

    for k in k_values:
        row(
            f"Hit@{k}",
            baseline["hit_mrr"].get(f"hit_{k}", 0.0),
            opt["hit_mrr"].get(f"hit_{k}", 0.0),
        )
    row("MRR", baseline["hit_mrr"]["mrr"], opt["hit_mrr"]["mrr"])
    print()
    row("None-Accuracy", baseline["none"]["none_accuracy"], opt["none"]["none_accuracy"])
    row("Known-Accuracy", baseline["none"]["known_accuracy"], opt["none"]["known_accuracy"])
    row("F1-known", baseline["none"]["f1_known"], opt["none"]["f1_known"])
    row("F1-unknown", baseline["none"]["f1_unknown"], opt["none"]["f1_unknown"])
    row("F1-macro", baseline["none"]["f1_macro"], opt["none"]["f1_macro"])
    print()
    row("AUROC", baseline["none"]["auroc"], opt["none"]["auroc"])
    row("FPR@95TPR", baseline["none"]["fpr_at_95tpr"], opt["none"]["fpr_at_95tpr"])
    print()
    print(f"  {'Threshold':<28}{0.5:>{col_w}.4f}{opt['threshold']:>{col_w}.4f}")
    print("=" * 70)


# ============================================================
# ОБНОВЛЕНИЕ EVAL JSON
# ============================================================

def update_eval_json(
    eval_json: dict,
    model_name: str,
    opt_threshold: float,
    opt_none_metrics: dict,
    opt_hit_mrr: dict,
    k_values: list,
) -> dict:
    """Добавляет метрики с оптимальным порогом в eval JSON.

    Изменения в eval JSON:
    - config.optimal_threshold_for_none — найденный порог
    - секция '{model_name}_opt_th' — все метрики при opt пороге
      (отдельно от основной секции модели и summary)
    """
    suffix = "_opt_th"

    # 1. Записываем найденный порог в config
    if "config" not in eval_json:
        eval_json["config"] = {}
    eval_json["config"]["optimal_threshold_for_none"] = opt_threshold

    # 2. Формируем метрики opt-порога
    opt_section_name = f"{model_name}_opt_th"
    opt_keys: dict = {
        "threshold": opt_threshold,
        f"{model_name}_none_accuracy{suffix}": (
            opt_none_metrics["none_accuracy"]
        ),
        f"{model_name}_known_accuracy{suffix}": (
            opt_none_metrics["known_accuracy"]
        ),
        f"{model_name}_f1_known{suffix}": opt_none_metrics["f1_known"],
        f"{model_name}_f1_unknown{suffix}": (
            opt_none_metrics["f1_unknown"]
        ),
        f"{model_name}_f1_macro{suffix}": opt_none_metrics["f1_macro"],
        f"{model_name}_unknown_auroc{suffix}": opt_none_metrics["auroc"],
        f"{model_name}_unknown_fpr_at_95tpr{suffix}": (
            opt_none_metrics["fpr_at_95tpr"]
        ),
    }
    for k in k_values:
        opt_keys[f"{model_name}_hit_{k}{suffix}"] = opt_hit_mrr.get(
            f"hit_{k}", 0.0
        )
    opt_keys[f"{model_name}_mrr{suffix}"] = opt_hit_mrr["mrr"]

    # 3. Пишем в отдельную секцию — не трогаем summary
    eval_json[opt_section_name] = opt_keys

    return eval_json


# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def run_threshold_analysis(
    cal_preds: list,
    eval_preds: list,
    model_name: str,
    k_values: list,
) -> dict:
    """Подбирает оптимальный порог на cal_preds, оценивает на eval_preds.

    Порог подбирается на cal-split (без data leakage).
    Все финальные метрики считаются на eval-split.
    Baseline (threshold=0.5) также считается на eval-split.

    Args:
        cal_preds: predictions для подбора порога (cal-split).
        eval_preds: predictions для финальной оценки (eval-split).
        model_name: имя модели для вывода.
        k_values: список K для Hit@K.

    Returns:
        dict с ключами 'baseline' и 'opt', каждый содержит
        'hit_mrr', 'none', 'threshold'.
    """
    empty_map: dict[int, list] = {}

    print(f"\n{'='*60}")
    print(f"Анализ порога: {model_name}")
    n_cal = len(cal_preds)
    n_eval = len(eval_preds)
    print(f"  Cal-split:  {n_cal} predictions (подбор порога)")
    print(f"  Eval-split: {n_eval} predictions (финальная оценка)")
    cal_known = sum(1 for p in cal_preds if p["target_idx"] != -1)
    cal_unknown = n_cal - cal_known
    eval_known = sum(1 for p in eval_preds if p["target_idx"] != -1)
    eval_unknown = n_eval - eval_known
    print(f"  Cal  — known: {cal_known}, unknown: {cal_unknown}")
    print(f"  Eval — known: {eval_known}, unknown: {eval_unknown}")
    print(f"{'='*60}")

    # --- Baseline на eval-split (threshold=0.5) ---
    print("\n" + "-" * 50)
    print("BASELINE на eval-split (threshold=0.5)")
    print("-" * 50)

    base_hit_mrr = compute_hit_mrr(eval_preds, empty_map, k_values)
    base_none = compute_none_metrics(eval_preds, empty_map, 0.5)

    for k in k_values:
        print(f"  Hit@{k}:          {base_hit_mrr[f'hit_{k}']:.4f}")
    print(f"  MRR:             {base_hit_mrr['mrr']:.4f}")
    print(f"  None-Accuracy:   {base_none['none_accuracy']:.4f}")
    print(f"  Known-Accuracy:  {base_none['known_accuracy']:.4f}")
    print(f"  F1-macro:        {base_none['f1_macro']:.4f}")
    print(f"  AUROC:           {base_none['auroc']:.4f}")
    print(f"  FPR@95TPR:       {base_none['fpr_at_95tpr']:.4f}")

    # --- Поиск оптимального порога на cal-split ---
    print("\n" + "-" * 50)
    print("ПОИСК ПОРОГА на cal-split (по F1-macro)")
    print("-" * 50)
    print("  Кандидаты: уникальные max(scores) из cal-split")

    opt_t, opt_f1_cal, all_th_results = find_optimal_threshold(
        cal_preds, empty_map, metric="f1_macro"
    )
    print(
        f"  Оптимальный порог: {opt_t:.6f}"
        f"  (F1-macro на cal={opt_f1_cal:.4f})"
    )

    # Топ-5 порогов по F1-macro на cal-split
    sorted_results = sorted(
        all_th_results,
        key=lambda x: x.get("f1_macro", 0.0),
        reverse=True,
    )
    print("\n  Топ-5 порогов (cal-split):")
    print(
        f"  {'threshold':>12}  {'F1-macro':>10}"
        f"  {'None-Acc':>10}  {'Known-Acc':>10}"
    )
    for r in sorted_results[:5]:
        print(
            f"  {r['threshold']:>12.6f}  "
            f"{r.get('f1_macro', 0.0):>10.4f}  "
            f"{r['none_accuracy']:>10.4f}  "
            f"{r['known_accuracy']:>10.4f}"
        )

    # --- Финальные метрики с opt порогом на eval-split ---
    print("\n" + "-" * 50)
    print(
        f"МЕТРИКИ НА EVAL-SPLIT (threshold={opt_t:.6f},"
        f" подобран на cal-split)"
    )
    print("-" * 50)

    opt_none = compute_none_metrics(eval_preds, empty_map, opt_t)
    # Hit@K и MRR не зависят от порога none-detection
    opt_hit_mrr = base_hit_mrr

    for k in k_values:
        print(
            f"  Hit@{k}:          {opt_hit_mrr[f'hit_{k}']:.4f}"
            f"  (не зависит от порога)"
        )
    print(
        f"  MRR:             {opt_hit_mrr['mrr']:.4f}"
        f"  (не зависит от порога)"
    )
    print()
    delta_none = opt_none["none_accuracy"] - base_none["none_accuracy"]
    delta_known = (
        opt_none["known_accuracy"] - base_none["known_accuracy"]
    )
    delta_f1 = opt_none["f1_macro"] - base_none["f1_macro"]
    print(
        f"  None-Accuracy:   {opt_none['none_accuracy']:.4f}"
        f"  ({delta_none:+.4f})"
    )
    print(
        f"  Known-Accuracy:  {opt_none['known_accuracy']:.4f}"
        f"  ({delta_known:+.4f})"
    )
    print(f"  F1-known:        {opt_none['f1_known']:.4f}")
    print(f"  F1-unknown:      {opt_none['f1_unknown']:.4f}")
    print(
        f"  F1-macro:        {opt_none['f1_macro']:.4f}"
        f"  ({delta_f1:+.4f})"
    )
    print(f"  AUROC:           {opt_none['auroc']:.4f}")
    print(f"  FPR@95TPR:       {opt_none['fpr_at_95tpr']:.4f}")

    results = {
        "baseline": {
            "hit_mrr": base_hit_mrr,
            "none": base_none,
            "threshold": 0.5,
        },
        "opt": {
            "hit_mrr": opt_hit_mrr,
            "none": opt_none,
            "threshold": opt_t,
        },
    }

    print_comparison_table(
        results["baseline"], results["opt"], model_name, k_values
    )

    return results


# ============================================================
# ТОЧКА ВХОДА
# ============================================================

if __name__ == "__main__":
    import os

    print("Загрузка predictions...")
    all_predictions = load_predictions(_PREDICTIONS_PATH)
    print(
        f"Загружено {len(all_predictions)} predictions"
        f" из {_PREDICTIONS_PATH}"
    )

    # Стратифицированный split: cal для подбора порога, eval для оценки
    cal_predictions, eval_predictions = stratified_cal_eval_split(
        all_predictions,
        cal_ratio=_CAL_SPLIT_RATIO,
        seed=_RANDOM_SEED,
    )
    print(
        f"Split: cal={len(cal_predictions)},"
        f" eval={len(eval_predictions)}"
        f" (ratio={_CAL_SPLIT_RATIO})"
    )

    results = run_threshold_analysis(
        cal_preds=cal_predictions,
        eval_preds=eval_predictions,
        model_name=_MODEL_NAME,
        k_values=_K_VALUES,
    )

    # Обновляем eval JSON с метриками при оптимальном пороге
    print(f"\nОбновление eval JSON: {_EVAL_JSON_PATH}")
    eval_json = load_eval_json(_EVAL_JSON_PATH)

    eval_json = update_eval_json(
        eval_json=eval_json,
        model_name=_MODEL_NAME,
        opt_threshold=results["opt"]["threshold"],
        opt_none_metrics=results["opt"]["none"],
        opt_hit_mrr=results["opt"]["hit_mrr"],
        k_values=_K_VALUES,
    )

    # Атомарная запись: сначала во временный файл, затем os.replace
    tmp_path = _EVAL_JSON_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(eval_json, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, _EVAL_JSON_PATH)
    print(
        f"✓ Метрики с оптимальным порогом добавлены в {_EVAL_JSON_PATH}"
    )

    # Итог
    opt_t = results["opt"]["threshold"]
    opt_f1 = results["opt"]["none"]["f1_macro"]
    base_f1 = results["baseline"]["none"]["f1_macro"]
    print("\nИтог:")
    print(f"  Оптимальный порог (подобран на cal-split): {opt_t:.4f}")
    print(
        f"  F1-macro (eval-split):"
        f" {base_f1:.4f} (t=0.5)"
        f" -> {opt_f1:.4f} (opt t)"
        f"  ({opt_f1 - base_f1:+.4f})"
    )
