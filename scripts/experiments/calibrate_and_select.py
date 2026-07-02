# -*- coding: utf-8 -*-
"""
Калибровка VLM-скоров и выбор оптимального порога отсечки.

Что делает скрипт:
1. Загружает predictions JSON (scores = сырые P(Yes) от VLM)
2. Применяет три метода калибровки:
   - Temperature Scaling (масштабирование логитов)
   - Platt Scaling (логистическая регрессия)
   - Isotonic Regression (непараметрическая)
3. Подбирает оптимальный порог для none-of-the-above detection по F1-macro
4. Пересчитывает все метрики с откалиброванными скорами
5. Сравнивает: исходные (threshold=0.5) vs калиброванные (optimal threshold)
6. Выводит итоговую таблицу сравнения

Запуск:
    python experiments/calibrate_and_select.py

Примечание: калибровка и оценка на одних и тех же данных (val set) -
это in-sample оценка. Для честной оценки нужен отдельный cal-сплит.
Результаты показывают потенциал калибровки, но могут быть оптимистичными.
"""
import json
import os
import sys
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit  # sigmoid
from sklearn.metrics import (
    f1_score, roc_auc_score, roc_curve,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

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

# None = in-sample (калибровка и оценка на всех данных, оптимистично)
# 0.3  = 30% на калибровку, 70% на оценку (честнее)
_CAL_SPLIT = None


# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================

def load_predictions(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_flat_pairs(predictions: list) -> tuple:
    """Плоские массивы (score, label) для калибровки reranker-скоров.

    Включает только known-сэмплы (target_idx != -1).
    label=1 если кандидат является target, иначе 0.

    Returns:
        scores_flat: np.ndarray [N]
        labels_flat: np.ndarray [N]
    """
    scores_flat = []
    labels_flat = []
    for pred in predictions:
        target_idx = pred["target_idx"]
        if target_idx == -1:
            continue
        for j, sc in enumerate(pred["scores"]):
            scores_flat.append(sc)
            labels_flat.append(1 if j == target_idx else 0)
    return (
        np.array(scores_flat, dtype=np.float64),
        np.array(labels_flat, dtype=np.float64),
    )


# ============================================================
# МЕТОДЫ КАЛИБРОВКИ
# ============================================================

def temperature_scaling_fit(scores: np.ndarray, labels: np.ndarray) -> float:
    """Находит оптимальную температуру T минимизацией NLL.

    logit(p) / T -> sigmoid -> P(Yes|calibrated)
    T > 1: модель была overconfident (сжимает к 0.5)
    T < 1: модель была underconfident (растягивает от 0.5)
    """
    eps = 1e-7
    sc = np.clip(scores, eps, 1 - eps)
    logits = np.log(sc / (1 - sc))

    def nll(log_T):
        T = np.exp(log_T)
        cal_probs = np.clip(expit(logits / T), eps, 1 - eps)
        return -np.mean(
            labels * np.log(cal_probs) + (1 - labels) * np.log(1 - cal_probs)
        )

    result = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    return float(np.exp(result.x))


def apply_temperature(scores: np.ndarray, T: float) -> np.ndarray:
    eps = 1e-7
    sc = np.clip(scores, eps, 1 - eps)
    logits = np.log(sc / (1 - sc))
    return expit(logits / T)


def platt_scaling_fit(scores: np.ndarray, labels: np.ndarray) -> tuple:
    """Platt scaling: LR поверх скоров. Возвращает (a, b)."""
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(scores.reshape(-1, 1), labels)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    return expit(a * scores + b)


def isotonic_fit(scores: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(scores, labels)
    return ir


# ============================================================
# ПРИМЕНЕНИЕ КАЛИБРОВКИ К PREDICTIONS
# ============================================================

def calibrate_predictions(predictions: list, method: str, params) -> dict:
    """Возвращает dict {pred_index: [calibrated_scores]}.

    Args:
        method: "temperature", "platt", "isotonic"
        params: T (float) | (a, b) (tuple) | IsotonicRegression
    """
    cal_map = {}
    for i, pred in enumerate(predictions):
        raw = np.array(pred["scores"], dtype=np.float64)
        if method == "temperature":
            cal = apply_temperature(raw, params)
        elif method == "platt":
            a, b = params
            cal = apply_platt(raw, a, b)
        elif method == "isotonic":
            cal = params.predict(raw)
        else:
            cal = raw
        cal_map[i] = cal.tolist()
    return cal_map


# ============================================================
# МЕТРИКИ
# ============================================================

def compute_hit_mrr(predictions: list, cal_map: dict, k_values: list) -> dict:
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
        m["auroc"] = float(roc_auc_score(true_arr, max_arr))
        fpr_v, tpr_v, _ = roc_curve(true_arr, max_arr)
        m["fpr_at_95tpr"] = float(np.interp(0.95, tpr_v, fpr_v))
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
    """Перебирает пороги, возвращает (best_threshold, best_score, all_results)."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 37).tolist()

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


def compute_brier(predictions: list, cal_map: dict) -> float:
    """Brier Score для reranking-калибровки."""
    sq_errors = []
    for i, pred in enumerate(predictions):
        target_idx = pred["target_idx"]
        if target_idx == -1:
            continue
        scores = cal_map.get(i, pred["scores"])
        for j, sc in enumerate(scores):
            label = 1.0 if j == target_idx else 0.0
            sq_errors.append((float(sc) - label) ** 2)
    return float(np.mean(sq_errors)) if sq_errors else 0.0


def compute_ece(scores_flat: np.ndarray, labels_flat: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (scores_flat >= lo) & (scores_flat <= hi if i == n_bins - 1 else scores_flat < hi)
        if mask.sum() > 0:
            acc = labels_flat[mask].mean()
            conf = scores_flat[mask].mean()
            ece += abs(conf - acc) * mask.mean()
    return float(ece)


# ============================================================
# СВОДНАЯ ТАБЛИЦА
# ============================================================

def print_summary_table(results: dict, k_values: list):
    """Печатает итоговую таблицу сравнения методов."""
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ ТАБЛИЦА СРАВНЕНИЯ")
    print("=" * 80)

    methods = list(results.keys())
    col_w = 18

    # Заголовок
    header = f"{'Метрика':<28}" + "".join(f"{m:>{col_w}}" for m in methods)
    print(header)
    print("-" * len(header))

    def row(label, key_fn):
        vals = []
        for m in methods:
            v = key_fn(results[m])
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        print(f"  {label:<26}" + "".join(f"{v:>{col_w}}" for v in vals))

    for k in k_values:
        row(f"Hit@{k}", lambda r, k=k: r["hit_mrr"].get(f"hit_{k}", 0.0))
    row("MRR", lambda r: r["hit_mrr"]["mrr"])
    print()
    row("None-Accuracy (t=0.5)", lambda r: r["none_05"]["none_accuracy"])
    row("Known-Accuracy (t=0.5)", lambda r: r["none_05"]["known_accuracy"])
    row("F1-macro (t=0.5)", lambda r: r["none_05"]["f1_macro"])
    print()
    row("Opt. threshold", lambda r: r["opt_threshold"])
    row("None-Accuracy (opt t)", lambda r: r["none_opt"]["none_accuracy"])
    row("Known-Accuracy (opt t)", lambda r: r["none_opt"]["known_accuracy"])
    row("F1-macro (opt t)", lambda r: r["none_opt"]["f1_macro"])
    print()
    row("AUROC", lambda r: r["none_opt"]["auroc"])
    row("FPR@95TPR", lambda r: r["none_opt"]["fpr_at_95tpr"])
    row("Brier Score", lambda r: r["brier"])
    row("ECE", lambda r: r["ece"])
    print("=" * 80)


# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def run_calibration_analysis(
    predictions: list,
    model_name: str,
    k_values: list,
    cal_split: float = None,
) -> dict:
    """Полный анализ калибровки.

    Returns:
        dict с результатами всех методов для дальнейшего использования
    """
    print(f"\n{'='*60}")
    print(f"Калибровка: {model_name}")
    print(f"Всего predictions: {len(predictions)}")
    n_known = sum(1 for p in predictions if p["target_idx"] != -1)
    n_unknown = sum(1 for p in predictions if p["target_idx"] == -1)
    print(f"  Known  (target_idx != -1): {n_known}")
    print(f"  Unknown (target_idx == -1): {n_unknown}")
    print(f"{'='*60}")

    # --- Разбивка cal/eval ---
    if cal_split is not None:
        n_cal = int(len(predictions) * cal_split)
        cal_preds = predictions[:n_cal]
        eval_preds = predictions[n_cal:]
        print(f"\nРазбивка: {n_cal} cal / {len(eval_preds)} eval")
    else:
        cal_preds = predictions
        eval_preds = predictions
        print("\nIn-sample калибровка (cal == eval, результаты оптимистичны)")

    # Плоские пары для обучения калибраторов
    cal_scores_flat, cal_labels_flat = extract_flat_pairs(cal_preds)
    eval_scores_flat, eval_labels_flat = extract_flat_pairs(eval_preds)

    print(f"\nПар (score, label) для калибровки: {len(cal_scores_flat)}")
    print(f"  Позитивов: {int(cal_labels_flat.sum())}")
    print(f"  Негативов: {int((1 - cal_labels_flat).sum())}")
    print(f"  Доля позитивов: {cal_labels_flat.mean():.4f}")

    # Статистика сырых скоров
    print(f"\nСтатистика сырых P(Yes):")
    print(f"  mean={cal_scores_flat.mean():.4f}  std={cal_scores_flat.std():.4f}")
    print(f"  min={cal_scores_flat.min():.4f}  max={cal_scores_flat.max():.4f}")
    pos_mask = cal_labels_flat == 1
    neg_mask = cal_labels_flat == 0
    print(f"  Позитивы: mean={cal_scores_flat[pos_mask].mean():.4f}  "
          f"std={cal_scores_flat[pos_mask].std():.4f}")
    print(f"  Негативы: mean={cal_scores_flat[neg_mask].mean():.4f}  "
          f"std={cal_scores_flat[neg_mask].std():.4f}")

    # ============================================================
    # ИСХОДНЫЕ МЕТРИКИ (без калибровки)
    # ============================================================
    empty_map = {}
    orig_hit_mrr = compute_hit_mrr(eval_preds, empty_map, k_values)
    orig_none_05 = compute_none_metrics(eval_preds, empty_map, 0.5)
    orig_brier = compute_brier(eval_preds, empty_map)
    orig_ece = compute_ece(eval_scores_flat, eval_labels_flat)
    orig_opt_t, orig_opt_f1, _ = find_optimal_threshold(
        eval_preds, empty_map, metric="f1_macro"
    )
    orig_none_opt = compute_none_metrics(eval_preds, empty_map, orig_opt_t)

    print("\n" + "-" * 50)
    print("ИСХОДНЫЕ МЕТРИКИ (без калибровки)")
    print("-" * 50)
    for k in k_values:
        print(f"  Hit@{k}:          {orig_hit_mrr[f'hit_{k}']:.4f}")
    print(f"  MRR:             {orig_hit_mrr['mrr']:.4f}")
    print(f"  Brier Score:     {orig_brier:.4f}")
    print(f"  ECE:             {orig_ece:.4f}")
    print(f"  AUROC:           {orig_none_05['auroc']:.4f}")
    print(f"  FPR@95TPR:       {orig_none_05['fpr_at_95tpr']:.4f}")
    print(f"\n  При threshold=0.5:")
    print(f"    None-Accuracy:  {orig_none_05['none_accuracy']:.4f}")
    print(f"    Known-Accuracy: {orig_none_05['known_accuracy']:.4f}")
    print(f"    F1-macro:       {orig_none_05['f1_macro']:.4f}")
    print(f"\n  Оптимальный порог (без калибровки): {orig_opt_t:.3f}")
    print(f"    None-Accuracy:  {orig_none_opt['none_accuracy']:.4f}")
    print(f"    Known-Accuracy: {orig_none_opt['known_accuracy']:.4f}")
    print(f"    F1-macro:       {orig_opt_f1:.4f}")

    all_results = {
        "Baseline (t=0.5)": {
            "hit_mrr": orig_hit_mrr,
            "none_05": orig_none_05,
            "none_opt": orig_none_opt,
            "opt_threshold": orig_opt_t,
            "brier": orig_brier,
            "ece": orig_ece,
        }
    }

    # ============================================================
    # МЕТОД 1: TEMPERATURE SCALING
    # ============================================================
    print("\n" + "-" * 50)
    print("МЕТОД 1: Temperature Scaling")
    print("-" * 50)

    T = temperature_scaling_fit(cal_scores_flat, cal_labels_flat)
    print(f"  Оптимальная температура T = {T:.4f}")
    if T > 1.05:
        print(f"  -> Модель была overconfident (T>1 сжимает вероятности к 0.5)")
    elif T < 0.95:
        print(f"  -> Модель была underconfident (T<1 растягивает вероятности)")
    else:
        print(f"  -> Модель хорошо откалибрована (T~1)")

    ts_map = calibrate_predictions(eval_preds, "temperature", T)
    ts_scores_cal = apply_temperature(eval_scores_flat, T)

    ts_hit_mrr = compute_hit_mrr(eval_preds, ts_map, k_values)
    ts_brier = compute_brier(eval_preds, ts_map)
    ts_ece = compute_ece(ts_scores_cal, eval_labels_flat)
    ts_none_05 = compute_none_metrics(eval_preds, ts_map, 0.5)
    ts_opt_t, ts_opt_f1, _ = find_optimal_threshold(
        eval_preds, ts_map, metric="f1_macro"
    )
    ts_none_opt = compute_none_metrics(eval_preds, ts_map, ts_opt_t)

    print(f"\n  При threshold=0.5:")
    print(f"    None-Accuracy:  {ts_none_05['none_accuracy']:.4f}  "
          f"(было {orig_none_05['none_accuracy']:.4f})")
    print(f"    Known-Accuracy: {ts_none_05['known_accuracy']:.4f}  "
          f"(было {orig_none_05['known_accuracy']:.4f})")
    print(f"    F1-macro:       {ts_none_05['f1_macro']:.4f}  "
          f"(было {orig_none_05['f1_macro']:.4f})")
    print(f"\n  Оптимальный порог: {ts_opt_t:.3f}")
    print(f"    None-Accuracy:  {ts_none_opt['none_accuracy']:.4f}")
    print(f"    Known-Accuracy: {ts_none_opt['known_accuracy']:.4f}")
    print(f"    F1-macro:       {ts_opt_f1:.4f}")
    print(f"\n  Качество калибровки:")
    print(f"    Brier Score:    {ts_brier:.4f}  (было {orig_brier:.4f})")
    print(f"    ECE:            {ts_ece:.4f}  (было {orig_ece:.4f})")
    for k in k_values:
        delta = ts_hit_mrr[f"hit_{k}"] - orig_hit_mrr[f"hit_{k}"]
        sign = "+" if delta >= 0 else ""
        print(f"    Hit@{k}:         {ts_hit_mrr[f'hit_{k}']:.4f}  "
              f"({sign}{delta:.4f})")
    delta_mrr = ts_hit_mrr["mrr"] - orig_hit_mrr["mrr"]
    sign = "+" if delta_mrr >= 0 else ""
    print(f"    MRR:            {ts_hit_mrr['mrr']:.4f}  ({sign}{delta_mrr:.4f})")

    all_results["TS (opt t)"] = {
        "hit_mrr": ts_hit_mrr,
        "none_05": ts_none_05,
        "none_opt": ts_none_opt,
        "opt_threshold": ts_opt_t,
        "brier": ts_brier,
        "ece": ts_ece,
    }

    # ============================================================
    # МЕТОД 2: PLATT SCALING
    # ============================================================
    print("\n" + "-" * 50)
    print("МЕТОД 2: Platt Scaling (Logistic Regression)")
    print("-" * 50)

    a_p, b_p = platt_scaling_fit(cal_scores_flat, cal_labels_flat)
    print(f"  Параметры: a={a_p:.4f}, b={b_p:.4f}")

    platt_map = calibrate_predictions(eval_preds, "platt", (a_p, b_p))
    platt_scores_cal = apply_platt(eval_scores_flat, a_p, b_p)

    platt_hit_mrr = compute_hit_mrr(eval_preds, platt_map, k_values)
    platt_brier = compute_brier(eval_preds, platt_map)
    platt_ece = compute_ece(platt_scores_cal, eval_labels_flat)
    platt_none_05 = compute_none_metrics(eval_preds, platt_map, 0.5)
    platt_opt_t, platt_opt_f1, _ = find_optimal_threshold(
        eval_preds, platt_map, metric="f1_macro"
    )
    platt_none_opt = compute_none_metrics(eval_preds, platt_map, platt_opt_t)

    print(f"\n  При threshold=0.5:")
    print(f"    None-Accuracy:  {platt_none_05['none_accuracy']:.4f}  "
          f"(было {orig_none_05['none_accuracy']:.4f})")
    print(f"    Known-Accuracy: {platt_none_05['known_accuracy']:.4f}  "
          f"(было {orig_none_05['known_accuracy']:.4f})")
    print(f"    F1-macro:       {platt_none_05['f1_macro']:.4f}  "
          f"(было {orig_none_05['f1_macro']:.4f})")
    print(f"\n  Оптимальный порог: {platt_opt_t:.3f}")
    print(f"    None-Accuracy:  {platt_none_opt['none_accuracy']:.4f}")
    print(f"    Known-Accuracy: {platt_none_opt['known_accuracy']:.4f}")
    print(f"    F1-macro:       {platt_opt_f1:.4f}")
    print(f"\n  Качество калибровки:")
    print(f"    Brier Score:    {platt_brier:.4f}  (было {orig_brier:.4f})")
    print(f"    ECE:            {platt_ece:.4f}  (было {orig_ece:.4f})")
    for k in k_values:
        delta = platt_hit_mrr[f"hit_{k}"] - orig_hit_mrr[f"hit_{k}"]
        sign = "+" if delta >= 0 else ""
        print(f"    Hit@{k}:         {platt_hit_mrr[f'hit_{k}']:.4f}  "
              f"({sign}{delta:.4f})")
    delta_mrr = platt_hit_mrr["mrr"] - orig_hit_mrr["mrr"]
    sign = "+" if delta_mrr >= 0 else ""
    print(f"    MRR:            {platt_hit_mrr['mrr']:.4f}  ({sign}{delta_mrr:.4f})")

    all_results["Platt (opt t)"] = {
        "hit_mrr": platt_hit_mrr,
        "none_05": platt_none_05,
        "none_opt": platt_none_opt,
        "opt_threshold": platt_opt_t,
        "brier": platt_brier,
        "ece": platt_ece,
    }

    # ============================================================
    # МЕТОД 3: ISOTONIC REGRESSION
    # ============================================================
    print("\n" + "-" * 50)
    print("МЕТОД 3: Isotonic Regression (непараметрическая)")
    print("-" * 50)

    ir = isotonic_fit(cal_scores_flat, cal_labels_flat)

    iso_map = calibrate_predictions(eval_preds, "isotonic", ir)
    iso_scores_cal = ir.predict(eval_scores_flat)

    iso_hit_mrr = compute_hit_mrr(eval_preds, iso_map, k_values)
    iso_brier = compute_brier(eval_preds, iso_map)
    iso_ece = compute_ece(iso_scores_cal, eval_labels_flat)
    iso_none_05 = compute_none_metrics(eval_preds, iso_map, 0.5)
    iso_opt_t, iso_opt_f1, _ = find_optimal_threshold(
        eval_preds, iso_map, metric="f1_macro"
    )
    iso_none_opt = compute_none_metrics(eval_preds, iso_map, iso_opt_t)

    print("\n  При threshold=0.5:")
    print(f"    None-Accuracy:  {iso_none_05['none_accuracy']:.4f}  "
          f"(было {orig_none_05['none_accuracy']:.4f})")
    print(f"    Known-Accuracy: {iso_none_05['known_accuracy']:.4f}  "
          f"(было {orig_none_05['known_accuracy']:.4f})")
    print(f"    F1-macro:       {iso_none_05['f1_macro']:.4f}  "
          f"(было {orig_none_05['f1_macro']:.4f})")
    print(f"\n  Оптимальный порог: {iso_opt_t:.3f}")
    print(f"    None-Accuracy:  {iso_none_opt['none_accuracy']:.4f}")
    print(f"    Known-Accuracy: {iso_none_opt['known_accuracy']:.4f}")
    print(f"    F1-macro:       {iso_opt_f1:.4f}")
    print("\n  Качество калибровки:")
    print(f"    Brier Score:    {iso_brier:.4f}  (было {orig_brier:.4f})")
    print(f"    ECE:            {iso_ece:.4f}  (было {orig_ece:.4f})")
    for k in k_values:
        delta = iso_hit_mrr[f"hit_{k}"] - orig_hit_mrr[f"hit_{k}"]
        sign = "+" if delta >= 0 else ""
        print(f"    Hit@{k}:         {iso_hit_mrr[f'hit_{k}']:.4f}  "
              f"({sign}{delta:.4f})")
    delta_mrr = iso_hit_mrr["mrr"] - orig_hit_mrr["mrr"]
    sign = "+" if delta_mrr >= 0 else ""
    print(f"    MRR:            {iso_hit_mrr['mrr']:.4f}  ({sign}{delta_mrr:.4f})")

    all_results["Isotonic (opt t)"] = {
        "hit_mrr": iso_hit_mrr,
        "none_05": iso_none_05,
        "none_opt": iso_none_opt,
        "opt_threshold": iso_opt_t,
        "brier": iso_brier,
        "ece": iso_ece,
    }

    # ============================================================
    # ИТОГОВАЯ ТАБЛИЦА
    # ============================================================
    print_summary_table(all_results, k_values)

    # ============================================================
    # РЕКОМЕНДАЦИЯ
    # ============================================================
    print("\n" + "=" * 60)
    print("РЕКОМЕНДАЦИЯ")
    print("=" * 60)

    # Выбираем лучший метод по F1-macro (none-detection)
    best_method = max(
        all_results.items(),
        key=lambda x: x[1]["none_opt"]["f1_macro"]
    )
    best_name, best_res = best_method

    # Проверяем, улучшает ли калибровка Brier Score
    orig_brier_val = all_results["Baseline (t=0.5)"]["brier"]
    best_brier = best_res["brier"]
    brier_improved = best_brier < orig_brier_val

    print(f"\n  Лучший метод по F1-macro: {best_name}")
    print(f"    F1-macro:       {best_res['none_opt']['f1_macro']:.4f}  "
          f"(baseline: {orig_none_05['f1_macro']:.4f})")
    print(f"    Opt. threshold: {best_res['opt_threshold']:.3f}")
    print(f"    Brier Score:    {best_brier:.4f}  "
          f"({'улучшился' if brier_improved else 'ухудшился'} с {orig_brier_val:.4f})")

    # Проверяем, не деградировал ли Hit@1
    orig_hit1 = all_results["Baseline (t=0.5)"]["hit_mrr"]["hit_1"]
    best_hit1 = best_res["hit_mrr"]["hit_1"]
    hit1_delta = best_hit1 - orig_hit1

    print(f"\n  Влияние на reranking:")
    print(f"    Hit@1: {best_hit1:.4f}  (delta={hit1_delta:+.4f})")
    print(f"    MRR:   {best_res['hit_mrr']['mrr']:.4f}  "
          f"(delta={best_res['hit_mrr']['mrr'] - orig_hit_mrr['mrr']:+.4f})")

    if abs(hit1_delta) < 0.001:
        print("\n  Калибровка не влияет на reranking (Hit@1 не изменился).")
        print("  Это ожидаемо: калибровка меняет абсолютные значения скоров,")
        print("  но не их относительный порядок (монотонное преобразование).")
    elif hit1_delta < -0.005:
        print("\n  ВНИМАНИЕ: калибровка ухудшила Hit@1. Проверьте данные.")

    print("\n  Итог:")
    if best_name == "Baseline (t=0.5)":
        print("  Калибровка не улучшает метрики. Рекомендуется использовать")
        print(f"  оптимальный порог {orig_opt_t:.3f} без калибровки.")
    else:
        print(f"  Рекомендуется применить {best_name} с порогом "
              f"{best_res['opt_threshold']:.3f}.")
        print(f"  Это улучшает F1-macro с {orig_none_05['f1_macro']:.4f} "
              f"до {best_res['none_opt']['f1_macro']:.4f}.")
        if brier_improved:
            print(f"  Brier Score улучшился: {orig_brier_val:.4f} -> {best_brier:.4f}.")

    print("=" * 60)

    return all_results


# ============================================================
# ТОЧКА ВХОДА
# ============================================================

if __name__ == "__main__":
    print("Загрузка predictions...")
    predictions = load_predictions(_PREDICTIONS_PATH)
    print(f"Загружено {len(predictions)} predictions из {_PREDICTIONS_PATH}")

    results = run_calibration_analysis(
        predictions=predictions,
        model_name=_MODEL_NAME,
        k_values=_K_VALUES,
        cal_split=_CAL_SPLIT,
    )