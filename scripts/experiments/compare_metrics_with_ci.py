# experiments/compare_metrics_with_ci.py
"""
Сравнение метрик двух моделей с доверительными интервалами.

Использует:
1. Bootstrap CI для разницы метрик (paired data)
2. Пермутационный тест для p-value
3. Визуализацию распределения разницы

Запуск:
    python experiments/compare_metrics_with_ci.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


# КОНФИГУРАЦИЯ

# Пути к результатам e2e_eval.py
SIGLIP_RESULTS_PATH = "/home/jupyter/s3/ai-tour-guide/dataset/results/e2e_rerank_exp_r16_alpha32_lr2e-5_rerank_attn_448_results_by_dinov2.json"
DINOV2_RESULTS_PATH = "/home/jupyter/s3/ai-tour-guide/dataset/results/e2e_rerank_exp_r16_alpha32_lr2e-5_dinov2_rerank_attn_448_by_siglip.json"

# Параметры bootstrap
N_BOOTSTRAP = 10000  # Больше = точнее, но медленнее
CONFIDENCE = 0.95
RANDOM_SEED = 42

# Метрики для сравнения
METRICS_TO_COMPARE = [
    "hit_1",
    "unknown_detection_accuracy",
    "e2e_accuracy",
]

# BOOTSTRAP ДЛЯ РАЗНИЦЫ (PAIRED)

def bootstrap_difference_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict:
    """
    Bootstrap CI для разницы двух метрик (paired data).
    """
    if len(values_a) != len(values_b):
        raise ValueError("values_a и values_b должны иметь одинаковую длину")
    
    rng = np.random.default_rng(seed)
    n = len(values_a)
    
    differences = values_b - values_a
    
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        bootstrap_diffs.append(np.mean(differences[indices]))
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    alpha = (1 - confidence) / 2
    ci_lower = np.percentile(bootstrap_diffs, alpha * 100)
    ci_upper = np.percentile(bootstrap_diffs, (1 - alpha) * 100)
    
    return {
        "mean_diff": np.mean(differences),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_means": bootstrap_diffs,
        "confidence": confidence,
    }


# ПЕРМУТАЦИОННЫЙ ТЕСТ (PAIRED)

def paired_permutation_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict:
    """
    Paired пермутационный тест для проверки значимости разницы.
    
    Для paired данных: случайно инвертируем знак разницы для каждой пары.
    """
    if len(values_a) != len(values_b):
        raise ValueError("values_a и values_b должны иметь одинаковую длину")
    
    rng = np.random.default_rng(seed)
    
    differences = values_b - values_a
    observed_diff = np.mean(differences)
    
    permuted_diffs = []
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(differences))
        permuted_diff = np.mean(differences * signs)
        permuted_diffs.append(permuted_diff)
    
    permuted_diffs = np.array(permuted_diffs)
    
    p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
    
    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "is_significant": p_value < 0.05,
        "permuted_diffs": permuted_diffs,
    }


# ЗАГРУЗКА РЕЗУЛЬТАТОВ

def load_e2e_results(path: str) -> List[Dict]:
    """Загружает результаты e2e_eval.py."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("predictions", [])


def extract_paired_metric_values(
    siglip_predictions: List[Dict],
    dinov2_predictions: List[Dict],
    metric_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Извлекает paired значения метрики только для общих samples.
    
    Логика фильтрации по метрике:
    - hit_1: только общие known samples
    - unknown_detection_accuracy: только общие unknown samples
    - e2e_accuracy: все общие samples (и known, и unknown)
    """
    siglip_map = {p["query_image"]: p for p in siglip_predictions}
    dinov2_map = {p["query_image"]: p for p in dinov2_predictions}
    
    common_queries = sorted(set(siglip_map.keys()) & set(dinov2_map.keys()))
    
    siglip_values = []
    dinov2_values = []
    
    for query_image in common_queries:
        siglip_pred = siglip_map[query_image]
        dinov2_pred = dinov2_map[query_image]
        
        siglip_type = siglip_pred.get("sample_type", "known")
        dinov2_type = dinov2_pred.get("sample_type", "known")
        
        # Фильтрация по типу sample в зависимости от метрики
        if metric_name == "hit_1":
            # Только общие known samples
            if siglip_type != "known" or dinov2_type != "known":
                continue
        elif metric_name == "unknown_detection_accuracy":
            # Только общие unknown samples
            if siglip_type != "unknown" or dinov2_type != "unknown":
                continue
        elif metric_name == "e2e_accuracy":
            # Все общие samples (и known, и unknown)
            pass
        else:
            raise ValueError(f"Неизвестная метрика: {metric_name}")
        
        # Извлекаем значение метрики
        siglip_correct = 1 if siglip_pred.get("is_correct", False) else 0
        dinov2_correct = 1 if dinov2_pred.get("is_correct", False) else 0
        
        siglip_values.append(siglip_correct)
        dinov2_values.append(dinov2_correct)
    
    return np.array(siglip_values), np.array(dinov2_values)


# СРАВНЕНИЕ МЕТРИК

def compare_metrics(
    siglip_predictions: List[Dict],
    dinov2_predictions: List[Dict],
    metrics: List[str],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict:
    """Сравнивает метрики двух моделей с CI и p-value."""
    results = {}
    
    for metric_name in metrics:
        print(f"\n{'='*70}")
        print(f"Метрика: {metric_name}")
        print(f"{'='*70}")
        
        siglip_values, dinov2_values = extract_paired_metric_values(
            siglip_predictions, dinov2_predictions, metric_name
        )
        
        if len(siglip_values) == 0:
            print(f"  Нет общих samples для {metric_name}")
            continue
        
        print(f"  Общие samples: {len(siglip_values)}")
        print(f"  SigLIP:  {np.mean(siglip_values):.4f}")
        print(f"  DINOv2:  {np.mean(dinov2_values):.4f}")
        print(f"  Разница: {np.mean(dinov2_values) - np.mean(siglip_values):+.4f}")
        
        # Bootstrap CI (paired)
        ci_result = bootstrap_difference_ci(
            siglip_values, dinov2_values,
            n_bootstrap=n_bootstrap, confidence=confidence, seed=seed
        )
        
        print(f"\n  Bootstrap CI ({confidence*100:.0f}%):")
        print(f"    [{ci_result['ci_lower']:+.4f}, {ci_result['ci_upper']:+.4f}]")
        
        if ci_result['ci_lower'] > 0:
            print(f"    DINOv2 значительно лучше")
        elif ci_result['ci_upper'] < 0:
            print(f"    SigLIP значительно лучше")
        else:
            print(f"    Разница не значима")
        
        # Paired пермутационный тест
        perm_result = paired_permutation_test(
            siglip_values, dinov2_values,
            n_permutations=n_bootstrap, seed=seed
        )
        
        print(f"\n  Пермутационный тест (paired):")
        print(f"    p-value: {perm_result['p_value']:.4f}")
        print(f"    {'Значимо' if perm_result['is_significant'] else 'Не значимо'}")
        
        results[metric_name] = {
            "siglip_mean": float(np.mean(siglip_values)),
            "dinov2_mean": float(np.mean(dinov2_values)),
            "difference": float(np.mean(dinov2_values) - np.mean(siglip_values)),
            "n_samples": len(siglip_values),
            "ci_lower": float(ci_result['ci_lower']),
            "ci_upper": float(ci_result['ci_upper']),
            "p_value": float(perm_result['p_value']),
            "is_significant": perm_result['is_significant'],
            "bootstrap_means": ci_result['bootstrap_means'].tolist(),
        }
    
    return results


# ПОПРАВКА БОНФЕРРОНИ

def apply_bonferroni_correction(results: Dict, alpha: float = 0.05) -> Dict:
    """Применяет поправку Бонферрони к p-values."""
    n_tests = len(results)
    corrected_alpha = alpha / n_tests
    
    print(f"\n{'='*70}")
    print(f"ПОПРАВКА БОНФЕРРОНИ")
    print(f"{'='*70}")
    print(f"  Количество тестов: {n_tests}")
    print(f"  Исходный α: {alpha:.3f}")
    print(f"  Скорректированный α: {corrected_alpha:.4f}")
    print(f"{'='*70}")
    
    for metric_name, result in results.items():
        p_value = result['p_value']
        corrected_p = min(p_value * n_tests, 1.0)
        
        is_significant_corrected = corrected_p < alpha
        
        result['p_value_bonferroni'] = corrected_p
        result['is_significant_bonferroni'] = is_significant_corrected
        result['alpha_bonferroni'] = corrected_alpha
        
        print(f"\n  {metric_name}:")
        print(f"    Исходный p-value:      {p_value:.4f}")
        print(f"    Скорректированный:     {corrected_p:.4f}")
        print(f"    Значимо (α=0.05):      {'да' if p_value < 0.05 else 'нет'}")
        print(f"    Значимо (Бонферрони):  {'да' if is_significant_corrected else 'нет'}")
    
    return results


# ВИЗУАЛИЗАЦИЯ

# ВИЗУАЛИЗАЦИЯ

def plot_comparison(results: Dict, output_path: str = "metrics_comparison.png"):
    """Визуализирует результаты сравнения."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    metrics = list(results.keys())
    y_pos = np.arange(len(metrics))
    
    differences = [results[m]['difference'] for m in metrics]
    ci_lowers = [results[m]['ci_lower'] for m in metrics]
    ci_uppers = [results[m]['ci_upper'] for m in metrics]
    p_values = [results[m]['p_value'] for m in metrics]
    
    # 1. Forest plot
    ax = axes[0]
    
    # Строим каждую точку отдельно с правильным цветом
    for i, m in enumerate(metrics):
        diff = differences[i]
        ci_lower = ci_lowers[i]
        ci_upper = ci_uppers[i]
        p_val = p_values[i]
        is_significant = results[m]['is_significant']
        
        # Определяем цвет
        if not is_significant:
            color = 'gray'
        elif diff > 0:
            color = 'green'
        else:
            color = 'red'
        
        # Вычисляем асимметричные ошибки
        xerr_lower = abs(diff - ci_lower)
        xerr_upper = abs(ci_upper - diff)
        
        # Строим точку с error bar
        ax.errorbar(
            diff, i,
            xerr=[[xerr_lower], [xerr_upper]],
            fmt='o', color=color, capsize=5, markersize=10,
            label=f'{m} (p={p_val:.3f})'
        )
        
        # Добавляем текст с p-value
        ax.text(
            diff, i,
            f'  p={p_val:.3f}',
            va='center', ha='left', fontsize=9
        )
    
    ax.axvline(0, color='black', linestyle='--', linewidth=1, label='Нет разницы')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics)
    ax.set_xlabel('Разница (DINOv2 - SigLIP)')
    ax.set_title(f'Bootstrap CI ({CONFIDENCE*100:.0f}%)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # 2. Гистограммы bootstrap распределений
    ax = axes[1]
    for i, m in enumerate(metrics):
        bootstrap_means = np.array(results[m]['bootstrap_means'])
        ax.hist(
            bootstrap_means,
            bins=50,
            alpha=0.5,
            label=m,
            density=True
        )
    
    ax.axvline(0, color='black', linestyle='--', linewidth=1, label='Нет разницы')
    ax.set_xlabel('Разница (DINOv2 - SigLIP)')
    ax.set_ylabel('Плотность')
    ax.set_title('Bootstrap распределения разницы')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nГрафик сохранён: {output_path}")
    plt.close()

# АНАЛИЗ RETRIEVAL QUALITY

def analyze_retrieval_quality(
    siglip_predictions: List[Dict],
    dinov2_predictions: List[Dict],
):
    """Анализирует, почему разное количество known/unknown samples."""
    print(f"\n{'='*70}")
    print("АНАЛИЗ RETRIEVAL QUALITY")
    print(f"{'='*70}")
    
    # Подсчёт по типам
    siglip_known = sum(1 for p in siglip_predictions if p.get("sample_type") == "known")
    siglip_unknown = sum(1 for p in siglip_predictions if p.get("sample_type") == "unknown")
    dinov2_known = sum(1 for p in dinov2_predictions if p.get("sample_type") == "known")
    dinov2_unknown = sum(1 for p in dinov2_predictions if p.get("sample_type") == "unknown")
    
    print(f"\n  SigLIP:")
    print(f"    Known samples:   {siglip_known:>6} ({siglip_known/len(siglip_predictions)*100:.1f}%)")
    print(f"    Unknown samples: {siglip_unknown:>6} ({siglip_unknown/len(siglip_predictions)*100:.1f}%)")
    
    print(f"\n  DINOv2:")
    print(f"    Known samples:   {dinov2_known:>6} ({dinov2_known/len(dinov2_predictions)*100:.1f}%)")
    print(f"    Unknown samples: {dinov2_unknown:>6} ({dinov2_unknown/len(dinov2_predictions)*100:.1f}%)")
    
    print(f"\n  Разница:")
    print(f"    Known:   {dinov2_known - siglip_known:+d}")
    print(f"    Unknown: {dinov2_unknown - siglip_unknown:+d}")
    
    # Анализ общих samples
    siglip_map = {p["query_image"]: p for p in siglip_predictions}
    dinov2_map = {p["query_image"]: p for p in dinov2_predictions}
    common_queries = set(siglip_map.keys()) & set(dinov2_map.keys())
    
    # Распределение типов в общих samples
    common_known = 0
    common_unknown = 0
    siglip_known_dinov2_unknown = 0
    siglip_unknown_dinov2_known = 0
    
    for q in common_queries:
        s_type = siglip_map[q].get("sample_type", "known")
        d_type = dinov2_map[q].get("sample_type", "known")
        
        if s_type == "known" and d_type == "known":
            common_known += 1
        elif s_type == "unknown" and d_type == "unknown":
            common_unknown += 1
        elif s_type == "known" and d_type == "unknown":
            siglip_known_dinov2_unknown += 1
        elif s_type == "unknown" and d_type == "known":
            siglip_unknown_dinov2_known += 1
    
    print(f"\n  Общие samples ({len(common_queries)}):")
    print(f"    Обе known:                  {common_known:>6}")
    print(f"    Обе unknown:                {common_unknown:>6}")
    print(f"    SigLIP known, DINOv2 unknown: {siglip_known_dinov2_unknown:>6}  <- retrieval DINOv2 хуже")
    print(f"    SigLIP unknown, DINOv2 known: {siglip_unknown_dinov2_known:>6}  <- retrieval SigLIP хуже")
    
    print(f"{'='*70}")


# ТОЧКА ВХОДА

if __name__ == "__main__":
    print("="*70)
    print("СРАВНЕНИЕ МЕТРИК С ДОВЕРИТЕЛЬНЫМИ ИНТЕРВАЛАМИ")
    print("="*70)
    
    # Загружаем результаты
    print(f"\n1. Загрузка SigLIP результатов: {SIGLIP_RESULTS_PATH}")
    siglip_predictions = load_e2e_results(SIGLIP_RESULTS_PATH)
    print(f"   Загружено {len(siglip_predictions)} predictions")
    
    print(f"\n2. Загрузка DINOv2 результатов: {DINOV2_RESULTS_PATH}")
    dinov2_predictions = load_e2e_results(DINOV2_RESULTS_PATH)
    print(f"   Загружено {len(dinov2_predictions)} predictions")
    
    # Анализ retrieval quality
    analyze_retrieval_quality(siglip_predictions, dinov2_predictions)
    
    # Сравниваем метрики
    print(f"\n3. Сравнение метрик...")
    results = compare_metrics(
        siglip_predictions=siglip_predictions,
        dinov2_predictions=dinov2_predictions,
        metrics=METRICS_TO_COMPARE,
        n_bootstrap=N_BOOTSTRAP,
        confidence=CONFIDENCE,
        seed=RANDOM_SEED,
    )
    
    # Применяем поправку Бонферрони
    results = apply_bonferroni_correction(results, alpha=0.05)
    
    # Итоговая таблица
    print(f"\n{'='*70}")
    print("ИТОГОВАЯ ТАБЛИЦА (С ПОПРАВКОЙ БОНФЕРРОНИ)")
    print(f"{'='*70}")
    print(f"{'Метрика':<28} {'N':>6} {'SigLIP':>8} {'DINOv2':>8} {'Разница':>9} {'p_corr':>8} {'Значимо':>10}")
    print("-"*80)
    
    for metric_name, result in results.items():
        sig = "да" if result['is_significant_bonferroni'] else "нет"
        print(
            f"{metric_name:<28} "
            f"{result['n_samples']:>6} "
            f"{result['siglip_mean']:>8.4f} "
            f"{result['dinov2_mean']:>8.4f} "
            f"{result['difference']:>+9.4f} "
            f"{result['p_value_bonferroni']:>8.4f} "
            f"{sig:>10}"
        )
    
    print(f"{'='*70}")
    
    # Визуализация
    print(f"\n4. Построение графика...")
    plot_comparison(
        results,
        output_path="/home/jupyter/s3/ai-tour-guide/dataset/results/metrics_comparison_cross.png"
    )
    
    # Сохраняем результаты
      
        # Сохраняем результаты
    output_data = {
        "config": {
            "siglip_results": SIGLIP_RESULTS_PATH,
            "dinov2_results": DINOV2_RESULTS_PATH,
            "n_bootstrap": N_BOOTSTRAP,
            "confidence": CONFIDENCE,
            "seed": RANDOM_SEED,
        },
        "metrics": {
            name: {
                "siglip_mean": float(result['siglip_mean']),
                "dinov2_mean": float(result['dinov2_mean']),
                "difference": float(result['difference']),
                "n_samples": int(result['n_samples']),
                "ci_lower": float(result['ci_lower']),
                "ci_upper": float(result['ci_upper']),
                "p_value": float(result['p_value']),
                "p_value_bonferroni": float(result['p_value_bonferroni']),
                # Явно приводим к стандартному bool Python
                "is_significant": bool(result['is_significant']),
                "is_significant_bonferroni": bool(result['is_significant_bonferroni']),
            }
            for name, result in results.items()
        }
    }
    
    output_path = Path("/home/jupyter/s3/ai-tour-guide/dataset/results/metrics_comparison_cross.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nРезультаты сохранены: {output_path}")
    print(f"{'='*70}")