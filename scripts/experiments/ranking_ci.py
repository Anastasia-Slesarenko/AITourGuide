# scripts/experiments/ranking_ci.py
"""
Доверительные интервалы для метрик ранжирования реранкера (bootstrap).

Считает по пер-запросным предсказаниям из results/eval/*_lora_predictions.json:
  - Hit@1 и MRR каждой конфигурации sweep с 95% bootstrap-CI;
  - парную разницу Hit@1 итоговой модели против каждого конкурента
    (paired bootstrap на одних и тех же landmarks).

Метрики ранжирования от порога не зависят и считаются на общем наборе кандидатов
step6, поэтому конфигурации сравнимы напрямую. Ресэмпл кластерный, по landmark
(несколько фото одного объекта коррелируют); в текущем валидном наборе на landmark
приходится один запрос, так что это эквивалентно ресэмплу по запросам.

Каждый расчёт берёт свежий генератор с одним и тем же seed, поэтому CI конфигурации
не зависят от порядка и состава CONFIGS (добавление/удаление прогона не сдвигает
числа остальных). Воспроизводит docs/MODEL_SELECTION.md при seed=42, N_BOOTSTRAP=5000.

Запуск:
    venv/bin/python scripts/experiments/ranking_ci.py
"""

import json
from pathlib import Path

import numpy as np

# Конфигурация

RESULTS_DIR = Path(__file__).parent / "results" / "eval"

# Ярлык -> файл предсказаний. Первый ключ — итоговая модель, с ней идёт сравнение.
CONFIGS: dict[str, str] = {
    "full LoRA 448 (итоговая)": "val_rerank_exp_r16_alpha32_lr2e-5_rerank_full_lora_448_lora_predictions.json",
    "attn 448": "val_rerank_exp_r16_alpha32_lr2e-5_rerank_attn_448_lora_predictions.json",
    "attn+MLP lr1e-5, 336": "val_rerank_exp_r16_alpha32_lr1e-5_rerank_attn_and_mlp_module_r16_lora_predictions.json",
    "attn only, 336": "val_rerank_exp_r16_alpha32_lr2e-5_rerank_attn_module_balanced_lora_predictions.json",
    "r8 attn+MLP, 336": "val_rerank_exp_r8_alpha16_lr2e-5_rerank_attn_and_mlp_module_lora_predictions.json",
}

N_BOOTSTRAP = 5000
CONFIDENCE = 0.95
RANDOM_SEED = 42


def load_valid_by_landmark(path: Path) -> dict[str, list[tuple[float, float]]]:
    """
    Читает предсказания и группирует валидные запросы по landmark.

    Валидный запрос — тот, где истинный кандидат присутствует в списке
    (target_idx есть в ranked_indices); остальные в метрики ранжирования не
    входят. landmark берётся из имени query_image ('<landmark>_<кадр>.jpg').

    Returns:
        landmark -> список (hit1, reciprocal_rank) по его запросам.
    """
    records = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[tuple[float, float]]] = {}
    for r in records:
        target = r.get("target_idx")
        ranked = r.get("ranked_indices") or []
        if target is None or target < 0 or target not in ranked:
            continue
        landmark = r["query_image"].split("_")[0]
        rank = ranked.index(target) + 1
        hit1 = 1.0 if r.get("correct") else 0.0
        grouped.setdefault(landmark, []).append((hit1, 1.0 / rank))
    return grouped


def landmark_sums(
    grouped: dict[str, list[tuple[float, float]]],
    landmarks: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Суммы hit1/rr и число запросов на каждый landmark (вход для bootstrap)."""
    sum_hit1 = np.array([sum(h for h, _ in grouped[lm]) for lm in landmarks])
    sum_rr = np.array([sum(rr for _, rr in grouped[lm]) for lm in landmarks])
    counts = np.array([len(grouped[lm]) for lm in landmarks], dtype=float)
    return sum_hit1, sum_rr, counts


def percentile_ci(values: np.ndarray, confidence: float) -> tuple[float, float]:
    """Перцентильный CI по массиву bootstrap-оценок."""
    tail = (1.0 - confidence) / 2.0
    return np.percentile(values, tail * 100), np.percentile(values, (1 - tail) * 100)


def bootstrap_config(
    sum_hit1: np.ndarray,
    sum_rr: np.ndarray,
    counts: np.ndarray,
    n_boot: int,
    confidence: float,
    seed: int,
) -> dict[str, tuple[float, float, float]]:
    """
    Hit@1 и MRR с CI за один проход bootstrap (общие resample-индексы на обе
    метрики). Свежий генератор с фиксированным seed — результат не зависит от
    порядка вызовов.

    Returns:
        {"hit1": (point, lo, hi), "mrr": (point, lo, hi)}.
    """
    rng = np.random.default_rng(seed)
    n_landmarks = len(counts)
    hit1_samples = np.empty(n_boot)
    mrr_samples = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_landmarks, size=n_landmarks)
        total = counts[idx].sum()
        hit1_samples[b] = sum_hit1[idx].sum() / total
        mrr_samples[b] = sum_rr[idx].sum() / total
    h_lo, h_hi = percentile_ci(hit1_samples, confidence)
    m_lo, m_hi = percentile_ci(mrr_samples, confidence)
    return {
        "hit1": (sum_hit1.sum() / counts.sum(), h_lo, h_hi),
        "mrr": (sum_rr.sum() / counts.sum(), m_lo, m_hi),
    }


def paired_hit1_diff_ci(
    grouped_a: dict[str, list[tuple[float, float]]],
    grouped_b: dict[str, list[tuple[float, float]]],
    n_boot: int,
    confidence: float,
    seed: int,
) -> tuple[float, float, float]:
    """CI разницы Hit@1 (a - b), paired bootstrap на общих landmarks."""
    rng = np.random.default_rng(seed)
    common = sorted(set(grouped_a) & set(grouped_b))
    sum_a, _, n_a = landmark_sums(grouped_a, common)
    sum_b, _, n_b = landmark_sums(grouped_b, common)
    n_landmarks = len(common)
    point = sum_a.sum() / n_a.sum() - sum_b.sum() / n_b.sum()
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_landmarks, size=n_landmarks)
        diffs[b] = sum_a[idx].sum() / n_a[idx].sum() - sum_b[idx].sum() / n_b[idx].sum()
    lo, hi = percentile_ci(diffs, confidence)
    return point, lo, hi


def main() -> None:
    grouped: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for label, fname in CONFIGS.items():
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"[пропуск] файл не найден: {path}")
            continue
        grouped[label] = load_valid_by_landmark(path)

    if not grouped:
        print(f"Нет предсказаний в {RESULTS_DIR}")
        return

    pct = int(CONFIDENCE * 100)
    print(
        f"Hit@1 и MRR, {pct}% bootstrap-CI "
        f"({N_BOOTSTRAP} ресэмплов по landmark, seed={RANDOM_SEED})\n"
    )
    for label, g in grouped.items():
        landmarks = list(g.keys())
        sum_hit1, sum_rr, counts = landmark_sums(g, landmarks)
        res = bootstrap_config(
            sum_hit1, sum_rr, counts, N_BOOTSTRAP, CONFIDENCE, RANDOM_SEED
        )
        h, h_lo, h_hi = res["hit1"]
        m, m_lo, m_hi = res["mrr"]
        print(
            f"  {label:24} n={int(counts.sum()):5} landmarks={len(landmarks):5}  "
            f"Hit@1={h * 100:5.2f}% [{h_lo * 100:.1f}, {h_hi * 100:.1f}]  "
            f"MRR={m:.3f} [{m_lo:.3f}, {m_hi:.3f}]"
        )

    winner = next(iter(grouped))
    print(
        f"\nПарная разница Hit@1: '{winner}' минус конкурент "
        f"(paired bootstrap, те же landmarks)\n"
    )
    for label, g in grouped.items():
        if label == winner:
            continue
        diff, lo, hi = paired_hit1_diff_ci(
            grouped[winner], g, N_BOOTSTRAP, CONFIDENCE, RANDOM_SEED
        )
        verdict = "значимо" if (lo > 0 or hi < 0) else "не значимо"
        print(
            f"  vs {label:24} {diff * 100:+5.2f} пп  "
            f"CI [{lo * 100:+.2f}, {hi * 100:+.2f}]  {verdict}"
        )


if __name__ == "__main__":
    main()
