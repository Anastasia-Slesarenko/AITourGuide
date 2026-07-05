# -*- coding: utf-8 -*-
"""
Калибровка уверенности реранкера: reliability diagram + ECE + temperature scaling.

Проверяет заявку «калиброванная уверенность»: когда пайплайн говорит P(yes)=0.8,
действительно ли top-1 верен в ~80% случаев. Это прямая калибровка того скора,
который реально используется для решения known/unknown (max P(yes) top-1
кандидата), в отличие от pairwise-Brier по всем кандидатам.

Вход — обогащённые e2e-предсказания VAL:
  * known-val (e2e на val.json)          — объект В индексе,
  * novel-val (e2e на novel_val_unknown) — объекта НЕТ в индексе.

Цель калибровки (бинарная):
  x = confidence_score (= P(yes) top-1 кандидата),
  y = 1, если top-1 — истинный объект (known И gt_reranked_rank==1), иначе 0.
      Для novel y всегда 0 (истинного совпадения в базе нет).

Что делает:
  1. ECE / Brier / reliability по сырым P(yes).
  2. Temperature scaling: p_T = sigmoid(logit(p)/T), T подбирается на VAL по NLL.
  3. ECE / Brier после калибровки, сравнение.
  4. (опц.) reliability-diagram PNG, если доступен matplotlib.

Применение T: confidence на TEST пересчитывается как sigmoid(logit(conf)/T)
ПЕРЕД подбором/применением порога. Temperature-scaling монотонна, поэтому порог
и метрики ранжирования не ломает — только делает уверенность честной.

Замечание: калибровка осмысленна для P(yes) реранкера (zero-shot/LoRA). Для
retrieval baseline confidence — косинусная близость, а не вероятность.

Запуск:
    python experiments/calibration.py
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Tuple

_EPS = 1e-6


# ============================================================
# ЗАГРУЗКА И ПОСТРОЕНИЕ ПАР (confidence, label)
# ============================================================

def _load_predictions(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("predictions", [])


def build_calibration_pairs(
    known_preds: List[Dict],
    novel_preds: List[Dict],
) -> List[Tuple[float, int]]:
    """Строит пары (confidence, label) для калибровки.

    label = 1, если top-1 кандидат — истинный объект (known и gt_reranked_rank==1).
    Для novel и для known с промахом ранжирования label = 0.
    """
    pairs: List[Tuple[float, int]] = []
    for p in known_preds:
        conf = float(p["confidence_score"])
        label = 1 if p["gt_reranked_rank"] == 1 else 0
        pairs.append((conf, label))
    for p in novel_preds:
        pairs.append((float(p["confidence_score"]), 0))  # истинного совпадения нет
    return pairs


# ============================================================
# МЕТРИКИ КАЛИБРОВКИ
# ============================================================

def _clamp(p: float) -> float:
    return min(1.0 - _EPS, max(_EPS, p))


def reliability(pairs: List[Tuple[float, int]], n_bins: int = 10) -> Dict:
    """Reliability по равным бинам [0,1]. Возвращает бины + ECE + MCE."""
    bins = [{"lo": i / n_bins, "hi": (i + 1) / n_bins,
             "n": 0, "sum_conf": 0.0, "sum_acc": 0} for i in range(n_bins)]
    for conf, label in pairs:
        idx = min(n_bins - 1, int(conf * n_bins))
        b = bins[idx]
        b["n"] += 1
        b["sum_conf"] += conf
        b["sum_acc"] += label

    n_total = len(pairs)
    ece = mce = 0.0
    rows = []
    for b in bins:
        if b["n"] == 0:
            rows.append({**{k: b[k] for k in ("lo", "hi", "n")},
                         "conf": 0.0, "acc": 0.0, "gap": 0.0})
            continue
        conf = b["sum_conf"] / b["n"]
        acc = b["sum_acc"] / b["n"]
        gap = abs(conf - acc)
        ece += (b["n"] / n_total) * gap
        mce = max(mce, gap)
        rows.append({"lo": b["lo"], "hi": b["hi"], "n": b["n"],
                     "conf": conf, "acc": acc, "gap": gap})
    return {"bins": rows, "ece": ece, "mce": mce, "n": n_total}


def brier(pairs: List[Tuple[float, int]]) -> float:
    if not pairs:
        return 0.0
    return sum((c - y) ** 2 for c, y in pairs) / len(pairs)


def nll(pairs: List[Tuple[float, int]]) -> float:
    """Средний бинарный NLL (log loss)."""
    if not pairs:
        return 0.0
    s = 0.0
    for c, y in pairs:
        c = _clamp(c)
        s += -(y * math.log(c) + (1 - y) * math.log(1 - c))
    return s / len(pairs)


# ============================================================
# TEMPERATURE SCALING
# ============================================================

def _logit(p: float) -> float:
    p = _clamp(p)
    return math.log(p / (1.0 - p))


def apply_temperature(conf: float, T: float) -> float:
    """p_T = sigmoid(logit(conf)/T). Монотонна по conf при T>0."""
    z = _logit(conf) / T
    return 1.0 / (1.0 + math.exp(-z))


def _nll_at_T(pairs: List[Tuple[float, int]], T: float) -> float:
    s = 0.0
    for c, y in pairs:
        p = _clamp(apply_temperature(c, T))
        s += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return s / len(pairs)


def fit_temperature(
    pairs: List[Tuple[float, int]],
    t_min: float = 0.1,
    t_max: float = 10.0,
    coarse_steps: int = 100,
) -> Tuple[float, float]:
    """Подбирает T по минимуму NLL. Грубая сетка (log) + локальное уточнение.

    Returns:
        (best_T, best_nll)
    """
    # Грубый лог-скан.
    lo, hi = math.log(t_min), math.log(t_max)
    best_T, best = 1.0, float("inf")
    for i in range(coarse_steps + 1):
        T = math.exp(lo + (hi - lo) * i / coarse_steps)
        v = _nll_at_T(pairs, T)
        if v < best:
            best, best_T = v, T
    # Локальное уточнение вокруг best_T.
    span = best_T * 0.2
    for i in range(41):
        T = max(t_min, best_T - span + 2 * span * i / 40)
        v = _nll_at_T(pairs, T)
        if v < best:
            best, best_T = v, T
    return best_T, best


# ============================================================
# ВЫВОД
# ============================================================

def _print_reliability(rel: Dict, title: str) -> None:
    print(f"\n  {title}")
    print(f"  {'бин':>12}  {'n':>7}  {'conf':>7}  {'acc':>7}  {'|gap|':>7}")
    for b in rel["bins"]:
        if b["n"] == 0:
            continue
        print(f"  [{b['lo']:.1f},{b['hi']:.1f}]  {b['n']:>7}  "
              f"{b['conf']:>7.3f}  {b['acc']:>7.3f}  {b['gap']:>7.3f}")
    print(f"  ECE={rel['ece']:.4f}  MCE={rel['mce']:.4f}")


def _maybe_plot(rel_raw: Dict, rel_cal: Dict, T: float, out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  (matplotlib недоступен — PNG пропущен, данные в JSON)")
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="идеал")
    for rel, name, mk in ((rel_raw, f"сырые (ECE={rel_raw['ece']:.3f})", "o"),
                          (rel_cal, f"T={T:.2f} (ECE={rel_cal['ece']:.3f})", "s")):
        xs = [b["conf"] for b in rel["bins"] if b["n"] > 0]
        ys = [b["acc"] for b in rel["bins"] if b["n"] > 0]
        ax.plot(xs, ys, marker=mk, label=name)
    ax.set_xlabel("Уверенность (P(yes))")
    ax.set_ylabel("Доля верных (accuracy)")
    ax.set_title("Reliability diagram")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Reliability diagram: {out_path}")


# ============================================================
# ТОЧКА ВХОДА
# ============================================================

if __name__ == "__main__":
    # ===================== КОНФИГ (без CLI) =====================
    KNOWN_VAL_JSON = "data/eval/e2e_lora_val_known.json"     # e2e на val.json
    NOVEL_VAL_JSON = "data/eval/e2e_lora_val_novel.json"     # e2e на novel_val_unknown
    N_BINS = 10
    PLOT_PATH = "data/eval/reliability_lora.png"
    SAVE_JSON = "data/eval/calibration_lora.json"
    # ===========================================================

    print("=" * 60)
    print("КАЛИБРОВКА УВЕРЕННОСТИ (temperature scaling на VAL)")
    print("=" * 60)
    known = _load_predictions(KNOWN_VAL_JSON)
    novel = _load_predictions(NOVEL_VAL_JSON)
    pairs = build_calibration_pairs(known, novel)
    n_pos = sum(y for _, y in pairs)
    print(f"  known={len(known)}  novel={len(novel)}  пар={len(pairs)}  "
          f"положительных (top-1 верен)={n_pos}")

    rel_raw = reliability(pairs, N_BINS)
    _print_reliability(rel_raw, "Сырые P(yes):")
    print(f"  Brier={brier(pairs):.4f}  NLL={nll(pairs):.4f}")

    T, best_nll = fit_temperature(pairs)
    cal_pairs = [(apply_temperature(c, T), y) for c, y in pairs]
    rel_cal = reliability(cal_pairs, N_BINS)
    _print_reliability(rel_cal, f"После temperature scaling (T={T:.3f}):")
    print(f"  Brier={brier(cal_pairs):.4f}  NLL={nll(cal_pairs):.4f}")

    print("\n" + "=" * 60)
    print(f"  T = {T:.4f}   (T>1 → модель была переуверена)")
    print(f"  ECE:   {rel_raw['ece']:.4f} → {rel_cal['ece']:.4f}")
    print(f"  Brier: {brier(pairs):.4f} → {brier(cal_pairs):.4f}")
    print("  Применить к TEST: confidence' = apply_temperature(confidence, T)")
    print("  (до подбора/применения порога; порог тюнить уже на калиброванных)")
    print("=" * 60)

    _maybe_plot(rel_raw, rel_cal, T, PLOT_PATH)

    with open(SAVE_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "temperature": T,
            "known_val": KNOWN_VAL_JSON, "novel_val": NOVEL_VAL_JSON,
            "raw": {"ece": rel_raw["ece"], "mce": rel_raw["mce"],
                    "brier": brier(pairs), "nll": nll(pairs)},
            "calibrated": {"ece": rel_cal["ece"], "mce": rel_cal["mce"],
                           "brier": brier(cal_pairs), "nll": nll(cal_pairs)},
            "reliability_raw": rel_raw["bins"],
            "reliability_calibrated": rel_cal["bins"],
        }, f, indent=2, ensure_ascii=False)
    print(f"✓ Данные калибровки: {SAVE_JSON}")
