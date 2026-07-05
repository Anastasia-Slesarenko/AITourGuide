# -*- coding: utf-8 -*-
"""
Offline-пересчёт e2e-метрик из обогащённых предсказаний (вариант A).

Зачем:
    e2e_eval_v2.py сохраняет в каждом predictions-элементе ранги истинного
    объекта (gt_retrieval_rank, gt_reranked_rank) и решение пайплайна
    (pipeline_status). Этого достаточно, чтобы пересчитать все метрики под
    ЛЮБОЙ known/unknown-знаменатель БЕЗ повторного дорогого VLM-прогона.

    Дорогой inference (zero-shot / LoRA) делаем ОДИН раз с сохранением
    обогащённых предсказаний, а сравнение определений (retrieval_pool против
    reranked_top5) и любые K-отсечки считаем здесь мгновенно.

Что считает:
    - known = истинный объект найден ретривером с рангом <= known_k
      (retrieval_pool) либо в reranked top-k (reranked_top5).
    - Accuracy (общий знаменатель = все сэмплы), Hit@1 / MRR (на known),
      Unknown accuracy (на unknown), retrieval recall (потолок).

Требования к входному файлу:
    predictions[i] должны содержать поля gt_retrieval_rank, gt_reranked_rank,
    pipeline_status, predicted_landmark_id, true_landmark_id. Их сохраняет
    текущий e2e_eval_v2.py. Старые файлы без этих полей не подходят — их нужно
    перегенерировать одним прогоном e2e_eval_v2.py.

Два режима (настраиваются в конфиг-блоке ниже, без CLI):
    MODE = "single"    — один файл, метка known/unknown по retrieval_pool-прокси.
    MODE = "open_set"  — known-файл (объект В индексе) + novel-файл (объекта НЕТ).
                         Истинная метка берётся ПО ПРОИСХОЖДЕНИЮ файла — это
                         корректная open-set оценка (см. novel_*_unknown.json).

Запуск:
    python experiments/recompute_e2e_metrics.py
"""
from __future__ import annotations

import json
from typing import Dict, List

_REQUIRED_FIELDS = (
    "gt_retrieval_rank",
    "gt_reranked_rank",
    "pipeline_status",
    "true_landmark_id",
    "predicted_landmark_id",
)


def _validate(predictions: List[Dict], need_confidence: bool = False) -> None:
    """Проверяет, что предсказания обогащены нужными полями."""
    if not predictions:
        raise ValueError("Пустой список predictions")
    required = _REQUIRED_FIELDS + (("confidence_score",) if need_confidence else ())
    missing = [f for f in required if f not in predictions[0]]
    if missing:
        raise ValueError(
            "Predictions не обогащены полями "
            f"{missing}. Перегенерируйте их текущим e2e_eval_v2.py "
            "(старые файлы без gt_retrieval_rank/gt_reranked_rank не подходят)."
        )


def _is_known(
    p: Dict, known_definition: str, known_k: int, rerank_top_k: int
) -> bool:
    """known/unknown-метка сэмпла по выбранному определению (не зависит от порога)."""
    if known_definition == "retrieval_pool":
        return 1 <= p["gt_retrieval_rank"] <= known_k
    return 1 <= p["gt_reranked_rank"] <= rerank_top_k


def recompute(
    predictions: List[Dict],
    known_definition: str = "retrieval_pool",
    known_k: int = 10,
    rerank_top_k: int = 5,
    threshold: float | None = None,
) -> Dict:
    """Пересчитывает e2e-метрики под заданный known/unknown-знаменатель.

    Args:
        predictions: обогащённые предсказания из e2e_eval_v2.py.
        known_definition: "retrieval_pool" (по рангу в сыром пуле) или
            "reranked_top5" (по рангу в reranked-списке).
        known_k: сэмпл known, если gt_retrieval_rank в 1..known_k
            (для known_definition="retrieval_pool").
        rerank_top_k: сэмпл known, если gt_reranked_rank в 1..rerank_top_k
            (для known_definition="reranked_top5").
        threshold: если задан — решение accept/reject ПЕРЕсчитывается offline
            как confidence_score >= threshold (порог инференса игнорируется).
            Позволяет свипать порог без повторного VLM-прогона. Если None —
            берётся сохранённое решение пайплайна (pipeline_status).
    """
    if known_definition not in ("retrieval_pool", "reranked_top5"):
        raise ValueError(
            "known_definition должен быть 'retrieval_pool' или 'reranked_top5'"
        )

    known_total = known_correct = 0
    unknown_total = unknown_correct = 0
    mrr_sum = 0.0        # MRR по рангу истинного объекта в reranked-списке
    rank_hit_1 = 0       # ranking hit@1: gt_reranked_rank == 1 (без учёта порога)

    for p in predictions:
        gt_rer_rank = p["gt_reranked_rank"]
        true_id = p["true_landmark_id"]

        # accept/reject: либо сохранённое решение, либо пересчёт под новый порог.
        if threshold is None:
            accepted = p["pipeline_status"] == "success"
            pred_correct = p["predicted_landmark_id"] == true_id
        else:
            accepted = float(p["confidence_score"]) >= threshold
            # accept + истинный объект первый в reranked-порядке → hit
            pred_correct = accepted and gt_rer_rank == 1

        is_known = _is_known(p, known_definition, known_k, rerank_top_k)

        if is_known:
            known_total += 1
            # hit@1 — end-to-end (reject = промах)
            if pred_correct:
                known_correct += 1
            # MRR и ranking hit@1 — по рангу в reranked-списке, без учёта порога
            if gt_rer_rank != -1:
                mrr_sum += 1.0 / gt_rer_rank
                if gt_rer_rank == 1:
                    rank_hit_1 += 1
        else:
            unknown_total += 1
            # unknown detected correctly = пайплайн отклонил (reject)
            if not accepted:
                unknown_correct += 1

    total = known_total + unknown_total
    correct = known_correct + unknown_correct

    def _safe(a: int, b: int) -> float:
        return a / b if b > 0 else 0.0

    return {
        "known_definition": known_definition,
        "known_k": known_k if known_definition == "retrieval_pool" else rerank_top_k,
        "threshold": threshold,  # None → сохранённое решение пайплайна
        "e2e_accuracy": _safe(correct, total),
        "e2e_hit_1": _safe(known_correct, known_total),
        "e2e_mrr": mrr_sum / known_total if known_total > 0 else 0.0,
        "rank_hit_1": _safe(rank_hit_1, known_total),
        "retrieval_recall": _safe(known_total, total),
        "unknown_detection_accuracy": _safe(unknown_correct, unknown_total),
        "unknown_detection_rate": _safe(unknown_total, total),
        "total_samples": total,
        "known_samples": known_total,
        "unknown_samples": unknown_total,
        "known_correct": known_correct,
        "unknown_correct": unknown_correct,
    }


def _f1_macro_at(
    predictions: List[Dict],
    known_definition: str,
    known_k: int,
    rerank_top_k: int,
    threshold: float,
) -> Dict:
    """F1-macro задачи детекции known/unknown при данном пороге.

    Бинарная задача: pred_known = confidence >= threshold, true_known = _is_known.
    Возвращает f1_macro / f1_known / f1_unknown (тот же критерий, что в find_th).
    """
    # Матрица ошибок с точки зрения "known как положительный класс".
    tp = fp = fn = tn = 0
    for p in predictions:
        true_known = _is_known(p, known_definition, known_k, rerank_top_k)
        pred_known = float(p["confidence_score"]) >= threshold
        if true_known and pred_known:
            tp += 1
        elif (not true_known) and pred_known:
            fp += 1
        elif true_known and (not pred_known):
            fn += 1
        else:
            tn += 1

    def _f1(tp_: int, fp_: int, fn_: int) -> float:
        prec = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0.0
        rec = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    f1_known = _f1(tp, fp, fn)
    # "unknown как положительный": TP_u=tn, FP_u=fn, FN_u=fp
    f1_unknown = _f1(tn, fn, fp)
    return {
        "threshold": threshold,
        "f1_macro": (f1_known + f1_unknown) / 2.0,
        "f1_known": f1_known,
        "f1_unknown": f1_unknown,
    }


def sweep_threshold(
    predictions: List[Dict],
    known_definition: str = "retrieval_pool",
    known_k: int = 10,
    rerank_top_k: int = 5,
) -> tuple:
    """Подбирает порог по максимуму F1-macro. ЗАПУСКАТЬ ТОЛЬКО НА VAL.

    Кандидаты порогов — уникальные confidence_score (+ границы), поэтому
    оптимум не пропускается. Работает одинаково для любого скора в
    confidence_score: P(yes) реранкера ИЛИ retrieval-скор baseline.

    Returns:
        (best_threshold, best_f1_macro, all_rows) — all_rows отсортирован по
        убыванию F1-macro.
    """
    confs = sorted({float(p["confidence_score"]) for p in predictions})
    if not confs:
        return 0.5, 0.0, []
    candidates = [confs[0] - 1e-6] + confs + [confs[-1] + 1e-6]

    rows = [
        _f1_macro_at(predictions, known_definition, known_k, rerank_top_k, t)
        for t in candidates
    ]
    rows.sort(key=lambda r: r["f1_macro"], reverse=True)
    best = rows[0]
    return best["threshold"], best["f1_macro"], rows


def _accepted(p: Dict, threshold: float | None) -> bool:
    """Решение accept/reject: сохранённое (threshold=None) или под новый порог."""
    if threshold is None:
        return p["pipeline_status"] == "success"
    return float(p["confidence_score"]) >= threshold


def open_set_metrics(
    known_preds: List[Dict],
    unknown_preds: List[Dict],
    threshold: float | None = None,
    known_k: int = 10,
) -> Dict:
    """Open-set метрики с ИСТИННОЙ меткой по происхождению файла.

    known_preds   — e2e-предсказания на known-наборе (объект В индексе).
    unknown_preds — e2e-предсказания на novel-наборе (объекта НЕТ в индексе).
    threshold     — accept = confidence >= threshold; None → сохранённое решение.
    known_k       — known-запрос «найден», если gt_retrieval_rank в 1..known_k.

    Метка known/unknown НЕ зависит от retrieval (в отличие от retrieval_pool):
    novel всегда unknown, известные всегда known. retrieval-промах на известном
    объекте остаётся known, но идёт в потолок retrieval_recall.
    """
    n_known = len(known_preds)
    n_unknown = len(unknown_preds)

    retrieved = hit = known_accepted = 0
    mrr_sum = 0.0
    for p in known_preds:
        acc = _accepted(p, threshold)
        known_accepted += int(acc)
        rr = p["gt_reranked_rank"]
        if 1 <= p["gt_retrieval_rank"] <= known_k:
            retrieved += 1
        if rr != -1:
            mrr_sum += 1.0 / rr
        if acc and rr == 1:
            hit += 1

    unknown_rejected = sum(1 for p in unknown_preds if not _accepted(p, threshold))

    def _safe(a: int, b: int) -> float:
        return a / b if b > 0 else 0.0

    # Детекция accept(known)/reject(unknown).
    tp, fn = known_accepted, n_known - known_accepted
    tn, fp = unknown_rejected, n_unknown - unknown_rejected

    def _f1(tp_: int, fp_: int, fn_: int) -> float:
        prec = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0.0
        rec = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    f1_known = _f1(tp, fp, fn)
    f1_unknown = _f1(tn, fn, fp)
    hit_rate = _safe(hit, n_known)
    unk_acc = _safe(unknown_rejected, n_unknown)

    return {
        "threshold": threshold,
        "n_known": n_known,
        "n_unknown": n_unknown,
        "retrieval_recall": _safe(retrieved, n_known),      # потолок ретривера
        "e2e_hit_1": hit_rate,                              # answered AND correct
        "e2e_mrr": mrr_sum / n_known if n_known > 0 else 0.0,  # ранжирование known
        "known_accept_rate": _safe(known_accepted, n_known),  # детекция TPR
        "unknown_detection_accuracy": unk_acc,             # детекция TNR (reject)
        "detection_f1_macro": (f1_known + f1_unknown) / 2.0,
        "detection_f1_known": f1_known,
        "detection_f1_unknown": f1_unknown,
        # ratio-free сводная и «сырая» combined (зависит от соотношения в файлах)
        "balanced_accuracy": (hit_rate + unk_acc) / 2.0,
        "e2e_accuracy": _safe(hit + unknown_rejected, n_known + n_unknown),
    }


def open_set_sweep(
    known_preds: List[Dict],
    unknown_preds: List[Dict],
    known_k: int = 10,
) -> tuple:
    """Подбор порога по максимуму detection-F1-macro. ТОЛЬКО НА VAL.

    Порог отделяет accept(known) от reject(unknown) — тот же смысл, что в find_th,
    но с истинной меткой по происхождению.
    """
    confs = sorted({
        float(p["confidence_score"]) for p in (known_preds + unknown_preds)
    })
    if not confs:
        return 0.5, 0.0, []
    candidates = [confs[0] - 1e-6] + confs + [confs[-1] + 1e-6]

    rows = []
    for t in candidates:
        m = open_set_metrics(known_preds, unknown_preds, threshold=t, known_k=known_k)
        rows.append({
            "threshold": t,
            "detection_f1_macro": m["detection_f1_macro"],
            "detection_f1_known": m["detection_f1_known"],
            "detection_f1_unknown": m["detection_f1_unknown"],
        })
    rows.sort(key=lambda r: r["detection_f1_macro"], reverse=True)
    best = rows[0]
    return best["threshold"], best["detection_f1_macro"], rows


def open_set_bootstrap_ci(
    known_preds: List[Dict],
    unknown_preds: List[Dict],
    threshold: float | None = None,
    known_k: int = 10,
    n_boot: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> Dict:
    """Bootstrap-CI headline-метрик с кластеризацией по landmark_id.

    Ресемплим ОБЪЕКТЫ (не отдельные фото) с возвращением, раздельно known и
    novel — фото одного объекта коррелированы, поэлементный bootstrap занизил бы
    CI. Возвращает по каждой метрике {point, lo, hi}.
    """
    import random as _random

    rng = _random.Random(seed)

    def _clusters(preds: List[Dict]) -> List[List[Dict]]:
        by: Dict[str, List[Dict]] = {}
        for p in preds:
            by.setdefault(p["true_landmark_id"], []).append(p)
        return list(by.values())

    known_cl = _clusters(known_preds)
    unknown_cl = _clusters(unknown_preds)
    nk, nu = len(known_cl), len(unknown_cl)

    keys = [
        "e2e_hit_1", "unknown_detection_accuracy", "balanced_accuracy",
        "detection_f1_macro", "retrieval_recall", "e2e_accuracy",
    ]
    draws: Dict[str, List[float]] = {k: [] for k in keys}

    for _ in range(n_boot):
        kp: List[Dict] = []
        for _ in range(nk):
            kp.extend(known_cl[rng.randrange(nk)])
        up: List[Dict] = []
        for _ in range(nu):
            up.extend(unknown_cl[rng.randrange(nu)])
        m = open_set_metrics(kp, up, threshold=threshold, known_k=known_k)
        for k in keys:
            draws[k].append(m[k])

    def _pct(vals: List[float], q: float) -> float:
        s = sorted(vals)
        idx = int(round(q * (len(s) - 1)))
        return s[min(max(idx, 0), len(s) - 1)]

    alpha = (1.0 - confidence) / 2.0
    point = open_set_metrics(known_preds, unknown_preds, threshold=threshold, known_k=known_k)
    return {
        k: {"point": point[k], "lo": _pct(draws[k], alpha), "hi": _pct(draws[k], 1 - alpha)}
        for k in keys
    }


def _print_metrics(m: Dict) -> None:
    thr = m.get("threshold")
    thr_str = f", thr={thr}" if thr is not None else ", thr=сохранённый"
    print(
        f"  [{m['known_definition']}, k={m['known_k']}{thr_str}]  "
        f"n={m['total_samples']} (known={m['known_samples']}, "
        f"unknown={m['unknown_samples']})"
    )
    print(f"    Accuracy:            {m['e2e_accuracy']:.4f}")
    print(f"    Hit@1 (known):       {m['e2e_hit_1']:.4f}")
    print(f"    MRR (known):         {m['e2e_mrr']:.4f}")
    print(f"    Ranking Hit@1:       {m['rank_hit_1']:.4f}")
    print(f"    Retrieval recall:    {m['retrieval_recall']:.4f}")
    print(f"    Unknown accuracy:    {m['unknown_detection_accuracy']:.4f}")


def _print_open_set(m: Dict) -> None:
    thr = m.get("threshold")
    thr_str = f"{thr:.4f}" if thr is not None else "сохранённый"
    print(f"  n_known={m['n_known']}  n_unknown={m['n_unknown']}  thr={thr_str}")
    print(f"    Retrieval recall (потолок):      {m['retrieval_recall']:.4f}")
    print(f"    Hit@1 (known, e2e):              {m['e2e_hit_1']:.4f}")
    print(f"    MRR (known):                     {m['e2e_mrr']:.4f}")
    print(f"    Unknown accuracy (reject novel): {m['unknown_detection_accuracy']:.4f}")
    print(f"    Detection F1-macro:              {m['detection_f1_macro']:.4f}")
    print(f"    Balanced accuracy (ratio-free):  {m['balanced_accuracy']:.4f}")
    print(f"    Combined accuracy (соотн. файлов): {m['e2e_accuracy']:.4f}")


def _load_predictions(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("predictions", [])


def _run_single(cfg: Dict) -> None:
    """MODE=single: один файл, метка known/unknown по retrieval_pool-прокси."""
    predictions = _load_predictions(cfg["RESULTS_JSON"])
    _validate(
        predictions,
        need_confidence=cfg["THRESHOLD"] is not None or cfg["SWEEP_THRESHOLD"],
    )
    print("=" * 60)
    print(f"[single] Файл: {cfg['RESULTS_JSON']}  (предсказаний: {len(predictions)})")
    print("=" * 60)

    kd, kk = cfg["KNOWN_DEFINITION"], cfg["KNOWN_K"]

    if cfg["SWEEP_THRESHOLD"]:
        print("⚠️  SWEEP: подбор порога — ТОЛЬКО на VAL, не на тесте!")
        best_t, best_f1, rows = sweep_threshold(predictions, kd, kk, kk)
        print(f"\n  Топ-5 порогов по F1-macro ({kd}):")
        print(f"  {'threshold':>12}  {'F1-macro':>9}  {'F1-known':>9}  {'F1-unknown':>10}")
        for r in rows[:5]:
            print(f"  {r['threshold']:>12.6f}  {r['f1_macro']:>9.4f}  "
                  f"{r['f1_known']:>9.4f}  {r['f1_unknown']:>10.4f}")
        print(f"\n  ✓ Оптимальный порог: {best_t:.6f} (F1-macro={best_f1:.4f})")
        _print_metrics(recompute(predictions, kd, kk, kk, threshold=best_t))
        return

    if cfg["COMPARE"]:
        for defn, k in (("retrieval_pool", kk), ("reranked_top5", 5)):
            _print_metrics(recompute(predictions, defn, k, k, threshold=cfg["THRESHOLD"]))
            print()
        return

    _print_metrics(recompute(predictions, kd, kk, kk, threshold=cfg["THRESHOLD"]))


def _run_open_set(cfg: Dict) -> None:
    """MODE=open_set: known-файл + novel-файл, истинная метка по происхождению."""
    known = _load_predictions(cfg["KNOWN_PREDS_JSON"])
    unknown = _load_predictions(cfg["UNKNOWN_PREDS_JSON"])
    need_conf = cfg["THRESHOLD"] is not None or cfg["SWEEP_THRESHOLD"]
    _validate(known, need_confidence=need_conf)
    _validate(unknown, need_confidence=need_conf)

    print("=" * 60)
    print(f"[open_set] known:   {cfg['KNOWN_PREDS_JSON']}  ({len(known)})")
    print(f"[open_set] unknown: {cfg['UNKNOWN_PREDS_JSON']}  ({len(unknown)})")
    print("=" * 60)

    kk = cfg["KNOWN_K"]
    threshold = cfg["THRESHOLD"]

    if cfg["SWEEP_THRESHOLD"]:
        print("⚠️  SWEEP: порог подбирается ТОЛЬКО на VAL (known-val + novel-val)!")
        best_t, best_f1, rows = open_set_sweep(known, unknown, known_k=kk)
        print(f"\n  Топ-5 порогов по detection-F1-macro:")
        print(f"  {'threshold':>12}  {'F1-macro':>9}  {'F1-known':>9}  {'F1-unknown':>10}")
        for r in rows[:5]:
            print(f"  {r['threshold']:>12.6f}  {r['detection_f1_macro']:>9.4f}  "
                  f"{r['detection_f1_known']:>9.4f}  {r['detection_f1_unknown']:>10.4f}")
        print(f"\n  ✓ Оптимальный порог: {best_t:.6f} (detection F1-macro={best_f1:.4f})")
        _print_open_set(open_set_metrics(known, unknown, threshold=best_t, known_k=kk))
        print(f"\n  Применить порог к ТЕСТУ: THRESHOLD = {best_t:.6f}")
        return

    _print_open_set(open_set_metrics(known, unknown, threshold=threshold, known_k=kk))

    if cfg.get("BOOTSTRAP_CI"):
        print(f"\n  Bootstrap-CI 95% (кластерно по landmark, n_boot={cfg['N_BOOTSTRAP']}):")
        ci = open_set_bootstrap_ci(
            known, unknown, threshold=threshold, known_k=kk,
            n_boot=cfg["N_BOOTSTRAP"], seed=cfg["SEED"],
        )
        for key in ["e2e_hit_1", "unknown_detection_accuracy",
                    "balanced_accuracy", "detection_f1_macro", "e2e_accuracy"]:
            c = ci[key]
            print(f"    {key:<28} {c['point']:.4f}  [{c['lo']:.4f}, {c['hi']:.4f}]")


if __name__ == "__main__":
    # ============================================================
    # КОНФИГ (без CLI)
    # ============================================================
    MODE = "open_set"           # "single" | "open_set"

    KNOWN_K = 10                # known-запрос «найден», если gt_retrieval_rank<=K
    THRESHOLD = None            # None → сохранённое решение; число → applied
    SWEEP_THRESHOLD = False     # True → подобрать порог (ТОЛЬКО на VAL!)

    # Bootstrap-CI для open_set (кластерно по landmark). Только при MODE=open_set
    # и SWEEP_THRESHOLD=False (т.е. на TEST при подобранном пороге).
    BOOTSTRAP_CI = True
    N_BOOTSTRAP = 1000
    SEED = 42

    # --- MODE="single": один файл (retrieval_pool-прокси) ---
    RESULTS_JSON = "data/eval/e2e_results.json"
    KNOWN_DEFINITION = "retrieval_pool"   # "retrieval_pool" | "reranked_top5"
    COMPARE = False             # сравнить оба определения на одном файле

    # --- MODE="open_set": known + novel unknown (истинная open-set оценка) ---
    KNOWN_PREDS_JSON = "data/eval/e2e_lora_test_known.json"     # e2e на test.json
    UNKNOWN_PREDS_JSON = "data/eval/e2e_lora_test_novel.json"   # e2e на novel_test_unknown
    # ============================================================

    cfg = {
        "MODE": MODE, "KNOWN_K": KNOWN_K, "THRESHOLD": THRESHOLD,
        "SWEEP_THRESHOLD": SWEEP_THRESHOLD, "RESULTS_JSON": RESULTS_JSON,
        "KNOWN_DEFINITION": KNOWN_DEFINITION, "COMPARE": COMPARE,
        "KNOWN_PREDS_JSON": KNOWN_PREDS_JSON, "UNKNOWN_PREDS_JSON": UNKNOWN_PREDS_JSON,
        "BOOTSTRAP_CI": BOOTSTRAP_CI, "N_BOOTSTRAP": N_BOOTSTRAP, "SEED": SEED,
    }

    if MODE == "single":
        _run_single(cfg)
    elif MODE == "open_set":
        _run_open_set(cfg)
    else:
        raise SystemExit(f"Неизвестный MODE: {MODE!r} (single | open_set)")
