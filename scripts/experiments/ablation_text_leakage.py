# -*- coding: utf-8 -*-
"""
Ablation: на что опирается реранкер — на ЗРЕНИЕ или на ТЕКСТ (name/caption)?

Реранкер получает: query-фото, candidate-фото, имя объекта и caption. Риск —
что модель отвечает Yes/No по совпадению ТЕКСТА, а не сравнивая изображения.
Скрипт прогоняет реранкер на подвыборке known-val в нескольких режимах и
сравнивает дискриминацию (Hit@1, AUROC, разрыв P(yes) позитив/негатив).

Режимы:
    full           — как в проде (контроль).
    blank_query    — query-фото заменено серым. Если метрики держатся → модель
                     НЕ использует query → решает по candidate-фото+тексту (плохо).
    noise_query    — query-фото = шум (альтернатива blank).
    blank_candidate— candidate-фото серое (проверка опоры на candidate-фото).
    no_text        — name→"this landmark", caption→"" . Если метрики держатся →
                     текст НЕ костыль, реранкер зрительный (хорошо).
    mismatch_query — query из ДРУГОГО случайного сэмпла. Если P(yes) остаётся
                     высоким → модель игнорирует query.

Интерпретация:
    full высокий, blank_query коллапсирует, no_text держится → честный vision-реранкер.
    full ≈ blank_query или full ≈ (no_text коллапс) → опора на текст/приоры, не зрение.

VLM-часть зеркалит e2e_eval_v2 (тот же промпт, позиция токена, softmax(No,Yes)).
Метрики — чистый python.

Запуск:
    python experiments/ablation_text_leakage.py
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ============================================================
# МЕТРИКИ (чистый python — тестируемо без VLM)
# ============================================================

def auroc(labels: List[int], scores: List[float]) -> float:
    """AUROC через ранги (Mann-Whitney U). labels ∈ {0,1}."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    # средние ранги (обработка ties)
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # ранги с 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(r for r, l in zip(ranks, labels) if l == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def hit_at_1(per_sample_scores: List[List[float]], target_idxs: List[int]) -> float:
    """Доля сэмплов, где argmax P(yes) == target_idx."""
    ok = tot = 0
    for scores, tgt in zip(per_sample_scores, target_idxs):
        if tgt < 0 or not scores:
            continue
        pred = max(range(len(scores)), key=lambda i: scores[i])
        ok += int(pred == tgt)
        tot += 1
    return ok / tot if tot else 0.0


def separation(per_sample_scores: List[List[float]], target_idxs: List[int]) -> Tuple[float, float]:
    """Средний P(yes) на позитивах и негативах (по всем сэмплам)."""
    pos, neg = [], []
    for scores, tgt in zip(per_sample_scores, target_idxs):
        for i, s in enumerate(scores):
            (pos if i == tgt else neg).append(s)
    mp = sum(pos) / len(pos) if pos else 0.0
    mn = sum(neg) / len(neg) if neg else 0.0
    return mp, mn


def flat_labels_scores(
    per_sample_scores: List[List[float]], target_idxs: List[int]
) -> Tuple[List[int], List[float]]:
    labels, scores = [], []
    for sc, tgt in zip(per_sample_scores, target_idxs):
        for i, s in enumerate(sc):
            labels.append(1 if i == tgt else 0)
            scores.append(s)
    return labels, scores


# ============================================================
# VLM-ЧАСТЬ (зеркалит e2e_eval_v2; запускается на боксе с torch)
# ============================================================

def _make_ablated_images(query_img, cand_img, mode: str, fixed_size, other_query_img):
    """Применяет ablation к паре изображений. Возвращает (q, c) как PIL."""
    from PIL import Image

    def _gray():
        return Image.new("RGB", fixed_size, (128, 128, 128))

    def _noise():
        import random as _r
        img = Image.new("RGB", fixed_size)
        img.putdata([(_r.randint(0, 255), _r.randint(0, 255), _r.randint(0, 255))
                     for _ in range(fixed_size[0] * fixed_size[1])])
        return img

    def _resize(im):
        return im.resize(fixed_size, Image.Resampling.BILINEAR)

    q = _resize(query_img)
    c = _resize(cand_img)
    if mode == "blank_query":
        q = _gray()
    elif mode == "noise_query":
        q = _noise()
    elif mode == "blank_candidate":
        c = _gray()
    elif mode == "mismatch_query" and other_query_img is not None:
        q = _resize(other_query_img)
    return q, c


def _ablated_text(name: str, caption: str, mode: str) -> Tuple[str, str]:
    if mode == "no_text":
        return "this landmark", ""
    return name, caption


def score_samples(model, processor, yes_id, no_id, samples, image_dir,
                  mode, fixed_size=(224, 224), batch_size=16, caption_max=300):
    """Считает P(yes) по всем кандидатам каждого сэмпла в заданном режиме.

    Возвращает per_sample_scores: List[List[float]] в порядке кандидатов.
    """
    import torch
    from PIL import Image

    # Пары (query, candidate) для батчинга, с адресацией назад в сэмплы.
    per_sample_scores: List[List[float]] = [[0.0] * len(s["candidates"]) for s in samples]
    all_query_paths = [s["query_image"] for s in samples]

    flat = []  # (s_idx, c_idx, name, caption)
    for si, s in enumerate(samples):
        for ci, cand in enumerate(s["candidates"]):
            flat.append((si, ci, cand))

    for start in range(0, len(flat), batch_size):
        batch = flat[start:start + batch_size]
        texts, images_flat, valid = [], [], []
        for (si, ci, cand) in batch:
            try:
                q_raw = Image.open(os.path.join(image_dir, samples[si]["query_image"])).convert("RGB")
                c_raw = Image.open(os.path.join(image_dir, cand["image"])).convert("RGB")
            except Exception:
                continue
            other = None
            if mode == "mismatch_query":
                oidx = (si + 1) % len(samples)
                try:
                    other = Image.open(os.path.join(image_dir, all_query_paths[oidx])).convert("RGB")
                except Exception:
                    other = None
            q_img, c_img = _make_ablated_images(q_raw, c_raw, mode, fixed_size, other)
            name, caption = _ablated_text(cand.get("name", ""), cand.get("caption", ""), mode)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Query Photo:"},
                    {"type": "image"},
                    {"type": "text", "text": "Candidate Photo:"},
                    {"type": "image"},
                    {"type": "text", "text": (
                        f"Question: Are these photos showing the same landmark: "
                        f"\"{name}\"?\nCandidate details: {caption[:caption_max]}\n"
                        f"Answer only with Yes or No.")},
                ],
            }]
            texts.append(processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
            images_flat.append(q_img)
            images_flat.append(c_img)
            valid.append((si, ci))

        if not texts:
            continue
        inputs = processor(text=texts, images=images_flat, return_tensors="pt",
                           padding=True if len(texts) > 1 else False).to(model.device)
        with torch.inference_mode():
            logits = model(**inputs).logits[:, -1, :]
            probs = torch.softmax(
                torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1), dim=1)
            yes = probs[:, 1].cpu().tolist()
        for (si, ci), p in zip(valid, yes):
            per_sample_scores[si][ci] = float(p)

    return per_sample_scores


# ============================================================
# ТОЧКА ВХОДА
# ============================================================

if __name__ == "__main__":
    # ===================== КОНФИГ (без CLI) =====================
    LORA_PATH = "experiments/results/val_rerank_exp_r16_alpha32_lr2e-5_rerank_full_lora_448"
    VAL_DATASET = "data/processed/dataset_v1/val.json"
    IMAGE_DIR = "images"
    N_SAMPLES = 800           # подвыборка known-сэмплов (target_idx != -1)
    SEED = 42
    MODES = ["full", "blank_query", "noise_query", "blank_candidate",
             "no_text", "mismatch_query"]
    OUT_JSON = "data/eval/ablation_text_leakage.json"
    # ===========================================================

    from e2e_eval_v2 import load_vlm_reranker

    with open(VAL_DATASET, "r", encoding="utf-8") as f:
        data = json.load(f)
    known = [s for s in data if s.get("target_idx", -1) != -1 and s.get("candidates")]
    random.Random(SEED).shuffle(known)
    samples = known[:N_SAMPLES]
    target_idxs = [s["target_idx"] for s in samples]
    print(f"Подвыборка known-сэмплов: {len(samples)}")

    print("Загрузка реранкера...")
    model, processor = load_vlm_reranker(lora_path=LORA_PATH)
    _tok = getattr(processor, "tokenizer", processor)
    yes_id = _tok.convert_tokens_to_ids("Yes")
    no_id = _tok.convert_tokens_to_ids("No")

    results = {}
    print("\n" + "=" * 72)
    print(f"{'режим':<16}{'Hit@1':>9}{'AUROC':>9}{'P+ ':>9}{'P- ':>9}{'разрыв':>9}")
    print("-" * 72)
    for mode in MODES:
        scores = score_samples(model, processor, yes_id, no_id, samples,
                               IMAGE_DIR, mode)
        h = hit_at_1(scores, target_idxs)
        labels, flat = flat_labels_scores(scores, target_idxs)
        a = auroc(labels, flat)
        mp, mn = separation(scores, target_idxs)
        results[mode] = {"hit_1": h, "auroc": a, "mean_pos": mp,
                         "mean_neg": mn, "gap": mp - mn}
        print(f"{mode:<16}{h:>9.3f}{a:>9.3f}{mp:>9.3f}{mn:>9.3f}{mp - mn:>9.3f}")
    print("=" * 72)

    full = results.get("full", {})
    print("\nИнтерпретация (относительно full):")
    for mode in MODES:
        if mode == "full":
            continue
        dh = results[mode]["hit_1"] - full.get("hit_1", 0)
        print(f"  {mode:<16} ΔHit@1 = {dh:+.3f}")
    print("  blank_query близко к full  → реранкер не использует query (плохо)")
    print("  no_text близко к full      → текст не костыль, зрительный (хорошо)")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"n_samples": len(samples), "modes": results}, f,
                  indent=2, ensure_ascii=False)
    print(f"\nСохранено: {OUT_JSON}")
