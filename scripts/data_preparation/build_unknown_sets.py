# -*- coding: utf-8 -*-
"""
Сборка НАСТОЯЩИХ unknown-наборов для open-set оценки.

Источник — novel_landmark.json: объекты, которых НЕТ в gallery-индексе
(проверено: 0 пересечений landmark_id с gallery_metadata). Их query-изображения
дают истинные unknown: правильного ответа в базе нет.

Что делает:
  1. Разбивает novel-ОБЪЕКТЫ на val/test (по landmark_id, непересекающиеся —
     иначе утечка при подборе порога reject).
  2. Для каждого изображения объекта строит e2e-сэмпл:
       {query_image, target_idx=-1, candidates=[], meta={landmark_id, ...}}.
     candidates не нужны — e2e делает свежий retrieval из индекса.
  3. (Опционально) проверяет существование файлов под image_dir.

Использование в оценке:
  порог  ← known-val (val.json) + novel-val-unknown
  метрики ← known-test (test.json) + novel-test-unknown
  Истинная метка: novel → unknown, известные → known (объект есть в индексе).

Запуск:
    python experiments/build_unknown_sets.py \
        --novel data/processed/novel_landmark.json \
        --out-dir data/processed/dataset_v1 \
        --image-dir images
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List


def build_unknown_items(
    landmarks: List[Dict],
    max_images_per_landmark: int = 0,
    image_dir: str | None = None,
) -> tuple[List[Dict], int]:
    """Строит unknown-сэмплы из списка novel-объектов.

    Args:
        landmarks: список novel-объектов (формат landmarks_*.json).
        max_images_per_landmark: 0 = все изображения, иначе кап на объект.
        image_dir: если задан — считает отсутствующие файлы (не отбрасывает).

    Returns:
        (items, n_missing)
    """
    items: List[Dict] = []
    n_missing = 0

    for lm in landmarks:
        lid = lm.get("landmark_id")
        if not lid:
            continue
        name = (lm.get("name_en") or lm.get("name_ru") or "").strip()

        imgs = lm.get("valid_images", [])
        if max_images_per_landmark > 0:
            imgs = imgs[:max_images_per_landmark]

        for img in imgs:
            path = img.get("path") if isinstance(img, dict) else img
            if not path:
                continue
            if image_dir is not None and not os.path.exists(os.path.join(image_dir, path)):
                n_missing += 1
                continue
            items.append({
                "query_image": path,
                "candidates": [],          # e2e делает свежий retrieval
                "target_idx": -1,          # unknown по построению
                "meta": {
                    "landmark_id": lid,     # НЕТ в индексе → истинный unknown
                    "landmark_name": name,
                    "is_novel_unknown": True,
                },
            })

    return items, n_missing


def split_landmarks(
    landmarks: List[Dict], val_ratio: float, seed: int
) -> tuple[List[Dict], List[Dict]]:
    """Делит объекты на val/test по landmark_id (непересекающиеся)."""
    shuffled = landmarks[:]
    random.Random(seed).shuffle(shuffled)
    n_val = round(len(shuffled) * val_ratio)
    return shuffled[:n_val], shuffled[n_val:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Сборка настоящих unknown-наборов из novel_landmark.json")
    parser.add_argument("--novel", default="data/processed/novel_landmark.json")
    parser.add_argument("--out-dir", default="data/processed/dataset_v1")
    parser.add_argument("--image-dir", default="images",
                        help="База для проверки существования файлов (None = не проверять)")
    parser.add_argument("--val-ratio", type=float, default=0.5,
                        help="Доля novel-ОБЪЕКТОВ в val (порог тюнится на val)")
    parser.add_argument("--max-images-per-landmark", type=int, default=0,
                        help="0 = все изображения; иначе кап (для баланса/кластеризации)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gallery-metadata", default=None,
                        help="Если задан — перепроверяет, что novel не в индексе")
    args = parser.parse_args()

    with open(args.novel, "r", encoding="utf-8") as f:
        novel = json.load(f)
    print(f"Загружено novel-объектов: {len(novel)}")

    # Страховочная проверка членства в индексе.
    if args.gallery_metadata:
        with open(args.gallery_metadata, "r", encoding="utf-8") as f:
            gm = json.load(f)
        gallery_ids = {m.get("landmark_id") for m in gm}
        overlap = {x.get("landmark_id") for x in novel} & gallery_ids
        if overlap:
            raise SystemExit(
                f"{len(overlap)} novel-объектов ЕСТЬ в индексе — это не unknown! "
                f"Примеры: {list(overlap)[:5]}"
            )
        print(f"Проверка членства: 0 пересечений с индексом ({len(gallery_ids)} объектов)")

    novel_val, novel_test = split_landmarks(novel, args.val_ratio, args.seed)
    print(f"Split объектов: val={len(novel_val)}  test={len(novel_test)}  (seed={args.seed})")

    image_dir = None if args.image_dir in ("", "None") else args.image_dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, lms in (("val", novel_val), ("test", novel_test)):
        items, missing = build_unknown_items(
            lms,
            max_images_per_landmark=args.max_images_per_landmark,
            image_dir=image_dir,
        )
        out_path = out_dir / f"novel_{split_name}_unknown.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        miss_str = f", пропущено отсутствующих файлов: {missing}" if missing else ""
        print(f"  {split_name}: {len(items)} unknown-сэмплов из {len(lms)} объектов{miss_str}")
        print(f"    → {out_path}")


if __name__ == "__main__":
    main()
