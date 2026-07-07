# scripts/experiments/build_calib_bundle.py
"""
Собирает МАЛЕНЬКИЙ калибровочный набор для GPTQ-квантизации в Colab.

Зачем: quantize_gptq_qwen2vl.py калибруется на реальных парах ранкера, но
тащить в Colab всю галерею (6 ГБ) незачем — нужно лишь N пар. Скрипт берёт
N примеров из манифеста, копирует только их query+candidate фото и пишет
мини-манифест того же формата. Результат ~15 МБ — загрузить в Colab легко.

Запуск локально (там, где есть images/ и манифест):
    python scripts/experiments/build_calib_bundle.py

Переменные:
    SRC_MANIFEST  манифест-источник (default: data/processed/dataset_v1/train.json)
    SRC_IMAGES    папка с фото       (default: images)
    OUT_DIR       куда сложить набор (default: calib_bundle)
    N             сколько пар        (default: 256)

Потом:
    tar -cf calib_bundle.tar -C calib_bundle .
    # загрузить calib_bundle.tar в Colab, распаковать, указать скрипту
    # квантизации CALIB_MANIFEST=<...>/calib.json  IMAGES_DIR=<...>/images
"""

import json
import os
import random
import shutil

SRC_MANIFEST = os.getenv("SRC_MANIFEST", "data/processed/dataset_v1/train.json")
SRC_IMAGES = os.getenv("SRC_IMAGES", "images")
OUT_DIR = os.getenv("OUT_DIR", "calib_bundle")
N = int(os.getenv("N", "256"))
SEED = 42
CAPTION_MAX = 300


def main() -> None:
    out_images = os.path.join(OUT_DIR, "images")
    os.makedirs(out_images, exist_ok=True)

    with open(SRC_MANIFEST, encoding="utf-8") as f:
        data = json.load(f)
    random.Random(SEED).shuffle(data)

    entries = []
    copied: set = set()
    for item in data:
        if len(entries) >= N:
            break
        cands = item.get("candidates") or []
        idx = item.get("target_idx", 0)
        if not cands:
            continue
        if idx is None or idx < 0 or idx >= len(cands):
            idx = 0
        cand = cands[idx]
        q_name = item["query_image"]
        c_name = cand["image"]
        qp = os.path.join(SRC_IMAGES, q_name)
        cp = os.path.join(SRC_IMAGES, c_name)
        if not (os.path.isfile(qp) and os.path.isfile(cp)):
            continue

        for name, src in ((q_name, qp), (c_name, cp)):
            if name not in copied:
                shutil.copy2(src, os.path.join(out_images, name))
                copied.add(name)

        # Оставляем только целевого кандидата — скрипт квантизации использует
        # именно его; так набор минимален.
        caption = (cand.get("caption") or cand.get("caption_landmark") or "")[:CAPTION_MAX]
        entries.append(
            {
                "query_image": q_name,
                "candidates": [
                    {"image": c_name, "name": cand.get("name", ""), "caption": caption}
                ],
                "target_idx": 0,
            }
        )

    if not entries:
        raise RuntimeError(
            f"Ни одной валидной пары — проверь SRC_MANIFEST={SRC_MANIFEST!r} "
            f"и SRC_IMAGES={SRC_IMAGES!r}"
        )

    out_manifest = os.path.join(OUT_DIR, "calib.json")
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False)

    total_mb = sum(
        os.path.getsize(os.path.join(out_images, n)) for n in os.listdir(out_images)
    ) / 1e6
    print(
        f"✅ Набор готов: {OUT_DIR}\n"
        f"   пар: {len(entries)}  фото: {len(copied)}  размер images/: {total_mb:.1f} МБ\n"
        f"   манифест: {out_manifest}\n\n"
        f"Упаковать и залить в Colab:\n"
        f"   tar -cf calib_bundle.tar -C {OUT_DIR} ."
    )


if __name__ == "__main__":
    main()
