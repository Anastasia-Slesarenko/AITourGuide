"""
Скрипт для добавления поля name_ru в landmarks_filtered.json.
- Переводит поле name_en с помощью Yandex Translate API.
- Исключает достопримечательности без названия
  (name_en отсутствует или пустое).
- Сохраняет результат обратно в тот же файл.
"""

import json
import os
import asyncio
import aiohttp
from typing import Optional
from tqdm.asyncio import tqdm as atqdm
from dotenv import load_dotenv

load_dotenv()

INPUT_FILE = "/Users/anastasiya/Documents/AITourGuide/data/processed/landmarks_with_guide_descriptions_filtred.json"
OUTPUT_FILE = "/Users/anastasiya/Documents/AITourGuide/data/processed/landmarks_with_guide_descriptions_filtred_2.json"

YC_API_KEY = os.getenv("YC_TRANSLATE_API_KEY")
YC_FOLDER_ID = os.getenv("YC_TRANSLATE_FOLDER_ID")

TRANSLATE_URL = (
    "https://translate.api.cloud.yandex.net/translate/v2/translate"
)

# Размер батча для Yandex Translate (макс. 250 строк за запрос)
BATCH_SIZE = 100
# Задержка между батчами (сек) для соблюдения rate-limit
BATCH_DELAY = 0.3
# Максимальное число параллельных запросов
CONCURRENCY = 5


async def translate_batch(
    session: aiohttp.ClientSession,
    texts: list[str],
    semaphore: asyncio.Semaphore,
    retries: int = 3,
) -> list[Optional[str]]:
    """Переводит батч текстов через Yandex Translate API."""
    headers = {
        "Authorization": f"Api-Key {YC_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "folderId": YC_FOLDER_ID,
        "texts": texts,
        "sourceLanguageCode": "en",
        "targetLanguageCode": "ru",
    }

    async with semaphore:
        for attempt in range(retries):
            try:
                async with session.post(
                    TRANSLATE_URL, json=body, headers=headers
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        translations = data.get("translations", [])
                        return [t.get("text") for t in translations]
                    elif resp.status in (429, 503):
                        wait = 2 ** attempt
                        print(f"  ⏳ HTTP {resp.status}, retry in {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        error_text = await resp.text()
                        print(f"  HTTP {resp.status}: {error_text}")
                        return [None] * len(texts)
            except Exception as e:
                print(f"  Request error: {e}")
                await asyncio.sleep(1 * (attempt + 1))

    return [None] * len(texts)


async def main():
    # ── 1. Загрузка данных ──────────────────────────────────────────────────
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        landmarks: list[dict] = json.load(f)

    print(f"   Total records: {len(landmarks)}")

    # ── 2. Фильтрация: исключаем записи без name_en ─────────────────────────
    before = len(landmarks)
    landmarks = [lm for lm in landmarks if lm.get("name_en", "").strip()]
    after = len(landmarks)
    print(f"   Excluded (no name_en): {before - after}")
    print(f"   Remaining: {after}")

    # ── 3. Определяем, какие записи уже имеют name_ru ──────────────────────
    need_translation = [
        (i, lm["name_en"])
        for i, lm in enumerate(landmarks)
        if not lm.get("name_ru", "").strip()
    ]
    print(f"   Need translation: {len(need_translation)}")

    if not need_translation:
        print("All records already have name_ru. Nothing to translate.")
    else:
        # ── 4. Перевод батчами ──────────────────────────────────────────────
        semaphore = asyncio.Semaphore(CONCURRENCY)
        connector = aiohttp.TCPConnector(limit=CONCURRENCY)

        batches = [
            need_translation[i:i + BATCH_SIZE]
            for i in range(0, len(need_translation), BATCH_SIZE)
        ]

        n_batches = len(batches)
        print(
            f"Translating {len(need_translation)} names "
            f"in {n_batches} batches..."
        )

        async with aiohttp.ClientSession(connector=connector) as session:
            ordered_results: list[list[Optional[str]]] = []
            for batch in atqdm(batches, desc="Translating batches"):
                texts = [item[1] for item in batch]
                translated = await translate_batch(
                    session, texts, semaphore
                )
                ordered_results.append(translated)
                await asyncio.sleep(BATCH_DELAY)

        # ── 5. Записываем переводы в landmarks ─────────────────────────────
        translated_count = 0
        failed_count = 0
        for batch, translations in zip(batches, ordered_results):
            for (idx, name_en), name_ru in zip(batch, translations):
                if name_ru:
                    landmarks[idx]["name_ru"] = name_ru
                    translated_count += 1
                else:
                    # Если перевод не получен — оставляем name_en как fallback
                    landmarks[idx]["name_ru"] = name_en
                    failed_count += 1

        print(f"Translated: {translated_count}")
        if failed_count:
            print(
                f"Failed (used name_en as fallback): {failed_count}"
            )

    # ── 6. Сохранение результата ────────────────────────────────────────────
    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(landmarks, f, ensure_ascii=False, indent=2)

    print(f"Done! Saved {len(landmarks)} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
