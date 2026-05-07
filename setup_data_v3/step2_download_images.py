import ast
import io
import pandas as pd
from pathlib import Path
from PIL import Image
import aiofiles
import aiohttp
import asyncio
from tqdm import tqdm


IMAGES_DIR = Path("/Users/anastasiya/Documents/AITourGuide/images")
HEADERS = {"User-Agent": "AIGuideBot/1.0 (slesarenko221999@gmail.com)"}
MAX_RETRIES = 3
MAX_PER_LANDMARK = 5
# Wikimedia: не более 200 запросов в секунду на IP,
# но на практике банят уже при ~5 параллельных.
# Семафор 3 + задержка 1с даёт ~3 req/s — безопасный темп.
SEMAPHORE_SIZE = 3
# минимальная пауза между запросами в одном слоте (секунды)
REQUEST_DELAY = 1.0
TIMEOUT = aiohttp.ClientTimeout(total=60)
CSV_PATH = (
    "/Users/anastasiya/Documents/AITourGuide/setup_data_v3/data/df_collected_landmark_id_imgs_gr.csv"
)


def _resize_image(data: bytes) -> bytes:
    """Масштабирует изображение до 448×448 (сохраняет пропорции)."""
    with Image.open(io.BytesIO(data)) as img:
        img.thumbnail((448, 448))
        buf = io.BytesIO()
        fmt = img.format or "JPEG"
        img.save(buf, format=fmt)
        return buf.getvalue()


async def download_image(
        url: str,
        save_path: Path,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        pbar: tqdm,
        ok_counter: list,
) -> bool:
    # пропускаем уже скачанные файлы
    if save_path.exists():
        ok_counter[0] += 1
        pbar.set_postfix(ok=ok_counter[0], refresh=False)
        pbar.update(1)
        return True

    for attempt in range(1, MAX_RETRIES + 1):
        async with semaphore:
            # равномерная пауза перед каждым запросом
            await asyncio.sleep(REQUEST_DELAY)
            try:
                async with session.get(url, headers=HEADERS) as response:
                    if response.status == 200:
                        data = await response.read()
                        loop = asyncio.get_event_loop()
                        data = await loop.run_in_executor(
                            None, _resize_image, data
                        )
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        async with aiofiles.open(save_path, "wb") as f:
                            await f.write(data)
                        ok_counter[0] += 1
                        pbar.set_postfix(ok=ok_counter[0], refresh=False)
                        pbar.update(1)
                        return True
                    elif response.status in (429, 503):
                        ra = response.headers.get("Retry-After", "")
                        wait = int(ra) if ra.isdigit() else 10 * attempt
                        print(
                            f"\n[{response.status}] повтор через {wait}с "
                            f"(попытка {attempt}/{MAX_RETRIES}): {url}"
                        )
                        await asyncio.sleep(wait)
                    else:
                        print(f"\n[{response.status}] пропускаем: {url}")
                        pbar.update(1)
                        return False
            except asyncio.TimeoutError:
                wait = 10 * attempt
                print(
                    f"\nТаймаут, повтор через {wait}с "
                    f"(попытка {attempt}/{MAX_RETRIES}): {url}"
                )
                await asyncio.sleep(wait)
            except Exception as e:
                print(f"\nОшибка [{type(e).__name__}]: {e} | {url}")
                pbar.update(1)
                return False

    print(f"\nНе удалось загрузить после {MAX_RETRIES} попыток: {url}")
    pbar.update(1)
    return False


async def main_download_image(df: pd.DataFrame) -> None:
    """Скачивает изображения из df (колонки: landmark_id, url, name).
    Колонки url и name могут быть строками-представлениями списков (из CSV).
    """
    # собираем пары (url, path), не более MAX_PER_LANDMARK на каждую
    pairs: list[tuple[str, Path]] = []
    for _, row in df.iterrows():
        urls = row["url"]
        names = row["name"]
        if isinstance(urls, str):
            urls = ast.literal_eval(urls)
        if isinstance(names, str):
            names = ast.literal_eval(names)
        for url, name in zip(
            urls[:MAX_PER_LANDMARK], names[:MAX_PER_LANDMARK]
        ):
            pairs.append((url, IMAGES_DIR / name))

    total = len(pairs)
    print(
        f"Всего задач: {total} "
        f"(до {MAX_PER_LANDMARK} фото × {len(df)} достопримечательностей)"
    )

    semaphore = asyncio.Semaphore(SEMAPHORE_SIZE)
    connector = aiohttp.TCPConnector(limit=SEMAPHORE_SIZE, ttl_dns_cache=300)

    ok_counter = [0]  # список для мутации внутри корутин
    with tqdm(total=total, desc="Скачивание изображений") as pbar:
        async with aiohttp.ClientSession(
            connector=connector, timeout=TIMEOUT
        ) as session:
            tasks = [
                download_image(
                    url, save_path, session, semaphore, pbar, ok_counter
                )
                for url, save_path in pairs
            ]
            results = await asyncio.gather(*tasks)

    ok = sum(results)
    print(f"Готово: {ok}/{total} изображений скачано успешно")


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    asyncio.run(main_download_image(df))
