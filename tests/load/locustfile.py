# tests/load/locustfile.py
"""
Нагрузочное тестирование AITourGuide API с помощью Locust.

Методика:
  - Реалистичный микс входов. Задержка сервиса бимодальна, и ветку выбирает
    контент: фото объекта ИЗ индекса идёт быстрым путём retrieval+rerank,
    фото объекта НЕ из индекса (novel) уводит запрос в дорогой fallback
    (Yandex + Wikipedia + повторная VLM-верификация). Поэтому пул набирается
    из двух наборов, а не из случайных/шумовых картинок:
      known  — query_image из test.json           (объекты есть в индексе),
      novel  — query_image из novel_test_unknown.json (объектов нет в индексе).
    Имена берутся из манифестов, байты — из LOAD_TEST_IMAGE_DIR (база фото).
  - Ступенчатая нагрузка. StepLoadShape поднимает число пользователей
    ступенями и держит каждую фиксированное время. Рабочая точка —
    последняя ступень, где error rate ≈ 0 и p95 в пределах SLO, а не плато
    на максимуме, где меряется лишь глубина очереди.
  - Строгий учёт статусов. Любой не-200 на /v1/predict считается провалом
    (429 — забыли отключить rate limiter; 503 — сработал backpressure;
    504 — таймаут). Так error rate честно показывает точку деградации.
  - Rate limiter на время теста отключается: RATE_LIMIT_ENABLED=false на
    стороне API (иначе один IP Locust упирается в общий бакет и меряется
    лимитер, а не модель).

Запуск (headless, ступенчатый профиль из StepLoadShape):
    RATE_LIMIT_ENABLED=false \
    locust -f tests/load/locustfile.py --host http://localhost:8000 \
        --headless --reset-stats \
        --csv tests/load/results/report --html tests/load/results/report.html

Переменные окружения теста (на стороне Locust):
    LOAD_TEST_IMAGE_DIR      — база фото (S3-маунт), к ней резолвятся query_image
                               (default: images)
    LOAD_TEST_KNOWN_MANIFEST — манифест known-набора, поле query_image
                               (default: data/processed/dataset_v1/test.json)
    LOAD_TEST_NOVEL_MANIFEST — манифест novel-набора
                               (default: data/processed/dataset_v1/novel_test_unknown.json)
    LOAD_TEST_POOL_SIZE      — фото на пул, держится в памяти (default: 64)
    LOAD_TEST_SCAN_LIMIT     — сколько имён вычитать из манифеста; ограничивает
                               чтение большого JSON с S3 (default: 2000)
    LOAD_TEST_STAGES         — ступени "users:seconds,..." (default: 5:60,10:60,20:60,40:60)
    LOAD_TEST_SPAWN_RATE     — скорость появления пользователей, польз/с (default: 10)

Манифесты и фото могут лежать в разных папках S3-маунта — пути независимы.

Переменная окружения API (на стороне сервера, не Locust):
    RATE_LIMIT_ENABLED=false — отключить rate limiter на время теста
"""

import io
import os
import random
import re

import requests
from locust import HttpUser, LoadTestShape, between, events, task
from PIL import Image

# --- Параметры теста из окружения -------------------------------------------

IMAGE_DIR = os.getenv("LOAD_TEST_IMAGE_DIR", "images")
KNOWN_MANIFEST = os.getenv(
    "LOAD_TEST_KNOWN_MANIFEST", "data/processed/dataset_v1/test.json"
)
NOVEL_MANIFEST = os.getenv(
    "LOAD_TEST_NOVEL_MANIFEST", "data/processed/dataset_v1/novel_test_unknown.json"
)
POOL_SIZE = int(os.getenv("LOAD_TEST_POOL_SIZE", "64"))
SPAWN_RATE = float(os.getenv("LOAD_TEST_SPAWN_RATE", "10"))
# Ступени "users:seconds", через запятую. Каждая держится указанное время.
STAGES_RAW = os.getenv("LOAD_TEST_STAGES", "5:60,10:60,20:60,40:60")
# Сколько имён вычитать из манифеста. test.json ~255 МБ; читаем потоково и
# останавливаемся, набрав окно, затем сэмплим пул из него — это ограничивает
# объём чтения с S3 началом файла.
SCAN_LIMIT = int(os.getenv("LOAD_TEST_SCAN_LIMIT", "2000"))

# Извлекаем имена query_image из манифеста без полного json.load.
_QUERY_RE = re.compile(rb'"query_image"\s*:\s*"([^"]+)"')

# Пулы изображений в памяти (заполняются в _on_test_start).
_KNOWN_POOL: list[bytes] = []
_NOVEL_POOL: list[bytes] = []


def _encode_jpeg(img: Image.Image, max_side: int = 1024, quality: int = 85) -> bytes:
    """Приводит изображение к реалистичному пользовательскому виду: RGB,
    сторона не больше max_side, JPEG. Возвращает байты."""
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _make_synthetic_jpeg(width: int = 224, height: int = 224) -> bytes:
    """Fallback: синтетический JPEG случайного цвета (нереалистичный вход)."""
    color = (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    )
    return _encode_jpeg(Image.new("RGB", (width, height), color=color))


def _scan_manifest_names(manifest_path: str, limit: int) -> list[str]:
    """Потоково извлекает до limit имён query_image из манифеста.

    Читает файл чанками и останавливается, набрав limit имён, — полный
    json.load неприемлем (test.json ~255 МБ, тем более с S3). Хвост без
    завершённого совпадения переносится в следующий чанк, поэтому имена
    на границе чанка не теряются и не дублируются.
    """
    names: list[str] = []
    buf = b""
    with open(manifest_path, "rb") as f:
        while len(names) < limit:
            chunk = f.read(1 << 20)  # 1 МиБ
            if not chunk:
                break
            buf += chunk
            last_end = 0
            for m in _QUERY_RE.finditer(buf):
                names.append(m.group(1).decode("utf-8"))
                last_end = m.end()
                if len(names) >= limit:
                    break
            buf = buf[last_end:]  # непросмотренный хвост
    return names


def _load_pool_from_manifest(label: str, manifest_path: str) -> list[bytes]:
    """Собирает пул байтов POOL_SIZE фото по именам из манифеста.

    Имена (query_image) резолвятся к IMAGE_DIR. Чтение с диска на каждый
    запрос исказило бы тайминги, поэтому пул грузится один раз на старте.
    Если манифест или база фото недоступны — fallback на синтетические
    изображения с явным предупреждением (нереалистичный вход).
    """
    if not os.path.isfile(manifest_path):
        print(
            f"⚠️  [{label}] манифест {manifest_path!r} не найден — "
            f"синтетические изображения (нереалистичный вход)"
        )
        return [_make_synthetic_jpeg() for _ in range(POOL_SIZE)]

    names = _scan_manifest_names(manifest_path, SCAN_LIMIT)
    if not names:
        print(f"⚠️  [{label}] в манифесте нет query_image — синтетические изображения")
        return [_make_synthetic_jpeg() for _ in range(POOL_SIZE)]

    sample = random.sample(names, min(POOL_SIZE, len(names)))
    pool: list[bytes] = []
    for name in sample:
        try:
            with Image.open(os.path.join(IMAGE_DIR, name)) as img:
                pool.append(_encode_jpeg(img))
        except (OSError, ValueError) as exc:  # noqa: PERF203
            print(f"⚠️  [{label}] пропущено {name}: {exc}")

    if not pool:
        print(
            f"⚠️  [{label}] ни одно фото не удалось прочитать из {IMAGE_DIR!r} — "
            f"синтетические изображения"
        )
        return [_make_synthetic_jpeg() for _ in range(POOL_SIZE)]

    print(f"✅ [{label}] пул: {len(pool)} фото ({manifest_path} → {IMAGE_DIR})")
    return pool


def _random_known() -> bytes:
    """Случайное known-фото (объект есть в индексе)."""
    return random.choice(_KNOWN_POOL)


def _random_novel() -> bytes:
    """Случайное novel-фото (объекта нет в индексе)."""
    return random.choice(_NOVEL_POOL)


@events.test_start.add_listener
def _on_test_start(environment, **kwargs):
    """Готовит пулы изображений и делает один прогрев модели перед нагрузкой.

    Первый /v1/predict загружает модель (cold start). Делаем ОДИН прогрев
    на весь тест напрямую через requests (не через Locust), чтобы cold start
    не пачкал статистику. В паре с --reset-stats это даёт чистые P50/P95/P99.
    """
    global _KNOWN_POOL, _NOVEL_POOL
    _KNOWN_POOL = _load_pool_from_manifest("known", KNOWN_MANIFEST)
    _NOVEL_POOL = _load_pool_from_manifest("novel", NOVEL_MANIFEST)

    host = (environment.host or "http://localhost:8000").rstrip("/")
    try:
        requests.get(f"{host}/v1/health", timeout=60)
        requests.post(
            f"{host}/v1/predict",
            files={"image": ("warmup.jpg", _random_known(), "image/jpeg")},
            data={"use_internet_search": "false"},
            timeout=120,
        )
    except requests.RequestException as exc:  # noqa: BLE001
        print(f"⚠️  Warmup request failed (продолжаем тест): {exc}")


class APIUser(HttpUser):
    """Симулирует пользователя сервиса: пауза 1–5 с между запросами."""

    wait_time = between(1, 5)

    def on_start(self):
        """Проверяем доступность сервиса (прогрев модели — в _on_test_start)."""
        self.client.get("/v1/health", name="/v1/health")

    @task(7)
    def predict_known(self):
        """Основной путь: объект есть в индексе → retrieval + rerank (вес 7).

        Самый частый в реальном трафике сценарий. use_internet_search=true
        (продовый дефолт), но при высокой уверенности fallback не срабатывает.
        """
        with self.client.post(
            "/v1/predict",
            files={"image": ("photo.jpg", _random_known(), "image/jpeg")},
            data={"use_internet_search": "true"},
            name="/v1/predict (known)",
            catch_response=True,
        ) as response:
            self._check_predict(response)

    @task(3)
    def predict_novel(self):
        """Fallback-путь: объекта нет в индексе → интернет-поиск (вес 3).

        Низкая уверенность на novel-фото уводит запрос в дорогой fallback
        (Yandex + Wikipedia + повторная VLM-верификация). Реже, чем known.
        """
        with self.client.post(
            "/v1/predict",
            files={"image": ("photo.jpg", _random_novel(), "image/jpeg")},
            data={"use_internet_search": "true"},
            name="/v1/predict (novel→internet)",
            catch_response=True,
        ) as response:
            self._check_predict(response)

    @task(1)
    def health_check(self):
        """Проверка состояния (вес 1 — имитирует мониторинг)."""
        self.client.get("/v1/health", name="/v1/health")

    @task(1)
    def metrics(self):
        """Prometheus-метрики (вес 1 — редкий запрос мониторинга)."""
        self.client.get("/metrics", name="/metrics")

    @staticmethod
    def _check_predict(response) -> None:
        """Строгая проверка ответа predict.

        Любой не-200 — провал, чтобы error rate честно показывал точку
        деградации:
          429 — включён rate limiter (для теста его надо отключить),
          503 — сработал backpressure (перегрузка, запрос отброшен),
          504 — таймаут предсказания.
        """
        if response.status_code != 200:
            response.failure(f"HTTP {response.status_code}")
            return
        try:
            data = response.json()
        except ValueError:
            response.failure("Невалидный JSON")
            return
        if "confidence" not in data:
            response.failure("Нет поля confidence")
        else:
            response.success()


def _parse_stages(raw: str) -> list[tuple[int, int]]:
    """Разбирает "users:seconds,..." в список (users, cumulative_end_seconds)."""
    stages: list[tuple[int, int]] = []
    elapsed = 0
    for chunk in raw.split(","):
        users_str, _, secs_str = chunk.strip().partition(":")
        users = int(users_str)
        secs = int(secs_str) if secs_str else 60
        elapsed += secs
        stages.append((users, elapsed))
    return stages


class StepLoadShape(LoadTestShape):
    """Ступенчатый профиль нагрузки.

    Держит каждую ступень фиксированное время, затем повышает число
    пользователей. Позволяет найти «колено» — максимальную нагрузку,
    при которой задержки ещё в пределах SLO.
    """

    stages = _parse_stages(STAGES_RAW)

    def tick(self):
        run_time = self.get_run_time()
        for users, end_time in self.stages:
            if run_time < end_time:
                return (users, SPAWN_RATE)
        return None
