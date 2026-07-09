# tests/unit/test_predict.py
"""
Юнит-тесты оркестратора AITourGuide.predict().

Ретривер и vLLM замоканы — реальные модели/индекс не нужны. Проверяется
бизнес-логика поверх реранкинга: выбор победителя по argmax(p_yes),
согласованность name/winner_images/retrieved_* (регресс на баг, когда
name брался от реранкера, а winner_images — от top-1 ретривера),
порог known/unknown, калибровка отдаваемой уверенности и кэш.
"""

import io
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from cachetools import TTLCache
from PIL import Image

from src.rag.landmark_retriever import GalleryImageMetadata, LandmarkRetrievalResult
from src.services.ai_tour_guide import AITourGuide
from src.services.calibration import ConfidenceCalibrator

THRESHOLD = 0.472656


def _jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (120, 90, 60)).save(buf, format="JPEG")
    return buf.getvalue()


def _gmeta(image_path, lid, name_en, name_ru, caption):
    return GalleryImageMetadata(
        image_id=0,
        image_path=image_path,
        landmark_id=lid,
        landmark_name=name_en,
        caption_landmark="",
        caption=caption,
        landmark_name_ru=name_ru,
    )


def _landmark(lid, name_en, name_ru, agg_score, images):
    """images: list[(score, image_path)] — старший score = top_image."""
    gallery = [
        (s, _gmeta(path, lid, name_en, name_ru, f"caption {path}"))
        for s, path in images
    ]
    return LandmarkRetrievalResult(
        landmark_id=lid,
        landmark_name=name_en,
        aggregated_score=agg_score,
        gallery_images=gallery,
    )


def _vllm_response(p_yes: float) -> dict:
    """Строит ответ vLLM с logprobs Yes/No, дающими нужный p_yes."""
    if p_yes >= 1.0:
        ly, ln = 20.0, 0.0
    elif p_yes <= 0.0:
        ly, ln = -20.0, 0.0
    else:
        ly, ln = math.log(p_yes / (1.0 - p_yes)), 0.0
    return {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "top_logprobs": [
                                {"token": "Yes", "logprob": ly},
                                {"token": "No", "logprob": ln},
                            ]
                        }
                    ]
                }
            }
        ]
    }


def _make_guide(retrieved, p_yes_by_name, calibrator=None, threshold=THRESHOLD):
    """
    Собирает AITourGuide без тяжёлого __init__: только атрибуты, которые
    использует predict(). chat_completion возвращает p_yes по EN-имени
    кандидата, извлечённому из текста промпта.
    """
    guide = object.__new__(AITourGuide)
    guide.config = SimpleNamespace(vlm_semaphore_limit=4)
    guide.vlm_threshold = threshold
    guide.top_k_retrieval = 10
    guide.faiss_k = 100
    guide.caption_max_length = 300
    guide.images_base_dir = None
    guide.calibrator = calibrator or ConfidenceCalibrator(curve_path=None)

    guide.retriever = MagicMock()
    guide.retriever.retrieve = MagicMock(return_value=retrieved)

    async def _chat_completion(messages, **kwargs):
        question = messages[0]["content"][-1]["text"]
        for name, p in p_yes_by_name.items():
            if f'"{name}"' in question:
                return _vllm_response(p)
        return _vllm_response(0.0)

    guide.vllm_client = MagicMock()
    guide.vllm_client.chat_completion = AsyncMock(side_effect=_chat_completion)

    # Изображение кандидата не читаем с диска — заглушка URI.
    guide._get_candidate_image_uri = AsyncMock(
        return_value="data:image/jpeg;base64,AAAA"
    )
    guide.internet_search = MagicMock()
    guide._gallery_image_cache = {}
    guide._predict_cache = TTLCache(maxsize=200, ttl=3600)
    return guide


# Сценарий: top-1 ретривера ≠ победитель реранкера.
#   A (Kremlin)   — retrieval top-1 (score 0.9), но p_yes низкий (0.20)
#   B (Hermitage) — retrieval #2 (0.7),   p_yes высокий (0.95) → победитель
#   C (Peterhof)  — retrieval #3 (0.6),   p_yes средний (0.40)
def _rerank_scenario():
    return [
        _landmark("A", "Kremlin", "Кремль", 0.9, [(0.9, "a1.jpg"), (0.8, "a2.jpg")]),
        _landmark("B", "Hermitage", "Эрмитаж", 0.7, [(0.7, "b1.jpg")]),
        _landmark("C", "Peterhof", "Петергоф", 0.6, [(0.6, "c1.jpg")]),
    ]


class TestRerankWinnerConsistency:
    """Победитель реранкера согласован во всех полях результата."""

    async def test_winner_is_reranker_argmax_not_retrieval_top1(self):
        guide = _make_guide(
            _rerank_scenario(),
            {"Kremlin": 0.20, "Hermitage": 0.95, "Peterhof": 0.40},
        )
        res = await guide.predict(_jpeg_bytes(), use_internet_search=False)

        assert res["unknown"] is False
        # name/description — от реранкера (B), не от top-1 ретривера (A)
        assert res["name"] == "Эрмитаж"
        # регресс-проверка: winner_images и id указывают на того же B
        assert res["winner_landmark_id"] == "B"
        assert res["winner_images"] == ["b1.jpg"]

    async def test_retrieved_lists_ordered_by_p_yes_winner_first(self):
        guide = _make_guide(
            _rerank_scenario(),
            {"Kremlin": 0.20, "Hermitage": 0.95, "Peterhof": 0.40},
        )
        res = await guide.predict(_jpeg_bytes(), use_internet_search=False)

        # индекс 0 = победитель, дальше по убыванию p_yes
        assert res["retrieved_names"] == ["Эрмитаж", "Петергоф", "Кремль"]
        assert res["retrieved_images"] == ["b1.jpg", "c1.jpg", "a1.jpg"]
        assert res["retrieved_names"][0] == res["name"]
        # p_yes невозрастающий, максимум — на позиции победителя
        assert res["retrieved_p_yes"] == sorted(res["retrieved_p_yes"], reverse=True)
        assert res["retrieved_p_yes"][0] == max(res["retrieved_p_yes"])

    async def test_confidence_is_winner_raw_p_yes_without_calibration(self):
        guide = _make_guide(
            _rerank_scenario(),
            {"Kremlin": 0.20, "Hermitage": 0.95, "Peterhof": 0.40},
        )
        res = await guide.predict(_jpeg_bytes(), use_internet_search=False)
        assert res["confidence"] == pytest.approx(0.95, abs=1e-3)
        assert res["confidence_band"] == "high"
        assert res["source"] == "retrieval"


class TestUnknown:
    """Все p_yes ниже порога → объект не распознан, поля очищены."""

    async def test_low_confidence_marks_unknown_and_clears_fields(self):
        guide = _make_guide(
            _rerank_scenario(),
            {"Kremlin": 0.20, "Hermitage": 0.30, "Peterhof": 0.10},
        )
        res = await guide.predict(_jpeg_bytes(), use_internet_search=False)

        assert res["unknown"] is True
        assert res["name"] == ""
        assert res["description"] == ""
        assert res["winner_images"] == []
        assert res["winner_landmark_id"] == ""
        assert res["retrieved_names"] == []
        assert res["confidence_band"] is None


class TestCalibrationApplied:
    """Решение known/unknown — на сыром p_yes; отдаётся калиброванный."""

    async def test_decision_on_raw_but_emits_calibrated_confidence(self):
        # кривая тянет 0.95 → 0.49 (переуверенность)
        cal = ConfidenceCalibrator(curve_path=None)
        cal._x = [0.0, 0.95, 1.0]
        cal._y = [0.0, 0.49, 1.0]
        guide = _make_guide(
            _rerank_scenario(),
            {"Kremlin": 0.20, "Hermitage": 0.95, "Peterhof": 0.40},
            calibrator=cal,
        )
        res = await guide.predict(_jpeg_bytes(), use_internet_search=False)

        # сырой 0.95 >= порога → принято (не unknown), название сохранено
        assert res["unknown"] is False
        assert res["name"] == "Эрмитаж"
        # но наружу отдаётся калиброванная уверенность
        assert res["confidence"] == pytest.approx(0.49, abs=1e-3)
        assert res["confidence_band"] == "medium"


class TestCache:
    """Одинаковый bytes-вход обслуживается из кэша — ретривер не дёргается."""

    async def test_second_call_hits_cache(self):
        guide = _make_guide(
            _rerank_scenario(),
            {"Kremlin": 0.20, "Hermitage": 0.95, "Peterhof": 0.40},
        )
        data = _jpeg_bytes()
        first = await guide.predict(data, use_internet_search=False)
        second = await guide.predict(data, use_internet_search=False)

        assert first["name"] == second["name"] == "Эрмитаж"
        assert guide.retriever.retrieve.call_count == 1
