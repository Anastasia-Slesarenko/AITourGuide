# scripts/experiments/quantize_gptq_qwen2vl.py
"""
GPTQ INT4-квантизация Qwen2-VL-2B ранкера для сервинга в vLLM на Tesla T4.

Почему GPTQ, а не AWQ:
  - AutoAWQ (продьюсер формата `--quantization awq`) не совместим с
    transformers>=4.5x.
  - llm-compressor выдаёт compressed-tensors → в vLLM это marlin-ядра
    (Ampere/sm_80+), на T4 (sm_75) не запускается.
  - GPTQ через GPTQModel ставится с новым transformers, квантует Qwen2-VL и
    бежит на T4 (у vLLM GPTQ-ядра работают на Turing). Формат — AutoGPTQ,
    в vLLM `--quantization gptq`.

Запускать в DataSphere (GPU). Проверено под целевой стек:
  transformers==4.57.1  accelerate==1.11.0  peft==0.17.1  pillow==11.3.0
  datasets==4.4.2  torch==2.7.1+cu118

Установка (в дополнение к стеку выше):
  pip install -U gptqmodel

Пайплайн:
  1. (опц.) мёрж LoRA-адаптера в базовую fp16-модель.
  2. Калибровка на РЕАЛЬНЫХ парах ранкера (query-фото + candidate-фото +
     тот же yes/no-промпт, что в проде) — это даёт активации, близкие к
     инференсу, и лучшую точность после квантизации.
  3. GPTQ w4g128, пропуская vision-башню и lm_head (как в текущем экспорте).
  4. Сохранение в AutoGPTQ-формате для vLLM.
"""

from __future__ import annotations

import json
import os
import random

import torch
from PIL import Image

# ── Конфиг ──────────────────────────────────────────────────────────────────

# База: либо уже смёрженная fp16-модель (ADAPTER_DIR=None),
# либо базовая модель + LoRA-адаптер (тогда сначала мёржим).
BASE_MODEL_DIR = os.getenv("BASE_MODEL_DIR", "Qwen/Qwen2-VL-2B-Instruct")
ADAPTER_DIR = os.getenv("ADAPTER_DIR", "")            # путь к LoRA, "" = не мёржить
MERGED_DIR = os.getenv("MERGED_DIR", "./qwen2-vl-2b-r16-merged")

# Данные для калибровки (те же манифест + фото, что в проде).
CALIB_MANIFEST = os.getenv("CALIB_MANIFEST", "data/processed/dataset_v1/train.json")
IMAGES_DIR = os.getenv("IMAGES_DIR", "images")
N_CALIB = int(os.getenv("N_CALIB", "128"))            # 64–256; меньше → меньше ОЗУ
CAPTION_MAX = 300                                     # как caption_max_length в проде
# Две картинки на пример точнее (как в проде), но вдвое больше vision-токенов
# и ОЗУ при калибровке. На маленькой RAM (free Colab ~12 ГБ) ставь =0.
CALIB_TWO_IMAGES = os.getenv("CALIB_TWO_IMAGES", "1") == "1"

# Выход.
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./qwen2-vl-2b-r16-gptq")

BITS = int(os.getenv("BITS", "8"))       # 4 или 8; на T4/exllama INT8 стабильнее — без NaN
GROUP_SIZE = int(os.getenv("GROUP_SIZE", "128"))
SEED = 42


# ── 1. Мёрж LoRA (опционально) ──────────────────────────────────────────────

def merge_lora() -> str:
    """Сливает LoRA-адаптер в базовую fp16-модель. Возвращает путь к результату."""
    if not ADAPTER_DIR:
        return BASE_MODEL_DIR

    from peft import PeftModel
    from transformers import (
        AutoProcessor,
        GenerationConfig,
        Qwen2VLForConditionalGeneration,
    )

    print(f"Мёрж LoRA {ADAPTER_DIR} в {BASE_MODEL_DIR} → {MERGED_DIR}")
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_DIR, torch_dtype=torch.float16, device_map="cpu"
    )
    merged = PeftModel.from_pretrained(base, ADAPTER_DIR).merge_and_unload()
    merged.save_pretrained(MERGED_DIR, safe_serialization=True)
    AutoProcessor.from_pretrained(BASE_MODEL_DIR).save_pretrained(MERGED_DIR)
    # Явно кладём generation_config.json из базы: иначе при загрузке
    # transformers строит его из model_config и падает на вложенном конфиге
    # Qwen2-VL (AttributeError: 'dict' object has no attribute 'to_dict').
    try:
        GenerationConfig.from_pretrained(BASE_MODEL_DIR).save_pretrained(MERGED_DIR)
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ generation_config не скопирован: {e}")
    del base, merged
    torch.cuda.empty_cache()
    return MERGED_DIR


# ── 2. Калибровочный датасет из реальных пар ранкера ────────────────────────

def _rerank_messages(query_path: str, cand_path: str, name: str, caption: str):
    """Формат сообщений как prepare_vlm_messages() в проде.

    CALIB_TWO_IMAGES=1 — query-фото + candidate-фото (точнее, как в инференсе).
    CALIB_TWO_IMAGES=0 — только candidate-фото (вдвое меньше vision-токенов и
    ОЗУ при калибровке; для GPTQ диапазонов активаций обычно достаточно).
    """
    content = []
    if CALIB_TWO_IMAGES:
        content += [
            {"type": "text", "text": "Query Photo:"},
            {"type": "image", "image": Image.open(query_path).convert("RGB")},
        ]
    content += [
        {"type": "text", "text": "Candidate Photo:"},
        {"type": "image", "image": Image.open(cand_path).convert("RGB")},
        {
            "type": "text",
            "text": (
                f'Question: Are these photos showing the same landmark: "{name}"?\n'
                f"Candidate details: {caption[:CAPTION_MAX]}\n"
                f"Answer only with Yes or No."
            ),
        },
    ]
    return [{"role": "user", "content": content}]


def build_calibration() -> list:
    """Строит N_CALIB реальных примеров ранкера из манифеста.

    Для каждого примера берём query_image и его целевого кандидата
    (target_idx) — это положительная пара «тот же объект», типичный вход.
    Пропускаем записи с отсутствующими файлами.
    """
    with open(CALIB_MANIFEST, encoding="utf-8") as f:
        data = json.load(f)

    random.Random(SEED).shuffle(data)
    samples: list = []
    for item in data:
        if len(samples) >= N_CALIB:
            break
        cands = item.get("candidates") or []
        idx = item.get("target_idx", 0)
        if not cands or idx is None or idx < 0 or idx >= len(cands):
            idx = 0
        if not cands:
            continue
        cand = cands[idx]
        qp = os.path.join(IMAGES_DIR, item["query_image"])
        cp = os.path.join(IMAGES_DIR, cand["image"])
        if not (os.path.isfile(qp) and os.path.isfile(cp)):
            continue
        caption = cand.get("caption") or cand.get("caption_landmark") or ""
        try:
            samples.append(
                _rerank_messages(qp, cp, cand.get("name", ""), caption)
            )
        except (OSError, ValueError):
            continue

    if not samples:
        raise RuntimeError(
            f"Не набрано ни одного калибровочного примера — проверь "
            f"CALIB_MANIFEST={CALIB_MANIFEST!r} и IMAGES_DIR={IMAGES_DIR!r}"
        )
    print(f"Калибровка: {len(samples)} реальных пар ранкера")
    return samples


# ── 3. GPTQ-квантизация ─────────────────────────────────────────────────────

def quantize(model_dir: str, calibration: list) -> None:
    from gptqmodel import GPTQModel, QuantizeConfig

    quant_config = QuantizeConfig(
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=True,            # симметричная квантизация — совместима с GPTQ-ядром vLLM
        # act-order (desc_act=True) на T4 (sm75) ломает применение g_idx в
        # GPTQ-ядре vLLM → модель выдаёт мусор. Отключаем: чуть ниже точность,
        # но формат реально запускается на Turing. desc_act вшит в упаковку
        # весов — поменять можно только повторной квантизацией.
        desc_act=False,
    )

    # GPTQModel для Qwen2-VL сам таргетит слои language_model и НЕ трогает
    # vision-башню и lm_head — это соответствует SKIP_MODULES=["visual","lm_head"]
    # из прежнего экспорта.
    model = GPTQModel.load(model_dir, quant_config)
    model.quantize(calibration, batch_size=1)
    model.save(OUTPUT_DIR)

    # Процессор/токенайзер нужны vLLM — сохраняем рядом.
    from transformers import AutoProcessor
    AutoProcessor.from_pretrained(model_dir).save_pretrained(OUTPUT_DIR)
    print(f"\n✅ GPTQ-модель сохранена: {OUTPUT_DIR}")


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)

    model_dir = merge_lora()
    calibration = build_calibration()
    quantize(model_dir, calibration)

    print(
        "\nСервинг в vLLM (T4):\n"
        f"  --model {OUTPUT_DIR}\n"
        "  --served-model-name qwen2-vl-2b-r16\n"
        "  --quantization gptq\n"
        "  --dtype float16 --trust-remote-code\n"
        "\nВыложи папку в ${S3_MOUNT}/models/qwen2-vl-2b-r16-gptq и обнови compose."
    )


if __name__ == "__main__":
    main()
