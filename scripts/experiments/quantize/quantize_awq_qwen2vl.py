"""
Merge LoRA + int4/int8 квантование Qwen2-VL через bitsandbytes для vLLM.

Пайплайн:
  1. Загружаем base модель + LoRA адаптер
  2. Merge LoRA в веса (merge_and_unload)
  3. Перезагружаем merged модель в int4 NF4 или int8 (bitsandbytes)
  4. Сохраняем в HuggingFace формате

Почему bitsandbytes:
  - Встроен в transformers, не требует доп. установки
  - Поддерживает любую архитектуру (включая multimodal Qwen2-VL)
  - auto-gptq не поддерживает qwen2_vl_text
  - autoawq/llmcompressor несовместимы с transformers==4.57.1

Сравнение режимов:
  int4 NF4: ~1.1GB, быстрее, чуть ниже качество
  int8:     ~2.0GB, медленнее, качество ближе к fp16

Требования:
  pip install bitsandbytes  (обычно уже есть в DataSphere)

Запуск:
  python experiments/export_model_awq.py

После квантования запускать vLLM:
  python -m vllm.entrypoints.openai.api_server \
    --model /path/to/qwen2-vl-2b-r16-int4 \
    --served-model-name qwen2-vl-2b-r16 \
    --quantization bitsandbytes \
    --dtype float16 \
    --port 30000
"""

import json
import torch
from pathlib import Path
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)
from peft import PeftModel

# КОНФИГУРАЦИЯ — редактируйте здесь

# Пути
BASE_MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct"
LORA_CHECKPOINT_PATH = (
    "/home/jupyter/s3/ai-tour-guide/qwen2vl-rerank-lora/"
    "rerank_exp_r16_alpha32_lr2e-5_rerank_full_lora_448"
)
# Промежуточная папка для merged fp16 модели
MERGED_FP16_DIR = Path(
    "/home/jupyter/s3/ai-tour-guide/models/qwen2-vl-2b-r16-merged"
)

# Режим квантования: "int4" или "int8"
#   int4 NF4: ~1.1GB, быстрее, чуть ниже качество
#   int8:     ~2.0GB, медленнее, качество ближе к fp16
QUANT_MODE = "int4"   # <-- меняйте здесь

# Финальная папка с квантованной моделью (подставляется автоматически)
QUANT_OUTPUT_DIR = Path(
    f"/home/jupyter/s3/ai-tour-guide/models/qwen2-vl-2b-r16-{QUANT_MODE}"
)


# Модули, которые НЕ квантуем:
# - visual — весь vision encoder (~300MB, небольшой, квантование нестабильно)
# - lm_head — выходной слой (квантование ухудшает качество генерации)
SKIP_MODULES = ["visual", "lm_head"]

# BitsAndBytes int4 конфиг
# NF4 (NormalFloat4) — лучшее качество для int4, разработан для LLM
BNB_INT4_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NF4 точнее обычного int4
    # двойная квантизация: ~0.4 бит/параметр экономии
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,  # вычисления в fp16
    llm_int8_skip_modules=SKIP_MODULES,    # исключаем visual и lm_head
)

# BitsAndBytes int8 конфиг
# LLM.int8() — поканальное масштабирование, качество близко к fp16
BNB_INT8_CONFIG = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_skip_modules=SKIP_MODULES,    # исключаем visual и lm_head
)

# Выбор конфига по QUANT_MODE
if QUANT_MODE not in ("int4", "int8"):
    raise ValueError(
        f"QUANT_MODE должен быть 'int4' или 'int8', получено: {QUANT_MODE!r}"
    )


def step1_merge_lora() -> Path:
    """
    Шаг 1: Загружаем base + LoRA, делаем merge_and_unload, сохраняем fp16.

    Возвращает путь к merged директории.
    """
    print("\n" + "=" * 60)
    print("Шаг 1: Merge LoRA -> fp16")
    print("=" * 60)

    if MERGED_FP16_DIR.exists():
        print(f"  Уже существует: {MERGED_FP16_DIR}")
        print("  Пропускаем merge (удалите папку чтобы пересоздать)")
        return MERGED_FP16_DIR

    MERGED_FP16_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Загружаем base модель: {BASE_MODEL_PATH}")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"  Загружаем LoRA: {LORA_CHECKPOINT_PATH}")
    model_with_lora = PeftModel.from_pretrained(
        base_model,
        LORA_CHECKPOINT_PATH,
        is_trainable=False,
    )

    print("  Merge LoRA в веса...")
    merged = model_with_lora.merge_and_unload()

    print(f"  Сохраняем merged fp16 -> {MERGED_FP16_DIR}")
    merged.save_pretrained(str(MERGED_FP16_DIR), safe_serialization=True)

    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True
    )
    processor.save_pretrained(str(MERGED_FP16_DIR))

    del merged, model_with_lora, base_model
    torch.cuda.empty_cache()

    print(f"  OK: Merged fp16 сохранён: {MERGED_FP16_DIR}")
    return MERGED_FP16_DIR


def step2_quantize(merged_dir: Path) -> Path:
    """
    Шаг 2: Загружаем merged fp16 модель в int4 NF4 или int8 (bitsandbytes)
    и сохраняем на диск.

    bitsandbytes встроен в transformers и поддерживает Qwen2-VL.
    Размер: ~4GB fp16 -> ~1.1GB int4 / ~2.0GB int8.

    Режим выбирается через QUANT_MODE в конфиге вверху файла.

    Возвращает путь к квантованной директории.
    """
    mode = "int8" if QUANT_MODE == "int8" else "int4 NF4"
    bnb_config = BNB_INT8_CONFIG if QUANT_MODE == "int8" else BNB_INT4_CONFIG

    print("\n" + "=" * 60)
    print(f"Шаг 2: {mode} квантование (bitsandbytes)")
    print("=" * 60)

    if QUANT_OUTPUT_DIR.exists():
        print(f"  Уже существует: {QUANT_OUTPUT_DIR}")
        print("  Пропускаем квантование (удалите папку чтобы пересоздать)")
        return QUANT_OUTPUT_DIR

    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        raise ImportError(
            "bitsandbytes не установлен.\n"
            "Установите: pip install bitsandbytes"
        )

    print(f"  Режим: {mode}")
    print(f"  Загружаем merged модель: {merged_dir}")
    print("  (загрузка сразу квантует веса — занимает ~2-5 мин)")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(merged_dir),
        quantization_config=bnb_config,
        device_map="cuda",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        str(merged_dir),
        trust_remote_code=True,
    )

    QUANT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Сохраняем {mode} модель -> {QUANT_OUTPUT_DIR}")
    # save_pretrained сохраняет квантованные веса + quantization_config.json
    model.save_pretrained(str(QUANT_OUTPUT_DIR), safe_serialization=True)
    processor.save_pretrained(str(QUANT_OUTPUT_DIR))

    del model
    torch.cuda.empty_cache()

    print(f"  OK: {mode} модель сохранена: {QUANT_OUTPUT_DIR}")
    return QUANT_OUTPUT_DIR


def step3_check_vision_weights(quant_dir: Path):
    """
    Шаг 3: Проверяем что visual веса присутствуют в квантованной директории.
    """
    print("\n" + "=" * 60)
    print("Шаг 3: Проверка vision encoder весов")
    print("=" * 60)

    shards = list(quant_dir.glob("*.safetensors"))
    if not shards:
        print("  WARN: директория пуста — пропускаем")
        return

    index_file = quant_dir / "model.safetensors.index.json"
    if not index_file.exists():
        print("  Один shard — visual веса внутри, OK")
        return

    with open(index_file) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    visual_keys = [k for k in weight_map if k.startswith("visual.")]

    if visual_keys:
        print(
            f"  OK: Visual веса присутствуют "
            f"({len(visual_keys)} тензоров)"
        )
    else:
        print("  WARN: Visual веса отсутствуют!")
        print(
            "  bitsandbytes не сохранил vision encoder.\n"
            "  Нужно скопировать visual веса вручную из merged fp16 модели."
        )


def print_summary(quant_dir: Path):
    """Выводит итоговую информацию и команды для запуска."""
    print("\n" + "=" * 60)
    print("ГОТОВО")
    print("=" * 60)

    total_size = sum(
        f.stat().st_size for f in quant_dir.rglob("*") if f.is_file()
    )
    print(f"  Путь:   {quant_dir}")
    print(f"  Размер: {total_size / 1e9:.2f} GB")

    cmd = (
        "python -m vllm.entrypoints.openai.api_server \\\n"
        f"  --model {quant_dir} \\\n"
        "  --served-model-name qwen2-vl-2b-r16 \\\n"
        "  --quantization bitsandbytes \\\n"
        "  --dtype float16 \\\n"
        "  --port 30000 \\\n"
        "  --host 0.0.0.0 \\\n"
        "  --gpu-memory-utilization 0.85 \\\n"
        "  --max-model-len 4096 \\\n"
        "  --trust-remote-code"
    )
    print("\nЗапуск vLLM:")
    print(cmd)

    suffix = QUANT_MODE
    dc_cmd = (
        "command: >\n"
        "  python -m vllm.entrypoints.openai.api_server\n"
        f"    --model /models/qwen2-vl-2b-r16-{suffix}\n"
        "    --served-model-name qwen2-vl-2b-r16\n"
        "    --quantization bitsandbytes\n"
        "    --dtype float16\n"
        "    --port 30000\n"
        "    --host 0.0.0.0\n"
        "    --gpu-memory-utilization 0.85\n"
        "    --max-model-len 4096\n"
        "    --trust-remote-code"
    )
    print("\ndocker-compose.yml:")
    print(dc_cmd)


def main():
    mode = "int8" if QUANT_MODE == "int8" else "int4 NF4"
    print("=" * 60)
    print(f"Qwen2-VL: LoRA merge + {mode} квантование (bitsandbytes)")
    print("=" * 60)
    print(f"  Base модель:  {BASE_MODEL_PATH}")
    print(f"  LoRA:         {LORA_CHECKPOINT_PATH}")
    print(f"  Merged fp16:  {MERGED_FP16_DIR}")
    print(f"  Output:       {QUANT_OUTPUT_DIR}")

    merged_dir = step1_merge_lora()
    quant_dir = step2_quantize(merged_dir)
    step3_check_vision_weights(quant_dir)
    print_summary(quant_dir)


if __name__ == "__main__":
    main()
