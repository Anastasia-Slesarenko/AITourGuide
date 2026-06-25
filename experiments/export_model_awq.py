"""
Merge LoRA + GPTQ int4 квантование Qwen2-VL для SGLang на T4 (sm75).

Пайплайн:
  1. Загружаем base модель + LoRA адаптер
  2. Merge LoRA в веса (merge_and_unload)
  3. Квантуем LLM-часть в GPTQ int4, vision encoder оставляем в fp16
  4. Сохраняем в HuggingFace формате — SGLang читает напрямую

Требования:
  pip install auto-gptq optimum

  auto-gptq совместим с transformers==4.57.1 и не ломает зависимости.

Запуск:
  python experiments/export_model_awq.py

После квантования запускать SGLang:
  python -m sglang.launch_server \
    --model-path /path/to/qwen2-vl-2b-r16-gptq \
    --served-model-name qwen2-vl-2b-r16 \
    --quantization gptq \
    --dtype float16 \
    --port 30000
"""

import json
import torch
from pathlib import Path
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

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
# Финальная папка с GPTQ моделью
GPTQ_OUTPUT_DIR = Path(
    "/home/jupyter/s3/ai-tour-guide/models/qwen2-vl-2b-r16-gptq"
)

# GPTQ параметры
GPTQ_CONFIG = {
    "bits": 4,
    "group_size": 128,
    "desc_act": False,   # False = быстрее на T4, True = чуть точнее
    # Vision encoder НЕ квантуем:
    # - он уже небольшой (~300MB в fp16)
    # - основной bottleneck — LLM-часть (28 слоёв)
    "modules_to_not_convert": [
        "visual",   # весь vision encoder Qwen2-VL
        "lm_head",  # выходной слой — не квантуем для стабильности
    ],
}


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


def step2_quantize_gptq(merged_dir: Path) -> Path:
    """
    Шаг 2: GPTQ int4 квантование LLM-части, vision encoder остаётся fp16.

    Использует auto-gptq — совместим с transformers==4.57.1.
    Установка: pip install auto-gptq optimum

    Размер модели: ~4GB fp16 → ~1.3GB int4 (только LLM-часть).

    Возвращает путь к GPTQ директории.
    """
    print("\n" + "=" * 60)
    print("Шаг 2: GPTQ int4 квантование (auto-gptq)")
    print("=" * 60)
    print(f"  Не квантуем: {GPTQ_CONFIG['modules_to_not_convert']}")

    if GPTQ_OUTPUT_DIR.exists():
        print(f"  Уже существует: {GPTQ_OUTPUT_DIR}")
        print("  Пропускаем квантование (удалите папку чтобы пересоздать)")
        return GPTQ_OUTPUT_DIR

    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    except ImportError:
        raise ImportError(
            "auto-gptq не установлен.\n"
            "Установите: pip install auto-gptq optimum\n\n"
            "auto-gptq совместим с transformers==4.57.1."
        )

    # Калибровочный датасет — небольшой набор текстов для определения
    # масштабов квантования. Используем стандартный c4/wikitext.
    from datasets import load_dataset

    print("  Загружаем калибровочный датасет (wikitext-2)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calibration_texts = [
        t for t in dataset["text"] if len(t.strip()) > 50
    ][:512]

    print(f"  Загружаем merged модель: {merged_dir}")
    processor = AutoProcessor.from_pretrained(
        str(merged_dir),
        trust_remote_code=True,
    )
    tokenizer = processor.tokenizer

    quantize_config = BaseQuantizeConfig(
        bits=GPTQ_CONFIG["bits"],
        group_size=GPTQ_CONFIG["group_size"],
        desc_act=GPTQ_CONFIG["desc_act"],
        model_file_base_name="model",
    )

    # auto-gptq загружает модель как CausalLM.
    # Visual encoder пропускается через modules_to_not_convert.
    model = AutoGPTQForCausalLM.from_pretrained(
        str(merged_dir),
        quantize_config=quantize_config,
        device_map="cuda",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        modules_to_not_convert=GPTQ_CONFIG["modules_to_not_convert"],
    )

    print(
        f"  GPTQ: bits={GPTQ_CONFIG['bits']}, "
        f"group_size={GPTQ_CONFIG['group_size']}, "
        f"desc_act={GPTQ_CONFIG['desc_act']}"
    )
    print("  Токенизируем калибровочные данные...")
    calibration_data = [
        tokenizer(
            text,
            return_tensors="pt",
            max_length=2048,
            truncation=True,
        )
        for text in calibration_texts
    ]

    print("  Запускаем калибровку и квантование (~5-15 мин)...")
    model.quantize(calibration_data)

    GPTQ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Сохраняем GPTQ модель -> {GPTQ_OUTPUT_DIR}")
    model.save_quantized(str(GPTQ_OUTPUT_DIR), use_safetensors=True)
    processor.save_pretrained(str(GPTQ_OUTPUT_DIR))

    del model
    torch.cuda.empty_cache()

    print(f"  OK: GPTQ модель сохранена: {GPTQ_OUTPUT_DIR}")
    return GPTQ_OUTPUT_DIR


def step3_check_vision_weights(gptq_dir: Path):
    """
    Шаг 3: Проверяем что visual веса присутствуют в GPTQ директории.

    auto-gptq может не сохранить visual веса если модель загружалась
    как CausalLM. Выводим предупреждение если visual весов нет.
    """
    print("\n" + "=" * 60)
    print("Шаг 3: Проверка vision encoder весов")
    print("=" * 60)

    gptq_shards = list(gptq_dir.glob("*.safetensors"))
    if not gptq_shards:
        print("  WARN: GPTQ директория пуста — пропускаем")
        return

    index_file = gptq_dir / "model.safetensors.index.json"
    if not index_file.exists():
        print("  Один shard — visual веса внутри, пропускаем")
        return

    with open(index_file) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    visual_keys = [k for k in weight_map if k.startswith("visual.")]

    if visual_keys:
        print(
            f"  OK: Visual веса присутствуют "
            f"({len(visual_keys)} тензоров в fp16)"
        )
    else:
        print("  WARN: Visual веса отсутствуют в GPTQ модели!")
        print(
            "  auto-gptq не сохранил vision encoder.\n"
            "  Нужно скопировать visual веса вручную из merged fp16 модели."
        )


def print_summary(gptq_dir: Path):
    """Выводит итоговую информацию и команды для запуска."""
    print("\n" + "=" * 60)
    print("ГОТОВО")
    print("=" * 60)

    total_size = sum(
        f.stat().st_size for f in gptq_dir.rglob("*") if f.is_file()
    )
    print(f"  Путь:   {gptq_dir}")
    print(f"  Размер: {total_size / 1e9:.2f} GB")

    cmd = (
        "python -m sglang.launch_server \\\n"
        f"  --model-path {gptq_dir} \\\n"
        "  --served-model-name qwen2-vl-2b-r16 \\\n"
        "  --quantization gptq \\\n"
        "  --dtype float16 \\\n"
        "  --port 30000 \\\n"
        "  --host 0.0.0.0 \\\n"
        "  --mem-fraction-static 0.85 \\\n"
        "  --max-total-tokens 4096 \\\n"
        "  --attention-backend triton"
    )
    print("\nЗапуск SGLang:")
    print(cmd)

    dc_cmd = (
        "command: >\n"
        "  python -m sglang.launch_server\n"
        "    --model-path /models/qwen2-vl-2b-r16-gptq\n"
        "    --served-model-name qwen2-vl-2b-r16\n"
        "    --quantization gptq\n"
        "    --dtype float16\n"
        "    --port 30000\n"
        "    --host 0.0.0.0\n"
        "    --mem-fraction-static 0.85\n"
        "    --max-total-tokens 4096\n"
        "    --attention-backend triton"
    )
    print("\ndocker-compose.yml:")
    print(dc_cmd)


def main():
    print("=" * 60)
    print("Qwen2-VL: LoRA merge + GPTQ int4 квантование")
    print("Vision encoder остаётся в fp16")
    print("=" * 60)
    print(f"  Base модель:  {BASE_MODEL_PATH}")
    print(f"  LoRA:         {LORA_CHECKPOINT_PATH}")
    print(f"  Merged fp16:  {MERGED_FP16_DIR}")
    print(f"  GPTQ output:  {GPTQ_OUTPUT_DIR}")

    merged_dir = step1_merge_lora()
    gptq_dir = step2_quantize_gptq(merged_dir)
    step3_check_vision_weights(gptq_dir)
    print_summary(gptq_dir)


if __name__ == "__main__":
    main()
