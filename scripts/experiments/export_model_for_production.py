"""
Экспорт Qwen2-VL модели с LoRA адаптером в формат GGUF.
ИСПОЛЬЗУЕТ NLLAMA.CPP - единственный рабочий путь для Qwen2-VL.
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
import torch

BASE_MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct"
LORA_CHECKPOINT_PATH = "/home/jupyter/s3/ai-tour-guide/qwen2vl-rerank-lora/rerank_exp_r16_alpha32_lr2e-5_rerank_full_lora_448"
OUTPUT_BASE = Path("/home/jupyter/s3/ai-tour-guide/models/gguf/qwen2-vl-2b-r16")
LLAMA_CPP_PATH = Path("/home/jupyter/project/llama.cpp")
PYTHON_EXECUTABLE = sys.executable


def find_convert_script():
    """Находит правильный скрипт конвертации."""
    
    possible_paths = [
        LLAMA_CPP_PATH / "convert-hf-to-gguf.py",
        LLAMA_CPP_PATH / "examples" / "qwen2" / "export_qwen2_vl_mmproj.py",
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"✅ Найден: {path}")
            return path
    
    # Если ничего нет — ищем где угодно
    for p in LLAMA_CPP_PATH.rglob("convert*.py"):
        if "hf" in str(p).lower() or "gguf" in str(p).lower():
            return p
    
    raise FileNotFoundError(
        "Скрипт конвертации не найден!\n"
        "Убедитесь что llama.cpp клонирован корректно:"
        "git clone --recurse-submodules https://github.com/ggerganov/llama.cpp"
    )


def merge_lora(temp_dir: Path, need: bool = False):
    """Сливает LoRA адаптер."""
    
    print("\n📦 Слияние LoRA...")
    
    base_model = AutoModelForVision2Seq.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    if need:
        model_with_lora = PeftModel.from_pretrained(
            base_model,
            LORA_CHECKPOINT_PATH,
            is_trainable=False,
        )

        merged = model_with_lora.merge_and_unload()
    
        merged.save_pretrained(str(temp_dir), safe_serialization=True)
        
    else:
        base_model.save_pretrained(str(temp_dir), safe_serialization=True)
    
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    processor.save_pretrained(str(temp_dir))
    
    return temp_dir


def export_llm(model_dir: Path, output_path: Path):
    """Конвертирует основную модель (LLM) в GGUF."""
    
    print("\n📦 Экспорт основной модели (LLM)...")
    
    convert_script = find_convert_script()
    
    cmd = [
        PYTHON_EXECUTABLE, str(convert_script),
        str(model_dir),
        "--outfile", str(output_path),
        "--outtype", "f16",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    
    if result.returncode != 0:
        raise RuntimeError(f"Ошибка LLM:\n{result.stderr[:500]}")
    
    return output_path


def create_mmproj_simple(model_dir: Path, output_path: Path):
    """
    Создает mmproj через convert-hf-to-gguf.py.
    Это работает для большинства VLM моделей!
    """
    
    print("\n🖼️  Создание mmproj...")
    
    try:
        convert_script = find_convert_script()
        
        cmd = [
            PYTHON_EXECUTABLE, str(convert_script),
            str(model_dir),
            "--outfile", str(output_path),
            "--outtype", "f16",
            "--mmproj",  # Ключевой флаг для проектора
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        
        if result.returncode == 0 and output_path.exists():
            size_mb = round(output_path.stat().st_size / 1e6, 1)
            print(f"✅ mmproj создан: {output_path.name} ({size_mb} MB)")
            return True
        
        else:
            print(f"⚠️  Стандартный конвертер не сработал")
            print(f"   stderr: {result.stderr[:200][:200]}")
            
    except Exception as e:
        print(f"⚠️  Ошибка создания mmproj: {e}")
    
    return False


def quantize(model_f16: Path, model_q5: Path):
    """Квантует GGUF файл."""
    
    if not model_f16.exists():
        return None
    
    quantize_bin = LLAMA_CPP_PATH / "build" / "bin" / "llama-quantize"
    
    if not quantize_bin.exists():
        print("❌ llama-quantize не найден!")
        return None
    
    cmd = [str(quantize_bin), str(model_f16), str(model_q5), "Q5_K_M"]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    
    if result.returncode == 0 and model_q5.exists():
        return model_q5
    
    return None


def main():
    """Главная функция экспорта."""
    
    print("="*70)
    print("🎯 ЭКСПОРТ FINAL FIXED (с mmproj)")
    print("="*70)
    
    temp_dir = OUTPUT_BASE / "temp_merged"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    
    # 1. Merge LoRA
    print("\n🔄 Шаг 1: Слияние LoRA...")
    merged_dir = merge_lora(temp_dir, need=True)  # need=True — слить LoRA в веса
    print(f"✅ Модель с LoRA готова")
    
    # 2. Export LLM
    print("\n📦 Шаг 2: Экспорт LLM...")
    gguf_main_f16 = OUTPUT_BASE / "model-f16.gguf"
    gguf_main_q5  = OUTPUT_BASE / "model-q5_k_m.gguf"

    export_llm(merged_dir, gguf_main_f16)

    # 3. Quantize
    quantized = quantize(gguf_main_f16, gguf_main_q5)
    if quantized:
        gguf_main_q5 = quantized
        print(f"✅ Квантизация завершена: {gguf_main_q5}")
    else:
        print("⚠️  Q5_K_M пропущен, использую f16 модель")
        gguf_main_q5 = gguf_main_f16
    
    # 3. Create mmproj
    print("\n🖼️  Шаг 3: Создание mmproj...")
    mmproj_f16 = OUTPUT_BASE / "mmproj-model-f16.gguf"
    
    created = create_mmproj_simple(merged_dir, mmproj_f16)
    
    if not created:
        print("⚠️  MMProj не создан — модель будет работать ТОЛЬКО с текстом!")
    
    # 4. Save metadata
    print("\n💾 Шаг 4: Метаданные...")
    
    with open(OUTPUT_BASE / "model_card.json", 'w', encoding='utf-8') as f:
        json.dump({
            "model": str(gguf_main_q5.name),
            "mmproj": str(mmproj_f16.name) if mmproj_f16.exists() else "не создан",
        }, f, indent=2)
    
    # Итог
    print("\n" + "="*70)
    print("📊 ФИНАЛ")
    print("="*70)
    print(f"✅ Модель: {gguf_main_q5.name}")
    if mmproj_f16.exists():
        print(f"✅ mmproj: {mmproj_f16.name}")
    else:
        print(f"⚠️  mmproj: НЕ СОЗДАН (работаем без изображений)")
    
    print(f"\n📁 Файлы в: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()