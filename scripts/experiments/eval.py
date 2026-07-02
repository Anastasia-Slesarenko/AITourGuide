# -*- coding: utf-8 -*-
"""
Оценка retrieval, zero-shot VLM и LoRA-адаптированного VLM reranking на данных из step6.

Оценивает:
1. Baseline: argmax(retrieval_score) → Hit@1, MRR
2. Zero-shot VLM: Qwen2-VL-2B-Instruct как reranker (без LoRA) - опционально
3. LoRA VLM: Qwen2-VL-2B-Instruct + LoRA reranker
4. None-of-the-above: корректная обработка target_idx = -1

Запуск:
    python experiments/eval.py
    (параметры задаются в блоке __main__ внизу файла)
"""
import gc
import warnings
import time

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="peft")

import os
import json
import torch
import numpy as np
import mlflow
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_recall_curve, roc_curve
)
import matplotlib
matplotlib.use('Agg')  # Неинтерактивный бэкенд (без GUI)
import matplotlib.pyplot as plt


# Модульный счётчик для профилировщика compute_rerank_scores_batch.
# Используем переменную модуля вместо атрибута функции, чтобы избежать
# mypy-ошибки "Callable has no attribute '_prof_count'".
_RERANK_PROF_COUNT: int = 0


def load_qwen2vl_processor(
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    lora_path: Optional[str] = None,
):
    """Загружает модель и процессор для VLM reranking (с LoRA или без)."""
    # PERF: T4 имеет нативные tensor cores для FP16, но не для BF16.
    # BF16 на T4 эмулируется через FP32 → в ~2x медленнее FP16.
    # Используем FP16 для inference (качество идентично BF16 для reranking).
    _dtype = torch.float16

    # PERF: attn_implementation="sdpa" использует PyTorch scaled_dot_product_attention
    # с memory-efficient attention — быстрее стандартного eager на T4.
    # Если установлен flash-attn, используем flash_attention_2 (ещё быстрее).
    _attn_impl = "sdpa"
    try:
        import flash_attn  # noqa: F401
        _attn_impl = "flash_attention_2"
        print("✓ flash_attention_2 доступен, используем его")
    except ImportError:
        print("ℹ flash_attn не установлен, используем sdpa")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=_dtype,
        attn_implementation=_attn_impl,
        device_map="auto",
    )

    if lora_path:
        print(f"Loading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

    # Используем AutoProcessor как в train.py — функционально эквивалентно
    # Qwen2VLProcessor, но унифицирует загрузку между train и eval.
    # Процессор всегда загружается из базовой модели model_id,
    # а не из lora_path. LoRA-адаптер не содержит токенизатор/процессор.
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

    # FIX batch_size>1: decoder-only модели требуют left-padding при батчинге,
    # чтобы логиты на позиции -1 соответствовали последнему токену промпта,
    # а не pad-токену. Устанавливаем один раз после загрузки процессора.
    processor.tokenizer.padding_side = "left"

    # FIX #8: явно переводим в eval + отключаем градиенты для всех параметров,
    # чтобы inference_mode работал корректно после PeftModel.from_pretrained
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, processor


def compute_rerank_scores_batch(
    model,
    processor,
    query_img_path: str,
    candidates: List[dict],
    image_base_dir: str,
    caption_max_length: int = 300,
    batch_size: int = 1,
    # FIX batch_size>1: Qwen2-VL использует динамическое разбиение на патчи —
    # количество патчей зависит от разрешения изображения. pixel_values —
    # плоский тензор [total_patches, C, H, W]. При разных размерах изображений
    # маппинг <image> токенов к патчам нарушается.
    # Решение: принудительный ресайз всех изображений до фиксированного
    # разрешения гарантирует одинаковое количество патчей для всех примеров
    # в батче. None = без ресайза (только batch_size=1 безопасен).
    fixed_image_size: Optional[tuple] = (448, 448),
) -> tuple:
    """Вычисляет scores для батча кандидатов с настоящим батчингом.

    FIXED: Оптимизирован для избежания дублирования query image и race conditions.
    FIXED batch_size>1: принудительный ресайз до fixed_image_size обеспечивает
    одинаковое количество патчей для всех примеров в батче.

    Args:
        model: VLM модель
        processor: Процессор модели
        query_img_path: Путь к query изображению
        candidates: Список кандидатов
        image_base_dir: Базовая директория
        caption_max_length: Максимальная длина caption
        batch_size: Размер батча для обработки
        fixed_image_size: Фиксированный размер (W, H) для ресайза изображений.
            Обязателен при batch_size>1 для корректного маппинга патчей.
            None отключает ресайз (только batch_size=1 безопасен).

    Returns:
        tuple: (scores: List[float], failed_indices: List[int])
            scores — скоры для каждого кандидата (0.0 для незагрузившихся)
            failed_indices — индексы кандидатов, которые не удалось загрузить
    """
    _t0_query_io = time.time()
    try:
        # FIX #18: используем контекстный менеджер, чтобы закрыть файловый
        # дескриптор сразу после convert() — иначе при тысячах изображений
        # возникает OSError: [Errno 24] Too many open files
        with Image.open(
            os.path.join(image_base_dir, query_img_path)
        ) as _img:
            query_image = _img.convert("RGB")
        # FIX batch_size>1: ресайз до фиксированного разрешения гарантирует
        # одинаковое количество патчей для всех примеров в батче.
        # BILINEAR — как в train.py (быстрее LANCZOS, качество достаточно).
        if fixed_image_size is not None:
            _resized = query_image.resize(
                fixed_image_size, Image.Resampling.BILINEAR
            )
            query_image.close()
            query_image = _resized
    except Exception as e:
        print(f"⚠ Ошибка загрузки query-изображения {query_img_path}: {e}")
        # Возвращаем пустые скоры и все индексы как failed
        return (
            [0.0] * len(candidates),
            list(range(len(candidates))),
        )
    _t_query_io = time.time() - _t0_query_io

    all_scores = []
    # FIX #22: глобальный счётчик незагрузившихся кандидатов для всего запроса
    all_failed_indices: List[int] = []

    # Кэшируем yes/no token ids один раз вне цикла батчей.
    # FIX #9: processor.tokenizer может быть недоступен в некоторых версиях
    # transformers — используем getattr с fallback на processor напрямую.
    # FIX #16: используем convert_tokens_to_ids вместо encode("Yes")[0],
    # так как encode может вернуть несколько токенов (BPE-сплит) или
    # зависеть от контекста (добавление пробела перед токеном).
    # convert_tokens_to_ids("Yes") возвращает ровно один id без контекста.
    _tokenizer = getattr(processor, "tokenizer", processor)
    yes_id = _tokenizer.convert_tokens_to_ids("Yes")
    no_id = _tokenizer.convert_tokens_to_ids("No")
    # Проверяем, что токены найдены (не UNK)
    _unk_id = getattr(_tokenizer, "unk_token_id", None)
    if yes_id == _unk_id or no_id == _unk_id:
        # Fallback: encode с явным отключением special tokens
        yes_id = _tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_id = _tokenizer.encode("No", add_special_tokens=False)[0]

    # PROF: счётчики времени по фазам
    _prof_io = 0.0       # чтение изображений кандидатов
    _prof_template = 0.0 # apply_chat_template
    _prof_processor = 0.0 # processor(text, images)
    _prof_forward = 0.0  # model forward pass
    _prof_transfer = 0.0 # .to(model.device)

    # Обрабатываем кандидатов батчами
    for batch_start in range(0, len(candidates), batch_size):
        batch_end = min(batch_start + batch_size, len(candidates))
        batch_candidates = candidates[batch_start:batch_end]

        batch_texts = []
        # FIX #2: изображения передаются как список пар [[q, c1], [q, c2], ...]
        # а не плоским списком — иначе процессор неправильно маппит <image> токены
        batch_images_grouped = []
        # FIX O(n²): используем set для O(1) lookup вместо O(n) list
        batch_valid_indices: set = set()

        for idx, cand in enumerate(batch_candidates):
            global_idx = batch_start + idx  # глобальный индекс кандидата
            _t0 = time.time()
            try:
                # FIX #18: аналогично закрываем дескриптор кандидата
                with Image.open(
                    os.path.join(image_base_dir, cand["image"])
                ) as _img:
                    cand_image = _img.convert("RGB")
                # FIX batch_size>1: аналогичный ресайз для кандидата
                # BILINEAR — как в train.py (быстрее LANCZOS, качество достаточно).
                # FIX PIL leak: закрываем исходный объект после ресайза.
                if fixed_image_size is not None:
                    _resized_c = cand_image.resize(
                        fixed_image_size, Image.Resampling.BILINEAR
                    )
                    cand_image.close()
                    cand_image = _resized_c
            except Exception as e:
                print(
                    f"⚠ Error loading candidate image "
                    f"{cand.get('image', 'unknown')} "
                    f"(global_idx={global_idx}): {e}"
                )
                # FIX #22: фиксируем глобальный индекс незагрузившегося кандидата
                all_failed_indices.append(global_idx)
                continue
            _prof_io += time.time() - _t0

            caption = cand["caption"][:caption_max_length]

            # FIX #7: в transformers==4.57.1 apply_chat_template для Qwen2-VL
            # не принимает PIL-объекты в поле "image" внутри messages.
            # PIL передаётся отдельно через processor(images=...).
            # В messages используем {"type": "image"} без поля "image".
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Query Photo:"},
                        {"type": "image"},
                        {"type": "text", "text": "Candidate Photo:"},
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": (
                                f"Question: Are these photos showing the same "
                                f"landmark: \"{cand['name']}\"?\n"
                                f"Candidate details: {caption}\n"
                                f"Answer only with Yes or No."
                            ),
                        },
                    ],
                }
            ]

            # Применяем chat template
            _t0 = time.time()
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            _prof_template += time.time() - _t0

            batch_texts.append(text)
            # FIX #8: processor(images=...) принимает плоский список PIL-объектов,
            # а не список пар. Каждый пример имеет 2 изображения: [query, cand].
            # FIX #17: оба изображения копируются, чтобы избежать in-place мутации
            # процессором (resize/normalize) при batch_size>1.
            # FIX PIL leak: cand_image закрывается после добавления копии в батч.
            batch_images_grouped.append(
                [query_image.copy(), cand_image.copy()]
            )
            cand_image.close()
            batch_valid_indices.add(idx)

        # Все кандидаты в батче не загрузились
        if not batch_texts:
            all_scores.extend([0.0] * len(batch_candidates))
            continue

        # FIX #8: processor принимает images как плоский список PIL-объектов.
        # batch_images_grouped = [[q1,c1], [q2,c2], ...] → разворачиваем в
        # [q1, c1, q2, c2, ...]. Количество изображений = 2 * len(batch_texts),
        # что совпадает с количеством <image> токенов в batch_texts.
        #
        # FIX batch_size>1: при fixed_image_size все изображения имеют
        # одинаковое разрешение → одинаковое количество патчей → корректный
        # маппинг <image> токенов к патчам в плоском pixel_values тензоре.
        # transformers>=4.47 корректно пересчитывает position_ids при padding.
        flat_images = [img for pair in batch_images_grouped for img in pair]

        _t0 = time.time()
        if len(batch_texts) == 1:
            inputs_cpu = processor(
                text=batch_texts,
                images=flat_images,
                return_tensors="pt",
            )
        else:
            # Используем встроенный padding процессора.
            # Корректно работает при transformers>=4.47 + fixed_image_size.
            inputs_cpu = processor(
                text=batch_texts,
                images=flat_images,
                return_tensors="pt",
                padding=True,
            )
        _prof_processor += time.time() - _t0

        _t0 = time.time()
        inputs = inputs_cpu.to(model.device)
        _prof_transfer += time.time() - _t0

        # PERF: используем прямой forward pass вместо generate(max_new_tokens=1).
        # generate() добавляет overhead планировщика (~5–15 мс на вызов):
        # инициализацию LogitsProcessorList, StoppingCriteriaList, beam search
        # структур — всё это лишнее при нужде только в логитах позиции [-1].
        # model(**inputs) возвращает сырые логиты последнего токена промпта
        # напрямую, что эквивалентно первому сгенерированному токену при
        # greedy decoding (авторегрессия: P(next|prompt) = softmax(logits[-1])).
        # При batch_size>1 + left-padding логит позиции [-1] соответствует
        # последнему реальному токену каждого примера (не pad-токену).
        _t0 = time.time()
        with torch.inference_mode():
            outputs = model(**inputs)
            # outputs.logits: [B, seq_len, V] — берём последнюю позицию
            logits = outputs.logits[:, -1, :]  # [B, V]

            logit_yes = logits[:, yes_id]  # [B]
            logit_no = logits[:, no_id]    # [B]

            logits_binary = torch.stack([logit_no, logit_yes], dim=1)  # [B, 2]
            probs = torch.softmax(logits_binary, dim=1)                 # [B, 2]
            probs_yes = probs[:, 1]                                     # [B]

            batch_scores = probs_yes.cpu().tolist()
        _prof_forward += time.time() - _t0

        # Освобождаем GPU-память: del тензоров достаточно при 34% VRAM.
        # empty_cache() убран из цикла — он вызывает CUDA sync после каждого
        # батча и нивелирует выигрыш от батчинга. Вызывается только при OOM.
        del (
            inputs, inputs_cpu, outputs, logits, logit_yes, logit_no,
            logits_binary, probs, probs_yes,
        )

        # Маппим скоры обратно на исходные индексы кандидатов
        score_idx = 0
        for idx in range(len(batch_candidates)):
            if idx in batch_valid_indices:
                all_scores.append(batch_scores[score_idx])
                score_idx += 1
            else:
                all_scores.append(0.0)  # кандидат не загрузился

    # PROF: выводим breakdown времени для первых 3 сэмплов.
    # FIX mypy: используем модульную переменную _RERANK_PROF_COUNT
    # вместо атрибута функции (mypy не поддерживает атрибуты Callable).
    global _RERANK_PROF_COUNT
    _RERANK_PROF_COUNT += 1
    if _RERANK_PROF_COUNT <= 3:
        total = _t_query_io + _prof_io + _prof_template + _prof_processor + _prof_transfer + _prof_forward
        print(
            f"\n[PROF sample #{_RERANK_PROF_COUNT}] "
            f"total={total:.2f}s | "
            f"query_io={_t_query_io:.2f}s | "
            f"cand_io={_prof_io:.2f}s | "
            f"template={_prof_template:.2f}s | "
            f"processor={_prof_processor:.2f}s | "
            f"transfer={_prof_transfer:.2f}s | "
            f"forward={_prof_forward:.2f}s"
        )

    return all_scores, all_failed_indices


def calculate_confidence_interval(
    values: List[float],
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> tuple:
    """Вычисляет confidence interval для метрики используя bootstrap.

    Args:
        values: Список значений метрики для каждого сэмпла
        confidence: Уровень доверия (по умолчанию 0.95)
        seed: Random seed для воспроизводимости bootstrap (None = случайный)

    Returns:
        tuple: (mean, lower_bound, upper_bound)
    """
    if not values:
        return 0.0, 0.0, 0.0

    values_array = np.array(values)
    mean = np.mean(values_array)

    # FIX #9: используем локальный Generator вместо глобального np.random,
    # чтобы bootstrap CI был воспроизводим независимо от внешнего состояния
    rng = np.random.default_rng(seed)

    n_bootstrap = 1000
    bootstrap_means = []

    for _ in range(n_bootstrap):
        bootstrap_sample = rng.choice(
            values_array, size=len(values_array), replace=True
        )
        bootstrap_means.append(np.mean(bootstrap_sample))

    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)

    return float(mean), float(lower_bound), float(upper_bound)


def evaluate_retrieval_only(
    samples: List[Dict],
    k_values=[1, 3, 5],
    compute_ci: bool = False,
    ci_seed: Optional[int] = None,
) -> Dict[str, float]:
    """Оценивает baseline: argmax(retrieval_score).
    
    Args:
        samples: Список сэмплов
        k_values: Значения K для метрик
        compute_ci: Вычислять ли confidence intervals (медленнее)
    
    Returns:
        Словарь с метриками
    """
    hits_at_k = {k: 0 for k in k_values}
    precision_at_k = {k: [] for k in k_values}  # NEW: Precision@K
    mrr_sum = 0.0
    mrr_values = []  # NEW: Для confidence interval
    total_valid = 0  # для которых target_idx != -1
    total_samples = len(samples)
    _ci_seed = ci_seed  # локальная переменная для передачи в CI
    
    for sample in samples:
        target_idx = sample["target_idx"]
        
        # Сортируем кандидатов по retrieval_score
        candidates = sample["candidates"]
        scores = [c["retrieval_score"] for c in candidates]
        ranked_indices = np.argsort(scores)[::-1]  # descending
        
        # === METRICS FOR TARGET EXISTENCE ===
        if target_idx != -1:
            # HIT@K / RECALL@K: находится ли позитив в топ-K
            for k in k_values:
                if target_idx in ranked_indices[:k]:
                    hits_at_k[k] += 1
                    # FIX #14: стандартная Precision@K = 1/K при одном
                    # релевантном документе (target в топ-K → 1/K, иначе 0).
                    precision_at_k[k].append(1.0 / k)
                else:
                    precision_at_k[k].append(0.0)
            
            # MRR: 1 / (rank_of_positive + 1)
            rank = np.where(ranked_indices == target_idx)[0][0] + 1
            reciprocal_rank = 1.0 / rank
            mrr_sum += reciprocal_rank
            mrr_values.append(reciprocal_rank)  # NEW: Для CI
            total_valid += 1
        else:
            # Для none-of-the-above: считаем successful, если top-K не содержит позитива
            # (т.е. модель не выбрала ложный позитив)
            # Но для baseline это всегда fail, т.к. argmax != -1
            pass
    
    # === CALCULATE METRICS ===
    # Примечание: для single-label задачи (один релевантный документ на запрос)
    # Hit@K == Recall@K. Оба ключа сохраняются для совместимости с внешними
    # системами, но вычисляются из одного значения.
    retrieval_metrics = {}
    for k in k_values:
        hit_rate = hits_at_k[k] / total_valid if total_valid > 0 else 0.0
        # Hit@K и Recall@K идентичны при одном релевантном документе на запрос
        retrieval_metrics[f"retrieval_hit_{k}"] = hit_rate
        retrieval_metrics[f"retrieval_recall_{k}"] = hit_rate

        # Precision@K
        if precision_at_k[k]:
            retrieval_metrics[f"retrieval_precision_{k}"] = float(
                np.mean(precision_at_k[k])
            )
        else:
            retrieval_metrics[f"retrieval_precision_{k}"] = 0.0

        # Confidence intervals для Hit@K
        if compute_ci and total_valid > 0:
            hit_values_ci = []
            for s in samples:
                s_target = s["target_idx"]
                if s_target == -1:
                    continue
                s_scores = [c["retrieval_score"] for c in s["candidates"]]
                s_ranked = np.argsort(s_scores)[::-1]
                hit_values_ci.append(
                    1.0 if s_target in s_ranked[:k] else 0.0
                )
            # FIX #20: передаём ci_seed для воспроизводимости bootstrap
            _, lower, upper = calculate_confidence_interval(
                hit_values_ci, seed=_ci_seed
            )
            retrieval_metrics[f"retrieval_hit_{k}_ci_lower"] = lower
            retrieval_metrics[f"retrieval_hit_{k}_ci_upper"] = upper
    
    # MRR
    retrieval_metrics["retrieval_mrr"] = (
        mrr_sum / total_valid if total_valid > 0 else 0.0
    )
    
    # Доверительный интервал для MRR
    if compute_ci and mrr_values:
        # FIX #20: передаём ci_seed для воспроизводимости bootstrap
        _, lower, upper = calculate_confidence_interval(
            mrr_values, seed=_ci_seed
        )
        retrieval_metrics["retrieval_mrr_ci_lower"] = lower
        retrieval_metrics["retrieval_mrr_ci_upper"] = upper
    
    retrieval_metrics["retrieval_total_samples"] = total_samples
    retrieval_metrics["retrieval_valid_samples"] = total_valid
    retrieval_metrics["retrieval_none_samples"] = total_samples - total_valid
    
    return retrieval_metrics


def calculate_mrr(
    samples: List[Dict], scores_list: List[List[float]]
) -> float:
    """Вычисляет Mean Reciprocal Rank (MRR).

    Примечание: для задачи с одним релевантным документом на запрос
    MRR == MAP. Функция переименована из calculate_map для ясности.

    Args:
        samples: List of samples with target_idx
        scores_list: Precomputed scores for each sample

    Returns:
        MRR score
    """
    reciprocal_ranks = []

    for sample, scores in zip(samples, scores_list):
        target_idx = sample["target_idx"]

        if target_idx == -1:
            continue  # Пропускаем сэмплы "ни один из вариантов"

        # Ранжируем кандидатов по скорам
        ranked_indices = np.argsort(scores)[::-1]

        # Находим позицию целевого кандидата
        rank = np.where(ranked_indices == target_idx)[0][0] + 1

        reciprocal_ranks.append(1.0 / rank)

    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def calculate_ndcg(
    samples: List[Dict],
    scores_list: List[List[float]],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """Calculate Normalized Discounted Cumulative Gain (nDCG@K).
    
    Args:
        samples: List of samples with target_idx
        scores_list: Precomputed scores for each sample
        k_values: K values for nDCG@K
    
    Returns:
        Dict with nDCG@K for each K
    """
    ndcg_scores = {k: [] for k in k_values}
    
    for sample, scores in zip(samples, scores_list):
        target_idx = sample["target_idx"]
        
        if target_idx == -1:
            continue  # Пропускаем сэмплы "ни один из вариантов"

        # Создаём массив релевантности (1 для цели, 0 для остальных)
        relevance = np.array([
            1.0 if i == target_idx else 0.0
            for i in range(len(scores))
        ])
        
        # Ранжируем по скорам
        ranked_indices = np.argsort(scores)[::-1]
        ranked_relevance = relevance[ranked_indices]

        # Вычисляем DCG и IDCG для каждого K
        for k in k_values:
            if k > len(scores):
                k_actual = len(scores)
            else:
                k_actual = k
            
            # DCG@K
            dcg = 0.0
            for i in range(k_actual):
                dcg += ranked_relevance[i] / np.log2(i + 2)
            
            # IDCG@K (идеальный DCG — цель на позиции 1)
            idcg = 1.0 / np.log2(2)  # Только один релевантный документ

            # nDCG@K
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores[k].append(ndcg)
    
    # Среднее nDCG@K по всем сэмплам
    result = {}
    for k in k_values:
        if ndcg_scores[k]:
            result[f"ndcg_{k}"] = float(np.mean(ndcg_scores[k]))
        else:
            result[f"ndcg_{k}"] = 0.0
    
    return result


# Веса компонентов confidence score для unknown detection.
# max_score: вероятность лучшего кандидата (VLM prob)
# low_entropy: 1 - нормализованная энтропия (пиковое распределение = уверенность)
# margin: разность между топ-1 и топ-2 скорами (чёткий победитель = уверенность)
_CONF_WEIGHT_MAX_SCORE: float = 0.5
_CONF_WEIGHT_LOW_ENTROPY: float = 0.3
_CONF_WEIGHT_MARGIN: float = 0.2
_CONF_ENTROPY_EPSILON: float = 1e-10


def _compute_confidence_score(scores: List[float]) -> float:
    """Вычисляет confidence score для unknown detection.

    FIX #13: использует сырые P(Yes) без нормализации по сумме.
    Нормализация по сумме искажает смысл: если все кандидаты имеют
    низкий P(Yes) (unknown), нормализованные значения всё равно
    дадут высокий max_score. Сырые P(Yes) корректно отражают
    абсолютную уверенность модели.

    Компоненты:
    - max_score: максимальный P(Yes) среди кандидатов (абсолютная уверенность)
    - low_entropy: 1 - нормализованная энтропия (пиковое распределение = уверенность)
    - margin: разность топ-1 и топ-2 скоров (чёткий победитель = уверенность)

    Args:
        scores: Список VLM scores P(Yes) для кандидатов [0, 1]

    Returns:
        Confidence score [0, 1] - высокий = confident it's known
    """
    if not scores:
        return 0.0

    scores_array = np.array(scores, dtype=float)

    # FIX #13: используем сырые P(Yes) напрямую.
    # max(P(Yes)) — абсолютная уверенность в лучшем кандидате.
    max_score = float(np.max(scores_array))

    # Margin между топ-1 и топ-2 на сырых скорах
    if len(scores_array) >= 2:
        sorted_scores = np.sort(scores_array)[::-1]
        margin = float(sorted_scores[0] - sorted_scores[1])
    else:
        margin = max_score

    # Нормализованная энтропия: H = -sum(p * log(p+eps)) / log(n)
    # low_entropy = 1 - H: высокое значение = уверенное (пиковое) распределение.
    # Клипируем scores в (eps, 1-eps) чтобы избежать log(0).
    n = len(scores_array)
    if n > 1:
        p = np.clip(scores_array, _CONF_ENTROPY_EPSILON, 1.0 - _CONF_ENTROPY_EPSILON)
        entropy = -float(np.sum(p * np.log(p))) / np.log(n)
        low_entropy = float(np.clip(1.0 - entropy, 0.0, 1.0))
    else:
        low_entropy = max_score

    # Взвешенная комбинация трёх компонентов
    confidence = (
        _CONF_WEIGHT_MAX_SCORE * max_score
        + _CONF_WEIGHT_LOW_ENTROPY * low_entropy
        + _CONF_WEIGHT_MARGIN * margin
    )

    return float(np.clip(confidence, 0.0, 1.0))


def calculate_additional_metrics_for_vlm_rerank(
    samples: List[Dict],
    precomputed_scores: List[List[float]],
    model_name: str = "vlm"
) -> Dict[str, float]:
    """Вычисляет AUROC, F1, FPR@95TPR, ECE, Median rank для VLM rerank.
    
    FIXED: Улучшена метрика unknown detection с использованием entropy.
    
    Args:
        samples: Список сэмплов
        precomputed_scores: Предвычисленные VLM скоры для каждого сэмпла
        model_name: Имя модели для префикса метрик
    
    Returns:
        Словарь с дополнительными метриками
    """
    all_probs = []
    all_labels = []
    all_ranks = []
    all_unknown_probs = []
    all_unknown_labels = []

    for sample, vl_scores in zip(samples, precomputed_scores):
        candidates = sample["candidates"]
        target_idx = sample["target_idx"]

        # === РАНЖИРОВАНИЕ ПО VLM СКОРАМ ===
        ranked_indices = np.argsort(vl_scores)[::-1]  # по убыванию

        # === СБОР ДАННЫХ ДЛЯ МЕТРИК ===
        if target_idx != -1:
            # --- Метрики для известных достопримечательностей ---
            # Метка: 1 для правильного кандидата, 0 для остальных
            labels_for_sample = [
                1 if i == target_idx else 0
                for i in range(len(candidates))
            ]
            all_labels.extend(labels_for_sample)
            all_probs.extend(vl_scores)

            # Позиция правильного кандидата в ранжировании
            rank_of_target = np.where(ranked_indices == target_idx)[0][0] + 1
            all_ranks.append(rank_of_target)

            # --- Метрики для задачи unknown (положительный класс = KNOWN) ---
            # Этот сэмпл "известный" → метка = 1
            all_unknown_labels.append(1)
            
            # Используем confidence на основе энтропии (низкая энтропия = высокая уверенность)
            confidence_known = _compute_confidence_score(vl_scores)
            all_unknown_probs.append(confidence_known)
        else:
            # --- Метрики для неизвестных достопримечательностей ---
            # Этот сэмпл "неизвестный" → метка = 0
            all_unknown_labels.append(0)
            
            # Confidence на основе энтропии
            confidence_known = _compute_confidence_score(vl_scores)
            all_unknown_probs.append(confidence_known)



    # --- CALCULATE METRICS ---
    metrics = {}

    # 1. Median Rank for known targets
    if all_ranks:
        metrics[f"{model_name}_median_rank"] = float(np.median(all_ranks))
    else:
        metrics[f"{model_name}_median_rank"] = 0.0

    # 2. MRR (для single-label задачи MRR == MAP)
    metrics[f"{model_name}_mrr_additional"] = calculate_mrr(
        samples, precomputed_scores
    )

    # 3. nDCG@K
    ndcg_metrics = calculate_ndcg(samples, precomputed_scores, k_values=[1, 3, 5, 10])
    for k, v in ndcg_metrics.items():
        metrics[f"{model_name}_{k}"] = v

    # 4. Brier Score — стандартная метрика калибровки для reranking.
    # FIX #18: заменяет PCE на Brier Score (mean squared error между
    # P(Yes) и бинарными метками target/non-target).
    # FIX #КРИТ5: ключ переименован из _ece в _brier_score, чтобы
    # не вводить в заблуждение при анализе результатов в MLflow/JSON.
    # В summary и print-выводе ключ обновлён соответственно.
    metrics[f"{model_name}_brier_score"] = calculate_brier_score(
        samples, precomputed_scores
    )

    # 3. AUROC, F1, FPR@95TPR for Unknown task (is_known vs is_unknown)
    if len(set(all_unknown_labels)) > 1: # Need both classes present
        try:
            metrics[f"{model_name}_unknown_auroc"] = roc_auc_score(all_unknown_labels, all_unknown_probs)
        except ValueError:
            metrics[f"{model_name}_unknown_auroc"] = 0.0 # Handle edge case if no variance

        # F1 Score: Find optimal threshold based on F1
        if len(set(all_unknown_labels)) == 2:
            precisions, recalls, thresholds_pr = precision_recall_curve(
                all_unknown_labels, all_unknown_probs
            )
            # f1_scores имеет len == len(precisions), thresholds_pr — на 1 меньше.
            # FIX #7: clamp индекса до len(thresholds_pr)-1, чтобы избежать
            # IndexError на последней точке кривой (precision=1, recall=0).
            f1_scores = (
                2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            )
            # Ищем оптимум только среди индексов, для которых есть порог
            optimal_idx_f1 = int(np.argmax(f1_scores[:len(thresholds_pr)]))
            optimal_threshold_f1 = float(thresholds_pr[optimal_idx_f1])
            y_pred_f1 = (
                np.array(all_unknown_probs) >= optimal_threshold_f1
            ).astype(int)
            metrics[f"{model_name}_unknown_f1"] = f1_score(
                all_unknown_labels, y_pred_f1
            )

            # FPR@95TPR через интерполяцию по ROC-кривой.
            # roc_curve возвращает (fpr, tpr, thresholds), где tpr
            # монотонно возрастает — корректный порядок для np.interp.
            fpr_vals, tpr_vals, _ = roc_curve(
                all_unknown_labels, all_unknown_probs
            )
            target_tpr = 0.95
            # np.interp(x, xp, fp): xp=tpr_vals (возрастает), fp=fpr_vals
            fpr_at_95tpr = np.interp(target_tpr, tpr_vals, fpr_vals)
            metrics[f"{model_name}_unknown_fpr_at_95tpr"] = float(
                fpr_at_95tpr
            )

        else:
            metrics[f"{model_name}_unknown_f1"] = 0.0
            metrics[f"{model_name}_unknown_fpr_at_95tpr"] = 0.0
    else:
        # AUROC не определён если присутствует только один класс
        metrics[f"{model_name}_unknown_auroc"] = 0.0
        metrics[f"{model_name}_unknown_f1"] = 0.0
        metrics[f"{model_name}_unknown_fpr_at_95tpr"] = 0.0

    return metrics


def calculate_brier_score(
    samples: List[Dict],
    precomputed_scores: List[List[float]],
) -> float:
    """Вычисляет Brier Score для reranking-калибровки.

    FIX #18: заменяет PCE на Brier Score — стандартную метрику калибровки.
    Brier Score = mean((P(Yes) - label)^2) по всем парам (кандидат, запрос).
    Для reranking: label=1 для target, label=0 для остальных.
    Диапазон [0, 1], меньше = лучше. Идеальная калибровка = 0.

    Args:
        samples: Список сэмплов с target_idx
        precomputed_scores: Предвычисленные VLM скоры P(Yes) для каждого сэмпла

    Returns:
        Brier Score (lower is better)
    """
    squared_errors: List[float] = []

    for sample, vl_scores in zip(samples, precomputed_scores):
        target_idx = sample["target_idx"]
        if target_idx == -1 or not vl_scores:
            continue

        for j, score in enumerate(vl_scores):
            label = 1.0 if j == target_idx else 0.0
            squared_errors.append((float(score) - label) ** 2)

    if not squared_errors:
        return 0.0

    return float(np.mean(squared_errors))


def calculate_ece(probs: np.ndarray, labels: np.ndarray,
                  n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (маргинальная калибровка).

    Используется для calibration plot в MLflow. Для основной метрики
    калибровки reranker используйте calculate_pairwise_calibration_error.

    Args:
        probs: Predicted probabilities [0, 1]
        labels: True binary labels {0, 1}
        n_bins: Number of bins for calibration

    Returns:
        ECE score (lower is better)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # Для последнего бина используем <= чтобы включить prob=1.0
        if i == len(bin_lowers) - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)
        
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def create_calibration_plot(
    probs: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    output_dir: str,
    n_bins: int = 10
) -> str:
    """Create and save calibration plot (reliability diagram).
    
    Args:
        probs: Predicted probabilities
        labels: True labels
        model_name: Name for the plot title
        output_dir: Directory to save the plot
        n_bins: Number of bins
    
    Returns:
        Path to saved plot
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        if i == len(bin_lowers) - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)
        
        count = in_bin.sum()
        if count > 0:
            accuracy = labels[in_bin].mean()
            confidence = probs[in_bin].mean()
            bin_accs.append(accuracy)
            bin_confs.append(confidence)
            bin_counts.append(count)
        else:
            bin_accs.append(0)
            bin_confs.append((bin_lower + bin_upper) / 2)
            bin_counts.append(0)
    
    # Создаём график
    fig, ax = plt.subplots(figsize=(8, 8))

    # Линия идеальной калибровки
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)

    # Фактическая калибровка
    ax.plot(bin_confs, bin_accs, 'o-', label=f'{model_name}',
            markersize=8, linewidth=2)
    
    # Столбчатая диаграмма числа сэмплов
    ax2 = ax.twinx()
    ax2.bar(bin_confs, bin_counts, alpha=0.3, width=0.08,
            color='gray', label='Sample count')
    ax2.set_ylabel('Sample count', fontsize=12)
    ax2.legend(loc='upper left')
    
    ax.set_xlabel('Predicted probability', fontsize=12)
    ax.set_ylabel('Actual accuracy', fontsize=12)
    ax.set_title(f'Calibration Plot - {model_name}', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Сохраняем график
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'calibration_{model_name}.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def _truncate_candidates(
    candidates: List[dict],
    target_idx: int,
    max_candidates: int,
) -> tuple:
    """Обрезает список кандидатов до max_candidates, гарантируя наличие позитива.

    Логика:
    - target_idx == -1 (none-of-the-above): берём первые max_candidates,
      target_idx остаётся -1.
    - target_idx < max_candidates: берём первые max_candidates без изменений.
    - target_idx >= max_candidates: заменяем candidates[max_candidates-1] на
      candidates[target_idx], новый target_idx = max_candidates - 1.
      Позитив всегда присутствует в усечённом списке.

    Args:
        candidates: Полный список кандидатов из датасета.
        target_idx: Индекс позитивного кандидата (-1 = none-of-the-above).
        max_candidates: Максимальное число кандидатов для inference.

    Returns:
        tuple: (truncated_candidates, new_target_idx)
    """
    if max_candidates is None or len(candidates) <= max_candidates:
        return candidates, target_idx

    truncated = list(candidates[:max_candidates])

    if target_idx == -1:
        # none-of-the-above: все кандидаты негативы, просто обрезаем
        return truncated, -1

    if target_idx < max_candidates:
        # позитив уже в усечённом списке
        return truncated, target_idx

    # позитив за пределами max_candidates — ставим его на последнее место
    truncated[max_candidates - 1] = candidates[target_idx]
    return truncated, max_candidates - 1


def evaluate_vlm_rerank(
    model,
    processor,
    samples: List[Dict],
    image_base_dir: str,
    k_values=[1, 3, 5],
    threshold_for_none: float = 0.5,
    model_name: str = "vlm",
    caption_max_length: int = 300,
    # batch_size>1 корректно работает при fixed_image_size != None
    # (transformers>=4.47 + фиксированный размер патчей).
    # Увеличивать только при достаточном VRAM (каждый пример = 2 изображения).
    batch_size: int = 1,
    fixed_image_size: Optional[tuple] = (448, 448),
    max_candidates: Optional[int] = None,
    save_predictions: bool = False,
    compute_ci: bool = False,
    ci_seed: Optional[int] = None,
) -> Tuple[
    Dict,
    List[List[float]],
    List[bool],
    Optional[List[Dict]],
    List[Dict],
]:
    """Оценивает VLM reranking (zero-shot или LoRA).

    Args:
        batch_size: Размер батча для inference (по умолчанию 1)
        fixed_image_size: Фиксированный размер (W, H) для ресайза изображений.
            Обязателен при batch_size>1. None отключает ресайз.
        max_candidates: Максимальное число кандидатов на запрос для inference.
            None = использовать все кандидаты из датасета.
            При target_idx != -1 позитив всегда включается в усечённый список.
            При target_idx == -1 берутся первые max_candidates негативов.
        save_predictions: Сохранять ли предсказания для анализа
        compute_ci: Вычислять ли confidence intervals (медленнее)

    Returns:
        tuple: (metrics_dict, precomputed_scores_list, predictions_list)
    """
    hits_at_k = {k: 0 for k in k_values}
    precision_at_k: Dict[int, List[float]] = {k: [] for k in k_values}
    mrr_sum = 0.0
    mrr_values: List[float] = []
    total_valid = 0
    none_correct = 0
    # FIX #СЕР6: отдельный счётчик none-samples, не включающий skipped.
    # total_samples - total_valid включает skipped samples, что занижает
    # none_accuracy, если skipped sample имел target_idx == -1.
    total_none_samples = 0
    total_samples = len(samples)
    latencies: List[float] = []
    all_scores: List[List[float]] = []
    # FIX #КРИТ6: явный флаг is_skipped вместо эвристики по нулевым скорам.
    # Если все скоры нулевые из-за низкой уверенности модели (не из-за ошибки
    # загрузки), sample ошибочно фильтровался бы как skipped.
    all_skipped_flags: List[bool] = []
    # FIX max_candidates: хранит обновлённый target_idx после _truncate_candidates.
    # Оригинальный sample["target_idx"] может указывать за пределы усечённого
    # списка кандидатов → IndexError в calculate_additional_metrics_for_vlm_rerank.
    all_effective_target_indices: List[int] = []
    predictions: Optional[List[Dict]] = [] if save_predictions else None
    _ci_seed = ci_seed  # локальная переменная для передачи в CI

    # FIX #22: счётчики ошибок загрузки изображений
    total_skipped_query = 0    # запросов пропущено из-за ошибки query image
    total_skipped_cand = 0     # кандидатов пропущено из-за ошибки загрузки
    total_skipped_samples = 0  # запросов исключено из метрик из-за ошибок

    for sample in tqdm(samples, desc=f"Evaluating {model_name} Rerank"):
        query_img_path = sample["query_image"]
        candidates = sample["candidates"]
        target_idx = sample["target_idx"]

        # Обрезаем кандидатов если задан max_candidates
        if max_candidates is not None:
            candidates, target_idx = _truncate_candidates(
                candidates, target_idx, max_candidates
            )

        start_time = time.time()

        # === COMPUTE VLM SCORES (BATCH) ===
        vl_scores, failed_indices = compute_rerank_scores_batch(
            model, processor, query_img_path, candidates,
            image_base_dir, caption_max_length, batch_size,
            fixed_image_size,
        )

        end_time = time.time()
        latencies.append(end_time - start_time)

        # FIX #22: если query image не загрузился — все кандидаты failed
        query_failed = (len(failed_indices) == len(candidates))
        if query_failed:
            total_skipped_query += 1
            total_skipped_samples += 1
            # Добавляем нулевые скоры для сохранения выравнивания all_scores
            all_scores.append(vl_scores)
            all_skipped_flags.append(True)
            all_effective_target_indices.append(target_idx)
            continue

        # FIX #22: если хотя бы один кандидат не загрузился — исключаем запрос
        # из метрик, чтобы не искажать ранжирование нулевыми скорами.
        if failed_indices:
            total_skipped_cand += len(failed_indices)
            total_skipped_samples += 1
            print(
                f"⚠ Skipping sample '{query_img_path}': "
                f"{len(failed_indices)}/{len(candidates)} candidates failed"
            )
            all_scores.append(vl_scores)
            all_skipped_flags.append(True)
            all_effective_target_indices.append(target_idx)
            continue

        all_scores.append(vl_scores)
        all_skipped_flags.append(False)
        all_effective_target_indices.append(target_idx)

        # === RANK BY VLM SCORES ===
        ranked_indices = np.argsort(vl_scores)[::-1]

        # === EVALUATE BASED ON TARGET TYPE ===
        if target_idx != -1:
            for k in k_values:
                if target_idx in ranked_indices[:k]:
                    hits_at_k[k] += 1
                    # FIX #14: стандартная Precision@K = 1/K при одном
                    # релевантном документе (target в топ-K → 1/K, иначе 0).
                    precision_at_k[k].append(1.0 / k)
                else:
                    precision_at_k[k].append(0.0)

            rank = np.where(ranked_indices == target_idx)[0][0] + 1
            reciprocal_rank = 1.0 / rank
            mrr_sum += reciprocal_rank
            mrr_values.append(reciprocal_rank)
            total_valid += 1
        else:
            # none-of-the-above: считаем верным, если все скоры < порога.
            # Используем единую логику и в счётчике, и в save_predictions.
            none_pred_correct = (
                max(vl_scores) < threshold_for_none if vl_scores else True
            )
            if none_pred_correct:
                none_correct += 1
            # FIX #СЕР6: считаем только не-skipped none-samples
            total_none_samples += 1

        # === SAVE PREDICTIONS ===
        if save_predictions and predictions is not None:
            if target_idx != -1:
                is_correct = int(ranked_indices[0]) == target_idx
            else:
                is_correct = (
                    max(vl_scores) < threshold_for_none if vl_scores else True
                )
            predictions.append({
                "query_image": query_img_path,
                "target_idx": target_idx,
                "scores": vl_scores,
                "ranked_indices": ranked_indices.tolist(),
                "predicted_idx": int(ranked_indices[0]),
                "predicted_score": float(vl_scores[ranked_indices[0]]),
                "correct": is_correct,
            })
    
    # === LOG IMAGE LOAD ERRORS ===
    if total_skipped_query > 0 or total_skipped_cand > 0:
        print(
            f"\n⚠ [{model_name}] Image load errors summary:\n"
            f"  Skipped due to query image failure: {total_skipped_query}\n"
            f"  Skipped due to candidate image failure: "
            f"{total_skipped_samples - total_skipped_query} samples "
            f"({total_skipped_cand} candidates total)\n"
            f"  Total samples excluded from metrics: {total_skipped_samples}"
        )

    # === CALCULATE LATENCY ===
    latency_p95 = np.percentile(latencies, 95) if latencies else 0.0

    # === AGGREGATE METRICS ===
    # FIX #17: кешируем hit_values_ci вне цикла по k — одинаковые данные
    # для всех k, пересчитывать каждый раз нет смысла (O(n*K) → O(n+K)).
    # Также фильтруем skipped samples через all_skipped_flags.
    if compute_ci and total_valid > 0:
        _hit_ci_cache: List[tuple] = []  # (s_target, s_ranked) для valid samples
        for _eff_tidx, _sc, _sk in zip(
            all_effective_target_indices, all_scores, all_skipped_flags
        ):
            if _sk or _eff_tidx == -1:
                continue
            _s_ranked = np.argsort(_sc)[::-1]
            _hit_ci_cache.append((_eff_tidx, _s_ranked))
    else:
        _hit_ci_cache = []

    vlm_metrics = {}
    for k in k_values:
        # Hit@K и Recall@K идентичны при одном релевантном документе на запрос
        hit_rate = hits_at_k[k] / total_valid if total_valid > 0 else 0.0
        vlm_metrics[f"{model_name}_hit_{k}"] = hit_rate
        vlm_metrics[f"{model_name}_recall_{k}"] = hit_rate

        # FIX #14: стандартная Precision@K = (число релевантных в топ-K) / K.
        # При одном релевантном документе: 1/K если target в топ-K, иначе 0.
        # Предыдущая формула 1/rank_in_topk — это AP@K, не Precision@K.
        if precision_at_k[k]:
            vlm_metrics[f"{model_name}_precision_{k}"] = float(
                np.mean(precision_at_k[k])
            )
        else:
            vlm_metrics[f"{model_name}_precision_{k}"] = 0.0

        # Confidence intervals для Hit@K
        if _hit_ci_cache:
            hit_values_ci = [
                1.0 if s_target in s_ranked[:k] else 0.0
                for s_target, s_ranked in _hit_ci_cache
            ]
            if hit_values_ci:
                # FIX #20: передаём ci_seed для воспроизводимости bootstrap
                _, lower, upper = calculate_confidence_interval(
                    hit_values_ci, seed=_ci_seed
                )
                vlm_metrics[f"{model_name}_hit_{k}_ci_lower"] = lower
                vlm_metrics[f"{model_name}_hit_{k}_ci_upper"] = upper
    
    # MRR
    vlm_metrics[f"{model_name}_mrr"] = (
        mrr_sum / total_valid if total_valid > 0 else 0.0
    )
    
    # NEW: Confidence interval для MRR
    if compute_ci and mrr_values:
        # FIX #20: передаём ci_seed для воспроизводимости bootstrap
        _, lower, upper = calculate_confidence_interval(
            mrr_values, seed=_ci_seed
        )
        vlm_metrics[f"{model_name}_mrr_ci_lower"] = lower
        vlm_metrics[f"{model_name}_mrr_ci_upper"] = upper
    
    # FIX #СЕР6: используем total_none_samples вместо (total_samples - total_valid),
    # чтобы skipped samples не искажали знаменатель none_accuracy.
    vlm_metrics[f"{model_name}_none_accuracy"] = (
        none_correct / total_none_samples
        if total_none_samples > 0 else 0.0
    )
    vlm_metrics[f"{model_name}_total_samples"] = total_samples
    vlm_metrics[f"{model_name}_valid_samples"] = total_valid
    vlm_metrics[f"{model_name}_none_samples"] = total_none_samples
    vlm_metrics[f"{model_name}_latency_p95"] = latency_p95
    # FIX #22: метрики ошибок загрузки изображений
    vlm_metrics[f"{model_name}_skipped_samples"] = total_skipped_samples
    vlm_metrics[f"{model_name}_skipped_query_errors"] = total_skipped_query
    vlm_metrics[f"{model_name}_skipped_cand_errors"] = (
        total_skipped_samples - total_skipped_query
    )
    
    # === ADDITIONAL METRICS (используем предвычисленные скоры) ===
    # FIX #КРИТ2+#КРИТ6: фильтруем skipped samples используя явный флаг
    # all_skipped_flags вместо эвристики по нулевым скорам.
    # Эвристика давала ложные срабатывания: sample с P(Yes)=0.0 для всех
    # кандидатов (низкая уверенность модели) ошибочно считался skipped.
    valid_samples_for_add: List[Dict] = []
    valid_scores_for_add: List[List[float]] = []
    for _s, _sc, _skipped, _eff_tidx in zip(
        samples, all_scores, all_skipped_flags, all_effective_target_indices
    ):
        if not _skipped:
            # FIX max_candidates: подменяем target_idx на обновлённый после
            # _truncate_candidates, чтобы он соответствовал усечённым скорам.
            _s_copy = dict(_s)
            _s_copy["target_idx"] = _eff_tidx
            valid_samples_for_add.append(_s_copy)
            valid_scores_for_add.append(_sc)
    additional_metrics = calculate_additional_metrics_for_vlm_rerank(
        valid_samples_for_add, valid_scores_for_add, model_name
    )
    vlm_metrics.update(additional_metrics)

    # FIX #КРИТ8: возвращаем valid_samples_for_add — sample'ы с обновлёнными
    # target_idx после _truncate_candidates. Вызывающий код должен использовать
    # их (а не оригинальные samples) при вызове evaluate_by_candidate_type,
    # чтобы target_idx соответствовал длине vl_scores.
    return (
        vlm_metrics,
        all_scores,
        all_skipped_flags,
        predictions,
        valid_samples_for_add,
    )


def evaluate_by_candidate_type(
    samples: List[Dict],
    precomputed_scores: List[List[float]],
    model_name: str = "vlm"
) -> Dict[str, float]:
    """Оценивает точность по типам кандидатов (easy/hard/semi_hard).

    Логика классификации запроса синхронизирована с train.py:
    сложность определяется по максимальному типу НЕГАТИВОВ в списке кандидатов:
    - "hard"      = есть хотя бы один hard-негатив
    - "semi_hard" = нет hard, но есть semi_hard-негатив
    - "easy"      = все негативы easy

    Это показывает, насколько reranker справляется со сложными
    (визуально похожими) негативами — в отличие от классификации
    по типу позитива, которая не отражает реальную сложность задачи.

    Args:
        samples: Список сэмплов
        precomputed_scores: Предвычисленные VLM скоры
        model_name: Имя модели для префикса метрик

    Returns:
        Словарь с метриками по типам
    """
    _KNOWN_TYPES = ("easy", "semi_hard", "hard")

    type_counts: Dict[str, int] = {t: 0 for t in _KNOWN_TYPES}
    type_correct: Dict[str, int] = {t: 0 for t in _KNOWN_TYPES}

    for sample, vl_scores in zip(samples, precomputed_scores):
        if sample["target_idx"] == -1:
            continue

        target_idx = sample["target_idx"]
        candidates = sample["candidates"]

        # FIX #КРИТ8: target_idx может указывать за пределы vl_scores,
        # если samples содержат оригинальный target_idx, а vl_scores —
        # скоры усечённого списка кандидатов (после _truncate_candidates).
        # Пропускаем такие sample'ы, чтобы избежать IndexError.
        if target_idx >= len(vl_scores):
            continue

        # Синхронизировано с train.py: определяем сложность по типу негативов,
        # а не по типу позитива. Это обеспечивает сопоставимость метрик
        # eval_hard_accuracy (train) и {model_name}_hard_accuracy (eval).
        neg_types = [
            c.get("candidate_type", "easy")
            for i, c in enumerate(candidates)
            if i != target_idx
        ]
        if "hard" in neg_types:
            query_difficulty = "hard"
        elif "semi_hard" in neg_types:
            query_difficulty = "semi_hard"
        else:
            query_difficulty = "easy"

        # Динамически добавляем неизвестные типы — не ломаем счётчики
        if query_difficulty not in type_counts:
            type_counts[query_difficulty] = 0
            type_correct[query_difficulty] = 0

        # Ранжируем по предвычисленным скорам
        ranked_indices = np.argsort(vl_scores)[::-1]
        pos = np.where(ranked_indices == target_idx)[0]
        # Дополнительная защита: target_idx должен быть в ranked_indices
        if len(pos) == 0:
            continue
        rank_of_target = pos[0]

        if rank_of_target == 0:  # top-1
            type_correct[query_difficulty] += 1
        type_counts[query_difficulty] += 1

    # Итерируем по всем собранным типам
    type_acc = {}
    for t in type_counts:
        acc = (type_correct[t] / type_counts[t]
               if type_counts[t] > 0 else 0.0)
        type_acc[f"{model_name}_{t}_accuracy"] = acc
        type_acc[f"{model_name}_{t}_count"] = type_counts[t]

    return type_acc


def find_optimal_none_threshold(
    samples: List[Dict],
    precomputed_scores: List[List[float]],
    thresholds: Optional[List[float]] = None,
) -> float:
    """Находит оптимальный порог для none-of-the-above на основе F1.

    Перебирает кандидатные пороги и выбирает тот, при котором F1-score
    для задачи «known vs unknown» максимален. Используется для подбора
    threshold_for_none на валидационной выборке вместо фиксированного 0.5.

    Args:
        samples: Список сэмплов с полем target_idx (-1 = unknown).
        precomputed_scores: Предвычисленные VLM скоры для каждого сэмпла.
        thresholds: Список порогов для перебора. По умолчанию np.linspace.

    Returns:
        Оптимальный порог (float).
    """
    if thresholds is None:
        thresholds = list(np.linspace(0.1, 0.9, 17))

    # Истинные метки: 1 = known (target_idx != -1), 0 = unknown
    true_labels = [
        0 if s["target_idx"] == -1 else 1
        for s in samples
    ]

    # Предсказанная метка при пороге t: 1 если max(scores) >= t, иначе 0
    best_threshold = 0.5
    best_f1 = -1.0

    for t in thresholds:
        preds = [
            # FIX #КРИТ7: max([]) бросает ValueError при пустом sc.
            # Пустой список возможен если все кандидаты sample'а не загрузились.
            # Используем max(sc, default=0.0) для безопасной обработки.
            1 if (max(sc, default=0.0) >= t) else 0
            for sc in precomputed_scores
        ]
        # Избегаем деления на ноль при вырожденных предсказаниях
        if len(set(preds)) < 2:
            continue
        score = f1_score(true_labels, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = t

    return float(best_threshold)


def evaluate_rerank(
    dataset_path: str,
    image_base_dir: str,
    output_path: str,
    lora_path: Optional[str] = None,
    skip_zero_shot: bool = False,
    threshold_for_none: float = 0.5,
    k_values: List[int] = [1, 3, 5],
    batch_size: int = 1,
    fixed_image_size: Optional[tuple] = (448, 448),
    caption_max_length: int = 64,
    max_candidates: Optional[int] = None,
    use_mlflow: bool = False,
    mlflow_experiment: str = "landmark_rerank_eval",
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_run_name: Optional[str] = None,
    random_seed: Optional[int] = 42,
    compute_ci: bool = False,
):
    """Полная оценка retrieval и VLM reranking.

    Args:
        dataset_path: Путь к датасету
        image_base_dir: Базовая директория для изображений
        output_path: Путь для сохранения результатов
        lora_path: Путь к LoRA адаптеру (опционально)
        skip_zero_shot: Пропустить zero-shot оценку
        threshold_for_none: Порог для none-of-the-above
        k_values: Значения K для метрик Hit@K, Recall@K
        batch_size: Размер батча для VLM inference (по умолчанию 1)
        fixed_image_size: Фиксированный размер (W, H) для ресайза изображений.
            Обязателен при batch_size>1. None отключает ресайз.
        use_mlflow: Использовать MLflow для логирования (по умолчанию False)
        mlflow_experiment: Имя MLflow эксперимента
        mlflow_tracking_uri: URI для MLflow tracking
        mlflow_run_name: Имя запуска MLflow
        random_seed: Random seed для воспроизводимости (None = без seed)
        compute_ci: Вычислять ли bootstrap confidence intervals (медленнее)
    """
    # Установка random seed для воспроизводимости
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        print(f"✓ Random seed установлен: {random_seed}")
    
    print("=" * 70)
    print("Оценка retrieval и VLM reranking")
    print("=" * 70)

    # Загрузка датасета
    print(f"\nЗагрузка датасета: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # Валидация структуры датасета
    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError(
            f"Датасет должен быть непустым списком, получено: "
            f"{type(samples).__name__}"
        )
    # FIX #5: проверяем структуру всех сэмплов, а не только первого —
    # ошибки в середине датасета иначе обнаружатся только во время GPU-инференса
    required_keys = {"query_image", "candidates", "target_idx"}
    for i, s in enumerate(samples):
        missing = required_keys - set(s.keys())
        if missing:
            raise ValueError(
                f"Сэмпл #{i} не содержит обязательных полей: {missing}"
            )
    print(f"Загружено {len(samples)} примеров")

    # === 1. EVALUATE RETRIEVAL ONLY (BASELINE) ===
    print("\n1. Evaluating retrieval baseline...")
    retrieval_metrics = evaluate_retrieval_only(
        samples, k_values,
        compute_ci=compute_ci,
        ci_seed=random_seed,
    )

    # === 2. EVALUATE ZERO-SHOT VLM RERANKING (опционально)===
    vlm_metrics = {}
    zs_type_metrics = {}
    zs_scores = []
    if not skip_zero_shot:
        print("\n2. Loading Qwen2-VL-2B-Instruct (zero-shot)...")
        model_zs, processor_zs = load_qwen2vl_processor(lora_path=None)

        print("\n3. Evaluating zero-shot VLM reranking...")
        (
            vlm_metrics,
            zs_scores,
            zs_skipped_flags,
            zs_predictions,
            zs_valid_samples,
        ) = evaluate_vlm_rerank(
            model_zs, processor_zs, samples, image_base_dir,
            k_values=k_values,
            threshold_for_none=threshold_for_none,
            model_name="zero_shot_vlm",
            caption_max_length=caption_max_length,
            batch_size=batch_size,
            fixed_image_size=fixed_image_size,
            max_candidates=max_candidates,
            save_predictions=True,
            compute_ci=compute_ci,
            ci_seed=random_seed,
        )

        # Сохранение predictions
        if zs_predictions:
            pred_path = output_path.replace(
                ".json", "_zero_shot_predictions.json"
            )
            with open(pred_path, 'w', encoding='utf-8') as f:
                json.dump(zs_predictions, f, indent=2, ensure_ascii=False)
            print(f"✓ Predictions сохранены: {pred_path}")

        # === 4. ANALYSIS BY CANDIDATE TYPE FOR ZERO-SHOT ===
        print("\n4. Evaluating zero-shot by candidate type...")
        # FIX #КРИТ8: используем zs_valid_samples из evaluate_vlm_rerank —
        # они содержат обновлённый target_idx после _truncate_candidates,
        # что гарантирует соответствие target_idx длине vl_scores.
        zs_valid_scores = [
            sc for sc, sk in zip(zs_scores, zs_skipped_flags) if not sk
        ]
        zs_type_metrics = evaluate_by_candidate_type(
            zs_valid_samples, zs_valid_scores, model_name="zero_shot_vlm"
        )
        
        # CRITICAL: Освобождаем GPU память перед загрузкой LoRA
        if lora_path:
            print("\n✓ Освобождение GPU памяти перед загрузкой LoRA...")
            del model_zs
            del processor_zs
            torch.cuda.empty_cache()
            gc.collect()
    else:
        print("\n2. Skipping zero-shot VLM evaluation.")

    # === 5. EVALUATE LoRA VLM RERANKING (только если lora_path указан) ===
    lora_metrics = {}
    lora_type_metrics = {}
    lora_scores = []
    if lora_path:
        print(f"\n3. Loading Qwen2-VL-2B-Instruct + LoRA from: {lora_path}")
        model_lora, processor_lora = load_qwen2vl_processor(
            lora_path=lora_path
        )

        print("\n4. Evaluating LoRA VLM reranking...")
        (
            lora_metrics,
            lora_scores,
            lora_skipped_flags,
            lora_predictions,
            lora_valid_samples,
        ) = evaluate_vlm_rerank(
            model_lora, processor_lora, samples, image_base_dir,
            k_values=k_values,
            threshold_for_none=threshold_for_none,
            model_name="lora_vlm",
            caption_max_length=caption_max_length,
            batch_size=batch_size,
            fixed_image_size=fixed_image_size,
            max_candidates=max_candidates,
            save_predictions=True,
            compute_ci=compute_ci,
            ci_seed=random_seed,
        )

        # Сохранение predictions
        if lora_predictions:
            pred_path = output_path.replace(
                ".json", "_lora_predictions.json"
            )
            with open(pred_path, 'w', encoding='utf-8') as f:
                json.dump(lora_predictions, f, indent=2, ensure_ascii=False)
            print(f"✓ Predictions сохранены: {pred_path}")

        # === 6. ANALYSIS BY CANDIDATE TYPE FOR LoRA ===
        print("\n5. Evaluating LoRA by candidate type...")
        # FIX #КРИТ8: используем lora_valid_samples из evaluate_vlm_rerank —
        # они содержат обновлённый target_idx после _truncate_candidates.
        lora_valid_scores = [
            sc for sc, sk in zip(lora_scores, lora_skipped_flags) if not sk
        ]
        lora_type_metrics = evaluate_by_candidate_type(
            lora_valid_samples, lora_valid_scores, model_name="lora_vlm"
        )

    # === 7. COMBINE RESULTS ===
    results = {
        "config": {
            "dataset": dataset_path,
            "image_base_dir": image_base_dir,
            "model": "Qwen/Qwen2-VL-2B-Instruct",
            "lora_path": lora_path,
            "skip_zero_shot": skip_zero_shot, # Добавлено в конфиг
            "threshold_for_none": threshold_for_none,
            "k_values": k_values
        },
        "retrieval_baseline": retrieval_metrics,
    }

    if not skip_zero_shot:
        results["zero_shot_vlm"] = vlm_metrics
        results["zero_shot_by_type"] = zs_type_metrics

    if lora_path:
        results["lora_vlm"] = lora_metrics
        results["lora_by_type"] = lora_type_metrics

    # Сводные метрики для быстрого сравнения
    results["summary"] = {
        # Метрики retrieval (hit@k и recall@k идентичны при одном релевантном документе)
        "retrieval_hit_1": retrieval_metrics.get("retrieval_hit_1", 0),
        "retrieval_hit_3": retrieval_metrics.get("retrieval_hit_3", 0),
        "retrieval_hit_5": retrieval_metrics.get("retrieval_hit_5", 0),
        "retrieval_recall_1": retrieval_metrics.get("retrieval_recall_1", 0),
        "retrieval_recall_3": retrieval_metrics.get("retrieval_recall_3", 0),
        "retrieval_recall_5": retrieval_metrics.get("retrieval_recall_5", 0),
        "retrieval_mrr": retrieval_metrics.get("retrieval_mrr", 0),
        
        # Метрики zero-shot VLM
        "zero_shot_vlm_hit_1": vlm_metrics.get(
            "zero_shot_vlm_hit_1", 0
        ) if not skip_zero_shot else None,
        "zero_shot_vlm_hit_3": vlm_metrics.get(
            "zero_shot_vlm_hit_3", 0
        ) if not skip_zero_shot else None,
        "zero_shot_vlm_hit_5": vlm_metrics.get(
            "zero_shot_vlm_hit_5", 0
        ) if not skip_zero_shot else None,
        "zero_shot_vlm_recall_1": vlm_metrics.get(
            "zero_shot_vlm_recall_1", 0
        ) if not skip_zero_shot else None,
        "zero_shot_vlm_recall_3": vlm_metrics.get(
            "zero_shot_vlm_recall_3", 0
        ) if not skip_zero_shot else None,
        "zero_shot_vlm_recall_5": vlm_metrics.get(
            "zero_shot_vlm_recall_5", 0
        ) if not skip_zero_shot else None,
        "zero_shot_vlm_mrr": vlm_metrics.get(
            "zero_shot_vlm_mrr", 0
        ) if not skip_zero_shot else None,
        "zero_shot_vlm_none_accuracy": vlm_metrics.get(
            "zero_shot_vlm_none_accuracy", 0
        ) if not skip_zero_shot else None,
        "zero_shot_vlm_latency_p95": vlm_metrics.get(
            "zero_shot_vlm_latency_p95", 0
        ) if not skip_zero_shot else None,
        "zero_shot_vlm_median_rank": vlm_metrics.get(
            "zero_shot_vlm_median_rank", 0
        ) if not skip_zero_shot else None,
        "zero_shot_vlm_brier_score": vlm_metrics.get(
            "zero_shot_vlm_brier_score", 0
        ) if not skip_zero_shot else None,
        "zero_shot_vlm_unknown_auroc": vlm_metrics.get(
            "zero_shot_vlm_unknown_auroc", 0
        ) if not skip_zero_shot else None,
        "zero_shot_vlm_unknown_f1": vlm_metrics.get(
            "zero_shot_vlm_unknown_f1", 0
        ) if not skip_zero_shot else None,
        "zero_shot_vlm_unknown_fpr_at_95tpr": vlm_metrics.get(
            "zero_shot_vlm_unknown_fpr_at_95tpr", 0
        ) if not skip_zero_shot else None,
        
        # LoRA VLM metrics
        "lora_vlm_hit_1": lora_metrics.get(
            "lora_vlm_hit_1", 0
        ) if lora_metrics else None,
        "lora_vlm_hit_3": lora_metrics.get(
            "lora_vlm_hit_3", 0
        ) if lora_metrics else None,
        "lora_vlm_hit_5": lora_metrics.get(
            "lora_vlm_hit_5", 0
        ) if lora_metrics else None,
        "lora_vlm_recall_1": lora_metrics.get(
            "lora_vlm_recall_1", 0
        ) if lora_metrics else None,
        "lora_vlm_recall_3": lora_metrics.get(
            "lora_vlm_recall_3", 0
        ) if lora_metrics else None,
        "lora_vlm_recall_5": lora_metrics.get(
            "lora_vlm_recall_5", 0
        ) if lora_metrics else None,
        "lora_vlm_mrr": lora_metrics.get(
            "lora_vlm_mrr", 0
        ) if lora_metrics else None,
        "lora_vlm_none_accuracy": lora_metrics.get(
            "lora_vlm_none_accuracy", 0
        ) if lora_metrics else None,
        "lora_vlm_latency_p95": lora_metrics.get(
            "lora_vlm_latency_p95", 0.0
        ) if lora_metrics else None,
        "lora_vlm_median_rank": lora_metrics.get(
            "lora_vlm_median_rank", 0
        ) if lora_metrics else None,
        "lora_vlm_brier_score": lora_metrics.get(
            "lora_vlm_brier_score", 0
        ) if lora_metrics else None,
        "lora_vlm_unknown_auroc": lora_metrics.get(
            "lora_vlm_unknown_auroc", 0
        ) if lora_metrics else None,
        "lora_vlm_unknown_f1": lora_metrics.get(
            "lora_vlm_unknown_f1", 0
        ) if lora_metrics else None,
        "lora_vlm_unknown_fpr_at_95tpr": lora_metrics.get(
            "lora_vlm_unknown_fpr_at_95tpr", 0
        ) if lora_metrics else None,
    }

    # Убираем None-значения для чистого JSON
    results["summary"] = {
        k: v for k, v in results["summary"].items() if v is not None
    }

    # === 8. SAVE AND PRINT RESULTS ===
    # FIX #2/#1: сохраняем JSON ДО открытия MLflow run, чтобы log_artifact
    # логировал актуальный файл внутри основного run, а не в отдельный.
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # === MLFLOW LOGGING (OPTIONAL) ===
    if use_mlflow:
        print("\nЛогирование в MLflow...")
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment)

        run_name = mlflow_run_name or (
            "eval_" + os.path.basename(output_path).replace(".json", "")
        )

        # Создаём директорию для графиков
        plots_dir = os.path.join(os.path.dirname(output_path), "plots")
        os.makedirs(plots_dir, exist_ok=True)

        with mlflow.start_run(run_name=run_name):
            # Логируем параметры
            mlflow.log_params({
                "dataset": dataset_path,
                "image_base_dir": image_base_dir,
                "lora_path": (lora_path or "none")[:500],
                "skip_zero_shot": skip_zero_shot,
                "threshold_for_none": threshold_for_none,
                "k_values": str(k_values),
                "model": "Qwen/Qwen2-VL-2B-Instruct",
                "dataset_size": len(samples),
                "random_seed": (
                    random_seed if random_seed is not None else "None"
                ),
            })

            # FIX #13: mlflow.log_metrics принимает только float —
            # явно приводим все значения, чтобы избежать MlflowException
            # на int-значениях (например retrieval_total_samples)
            mlflow.log_metrics(
                {k: float(v) for k, v in results["summary"].items()}
            )

            # Создаём и логируем calibration plots
            if not skip_zero_shot and vlm_metrics:
                print("Создание calibration plot для zero-shot...")
                # FIX #16: используем _s вместо sample, чтобы не перезаписывать
                # одноимённую переменную из внешнего контекста
                zs_probs, zs_labels = [], []
                for _s, _scores in zip(samples, zs_scores):
                    if _s["target_idx"] != -1:
                        # FIX #КРИТ9: используем len(_scores) вместо
                        # len(_s["candidates"]) — при max_candidates скоры
                        # усечены, а candidates в sample оригинальные.
                        # Несоответствие длин → IndexError в calibration plot.
                        labels = [
                            1 if i == _s["target_idx"] else 0
                            for i in range(len(_scores))
                        ]
                        zs_probs.extend(_scores)
                        zs_labels.extend(labels)

                if zs_probs:
                    plot_path = create_calibration_plot(
                        np.array(zs_probs), np.array(zs_labels),
                        "zero_shot_vlm", plots_dir
                    )
                    mlflow.log_artifact(plot_path, "plots")
                    print(f"✓ Calibration plot сохранён: {plot_path}")

            if lora_metrics and lora_scores:
                print("Создание calibration plot для LoRA...")
                # FIX #16: аналогично используем _s/_scores
                lora_probs, lora_labels = [], []
                for _s, _scores in zip(samples, lora_scores):
                    if _s["target_idx"] != -1:
                        # FIX #КРИТ9: аналогично используем len(_scores)
                        labels = [
                            1 if i == _s["target_idx"] else 0
                            for i in range(len(_scores))
                        ]
                        lora_probs.extend(_scores)
                        lora_labels.extend(labels)

                if lora_probs:
                    plot_path = create_calibration_plot(
                        np.array(lora_probs), np.array(lora_labels),
                        "lora_vlm", plots_dir
                    )
                    mlflow.log_artifact(plot_path, "plots")
                    print(f"✓ Calibration plot сохранён: {plot_path}")

            # FIX #1/#2: JSON уже записан выше — логируем артефакт внутри
            # основного run, а не в отдельный (mlflow.active_run() после
            # with-блока всегда None, что создавало второй run)
            mlflow.log_artifact(output_path, "results")
            print("✓ Метрики и артефакты залогированы в MLflow")
    else:
        print("\nMLflow логирование отключено (use_mlflow=False)")

    print("\n" + "="*70)
    print("ОЦЕНКА ЗАВЕРШЕНА")
    print("="*70)
    print("\nCONFIG:")
    print(f"  Skip Zero-Shot: {skip_zero_shot}")
    print(f"  LoRA Path: {lora_path or 'None'}")
    print("\nSUMMARY:")
    # FIX #21: используем .get() вместо прямого обращения по ключу, чтобы
    # избежать KeyError при k_values отличных от [1, 3, 5].
    summary = results["summary"]
    for k in k_values:
        v = summary.get(f"retrieval_hit_{k}")
        if v is not None:
            print(f"  Retrieval Hit_{k}: {v:.3f}")
    for k in k_values:
        v = summary.get(f"retrieval_recall_{k}")
        if v is not None:
            print(f"  Retrieval Recall_{k}: {v:.3f}")
    mrr_v = summary.get("retrieval_mrr")
    if mrr_v is not None:
        print(f"  Retrieval MRR: {mrr_v:.3f}")
    if not skip_zero_shot:
        for k in k_values:
            v = summary.get(f"zero_shot_vlm_hit_{k}")
            if v is not None:
                print(f"  Zero-shot VLM Hit_{k}: {v:.3f}")
        for k in k_values:
            v = summary.get(f"zero_shot_vlm_recall_{k}")
            if v is not None:
                print(f"  Zero-shot VLM Recall_{k}: {v:.3f}")
        for key, label in [
            ("zero_shot_vlm_mrr", "Zero-shot VLM MRR"),
            ("zero_shot_vlm_none_accuracy", "Zero-shot VLM None-Accuracy"),
            ("zero_shot_vlm_latency_p95", "Zero-shot VLM Latency P95"),
            ("zero_shot_vlm_median_rank", "Zero-shot VLM Median Rank"),
            ("zero_shot_vlm_brier_score", "Zero-shot VLM Brier Score"),
            ("zero_shot_vlm_unknown_auroc", "Zero-shot VLM Unknown AUROC"),
            ("zero_shot_vlm_unknown_f1", "Zero-shot VLM Unknown F1"),
            (
                "zero_shot_vlm_unknown_fpr_at_95tpr",
                "Zero-shot VLM Unknown FPR_95TPR",
            ),
        ]:
            v = summary.get(key)
            if v is not None:
                suffix = "s" if "latency" in key else ""
                print(f"  {label}: {v:.3f}{suffix}")
    if any(f"lora_vlm_hit_{k}" in summary for k in k_values):
        for k in k_values:
            v = summary.get(f"lora_vlm_hit_{k}")
            if v is not None:
                print(f"  LoRA VLM Hit_{k}: {v:.3f}")
        for k in k_values:
            v = summary.get(f"lora_vlm_recall_{k}")
            if v is not None:
                print(f"  LoRA VLM Recall_{k}: {v:.3f}")
        for key, label in [
            ("lora_vlm_mrr", "LoRA VLM MRR"),
            ("lora_vlm_none_accuracy", "LoRA VLM None-Accuracy"),
            ("lora_vlm_latency_p95", "LoRA VLM Latency P95"),
            ("lora_vlm_median_rank", "LoRA VLM Median Rank"),
            ("lora_vlm_brier_score", "LoRA VLM Brier Score"),
            ("lora_vlm_unknown_auroc", "LoRA VLM Unknown AUROC"),
            ("lora_vlm_unknown_f1", "LoRA VLM Unknown F1"),
            ("lora_vlm_unknown_fpr_at_95tpr", "LoRA VLM Unknown FPR_95TPR"),
        ]:
            v = summary.get(key)
            if v is not None:
                suffix = "s" if "latency" in key else ""
                print(f"  {label}: {v:.3f}{suffix}")
    if "zero_shot_vlm_easy_accuracy" in zs_type_metrics:
        for t_key, t_label in [
            ("zero_shot_vlm_easy_accuracy", "Zero-shot VLM Easy Accuracy"),
            (
                "zero_shot_vlm_semi_hard_accuracy",
                "Zero-shot VLM Semi-Hard Accuracy",
            ),
            ("zero_shot_vlm_hard_accuracy", "Zero-shot VLM Hard Accuracy"),
        ]:
            v = zs_type_metrics.get(t_key)
            if v is not None:
                print(f"  {t_label}: {v:.3f}")
    if "lora_vlm_easy_accuracy" in lora_type_metrics:
        for t_key, t_label in [
            ("lora_vlm_easy_accuracy", "LoRA VLM Easy Accuracy"),
            ("lora_vlm_semi_hard_accuracy", "LoRA VLM Semi-Hard Accuracy"),
            ("lora_vlm_hard_accuracy", "LoRA VLM Hard Accuracy"),
        ]:
            v = lora_type_metrics.get(t_key)
            if v is not None:
                print(f"  {t_label}: {v:.3f}")

    print(f"\nРезультаты сохранены: {output_path}")
    print("="*70)

    return results


# ============================================
# ТОЧКА ВХОДА
# ============================================

if __name__ == "__main__":
    # Конфигурация параметров
    _DATASET_PATH = "data/processed/test_samples/val.json"  # или test.json
    _IMAGE_BASE_DIR = "images"
    _OUTPUT_PATH = "data/eval/val_eval_results.json"
    _LORA_PATH = None  # Указать путь к LoRA адаптеру или None
    _SKIP_ZERO_SHOT = False  # Установить в True, чтобы пропустить zero-shot
    _THRESHOLD_FOR_NONE = 0.5
    _K_VALUES = [1, 3, 5]
    # Размер батча для VLM inference.
    # T4 16GB: при fixed_image_size=(224,224) оптимально batch_size=4-8.
    # При (448,448) каждый пример = ~256 патчей × 2 изображения → OOM при >4.
    # batch_size>1 корректно работает при fixed_image_size != None.
    _BATCH_SIZE = 8
    # Фиксированный размер изображений для батчинга.
    # (224, 224) → 1 тайл = 64 патча на изображение (в 4x меньше чем 448x448).
    # Даёт ~3-4x ускорение на T4 при незначительной потере качества.
    # (448, 448) — стандартный размер тайла, но тяжело для T4 при батчинге.
    _FIXED_IMAGE_SIZE = (224, 224)
    # Максимальная длина caption в символах.
    # 64 символа ≈ 16-20 токенов — достаточно для ключевых деталей ориентира.
    # 300 (дефолт) → ~75 токенов, увеличивает sequence length и замедляет attention.
    _CAPTION_MAX_LENGTH = 64
    # Максимальное число кандидатов на запрос для VLM inference.
    # None = использовать все кандидаты из датасета (по умолчанию ~15).
    # При target_idx != -1 позитив всегда включается в усечённый список.
    # При target_idx == -1 берутся первые _MAX_CANDIDATES негативов.
    # Пример: 5 кандидатов вместо 15 → ~3x ускорение forward pass.
    _MAX_CANDIDATES = 10
    # Вычислять ли bootstrap confidence intervals (медленнее, но точнее).
    _COMPUTE_CI = False
    _USE_MLFLOW = False  # Установить в True для логирования в MLflow
    _MLFLOW_EXPERIMENT = "landmark_rerank_eval"
    _MLFLOW_TRACKING_URI = None  # Указать URI или None для локального сервера
    _MLFLOW_RUN_NAME = None  # Указать имя запуска или None для автогенерации

    evaluate_rerank(
        dataset_path=_DATASET_PATH,
        image_base_dir=_IMAGE_BASE_DIR,
        output_path=_OUTPUT_PATH,
        lora_path=_LORA_PATH,
        skip_zero_shot=_SKIP_ZERO_SHOT,
        threshold_for_none=_THRESHOLD_FOR_NONE,
        k_values=_K_VALUES,
        batch_size=_BATCH_SIZE,
        fixed_image_size=_FIXED_IMAGE_SIZE,
        caption_max_length=_CAPTION_MAX_LENGTH,
        max_candidates=_MAX_CANDIDATES,
        compute_ci=_COMPUTE_CI,
        use_mlflow=_USE_MLFLOW,
        mlflow_experiment=_MLFLOW_EXPERIMENT,
        mlflow_tracking_uri=_MLFLOW_TRACKING_URI,
        mlflow_run_name=_MLFLOW_RUN_NAME,
    )