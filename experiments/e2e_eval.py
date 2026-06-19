# experiments/e2e_eval.py
"""
End-to-End оценка полного пайплайна распознавания достопримечательностей.

Этот скрипт оценивает производительность всей системы:
1. Retrieval (SigLIP/DINOv2 + FAISS)
2. Reranking (Qwen2-VL с LoRA)
3. Unknown detection (обработка неизвестных достопримечательностей)

Формат входного датасета (абсолютный ground truth):
[
    {
        "query_image": "path/to/image.jpg",
        "target_landmark_id": "12345",   # str — известный объект
        "is_unknown": false
    },
    {
        "query_image": "path/to/unknown.jpg",
        "target_landmark_id": null,      # null — неизвестный объект
        "is_unknown": true
    }
]

Совместимость с форматом step6_setup_dataset.py:
Если датасет содержит поля "target_idx" и "candidates" (старый формат),
скрипт автоматически извлекает target_landmark_id из candidates[target_idx].
Для unknown-сэмплов (target_idx=-1) is_unknown=True.

ВАЖНО: unknown-сэмплы должны содержать изображения объектов, которых
НЕТ в production FAISS индексе. Использование outside_topk сэмплов из
step6 некорректно — там known объекты с искусственно убранным ответом.
Для честной оценки unknown detection используй отдельный датасет с
реально неизвестными объектами.

Метрики:
- Hit@1: Точность распознавания известных достопримечательностей
- MRR: Mean Reciprocal Rank
- Unknown Detection Rate: Доля корректно определённых неизвестных объектов
- P95 Latency: 95-й перцентиль задержки
- Throughput: Количество запросов в секунду
"""

import json
import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
from tqdm import tqdm

from src.rag.landmark_retriever import LandmarkRetriever
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)
from peft import PeftModel


@dataclass
class E2EResult:
    """Результат E2E оценки одного запроса."""
    query_image: str
    target_landmark_id: Optional[str]   # None для unknown
    predicted_landmark_id: Optional[str]
    confidence_score: float
    latency_ms: float
    is_correct: bool
    is_unknown: bool
    retrieval_rank: int  # Ранг правильного ответа среди кандидатов pipeline

    def to_dict(self) -> Dict:
        return {
            "query_image": self.query_image,
            "target_landmark_id": self.target_landmark_id,
            "predicted_landmark_id": self.predicted_landmark_id,
            "confidence_score": self.confidence_score,
            "latency_ms": self.latency_ms,
            "is_correct": self.is_correct,
            "is_unknown": self.is_unknown,
            "retrieval_rank": self.retrieval_rank,
        }


def extract_ground_truth(item: Dict) -> Tuple[Optional[str], bool]:
    """
    Извлекает абсолютный ground truth из записи датасета.

    Поддерживает два формата:
    1. Новый (абсолютный):
       {"target_landmark_id": "12345", "is_unknown": false}
    2. Старый (step6, относительный):
       {"target_idx": 0, "candidates": [{"landmark_id": "12345", ...}]}

    Returns:
        (target_landmark_id, is_unknown)
        target_landmark_id = None если is_unknown=True
    """
    # Новый формат — приоритет
    if "target_landmark_id" in item:
        tid = item["target_landmark_id"]
        is_unknown = item.get("is_unknown", tid is None)
        return (None if is_unknown else tid), is_unknown

    # Старый формат (step6_setup_dataset.py)
    target_idx = item.get("target_idx", -1)
    candidates = item.get("candidates", [])

    if target_idx == -1:
        # В step6 unknown-сэмплы генерируются outside_topk —
        # это known объекты с убранным правильным ответом.
        # Для E2E оценки unknown detection такие сэмплы некорректны
        # (production retrieval может найти правильный ответ).
        # Помечаем как unknown, но предупреждаем.
        return None, True

    if 0 <= target_idx < len(candidates):
        lid = candidates[target_idx].get("landmark_id")
        if lid:
            return str(lid), False

    # Не удалось извлечь — пропускаем
    return None, False


class ProductionPipeline:
    """
    Production-ready пайплайн для распознавания достопримечательностей.

    Включает:
    1. Retrieval: LandmarkRetriever (SigLIP/DINOv2 + FAISS)
    2. Reranking: VLM reranker (Qwen2-VL с LoRA) — батчинг всех кандидатов
    3. Unknown detection: Порог уверенности
    """

    def __init__(
        self,
        retriever: LandmarkRetriever,
        reranker_model,
        reranker_processor,
        image_base_dir: str,
        retrieval_top_k: int = 10,
        rerank_threshold: float = 0.5,
        caption_max_length: int = 300,
        # PERF: батчинг всех кандидатов одним forward pass.
        # При fixed_image_size все изображения имеют одинаковое число патчей
        # → корректный маппинг <image> токенов в плоском pixel_values тензоре.
        # 224px: 256 патчей/img × 2 img × batch = умеренный размер тензора.
        # 448px: 1024 патчей/img × 2 × 10 = 20480 патчей → OOM/swap.
        rerank_batch_size: int = 10,
        fixed_image_size: Optional[Tuple[int, int]] = (224, 224),
    ):
        self.retriever = retriever
        self.reranker_model = reranker_model
        self.reranker_processor = reranker_processor
        self.image_base_dir = Path(image_base_dir)
        self.retrieval_top_k = retrieval_top_k
        self.rerank_threshold = rerank_threshold
        self.caption_max_length = caption_max_length
        self.rerank_batch_size = rerank_batch_size
        self.fixed_image_size = fixed_image_size

        # FIX batch_size>1: left-padding обязателен для decoder-only моделей,
        # чтобы логит на позиции -1 соответствовал последнему токену промпта.
        _tokenizer = getattr(reranker_processor, "tokenizer", reranker_processor)
        _tokenizer.padding_side = "left"

        # Токены Yes/No — используем convert_tokens_to_ids как в eval.py
        _unk_id = getattr(_tokenizer, "unk_token_id", None)
        yes_id = _tokenizer.convert_tokens_to_ids("Yes")
        no_id = _tokenizer.convert_tokens_to_ids("No")
        if yes_id == _unk_id or no_id == _unk_id:
            yes_id = _tokenizer.encode("Yes", add_special_tokens=False)[0]
            no_id = _tokenizer.encode("No", add_special_tokens=False)[0]
        self.yes_id = yes_id
        self.no_id = no_id

    def _rerank_all_candidates(
        self,
        query_image: Image.Image,
        candidates_info: List[Dict],
    ) -> List[float]:
        """
        Вычисляет rerank scores для всех кандидатов батчами.

        PERF: вместо N отдельных forward pass — ceil(N/batch_size) батчей.
        При rerank_batch_size=10 и top_k=10 — один forward pass на запрос.

        Args:
            query_image: PIL-изображение запроса (уже загружено)
            candidates_info: список dict с ключами image_path, name, caption

        Returns:
            List[float]: scores[i] для candidates_info[i], 0.0 при ошибке загрузки
        """
        all_scores = [0.0] * len(candidates_info)

        # PROF: счётчики фаз (активны только первые 3 вызова)
        _prof_calls = getattr(self, "_prof_rerank_calls", 0)
        _do_prof = _prof_calls < 3
        self._prof_rerank_calls = _prof_calls + 1
        _p_io = _p_tmpl = _p_proc = _p_xfer = _p_fwd = 0.0

        # Ресайз query один раз для всего батча
        if self.fixed_image_size is not None:
            query_resized = query_image.resize(
                self.fixed_image_size, Image.Resampling.BILINEAR
            )
        else:
            query_resized = query_image.copy()

        def _load_one(
            idx_info: Tuple[int, Dict]
        ) -> Tuple[int, Optional[Image.Image]]:
            """Загружает и ресайзит одно изображение кандидата."""
            local_idx, info = idx_info
            try:
                with Image.open(
                    self.image_base_dir / info["image_path"]
                ) as _img:
                    img = _img.convert("RGB")
                if self.fixed_image_size is not None:
                    _r = img.resize(
                        self.fixed_image_size, Image.Resampling.BILINEAR
                    )
                    img.close()
                    img = _r
                return local_idx, img
            except Exception as e:
                print(
                    f"⚠️ Error loading candidate image "
                    f"{info.get('image_path')}: {e}"
                )
                return local_idx, None

        for batch_start in range(
            0, len(candidates_info), self.rerank_batch_size
        ):
            batch_end = min(
                batch_start + self.rerank_batch_size, len(candidates_info)
            )
            batch_info = candidates_info[batch_start:batch_end]

            # PERF: параллельная загрузка всех изображений батча
            _t = time.time()
            loaded: Dict[int, Image.Image] = {}
            with ThreadPoolExecutor(max_workers=min(8, len(batch_info))) as ex:
                for local_idx, img in ex.map(
                    _load_one, enumerate(batch_info)
                ):
                    if img is not None:
                        loaded[local_idx] = img
            _p_io += time.time() - _t

            batch_texts = []
            batch_images_flat = []
            batch_valid_local: List[int] = []

            for local_idx, info in enumerate(batch_info):
                if local_idx not in loaded:
                    continue
                cand_img = loaded[local_idx]

                caption = info["caption"][:self.caption_max_length]
                _t = time.time()
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
                                    "Question: Are these photos showing"
                                    " the same landmark: "
                                    f"\"{info['name']}\"?\n"
                                    f"Candidate details: {caption}\n"
                                    "Answer only with Yes or No."
                                ),
                            },
                        ],
                    }
                ]
                text = self.reranker_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                _p_tmpl += time.time() - _t

                batch_texts.append(text)
                # FIX #17: копируем query чтобы процессор не мутировал оригинал
                batch_images_flat.append(query_resized.copy())
                batch_images_flat.append(cand_img.copy())
                cand_img.close()
                batch_valid_local.append(local_idx)

            if not batch_texts:
                continue

            _t = time.time()
            if len(batch_texts) == 1:
                inputs_cpu = self.reranker_processor(
                    text=batch_texts,
                    images=batch_images_flat,
                    return_tensors="pt",
                )
            else:
                inputs_cpu = self.reranker_processor(
                    text=batch_texts,
                    images=batch_images_flat,
                    return_tensors="pt",
                    padding=True,
                )
            _p_proc += time.time() - _t

            _t = time.time()
            inputs = inputs_cpu.to(self.reranker_model.device)
            _p_xfer += time.time() - _t

            _t = time.time()
            with torch.inference_mode():
                outputs = self.reranker_model(**inputs)
                logits = outputs.logits[:, -1, :]
                logit_yes = logits[:, self.yes_id]
                logit_no = logits[:, self.no_id]
                logits_binary = torch.stack([logit_no, logit_yes], dim=1)
                probs = torch.softmax(logits_binary, dim=1)
                batch_scores = probs[:, 1].cpu().tolist()
            _p_fwd += time.time() - _t

            del inputs, outputs, logits
            del logit_yes, logit_no, logits_binary, probs

            for score_idx, local_idx in enumerate(batch_valid_local):
                global_idx = batch_start + local_idx
                all_scores[global_idx] = batch_scores[score_idx]

        if _do_prof:
            total = _p_io + _p_tmpl + _p_proc + _p_xfer + _p_fwd
            print(
                f"\n[RERANK PROF call#{_prof_calls}] "
                f"io={_p_io*1e3:.0f}ms "
                f"template={_p_tmpl*1e3:.0f}ms "
                f"processor={_p_proc*1e3:.0f}ms "
                f"transfer={_p_xfer*1e3:.0f}ms "
                f"forward={_p_fwd*1e3:.0f}ms "
                f"total={total*1e3:.0f}ms "
                f"(B={len(candidates_info)}, "
                f"size={self.fixed_image_size})"
            )

        query_resized.close()
        return all_scores

    def run(self, query_image: Image.Image) -> Dict:
        """
        Выполняет полный пайплайн распознавания.

        PERF: reranking всех кандидатов одним батчем вместо N forward pass.

        Returns:
            {
                "status": "success" | "unknown",
                "landmark_id": str или None,
                "landmark_name": str или None,
                "confidence": float,
                "candidates": List[Dict]  — топ кандидаты с scores
            }
        """
        # 1. Retrieval
        retrieval_results = self.retriever.retrieve(
            query_image,
            top_k=self.retrieval_top_k,
            faiss_k=50,
        )

        if not retrieval_results:
            return {
                "status": "unknown",
                "landmark_id": None,
                "landmark_name": None,
                "confidence": 0.0,
                "candidates": [],
            }

        # 2. Собираем кандидатов с валидными изображениями
        valid_results = []
        candidates_info = []
        for result in retrieval_results:
            top_image = result.get_top_image()
            if not top_image:
                continue
            valid_results.append(result)
            candidates_info.append({
                "image_path": top_image.image_path,
                "name": result.landmark_name,
                "caption": top_image.caption,
            })

        if not candidates_info:
            return {
                "status": "unknown",
                "landmark_id": None,
                "landmark_name": None,
                "confidence": 0.0,
                "candidates": [],
            }

        # 3. Reranking — батч всех кандидатов за один/несколько forward pass
        rerank_scores = self._rerank_all_candidates(
            query_image, candidates_info
        )

        reranked_candidates = []
        for result, score in zip(valid_results, rerank_scores):
            reranked_candidates.append({
                "landmark_id": result.landmark_id,
                "landmark_name": result.landmark_name,
                "retrieval_score": result.aggregated_score,
                "rerank_score": score,
                "metadata": result.get_metadata(),
            })

        reranked_candidates.sort(
            key=lambda x: float(x["rerank_score"]), reverse=True
        )

        # 4. Unknown detection
        top_score = float(
            reranked_candidates[0]["rerank_score"]
        ) if reranked_candidates else 0.0
        if not reranked_candidates or top_score < self.rerank_threshold:
            return {
                "status": "unknown",
                "landmark_id": None,
                "landmark_name": None,
                "confidence": top_score,
                "candidates": reranked_candidates[:5],
            }

        best = reranked_candidates[0]
        return {
            "status": "success",
            "landmark_id": best["landmark_id"],
            "landmark_name": best["landmark_name"],
            "confidence": float(best["rerank_score"]),
            "candidates": reranked_candidates[:5],
        }


def load_test_dataset(path: str) -> List[Dict]:
    """
    Загружает тестовый датасет для E2E оценки.

    Поддерживает оба формата (новый с target_landmark_id и старый из step6).
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_e2e(
    dataset: List[Dict],
    pipeline: ProductionPipeline,
    image_dir: str,
    save_predictions: bool = False,
    warn_step6_unknown: bool = True,
) -> Tuple[Dict, List[E2EResult]]:
    """
    Выполняет E2E оценку пайплайна.

    Args:
        dataset: Список примеров (новый или старый формат)
        pipeline: ProductionPipeline
        image_dir: Директория с изображениями
        save_predictions: Сохранять ли детальные предсказания
        warn_step6_unknown: Предупреждать о некорректных unknown из step6

    Returns:
        Tuple[Dict, List[E2EResult]]: (метрики, детальные результаты)
    """
    latencies = []
    results = []

    known_correct = 0
    known_total = 0
    unknown_correct = 0
    unknown_total = 0
    mrr_sum = 0.0

    # Считаем сколько unknown-сэмплов из старого формата (step6)
    step6_unknown_count = 0

    # PROF: счётчики времени по фазам (сбрасываются каждые 50 итераций)
    _prof_retrieval = 0.0
    _prof_rerank = 0.0
    _prof_n = 0
    _PROF_INTERVAL = 50

    for item in tqdm(dataset, desc="E2E Evaluation"):
        # Извлекаем абсолютный ground truth
        true_id, is_unknown = extract_ground_truth(item)

        # Предупреждение о некорректных unknown из step6
        if (
            warn_step6_unknown
            and is_unknown
            and "target_idx" in item
            and "target_landmark_id" not in item
        ):
            step6_unknown_count += 1

        img_path = os.path.join(image_dir, item["query_image"])
        try:
            query_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ Error loading image {img_path}: {e}")
            continue

        # Запускаем пайплайн с замером времени по фазам
        _t0 = time.time()
        retrieval_results = pipeline.retriever.retrieve(
            query_img,
            top_k=pipeline.retrieval_top_k,
            faiss_k=50,
        )
        _t_retrieval = time.time() - _t0

        _t1 = time.time()
        # Собираем кандидатов и rerank
        valid_results = []
        candidates_info = []
        for r in retrieval_results:
            top_image = r.get_top_image()
            if not top_image:
                continue
            valid_results.append(r)
            candidates_info.append({
                "image_path": top_image.image_path,
                "name": r.landmark_name,
                "caption": top_image.caption,
            })

        if candidates_info:
            rerank_scores = pipeline._rerank_all_candidates(
                query_img, candidates_info
            )
            reranked = []
            for r, score in zip(valid_results, rerank_scores):
                reranked.append({
                    "landmark_id": r.landmark_id,
                    "landmark_name": r.landmark_name,
                    "retrieval_score": r.aggregated_score,
                    "rerank_score": score,
                    "metadata": r.get_metadata(),
                })
            reranked.sort(
                key=lambda x: float(x["rerank_score"]), reverse=True
            )
            top_score = float(reranked[0]["rerank_score"]) if reranked else 0.0
            if not reranked or top_score < pipeline.rerank_threshold:
                result = {
                    "status": "unknown",
                    "landmark_id": None,
                    "landmark_name": None,
                    "confidence": top_score,
                    "candidates": reranked[:5],
                }
            else:
                best = reranked[0]
                result = {
                    "status": "success",
                    "landmark_id": best["landmark_id"],
                    "landmark_name": best["landmark_name"],
                    "confidence": float(best["rerank_score"]),
                    "candidates": reranked[:5],
                }
        else:
            result = {
                "status": "unknown",
                "landmark_id": None,
                "landmark_name": None,
                "confidence": 0.0,
                "candidates": [],
            }
        _t_rerank = time.time() - _t1

        latency = (_t_retrieval + _t_rerank) * 1000
        latencies.append(latency)
        query_img.close()

        # PROF: накапливаем и выводим каждые _PROF_INTERVAL итераций
        _prof_retrieval += _t_retrieval
        _prof_rerank += _t_rerank
        _prof_n += 1
        if _prof_n % _PROF_INTERVAL == 0:
            avg_r = _prof_retrieval / _prof_n * 1000
            avg_rr = _prof_rerank / _prof_n * 1000
            tqdm.write(
                f"[PROF @{_prof_n}] "
                f"retrieval={avg_r:.0f}ms  "
                f"rerank={avg_rr:.0f}ms  "
                f"total={avg_r+avg_rr:.0f}ms/it"
            )

        pred_id = result.get("landmark_id")
        confidence = result.get("confidence", 0.0)
        is_correct = False
        retrieval_rank = -1

        if is_unknown:
            # Правильно если пайплайн вернул "unknown"
            unknown_total += 1
            if result["status"] == "unknown":
                unknown_correct += 1
                is_correct = True
        else:
            # Правильно если landmark_id совпадает с абсолютным ground truth
            known_total += 1
            if pred_id == true_id:
                known_correct += 1
                is_correct = True
                mrr_sum += 1.0
            else:
                # Ищем правильный ответ среди кандидатов pipeline
                for rank, cand in enumerate(
                    result.get("candidates", []), start=1
                ):
                    if cand.get("landmark_id") == true_id:
                        retrieval_rank = rank
                        mrr_sum += 1.0 / rank
                        break

        if save_predictions:
            results.append(E2EResult(
                query_image=item["query_image"],
                target_landmark_id=true_id,
                predicted_landmark_id=pred_id,
                confidence_score=confidence,
                latency_ms=latency,
                is_correct=is_correct,
                is_unknown=is_unknown,
                retrieval_rank=retrieval_rank,
            ))

    if warn_step6_unknown and step6_unknown_count > 0:
        print(
            f"\n⚠️  WARNING: {step6_unknown_count} unknown-сэмплов из "
            f"step6 формата (outside_topk). Эти сэмплы некорректны для "
            f"E2E unknown detection — production retrieval может найти "
            f"правильный ответ. Используй датасет с реально неизвестными "
            f"объектами для честной оценки unknown_detection_accuracy."
        )

    total = known_total + unknown_total
    correct = known_correct + unknown_correct

    metrics = {
        "e2e_accuracy": correct / total if total > 0 else 0.0,
        "e2e_hit_1": known_correct / known_total if known_total > 0 else 0.0,
        "e2e_mrr": mrr_sum / known_total if known_total > 0 else 0.0,
        "unknown_detection_accuracy": (
            unknown_correct / unknown_total if unknown_total > 0 else 0.0
        ),
        "unknown_detection_rate": (
            unknown_total / total if total > 0 else 0.0
        ),
        "p50_latency_ms": (
            float(np.percentile(latencies, 50)) if latencies else 0.0
        ),
        "p95_latency_ms": (
            float(np.percentile(latencies, 95)) if latencies else 0.0
        ),
        "p99_latency_ms": (
            float(np.percentile(latencies, 99)) if latencies else 0.0
        ),
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "throughput_qps": (
            1000.0 / np.mean(latencies) if latencies else 0.0
        ),
        "total_samples": total,
        "known_samples": known_total,
        "unknown_samples": unknown_total,
        "known_correct": known_correct,
        "unknown_correct": unknown_correct,
        # Предупреждение о некорректных unknown из step6
        "step6_unknown_samples": step6_unknown_count,
        "unknown_detection_reliable": step6_unknown_count == 0,
    }

    return metrics, results


def load_vlm_reranker(
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    lora_path: Optional[str] = None,
):
    """Загружает VLM reranker (с LoRA или без).

    PERF: fp16 вместо bfloat16 — T4 имеет нативные tensor cores для fp16,
    bfloat16 эмулируется через fp32 (~2x медленнее).
    PERF: sdpa/flash_attention_2 — memory-efficient attention.
    """
    # PERF: fp16 быстрее bfloat16 на T4/V100 (нативные tensor cores)
    _dtype = torch.float16

    # PERF: sdpa использует PyTorch scaled_dot_product_attention
    _attn_impl = "sdpa"
    try:
        import flash_attn  # noqa: F401
        _attn_impl = "flash_attention_2"
        print("✓ flash_attention_2 доступен, используем его")
    except ImportError:
        print("ℹ flash_attn не установлен, используем sdpa")

    print(f"Loading VLM model: {model_id}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=_dtype,
        attn_implementation=_attn_impl,
        device_map="auto",
    )

    if lora_path:
        print(f"Loading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

    # Процессор всегда из базовой модели (LoRA не содержит токенизатор)
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    # NOTE: torch.compile несовместим с Qwen2-VL — модель использует
    # _local_scalar_dense (.item()) внутри vision encoder для динамических
    # патчей, что inductor не может скомпилировать (graph break + fallback).
    # Оставляем eager mode.
    return model, processor


if __name__ == "__main__":
    # ============================================================
    # КОНФИГУРАЦИЯ
    # ============================================================

    # Пути к данным
    INDEX_DIR = "data/index"           # Директория с production FAISS индексом
    IMAGE_DIR = "images"               # Директория с изображениями
    TEST_DATASET = "data/processed/dataset_v1/test.json"  # Тестовый датасет
    OUTPUT_PATH = "data/eval/e2e_results.json"            # Результаты

    # Параметры модели
    LORA_PATH = None   # Путь к LoRA адаптеру или None для zero-shot
    RETRIEVAL_TOP_K = 10
    RERANK_THRESHOLD = 0.5

    SAVE_PREDICTIONS = True

    print("=" * 70)
    print("E2E EVALUATION - Production Pipeline")
    print("=" * 70)
    print(f"  Dataset:   {TEST_DATASET}")
    print(f"  LoRA:      {LORA_PATH or 'zero-shot'}")
    print(f"  Threshold: {RERANK_THRESHOLD}")
    print("=" * 70)

    # 1. Загрузка компонентов
    print("\n1. Загрузка LandmarkRetriever...")
    # PERF: передаём device="cuda" чтобы SigLIP encoder работал на GPU
    # (по умолчанию IndexConfig.device="cpu" — это основной тормоз retrieval)
    _retrieval_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Retrieval device: {_retrieval_device}")
    from src.rag.indexing_v2 import IndexConfig
    retriever = LandmarkRetriever.from_index_dir(
        INDEX_DIR,
        index_config=IndexConfig(device=_retrieval_device),
    )

    print("\n2. Загрузка VLM Reranker...")
    reranker_model, reranker_processor = load_vlm_reranker(
        lora_path=LORA_PATH
    )

    print("\n3. Инициализация ProductionPipeline...")
    pipeline = ProductionPipeline(
        retriever=retriever,
        reranker_model=reranker_model,
        reranker_processor=reranker_processor,
        image_base_dir=IMAGE_DIR,
        retrieval_top_k=RETRIEVAL_TOP_K,
        rerank_threshold=RERANK_THRESHOLD,
    )

    # 2. Загрузка датасета
    print(f"\n4. Загрузка тестового датасета: {TEST_DATASET}")
    dataset = load_test_dataset(TEST_DATASET)
    print(f"   Загружено {len(dataset)} примеров")

    # 3. Запуск оценки
    print("\n5. Запуск E2E оценки...")
    metrics, detailed_results = evaluate_e2e(
        dataset=dataset,
        pipeline=pipeline,
        image_dir=IMAGE_DIR,
        save_predictions=SAVE_PREDICTIONS,
    )

    # 4. Сохранение результатов
    print("\n6. Сохранение результатов...")
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output = {
        "config": {
            "index_dir": INDEX_DIR,
            "image_dir": IMAGE_DIR,
            "test_dataset": TEST_DATASET,
            "lora_path": LORA_PATH,
            "retrieval_top_k": RETRIEVAL_TOP_K,
            "rerank_threshold": RERANK_THRESHOLD,
        },
        "metrics": metrics,
        "predictions": (
            [r.to_dict() for r in detailed_results]
            if SAVE_PREDICTIONS else []
        ),
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 5. Вывод результатов
    print("\n" + "=" * 70)
    print("E2E EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n📊 ОСНОВНЫЕ МЕТРИКИ:")
    print(f"  E2E Accuracy:              {metrics['e2e_accuracy']:.3f}")
    print(f"  E2E Hit@1 (known):         {metrics['e2e_hit_1']:.3f}")
    print(f"  E2E MRR:                   {metrics['e2e_mrr']:.3f}")

    reliable = metrics["unknown_detection_reliable"]
    reliability_note = "" if reliable else " ⚠️  (step6 unknown — ненадёжно)"
    print(
        f"  Unknown Detection Acc:     "
        f"{metrics['unknown_detection_accuracy']:.3f}{reliability_note}"
    )
    print(
        f"  Unknown Detection Rate:    {metrics['unknown_detection_rate']:.3f}"
    )

    print(f"\n⏱️  ПРОИЗВОДИТЕЛЬНОСТЬ:")
    print(f"  P50 Latency:               {metrics['p50_latency_ms']:.1f} ms")
    print(f"  P95 Latency:               {metrics['p95_latency_ms']:.1f} ms")
    print(f"  P99 Latency:               {metrics['p99_latency_ms']:.1f} ms")
    print(f"  Mean Latency:              {metrics['mean_latency_ms']:.1f} ms")
    print(f"  Throughput:                {metrics['throughput_qps']:.2f} QPS")

    print(f"\n📈 СТАТИСТИКА:")
    print(f"  Total Samples:             {metrics['total_samples']}")
    print(f"  Known Samples:             {metrics['known_samples']}")
    print(f"  Unknown Samples:           {metrics['unknown_samples']}")
    print(f"  Known Correct:             {metrics['known_correct']}")
    print(f"  Unknown Correct:           {metrics['unknown_correct']}")
    if metrics["step6_unknown_samples"] > 0:
        print(
            f"  Step6 Unknown (unreliable):{metrics['step6_unknown_samples']}"
        )

    print(f"\n✅ Результаты сохранены: {OUTPUT_PATH}")
    print("=" * 70)
