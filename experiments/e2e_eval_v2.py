# experiments/e2e_eval.py
"""
End-to-End оценка пайплайна распознавания достопримечательностей.

Использует test.json из step6_setup_dataset.py для production-like оценки.

Ключевое отличие от наивного подхода:
- Unknown samples (target_idx=-1) в step6 — это known объекты с искусственно
  убранным правильным ответом из candidates.
- Production retrieval может найти правильный landmark через meta["landmark_id"].
- Правильная логика:
  1. Для каждого сэмпла делаем retrieval через production index
  2. Проверяем, есть ли meta["landmark_id"] в top-k retrieval
  3. Если есть → known sample, оцениваем hit@1, MRR
  4. Если нет → настоящий unknown, оцениваем unknown detection accuracy

Это даёт честную production-like оценку без искусственных unknown.
"""

import json
import os
import time
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
from tqdm import tqdm

from src.rag.landmark_retriever import LandmarkRetriever
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel


@dataclass
class E2EResult:
    query_image: str
    true_landmark_id: Optional[str]
    predicted_landmark_id: Optional[str]
    confidence_score: float
    latency_ms: float
    is_correct: bool
    sample_type: str  # "known" или "unknown"

    def to_dict(self) -> Dict:
        return {
            "query_image": self.query_image,
            "true_landmark_id": self.true_landmark_id,
            "predicted_landmark_id": self.predicted_landmark_id,
            "confidence_score": self.confidence_score,
            "latency_ms": self.latency_ms,
            "is_correct": self.is_correct,
            "sample_type": self.sample_type,
        }


class ProductionPipeline:
    def __init__(
        self,
        retriever: LandmarkRetriever,
        reranker_model,
        reranker_processor,
        image_base_dir: str,
        retrieval_top_k: int = 10,
        rerank_threshold: float = 0.5,
        caption_max_length: int = 300,
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

        _tokenizer = getattr(reranker_processor, "tokenizer", reranker_processor)
        _tokenizer.padding_side = "left"

        self.yes_id = _tokenizer.convert_tokens_to_ids("Yes")
        self.no_id = _tokenizer.convert_tokens_to_ids("No")

    def _rerank_all_candidates(
        self,
        query_image: Image.Image,
        candidates_info: List[Dict],
    ) -> List[float]:
        all_scores = [0.0] * len(candidates_info)

        if self.fixed_image_size is not None:
            query_resized = query_image.resize(
                self.fixed_image_size, Image.Resampling.BILINEAR
            )
        else:
            query_resized = query_image.copy()

        for batch_start in range(0, len(candidates_info), self.rerank_batch_size):
            batch_end = min(batch_start + self.rerank_batch_size, len(candidates_info))
            batch_info = candidates_info[batch_start:batch_end]

            batch_texts = []
            batch_images_flat = []
            batch_valid_local = []

            for local_idx, info in enumerate(batch_info):
                try:
                    cand_img = Image.open(
                        self.image_base_dir / info["image_path"]
                    ).convert("RGB")
                    
                    if self.fixed_image_size is not None:
                        cand_img = cand_img.resize(
                            self.fixed_image_size, Image.Resampling.BILINEAR
                        )

                    caption = info["caption"][:self.caption_max_length]
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
                                        f"Question: Are these photos showing"
                                        f" the same landmark: \"{info['name']}\"?\n"
                                        f"Candidate details: {caption}\n"
                                        f"Answer only with Yes or No."
                                    ),
                                },
                            ],
                        }
                    ]
                    text = self.reranker_processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    batch_texts.append(text)
                    batch_images_flat.append(query_resized.copy())
                    batch_images_flat.append(cand_img.copy())
                    cand_img.close()
                    batch_valid_local.append(local_idx)
                except Exception as e:
                    print(f"⚠️ Error loading {info.get('image_path')}: {e}")

            if not batch_texts:
                continue

            inputs = self.reranker_processor(
                text=batch_texts,
                images=batch_images_flat,
                return_tensors="pt",
                padding=True if len(batch_texts) > 1 else False,
            ).to(self.reranker_model.device)

            with torch.inference_mode():
                outputs = self.reranker_model(**inputs)
                logits = outputs.logits[:, -1, :]
                logit_yes = logits[:, self.yes_id]
                logit_no = logits[:, self.no_id]
                logits_binary = torch.stack([logit_no, logit_yes], dim=1)
                probs = torch.softmax(logits_binary, dim=1)
                batch_scores = probs[:, 1].cpu().tolist()

            for score_idx, local_idx in enumerate(batch_valid_local):
                global_idx = batch_start + local_idx
                all_scores[global_idx] = batch_scores[score_idx]

        query_resized.close()
        return all_scores

    def run(self, query_image: Image.Image) -> Dict:
        retrieval_results = self.retriever.retrieve(
            query_image,
            top_k=self.retrieval_top_k,
            faiss_k=50,
        )

        if not retrieval_results:
            return {
                "status": "unknown",
                "landmark_id": None,
                "confidence": 0.0,
                "candidates": [],
            }

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
                "confidence": 0.0,
                "candidates": [],
            }

        rerank_scores = self._rerank_all_candidates(query_image, candidates_info)

        reranked_candidates = []
        for result, score in zip(valid_results, rerank_scores):
            reranked_candidates.append({
                "landmark_id": result.landmark_id,
                "landmark_name": result.landmark_name,
                "rerank_score": score,
            })

        reranked_candidates.sort(
            key=lambda x: float(x["rerank_score"]), reverse=True
        )

        top_score = float(reranked_candidates[0]["rerank_score"]) if reranked_candidates else 0.0
        
        if not reranked_candidates or top_score < self.rerank_threshold:
            return {
                "status": "unknown",
                "landmark_id": None,
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


def evaluate_e2e(
    dataset: List[Dict],
    pipeline: ProductionPipeline,
    image_dir: str,
    save_predictions: bool = False,
) -> Tuple[Dict, List[E2EResult]]:
    """
    Production-like оценка с правильной логикой unknown detection.
    
    Для каждого сэмпла:
    1. Берём ground truth landmark_id из meta["landmark_id"]
    2. Делаем retrieval через production index
    3. Проверяем, есть ли ground truth в top-k retrieval
    4. Если есть → known sample, оцениваем hit@1, MRR
    5. Если нет → unknown sample, оцениваем unknown detection accuracy
    """
    latencies = []
    results = []

    known_correct = 0
    known_total = 0
    unknown_correct = 0
    unknown_total = 0
    mrr_sum = 0.0

    # Статистика: сколько сэмплов из test.json оказались known/unknown после retrieval
    step6_unknown_became_known = 0  # target_idx=-1, но retrieval нашёл правильный
    step6_known_stayed_known = 0    # target_idx!=-1, и retrieval нашёл правильный
    step6_known_became_unknown = 0  # target_idx!=-1, но retrieval НЕ нашёл правильный

    for item in tqdm(dataset, desc="E2E Evaluation"):
        # Ground truth landmark_id всегда в meta["landmark_id"]
        meta = item.get("meta", {})
        true_landmark_id = meta.get("landmark_id")
        
        if not true_landmark_id:
            print(f"⚠️  No landmark_id in meta for {item.get('query_image')}")
            continue

        target_idx = item.get("target_idx", -1)

        img_path = os.path.join(image_dir, item["query_image"])
        try:
            query_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ Error loading {img_path}: {e}")
            continue

        # Запускаем пайплайн (включает retrieval + reranking)
        start_time = time.time()
        result = pipeline.run(query_img)
        latency = (time.time() - start_time) * 1000
        latencies.append(latency)
        query_img.close()

        # Проверяем, есть ли true_landmark_id в retrieval results
        retrieved_landmark_ids = [
            cand.get("landmark_id") for cand in result.get("candidates", [])
        ]
        is_in_retrieval = true_landmark_id in retrieved_landmark_ids

        # Определяем тип сэмпла на основе retrieval (не target_idx!)
        if is_in_retrieval:
            # Known sample: retrieval нашёл правильный landmark
            sample_type = "known"
            known_total += 1
            
            # Статистика
            if target_idx == -1:
                step6_unknown_became_known += 1
            else:
                step6_known_stayed_known += 1
            
            # Оцениваем hit@1, MRR
            pred_id = result.get("landmark_id")
            confidence = result.get("confidence", 0.0)
            is_correct = False
            
            if pred_id == true_landmark_id:
                known_correct += 1
                is_correct = True
                mrr_sum += 1.0
            else:
                # Ищем правильный ответ среди кандидатов
                for rank, cand in enumerate(result.get("candidates", []), start=1):
                    if cand.get("landmark_id") == true_landmark_id:
                        mrr_sum += 1.0 / rank
                        break
        else:
            # Unknown sample: retrieval НЕ нашёл правильный landmark
            sample_type = "unknown"
            unknown_total += 1
            
            # Статистика
            if target_idx != -1:
                step6_known_became_unknown += 1
            
            # Оцениваем unknown detection accuracy
            confidence = result.get("confidence", 0.0)
            is_correct = False
            
            if result["status"] == "unknown":
                unknown_correct += 1
                is_correct = True

        if save_predictions:
            results.append(E2EResult(
                query_image=item["query_image"],
                true_landmark_id=true_landmark_id,
                predicted_landmark_id=result.get("landmark_id"),
                confidence_score=confidence,
                latency_ms=latency,
                is_correct=is_correct,
                sample_type=sample_type,
            ))

    total = known_total + unknown_total
    correct = known_correct + unknown_correct

    metrics = {
        "e2e_accuracy": correct / total if total > 0 else 0.0,
        "e2e_hit_1": known_correct / known_total if known_total > 0 else 0.0,
        "e2e_mrr": mrr_sum / known_total if known_total > 0 else 0.0,
        "unknown_detection_accuracy": (
            unknown_correct / unknown_total if unknown_total > 0 else 0.0
        ),
        "unknown_detection_rate": unknown_total / total if total > 0 else 0.0,
        "p50_latency_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "total_samples": total,
        "known_samples": known_total,
        "unknown_samples": unknown_total,
        "known_correct": known_correct,
        "unknown_correct": unknown_correct,
        # Статистика трансформации сэмплов
        "step6_unknown_became_known": step6_unknown_became_known,
        "step6_known_stayed_known": step6_known_stayed_known,
        "step6_known_became_unknown": step6_known_became_unknown,
    }

    return metrics, results


def load_vlm_reranker(
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    lora_path: Optional[str] = None,
):
    print(f"Loading VLM model: {model_id}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if lora_path:
        print(f"Loading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model.eval()
    
    return model, processor


if __name__ == "__main__":
    # КОНФИГУРАЦИЯ
    INDEX_DIR = "data/processed"
    IMAGE_DIR = "images"
    TEST_DATASET = "data/processed/dataset_v1/test.json"
    OUTPUT_PATH = "data/eval/e2e_results.json"

    LORA_PATH = "experiments/results/val_rerank_exp_r16_alpha32_lr2e-5_rerank_full_lora_448"
    RETRIEVAL_TOP_K = 10
    RERANK_THRESHOLD = 0.50

    SAVE_PREDICTIONS = True

    print("=" * 70)
    print("E2E EVALUATION - Production Pipeline")
    print("=" * 70)
    print(f"  Dataset:   {TEST_DATASET}")
    print(f"  LoRA:      {LORA_PATH or 'zero-shot'}")
    print(f"  Threshold: {RERANK_THRESHOLD}")
    print("=" * 70)

    # Загрузка компонентов
    print("\n1. Загрузка LandmarkRetriever...")
    from src.rag.indexing_v2 import IndexConfig
    retriever = LandmarkRetriever.from_index_dir(
        INDEX_DIR,
        index_config=IndexConfig(device="cuda" if torch.cuda.is_available() else "cpu"),
    )

    print("\n2. Загрузка VLM Reranker...")
    reranker_model, reranker_processor = load_vlm_reranker(lora_path=LORA_PATH)

    print("\n3. Инициализация ProductionPipeline...")
    pipeline = ProductionPipeline(
        retriever=retriever,
        reranker_model=reranker_model,
        reranker_processor=reranker_processor,
        image_base_dir=IMAGE_DIR,
        retrieval_top_k=RETRIEVAL_TOP_K,
        rerank_threshold=RERANK_THRESHOLD,
    )

    # Загрузка датасета
    print(f"\n4. Загрузка тестового датасета: {TEST_DATASET}")
    with open(TEST_DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"   Загружено {len(dataset)} примеров")

    # Запуск оценки
    print("\n5. Запуск E2E оценки...")
    metrics, detailed_results = evaluate_e2e(
        dataset=dataset,
        pipeline=pipeline,
        image_dir=IMAGE_DIR,
        save_predictions=SAVE_PREDICTIONS,
    )

    # Сохранение результатов
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
        "predictions": [r.to_dict() for r in detailed_results] if SAVE_PREDICTIONS else [],
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Вывод результатов
    print("\n" + "=" * 70)
    print("E2E EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n📊 ОСНОВНЫЕ МЕТРИКИ:")
    print(f"  E2E Accuracy:              {metrics['e2e_accuracy']:.3f}")
    print(f"  E2E Hit@1 (known):         {metrics['e2e_hit_1']:.3f}")
    print(f"  E2E MRR:                   {metrics['e2e_mrr']:.3f}")
    print(f"  Unknown Detection Acc:     {metrics['unknown_detection_accuracy']:.3f}")
    print(f"  Unknown Detection Rate:    {metrics['unknown_detection_rate']:.3f}")

    print(f"\n⏱️  ПРОИЗВОДИТЕЛЬНОСТЬ:")
    print(f"  P50 Latency:               {metrics['p50_latency_ms']:.1f} ms")
    print(f"  P95 Latency:               {metrics['p95_latency_ms']:.1f} ms")
    print(f"  Mean Latency:              {metrics['mean_latency_ms']:.1f} ms")

    print(f"\n📈 СТАТИСТИКА:")
    print(f"  Total Samples:             {metrics['total_samples']}")
    print(f"  Known Samples:             {metrics['known_samples']}")
    print(f"  Unknown Samples:           {metrics['unknown_samples']}")
    print(f"  Known Correct:             {metrics['known_correct']}")
    print(f"  Unknown Correct:           {metrics['unknown_correct']}")
    
    print(f"\n🔄 ТРАНСФОРМАЦИЯ СЭМПЛОВ (step6 → retrieval):")
    print(f"  Step6 unknown → Known:     {metrics['step6_unknown_became_known']}")
    print(f"  Step6 known → Known:       {metrics['step6_known_stayed_known']}")
    print(f"  Step6 known → Unknown:     {metrics['step6_known_became_unknown']}")

    print(f"\n✅ Результаты сохранены: {OUTPUT_PATH}")
    print("=" * 70)