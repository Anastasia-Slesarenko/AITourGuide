"""
Скрипт обучения Qwen2-VL reranking-модели с LoRA fine-tuning.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="peft")

import os
import json
import random
import torch
import gc
import numpy as np
from PIL import Image
from typing import Dict, Tuple, List
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model
import mlflow
from torch.utils.data import Dataset
from tqdm import tqdm
from plotting import TrainingPlotter

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ----------------------------
# Параметры
MLFLOW_AVAILABLE = True
TRAIN_DATASET_FILE = "data/processed/dataset_v1/train.json"
VAL_DATASET_FILE = "data/processed/dataset_v1/val.json"
IMAGE_DIR = "images" 
OUTPUT_BASE_DIR = "checkpoints/qwen2vl-rerank-lora"

EXPERIMENT_NAME = "Qwen2-VL-Rerank-Finetuning"


def build_rerank_prompt(
        query_img_path: str, 
        candidate: dict, 
        image_base_dir: str
) -> Tuple[Image.Image, Image.Image, List] | Tuple[None, None, None]:
    """
    Формирует промпт для reranking пары (query, candidate).
    Использует тот же формат что и в eval.py с двумя изображениями.
    
    Returns:
        tuple: (query_image, candidate_image, messages) или (None, None, None) при ошибке
    """
    try:

        with Image.open(
            os.path.join(image_base_dir, query_img_path)
        ) as _img:
            query_image = _img.convert("RGB")
        with Image.open(
            os.path.join(image_base_dir, candidate["image"])
        ) as _img:
            candidate_image = _img.convert("RGB")

        caption = candidate["caption"][:300]  # truncate

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
                        "text": f"Question: Are these photos showing the same landmark: \"{candidate['name']}\"?\n"
                                f"Candidate details: {caption}\n"
                                f"Answer only with Yes or No."
                    }
                ]
            }
        ]

        return query_image, candidate_image, messages
    except FileNotFoundError as e:
        print(f"Файл изображения не найден: {e}")
        return None, None, None
    except Exception as e:
        print(f"Ошибка загрузки изображений: {e}")
        return None, None, None


def make_collate_fn(
    processor,
    image_base_dir: str,
    fixed_image_size: tuple = (448, 448),
    max_pairs_per_sample: int = 4,
):
    """
    Фабрика collate_fn: замыкает processor и image_base_dir.
    Использует messages API с двумя изображениями (как в eval.py).
    """
    # Decoder-only модели требуют left-padding при батчинге.
    # Устанавливаем один раз перед созданием DataLoader.
    processor.tokenizer.padding_side = "left"

    def _collate(batch):
        # batch: [{"query_image": str, "candidates": [...], "target_idx": int}, ...]
        # Разворачиваем в список пар (query_img, candidate, label)
        flat_batch = []
        for sample in batch:
            query_img_path = sample["query_image"]
            candidates = sample["candidates"]
            target_idx = sample["target_idx"]

            # сэмплируем 1 позитив + (max_pairs_per_sample-1) негативов
            # вместо всех ~15 кандидатов. Ускоряет шаг в ~15/max_pairs раз.
            positives = []
            negatives = []
            for i, cand in enumerate(candidates):
                if i == target_idx:
                    positives.append((i, cand))
                else:
                    negatives.append((i, cand))

            # Всегда берём позитив (если есть), остальное — случайные негативы
            selected = list(positives)
            n_neg = max_pairs_per_sample - len(selected)
            if negatives and n_neg > 0:
                selected_negs = random.sample(negatives, min(n_neg, len(negatives)))
                selected.extend(selected_negs)
            # Перемешиваем чтобы позитив не всегда был первым
            random.shuffle(selected)

            # FIX: вес позитива = n_neg (реальный дисбаланс в батче).
            # При max_pairs=12: 1 позитив + 11 негативов → weight_pos=11.
            # Это гарантирует, что суммарный градиент от позитива ≥ градиенту
            # от всех негативов вместе взятых, и модель не коллапсирует
            # к тривиальному решению "всё — No".
            n_neg_selected = len(selected) - len(positives)
            weight_pos = float(max(n_neg_selected, 1))

            for i, cand in selected:
                label = 1 if i == target_idx else 0
                if label == 1:  # позитив: вес пропорционален числу негативов
                    weight = weight_pos
                else:  # негативы
                    weight = 1.0
                    cand_type = cand.get("candidate_type", "easy")
                    if cand_type == "hard":
                        weight = 1.2
                    elif cand_type == "semi_hard":
                        weight = 1.1
                flat_batch.append({
                    "query_image": query_img_path,
                    "candidate": cand,
                    "label": label,
                    "weight": weight
                })

        texts = []
        # processor(images=...) принимает плоский список PIL-объектов,
        # а не список пар. Разворачиваем пары в flat_images как в eval.py
        all_images_grouped = []
        labels = []
        weights = []

        for item in flat_batch:
            query_img, cand_img, messages = build_rerank_prompt(
                item["query_image"], item["candidate"], image_base_dir
            )
            # Пропускаем пары с ошибками загрузки
            if query_img is None or cand_img is None or messages is None:
                continue

            # Применяем chat template
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # batch_size>1: ресайз до фиксированного разрешения гарантирует
            # одинаковое количество патчей для всех примеров в батче.
            # Без ресайза pixel_values — плоский тензор с разным числом патчей
            # на пример, что нарушает маппинг <image> токенов к патчам.
            if fixed_image_size is not None:
               
                query_img = query_img.resize(
                    fixed_image_size, Image.Resampling.BILINEAR
                )
                cand_img = cand_img.resize(
                    fixed_image_size, Image.Resampling.BILINEAR
                )

            texts.append(text)
            # копируем изображения чтобы избежать in-place
            # мутации процессором (resize/normalize) при batch_size>1
            all_images_grouped.append([query_img.copy(), cand_img.copy()])
            labels.append(item["label"])
            weights.append(item["weight"])

        # Если все изображения не загрузились — возвращаем пустой валидный батч.
        # None вызывает краш в Trainer.
        if not texts:
            return {
                "input_ids": torch.zeros((0, 1), dtype=torch.long),
                "attention_mask": torch.zeros((0, 1), dtype=torch.long),
                # bfloat16 — совпадает с dtype модели, иначе RuntimeError
                "pixel_values": torch.zeros(
                    (0, 3, 1, 1), dtype=torch.bfloat16
                ),
                "image_grid_thw": torch.zeros((0, 3), dtype=torch.long),
                "labels": torch.zeros(0, dtype=torch.long),
                "weights": torch.zeros(0, dtype=torch.float),
            }

        # разворачиваем пары в плоский список как в eval.py
        # [q1,c1,q2,c2,...] — количество совпадает с числом <image> токенов
        flat_images = [img for pair in all_images_grouped for img in pair]

        inputs = processor(
            text=texts,
            images=flat_images,
            return_tensors="pt",
            padding=True,
        )

        # явно закрываем PIL Images после токенизации
        for pair in all_images_grouped:
            for img in pair:
                img.close()

        # Labels для вычисления loss на Yes/No
        labels_tensor = torch.tensor(labels, dtype=torch.long)  # 0 or 1
        weights_tensor = torch.tensor(weights, dtype=torch.float)

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
            "labels": labels_tensor,
            "weights": weights_tensor,
        }
    return _collate


class RerankDataset(Dataset):
    def __init__(self, data_path: str, image_base_dir: str):
        self.data_path = data_path
        self.image_base_dir = image_base_dir
        
        # Валидация путей
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        if not os.path.exists(image_base_dir):
            raise FileNotFoundError(f"Image directory not found: {image_base_dir}")
        
        # Загрузка данных
        with open(data_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        
        # Валидация структуры данных
        if not self.samples:
            raise ValueError(f"Dataset is empty: {data_path}")
        
        # Проверка первого сэмпла на корректность структуры
        required_keys = {"query_image", "candidates", "target_idx"}
        if not required_keys.issubset(self.samples[0].keys()):
            missing = required_keys - set(self.samples[0].keys())
            raise ValueError(f"Missing required keys in dataset: {missing}")
        
        print(f"Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class RerankTrainer(Trainer):
    """
    Кастомный Trainer для Logit-based BCE Loss с label smoothing.

    Логиты берутся из outputs.logits[:, -1, :] — предсказание следующего
    токена после последнего токена промпта. Это корректно для teacher-forcing
    обучения decoder-only модели: модель видит весь промпт и предсказывает
    первый токен ответа (Yes/No).

    Для eval используется model(**inputs).logits[:, -1, :] — прямой forward
    pass без overhead планировщика generate(). Оба метода (train и eval)
    берут логиты одного токена — согласованный подход.
    """
    def __init__(
        self, *args, processor=None, metrics_callback=None,
        label_smoothing=0.0, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.metrics_callback = metrics_callback
        self.label_smoothing = label_smoothing

        # используем convert_tokens_to_ids как в eval.py FIX #16 —
        # возвращает ровно один id без BPE-контекста. encode() может добавлять
        # пробел перед токеном в зависимости от позиции (BPE-сплит).
        _tokenizer = getattr(processor, "tokenizer", processor)
        yes_id = _tokenizer.convert_tokens_to_ids("Yes")
        no_id = _tokenizer.convert_tokens_to_ids("No")
        # Запасной вариант: encode() если convert_tokens_to_ids вернул UNK
        _unk_id = getattr(_tokenizer, "unk_token_id", None)
        if yes_id == _unk_id or no_id == _unk_id:
            yes_id = _tokenizer.encode("Yes", add_special_tokens=False)[0]
            no_id = _tokenizer.encode("No", add_special_tokens=False)[0]

        # Валидация: убеждаемся, что токены найдены
        yes_tokens_check = _tokenizer.encode("Yes", add_special_tokens=False)
        no_tokens_check = _tokenizer.encode("No", add_special_tokens=False)
        if len(yes_tokens_check) != 1 or len(no_tokens_check) != 1:
            raise ValueError(
                f"Yes/No tokenization produced multiple tokens: "
                f"Yes={yes_tokens_check}, No={no_tokens_check}"
            )

        self.yes_id = yes_id
        self.no_id = no_id
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")    # (B,) 0 or 1
        weights = inputs.pop("weights")  # (B,)

        # используем model(**inputs) для forward pass с градиентами.
        # Берём логиты последнего токена промпта (позиция -1 при left-padding).
        # Это согласовано с eval.py: оба метода получают логиты одного токена —
        # первого сгенерированного. При left-padding позиция -1 всегда является
        # последним токеном промпта (перед генерацией).
        # Примечание: в отличие от eval.py (generate + output_logits), здесь
        # используем forward pass для поддержки backprop через LoRA-параметры.
        outputs = model(**inputs)
        # [B, T, V] → логиты на позиции -1 = предсказание первого токена ответа
        logits_last = outputs.logits[:, -1, :]

        logit_yes = logits_last[:, self.yes_id]  # [B]
        logit_no = logits_last[:, self.no_id]    # [B]

        # BCEWithLogitsLoss ожидает сырые логиты, а не вероятности.
        # Разность logit_yes - logit_no эквивалентна log-odds P(yes)/P(no)
        # и является численно устойчивым логитом для бинарной классификации.
        relevance_logits = logit_yes - logit_no  # [B]

        # применяем label smoothing к целевым меткам
        targets = labels.float()
        if self.label_smoothing > 0.0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        loss_unreduced = self.bce(relevance_logits, targets)  # [B]

        # стандартная взвешенная BCE — нормализуем веса так,
        # чтобы их среднее было 1.0 (эквивалентно делению на mean(weights))
        weights_norm = weights / (weights.mean() + 1e-8)
        loss = (loss_unreduced * weights_norm).mean()

        # Логируем train loss в callback для графиков.
        # MLflow-логирование вынесено из compute_loss (вызывается на каждом шаге)
        # в on_log callback, чтобы не блокировать горячий путь обучения.
        if (
            self.metrics_callback is not None
            and self.state.global_step % self.args.logging_steps == 0
        ):
            loss_value = loss.item()
            self.metrics_callback.train_loss_steps.append(
                self.state.global_step
            )
            self.metrics_callback.train_loss_values.append(loss_value)
            # MLflow: логируем асинхронно через on_log, не здесь

        return (loss, outputs) if return_outputs else loss


class MetricsCallback(TrainerCallback):
    def __init__(
        self, val_dataset, processor, model, output_dir,
        eval_every_n_steps=50, experiment_name="experiment",
        early_stopping_patience=5,
        fixed_image_size: tuple = (448, 448),
        none_threshold: float = 0.5,
    ):
        self.val_dataset = val_dataset
        self.processor = processor
        self.model = model
        self.eval_every_n_steps = eval_every_n_steps
        self.output_dir = output_dir
        # храним patience как атрибут, а не читаем из args
        self.patience = early_stopping_patience
        # batch_size>1: фиксированный размер для eval батчинга
        self.fixed_image_size = fixed_image_size
        # none_threshold синхронизирован с eval.py threshold_for_none
        self.none_threshold = none_threshold
        # используем convert_tokens_to_ids как в eval.py 
        _tokenizer = getattr(processor, "tokenizer", processor)
        _unk_id = getattr(_tokenizer, "unk_token_id", None)
        yes_id = _tokenizer.convert_tokens_to_ids("Yes")
        no_id = _tokenizer.convert_tokens_to_ids("No")
        if yes_id == _unk_id or no_id == _unk_id:
            yes_id = _tokenizer.encode("Yes", add_special_tokens=False)[0]
            no_id = _tokenizer.encode("No", add_special_tokens=False)[0]
        self.yes_id = yes_id
        self.no_id = no_id

        self.eval_steps_list: list = []
        self.metrics_history: Dict[str, list] = {
            "eval_loss": [],
            "eval_hit_1": [],
            "eval_mrr": [],
            "eval_none_accuracy": [],
            "eval_hard_accuracy": [],
            "eval_easy_accuracy": [],
            "eval_semi_hard_accuracy": [],
        }
        self.best_eval_loss = float('inf')
        # Инициализация для early stopping
        self.best_primary_metric = float('-inf')
        self.es_counter = 0
        
        # Инициализация plotter для графиков
        self.plotter = TrainingPlotter(output_dir, experiment_name)
        self.train_loss_steps = []
        self.train_loss_values = []

    def evaluate_on_subset(
        self,
        subset_size=50,
        none_threshold=0.5,
        fixed_image_size: tuple = (448, 448),
    ):
        """
        Оценка модели на подмножестве валидационных данных.
        Использует stratified sampling для сбалансированной оценки.
        """
        self.model.eval()

        # Stratified sampling: разделяем на valid и none queries
        valid_indices = []
        none_indices = []
        for idx in range(len(self.val_dataset)):
            sample = self.val_dataset[idx]
            if sample["target_idx"] != -1:
                valid_indices.append(idx)
            else:
                none_indices.append(idx)

        # Пропорциональный отбор
        n_valid = min(int(subset_size * 0.8), len(valid_indices))
        n_none = min(subset_size - n_valid, len(none_indices))

        subset_indices = []
        if valid_indices:
            subset_indices.extend(random.sample(valid_indices, n_valid))
        if none_indices and n_none > 0:
            subset_indices.extend(random.sample(none_indices, n_none))

        random.shuffle(subset_indices)

        hits_at_1 = 0
        mrr_sum = 0.0
        total_valid_queries = 0
        none_correct = 0
        total_none_queries = 0
        # FIX: метрики по hardness негативов в батче, а не по типу позитива.
        # "hard" случай = в кандидатах есть хотя бы один hard-негатив (score>=0.85)
        # "semi_hard" случай = нет hard, но есть semi_hard негатив
        # "easy" случай = все негативы easy (score<0.75) — лёгкий для reranker
        hard_correct = 0    # hit@1 когда есть hard-негативы
        easy_correct = 0    # hit@1 когда все негативы easy
        semi_hard_correct = 0  # hit@1 когда нет hard, но есть semi_hard
        hard_total = 0
        easy_total = 0
        semi_hard_total = 0
        eval_loss_sum = 0.0
        eval_loss_count = 0

        # PERF: device вычисляем один раз вне цикла
        device = next(self.model.parameters()).device
        # PERF: увеличен batch_size для eval — меньше forward pass-ов,
        # лучше утилизация GPU. T4 при inference_mode тянет 8 пар (16 изображений).
        eval_batch_size = 8

        # FIX #16: inference_mode быстрее и безопаснее no_grad (как в eval.py)
        with torch.inference_mode():
            pbar = tqdm(subset_indices, desc="Evaluating", leave=False)
            for idx in pbar:
                sample = self.val_dataset[idx]
                query_img_path = sample["query_image"]
                candidates = sample["candidates"]
                target_idx = sample["target_idx"]

                scores = []

                for batch_start in range(0, len(candidates), eval_batch_size):
                    batch_end = min(
                        batch_start + eval_batch_size, len(candidates)
                    )
                    batch_candidates = candidates[batch_start:batch_end]

                    batch_texts = []
                    # FIX #6: список пар изображений (как в eval.py)
                    batch_images_grouped = []
                    # FIX #14: отслеживаем валидные индексы для корректного
                    # маппинга скоров обратно на позиции кандидатов
                    batch_valid_indices = []

                    for cand_idx, cand in enumerate(batch_candidates):
                        query_img, cand_img, messages = build_rerank_prompt(
                            query_img_path,
                            cand,
                            self.val_dataset.image_base_dir
                        )
                        if query_img is None:
                            continue

                        # FIX batch_size>1: ресайз до фиксированного
                        # разрешения — одинаковое количество патчей в батче.
                        # PERF: BILINEAR быстрее LANCZOS при незначительной
                        # потере качества для eval (не для финального теста).
                        if fixed_image_size is not None:
                            query_img = query_img.resize(
                                fixed_image_size, Image.Resampling.BILINEAR
                            )
                            cand_img = cand_img.resize(
                                fixed_image_size, Image.Resampling.BILINEAR
                            )

                        text = self.processor.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        batch_texts.append(text)
                        # FIX #6: пара [query, candidate] как в eval.py
                        # FIX #17: копируем чтобы избежать in-place мутации
                        batch_images_grouped.append(
                            [query_img.copy(), cand_img.copy()]
                        )
                        batch_valid_indices.append(cand_idx)

                    if not batch_texts:
                        # Все кандидаты не загрузились — заполняем нулями
                        scores.extend([0.0] * len(batch_candidates))
                        continue

                    # FIX #3: разворачиваем пары в плоский список как в eval.py FIX #8
                    flat_batch_images = [
                        img
                        for pair in batch_images_grouped
                        for img in pair
                    ]

                    inputs = self.processor(
                        text=batch_texts,
                        images=flat_batch_images,
                        return_tensors="pt",
                        padding=True,
                    )

                    # FIX #13: закрываем PIL Images ПОСЛЕ передачи в процессор
                    for pair in batch_images_grouped:
                        for img in pair:
                            img.close()

                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    # PERF: прямой forward pass вместо generate(max_new_tokens=1).
                    # Синхронизирован с eval.py: outputs.logits[:, -1, :] —
                    # логиты последнего токена промпта = P(next token | prompt),
                    # что эквивалентно первому токену generate() при greedy.
                    # При left-padding позиция -1 всегда является последним
                    # реальным токеном (не pad-токеном).
                    outputs = self.model(**inputs)
                    logits = outputs.logits[:, -1, :]  # [B, V]

                    logit_yes = logits[:, self.yes_id]
                    logit_no = logits[:, self.no_id]

                    logits_binary = torch.stack([logit_no, logit_yes], dim=1)
                    probs = torch.softmax(logits_binary, dim=1)
                    batch_scores = probs[:, 1].cpu().tolist()

                    # Вычисляем eval_loss для текущего батча
                    # Метки: для каждого кандидата в батче определяем label
                    # (здесь нет прямого доступа к labels, поэтому считаем BCE
                    # по relevance_logits = logit_yes - logit_no)

                    # FIX #14: маппим скоры обратно на исходные индексы
                    # кандидатов (как в eval.py строки 212-218)
                    score_ptr = 0
                    for ci in range(len(batch_candidates)):
                        if ci in batch_valid_indices:
                            scores.append(batch_scores[score_ptr])
                            score_ptr += 1
                        else:
                            scores.append(0.0)

                    del inputs, outputs, logits
                    del logit_yes, logit_no, logits_binary, probs

                # Вычисляем eval_loss для текущего запроса
                # BCE loss: label=1 для target_idx, label=0 для остальных
                if scores:
                    scores_tensor = torch.tensor(scores, dtype=torch.float32)
                    if target_idx != -1:
                        labels_for_loss = torch.zeros(len(scores), dtype=torch.float32)
                        if target_idx < len(scores):
                            labels_for_loss[target_idx] = 1.0
                    else:
                        labels_for_loss = torch.zeros(len(scores), dtype=torch.float32)
                    # Переводим вероятности обратно в логиты для BCE
                    # scores — это P(yes), логит = log(p/(1-p))
                    eps = 1e-7
                    scores_clamped = scores_tensor.clamp(eps, 1.0 - eps)
                    logits_for_loss = torch.log(scores_clamped / (1.0 - scores_clamped))
                    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits_for_loss, labels_for_loss, reduction='mean'
                    )
                    eval_loss_sum += bce_loss.item()
                    eval_loss_count += 1

                scores_arr = np.array(scores)
                ranked_indices = np.argsort(scores_arr)[::-1]

                if target_idx != -1:
                    total_valid_queries += 1
                    if ranked_indices[0] == target_idx:
                        hits_at_1 += 1
                    rank_of_target = (
                        np.where(ranked_indices == target_idx)[0][0] + 1
                    )
                    mrr_sum += 1.0 / rank_of_target

                    # Определяем сложность запроса по максимальному типу
                    # негативов в списке кандидатов:
                    # "hard"      = есть хотя бы один hard-негатив (score>=0.85)
                    # "semi_hard" = нет hard, но есть semi_hard (0.75-0.85)
                    # "easy"      = все негативы easy (score<0.75)
                    # Это показывает насколько reranker справляется со
                    # сложными (визуально похожими) негативами.
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

                    hit = (ranked_indices[0] == target_idx)
                    if query_difficulty == "hard":
                        hard_total += 1
                        if hit:
                            hard_correct += 1
                    elif query_difficulty == "semi_hard":
                        semi_hard_total += 1
                        if hit:
                            semi_hard_correct += 1
                    else:
                        easy_total += 1
                        if hit:
                            easy_correct += 1
                else:
                    total_none_queries += 1
                    # FIX #2: порог 0.5 (вероятности из softmax всегда [0,1])
                    if len(scores) == 0 or float(np.max(scores_arr)) < none_threshold:
                        none_correct += 1

        # FIX #8: возвращаем модель в train() режим после оценки.
        self.model.train()

        eval_hit_1 = (
            hits_at_1 / total_valid_queries if total_valid_queries > 0 else 0.0
        )
        eval_mrr = (
            mrr_sum / total_valid_queries if total_valid_queries > 0 else 0.0
        )
        eval_none_acc = (
            none_correct / total_none_queries
            if total_none_queries > 0 else 0.0
        )
        eval_hard_acc = (
            hard_correct / hard_total if hard_total > 0 else 0.0
        )
        eval_easy_acc = (
            easy_correct / easy_total if easy_total > 0 else 0.0
        )
        eval_semi_hard_acc = (
            semi_hard_correct / semi_hard_total
            if semi_hard_total > 0 else 0.0
        )

        eval_loss = (
            eval_loss_sum / eval_loss_count if eval_loss_count > 0 else 0.0
        )

        return {
            "eval_loss": eval_loss,
            "eval_hit_1": eval_hit_1,
            "eval_mrr": eval_mrr,
            "eval_none_accuracy": eval_none_acc,
            "eval_hard_accuracy": eval_hard_acc,
            "eval_easy_accuracy": eval_easy_acc,
            "eval_semi_hard_accuracy": eval_semi_hard_acc,
        }


    def on_log(self, args, state, control, logs=None, **kwargs):
        """Логируем train_loss в MLflow через on_log — вне горячего пути."""
        if not MLFLOW_AVAILABLE or logs is None:
            return
        train_loss = logs.get("loss")
        if train_loss is not None:
            try:
                mlflow.log_metric(
                    "train_loss", train_loss,
                    step=state.global_step
                )
            except Exception as e:
                print(f"MLflow train_loss logging error: {e}")

    def on_step_end(self, args, state, control, **kwargs):
        # global_step=0 срабатывает до первого обновления весов — пропускаем.
        if state.global_step == 0:
            return
        if state.global_step % self.eval_every_n_steps != 0:
            return

        print(f"\nВычисление метрик на шаге {state.global_step}...")

        # FIX #2: передаём self.none_threshold для синхронизации с eval.py
        # FIX: subset_size=100 вместо 50 — уменьшает дисперсию метрик.
        # При 50 сэмплах Easy/Semi-Hard могут содержать 1-3 примера,
        # что даёт скачки 0→1→0. 100 сэмплов стабилизирует оценку.
        metrics = self.evaluate_on_subset(
            subset_size=100,
            none_threshold=self.none_threshold,
            fixed_image_size=self.fixed_image_size,
        )

        # Вывод в консоль
        print(f"Eval Loss: {metrics['eval_loss']:.4f}")
        print(f"Eval Hit_1: {metrics['eval_hit_1']:.3f}")
        print(f"Eval MRR: {metrics['eval_mrr']:.3f}")
        print(f"Eval None Accuracy: {metrics['eval_none_accuracy']:.3f}")
        print(f"Eval Hard Acc: {metrics['eval_hard_accuracy']:.3f}")
        print(f"Eval Easy Acc: {metrics['eval_easy_accuracy']:.3f}")
        print(f"Eval Semi-Hard Acc: {metrics['eval_semi_hard_accuracy']:.3f}")

        # Сохраняем историю метрик
        self.eval_steps_list.append(state.global_step)
        for k, v in metrics.items():
            self.metrics_history[k].append(v)

        # Логируем в MLflow
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_metrics({
                    "eval_loss": float(metrics["eval_loss"]),
                    "eval_hit_1": float(metrics["eval_hit_1"]),
                    "eval_mrr": float(metrics["eval_mrr"]),
                    "eval_none_accuracy": float(
                        metrics["eval_none_accuracy"]
                    ),
                    "eval_hard_accuracy": float(
                        metrics["eval_hard_accuracy"]
                    ),
                    "eval_easy_accuracy": float(
                        metrics["eval_easy_accuracy"]
                    ),
                    "eval_semi_hard_accuracy": float(
                        metrics["eval_semi_hard_accuracy"]
                    ),
                }, step=int(state.global_step))
            except Exception as e:
                import traceback
                print(f"MLflow eval logging error: {e}")
                traceback.print_exc()
        
        # Создаем и сохраняем графики
        try:
            self.plotter.plot_metrics(
                steps=self.eval_steps_list,
                metrics_history=self.metrics_history,
                train_loss_steps=self.train_loss_steps,
                train_loss_values=self.train_loss_values
            )
        except Exception as e:
            print(f"Ошибка построения графика: {e}")

        # Early stopping по основной метрике (eval_mrr)
        primary_metric_value = metrics["eval_mrr"]
        if primary_metric_value > self.best_primary_metric:
            self.best_primary_metric = primary_metric_value
            self.es_counter = 0
            print("Улучшение eval_mrr!")
            # Сохраняем лучший чекпоинт вручную — load_best_model_at_end=False
            # не сохраняет его автоматически. Сохраняем только LoRA-адаптер
            # (не полную модель) для экономии места.
            best_ckpt_dir = os.path.join(self.output_dir, "best_checkpoint")
            try:
                self.model.save_pretrained(best_ckpt_dir)
                print(f"Лучший чекпоинт сохранён: {best_ckpt_dir} "
                      f"(MRR={primary_metric_value:.3f}, "
                      f"step={state.global_step})")
            except Exception as e:
                print(f"Ошибка сохранения чекпоинта: {e}")
        else:
            self.es_counter += 1
            # FIX #3: self.patience вместо args.early_stopping_patience
            # (TrainingArguments не имеет такого атрибута — AttributeError)
            print(f"ES Counter: {self.es_counter}/{self.patience}")
            if self.es_counter >= self.patience:
                print("Ранняя остановка!")
                control.should_training_stop = True


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # deterministic=True + benchmark=False замедляют Vision Tower на 20-40%.
        # Для обучения детерминизм не критичен — отключаем.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def run_experiment(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    batch_size=2,  # Подбирается под объём видеопамяти (VRAM)
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.05,
    label_smoothing=0.05,
    exp_name_suffix="",
    seed=42,
    early_stopping_patience=5,
    eval_every_n_steps=50,
    # FIX batch_size>1: фиксированный размер изображений для корректного
    # батчинга в Qwen2-VL. (448, 448) — стандартный тайл Qwen2-VL.
    # None отключает ресайз (только batch_size=1 безопасен без ресайза).
    fixed_image_size: tuple = (448, 448),
    # FIX #2: none_threshold синхронизирован с eval.py threshold_for_none.
    # Порог для none-of-the-above: max(scores) < threshold → unknown.
    none_threshold: float = 0.5,
    # PERF: число пар (query, candidate) на сэмпл в collate.
    # Каждый сэмпл имеет ~15 кандидатов; обрабатывать все = 30 изображений/шаг.
    # 4 = 1 позитив + 3 негатива → 8 изображений/шаг, ~4x быстрее.
    max_pairs_per_sample: int = 4,
):
    set_seed(seed)

    lr_str = f"{learning_rate:.0e}".replace("+", "").replace("-0", "-")
    exp_name = f"rerank_exp_r{r}_alpha{lora_alpha}_lr{lr_str}_{exp_name_suffix}"
    output_dir = os.path.join(OUTPUT_BASE_DIR, exp_name)

    print(f"=== Запуск эксперимента: {exp_name} ===")

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(EXPERIMENT_NAME)

    def _mlflow_run_ctx():
        """Контекстный менеджер: MLflow run если доступен, иначе no-op."""
        if MLFLOW_AVAILABLE:
            return mlflow.start_run(run_name=exp_name)
        from contextlib import nullcontext
        return nullcontext()

    with _mlflow_run_ctx():
        if MLFLOW_AVAILABLE:
            mlflow.log_params({
                "r": r, "lora_alpha": lora_alpha, "lr": learning_rate,
                "epochs": num_train_epochs, "batch_size": batch_size,
                "seed": seed, "lora_dropout": lora_dropout,
                "target_modules": str(target_modules),
                "grad_acc": gradient_accumulation_steps,
            })

        torch.cuda.empty_cache()
        gc.collect()

        # Загружаем модель и процессор
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", use_fast=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            dtype=torch.bfloat16,
            device_map="auto"
        )

        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Загружаем датасеты
        train_dataset = RerankDataset(TRAIN_DATASET_FILE, IMAGE_DIR)
        val_dataset = RerankDataset(VAL_DATASET_FILE, IMAGE_DIR)
        train_dataset.image_base_dir = IMAGE_DIR
        val_dataset.image_base_dir = IMAGE_DIR

        # Функция сборки батча (collate)
        # FIX batch_size>1: передаём fixed_image_size для корректного батчинга
        # PERF: max_pairs_per_sample ограничивает число пар на сэмпл
        data_collator = make_collate_fn(
            processor, IMAGE_DIR,
            fixed_image_size=fixed_image_size,
            max_pairs_per_sample=max_pairs_per_sample,
        )

        # Создаем callback сначала
        # FIX #3: передаём early_stopping_patience в MetricsCallback
        # FIX batch_size>1: передаём fixed_image_size для eval батчинга
        # FIX #2: передаём none_threshold для синхронизации с eval.py
        callback = MetricsCallback(
            val_dataset, processor, model, output_dir,
            eval_every_n_steps=eval_every_n_steps,
            experiment_name=exp_name,
            early_stopping_patience=early_stopping_patience,
            fixed_image_size=fixed_image_size,
            none_threshold=none_threshold,
        )

        # FIX #11: передаём label_smoothing в RerankTrainer
        trainer = RerankTrainer(
            processor=processor,
            model=model,
            metrics_callback=callback,
            label_smoothing=label_smoothing,
            args=TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                num_train_epochs=num_train_epochs,
                save_steps=500,
                logging_steps=20,
                learning_rate=learning_rate,
                bf16=True,  # Используем bfloat16 согласно рекомендации
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                dataloader_num_workers=2,
                dataloader_prefetch_factor=1,
                dataloader_pin_memory=True,
                optim="adamw_torch_fused",
                report_to=[],
                disable_tqdm=False,
                remove_unused_columns=False,
                lr_scheduler_type=lr_scheduler_type,
                warmup_ratio=warmup_ratio,
                weight_decay=weight_decay,
                # Gradient clipping для стабильности обучения
                max_grad_norm=1.0,
                # Ранняя остановка обрабатывается через callback
                load_best_model_at_end=False,
            ),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,  # Напрямую тренером для eval не используется, но нужен для инициализации
            data_collator=data_collator,
        )

        trainer.add_callback(callback)

        trainer.train()

        # Сохраняем финальную модель
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        if MLFLOW_AVAILABLE:
            mlflow.log_artifacts(output_dir, artifact_path="final_model")

    print(f"Эксперимент {exp_name} завершён.")


if __name__ == "__main__":
    run_experiment(
        # Итог экспериментов с MLP:
        # - attn+mlp r=8  lr=2e-5: Hit@1=0.725, grad_norm=9.47  → хуже
        # - attn+mlp r=16 lr=1e-5: Hit@1=0.750, grad_norm=11.14 → хуже
        # MLP не помогает: reranking = сравнение двух изображений,
        # важен cross-attention, а не независимая трансформация токенов.
        # Возвращаемся к лучшей конфигурации: attn-only r=16 lr=2e-5
        # (Hit@1=0.775, MRR=0.88).
        # Следующее улучшение: увеличить разрешение 336→448 + уменьшить
        # max_pairs 12→8 чтобы не выйти за VRAM.
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
        ],
        # 448×448: больше деталей для различения похожих landmarks.
        # max_pairs=8 вместо 12: 2×8=16 пар = 32 изображения/шаг —
        # компенсирует рост VRAM от увеличения разрешения.
        # grad_acc=4 → эффективный батч = 2×4=8 сэмплов × 8 пар = 64 пары.
        batch_size=2,
        gradient_accumulation_steps=4,
        max_pairs_per_sample=8,
        # Возвращаем lr=2e-5 — лучший результат в attn-only конфигурации.
        learning_rate=2e-5,
        # FIX: epochs=2 вместо 5 — при epochs=5 деградация началась на epoch 0.43.
        # Модель достигает пика на ~700 шагах (~epoch 0.31) и затем переобучается.
        # epochs=2 даёт ~2100 шагов — достаточно с учётом early stopping.
        num_train_epochs=2,
        lr_scheduler_type="cosine",
        # FIX: увеличен warmup с 0.1 до 0.15 — провал Hit@1/MRR на шаге 100
        # совпадал с окончанием warmup при 0.1. Более длинный warmup стабилизирует
        # переход к полному LR.
        warmup_ratio=0.15,
        # FIX: увеличен weight_decay с 0.01 до 0.05 — усиливает регуляризацию,
        # предотвращает переобучение на Yes/No с высокой уверенностью.
        weight_decay=0.05,
        # FIX: label_smoothing=0.05 — без сглаживания модель давала None Accuracy=0
        # (всегда предсказывала Yes с высокой уверенностью). Сглаживание 0.05
        # снижает уверенность и позволяет модели предсказывать "нет совпадения".
        label_smoothing=0.05,
        exp_name_suffix="rerank_attn_448_v7",
        # patience=3 — возвращаемся к attn-only, деградация предсказуема
        early_stopping_patience=3,
        eval_every_n_steps=50,
        # 448×448 вместо 336×336: больше деталей для различения похожих
        # landmarks. max_pairs уменьшен до 8 для компенсации роста VRAM.
        fixed_image_size=(448, 448),
    )