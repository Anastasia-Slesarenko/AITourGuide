import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="peft")

import os
import json
import random
import torch
import torch.nn.functional as F
import gc
import re
from PIL import Image
from typing import Dict, Optional
import matplotlib.pyplot as plt
from IPython.display import clear_output
import difflib
import numpy as np

# Библиотеки для метрик
from bert_score import score
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

# Hugging Face & PEFT
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model
import mlflow


from rag.retriever import RAGRetriever
from rag.dataset import RAGMultimodalDataset

# ----------------------------
# Параметры
INDEX_PATH = "/home/jupyter/s3/ai-tour-guide/landmarks/clip_index"
FACTS_DB_PATH = "/home/jupyter/s3/ai-tour-guide/facts_db.pkl"
TRAIN_DATASET_FILE = "/home/jupyter/s3/ai-tour-guide/train_russian_landmarks.json"
VAL_DATASET_FILE = "/home/jupyter/s3/ai-tour-guide/val_russian_landmarks.json"
IMAGE_DIR = "/home/jupyter/s3/ai-tour-guide/landmarks/"
OUTPUT_BASE_DIR = "/home/jupyter/s3/ai-tour-guide/lora-qwen2-vl-multimodal"

EXPERIMENT_NAME = "Qwen2-VL-Finetuning"

def make_rag_prompt(retrieved_context: str) -> str:
    """Создаёт промпт с retrieved контекстом для RAG"""
    return f"""Ты — профессиональный гид. Вот справочная информация о возможных достопримечательностях:

{retrieved_context}

ЗАДАЧА:
1. Определи, какая достопримечательность из списка на фотографии.
2. Верни описание в формате JSON, используя ТОЛЬКО факты из справочной информации выше.

Ответ должен быть строго в формате JSON:
{{
    "name": "Название на русском",
    "description": "Описание (3-5 предложений)"
}}

Не добавляй никакой текст до или после JSON!"""


def set_seed(seed: int = 42):
    """
    Устанавливает фиксированный seed для воспроизводимости.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_json_response(text: str) -> Dict[str, str]:
    """
    Пытается извлечь JSON из текста ответа.
    Возвращает dict с ключами 'name' и 'description'.
    Если парсинг не удается, возвращает пустые строки.
    """
    text = text.strip()

    # Сначала убираем маркеры кода (до поиска JSON-блока,
    # иначе фигурные скобки ищутся внутри уже «чистого» текста)
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    # Извлекаем первый JSON-объект, если вокруг есть мусор
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        text = match.group(0)

    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            return {"name": "", "description": ""}
        return {
            "name": str(data.get("name", "")).strip(),
            "description": str(data.get("description", "")).strip(),
        }
    except json.JSONDecodeError:
        return {"name": "", "description": ""}
    
    
def make_collate_fn(processor):
    """
    Фабрика collate_fn: замыкает processor для вызова на батче.

    Маскировка промпта в labels:
    Loss считается только на токенах ответа assistant, промпт маскируется -100.
    Это стандартная практика для instruction fine-tuning — модель учится
    генерировать ответ, а не воспроизводить промпт.

    Использование:
        data_collator = make_collate_fn(processor)
    """
    def _collate(batch):
        images = [item["image"] for item in batch]
        texts = [item["text"] for item in batch]
        prompts = [item["prompt_only"] for item in batch]

        # Processor обрабатывает весь батч — pixel_values и image_grid_thw
        # формируются корректно для packed vision attention Qwen2-VL
        inputs = processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
        )

        # Токенизируем промпты с изображениями для точного определения
        # границы промпта в input_ids (vision токены увеличивают длину)
        prompt_inputs = processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
        )

        labels = inputs["input_ids"].clone()

        for i in range(len(batch)):
            # Длина промпта с vision-токенами (без паддинга)
            prompt_len = int(
                prompt_inputs["attention_mask"][i].sum().item()
            )
            seq_len = int(inputs["attention_mask"][i].sum().item())

            # Защита: если промпт >= всей последовательности,
            # labels будут все -100 → loss = 0 → обучение не идёт.
            # В этом случае маскируем только паддинг (fallback).
            if prompt_len >= seq_len:
                print(
                    f"⚠️ [collate] prompt_len ({prompt_len}) >= seq_len "
                    f"({seq_len}) для примера {i}. "
                    f"Маскируем только паддинг (fallback)."
                )
                labels[i, inputs["attention_mask"][i] == 0] = -100
            else:
                # Маскируем промпт: loss только на ответе assistant
                labels[i, :prompt_len] = -100
                # Маскируем паддинг (attention_mask == 0)
                labels[i, inputs["attention_mask"][i] == 0] = -100

            # Финальная проверка: хотя бы один токен должен быть не -100
            n_valid = (labels[i] != -100).sum().item()
            if n_valid == 0:
                print(
                    f"⚠️ [collate] labels[{i}] состоит из одних -100! "
                    f"prompt_len={prompt_len}, seq_len={seq_len}. "
                    f"Сбрасываем маскировку промпта."
                )
                # Fallback: маскируем только паддинг
                labels[i] = inputs["input_ids"][i].clone()
                labels[i, inputs["attention_mask"][i] == 0] = -100

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],      # packed (sum_N, D)
            "image_grid_thw": inputs["image_grid_thw"],  # (B, 3)
            "labels": labels,
            "true_name": [item["true_name"] for item in batch],
            "true_desc": [item["true_desc"] for item in batch],
        }

    return _collate


def compute_metrics(preds_raw, true_names, true_descs):
    """
    Вычисляет метрики:
    1. Text metrics (BERT, BLEU, ROUGE) для поля 'description'.
    2. Accuracy для поля 'name'.
    Логика идентична evaluation_model.py.
    """
    pred_descs = []
    pred_names = []

    for p_raw in preds_raw:
        parsed = parse_json_response(p_raw)
        pred_descs.append(parsed["description"])
        pred_names.append(parsed["name"])

    # --- 1. Метрики для описания ---
    preds_clean = [p if p.strip() else "пусто" for p in pred_descs]
    labels_clean = [
        label if label.strip() else "пусто" for label in true_descs
    ]

    # BERTScore на GPU (модель уже переведена в eval, GPU свободен для метрик)
    try:
        bert_device = "cuda" if torch.cuda.is_available() else "cpu"
        _, _, F1 = score(
            preds_clean, labels_clean, lang="ru", verbose=False,
            device=bert_device
        )
        avg_f1 = F1.mean().item()
    except Exception as e:
        print(f"BERTScore error: {e}")
        avg_f1 = 0.0

    # BLEU (sacrebleu с char-токенизацией — лучше для русского)
    try:
        bleu_metric = BLEU(tokenize='char')
        bleu_result = bleu_metric.corpus_score(preds_clean, [labels_clean])
        bleu_score = bleu_result.score / 100  # sacrebleu возвращает 0–100
    except Exception as e:
        print(f"BLEU error: {e}")
        bleu_score = 0.0

    # ROUGE
    try:
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=False
        )
        rouge_scores = [
            scorer.score(ref, pred)
            for ref, pred in zip(labels_clean, preds_clean)
        ]
        n = len(rouge_scores)
        rouge1_f = (
            sum(s['rouge1'].fmeasure for s in rouge_scores) / n if n else 0
        )
        rouge2_f = (
            sum(s['rouge2'].fmeasure for s in rouge_scores) / n if n else 0
        )
        rougeL_f = (
            sum(s['rougeL'].fmeasure for s in rouge_scores) / n if n else 0
        )
    except Exception as e:
        print(f"ROUGE error: {e}")
        rouge1_f = rouge2_f = rougeL_f = 0.0

    # --- 2. Accuracy для названия ---
    name_matches = 0
    valid_names = 0

    for p_name, t_name in zip(pred_names, true_names):
        p_name = (p_name or "").strip().lower()
        t_name = (t_name or "").strip().lower()

        if not t_name:
            continue

        valid_names += 1

        # Прямое совпадение
        if p_name == t_name:
            name_matches += 1
            continue

        # Нечеткое совпадение (difflib):
        # одно содержится в другом или высокий коэффициент схожести
        ratio = difflib.SequenceMatcher(None, p_name, t_name).ratio()
        if t_name in p_name or p_name in t_name or ratio > 0.85:
            name_matches += 1

    # name_accuracy в диапазоне 0–1
    name_accuracy = name_matches / valid_names if valid_names > 0 else 0.0

    return {
        "bertscore_f1": round(avg_f1, 4),
        "bleu": round(bleu_score, 4),
        "rouge1": round(rouge1_f, 4),
        "rouge2": round(rouge2_f, 4),
        "rougeL": round(rougeL_f, 4),
        "name_accuracy": round(name_accuracy, 4)
    }


class MetricsCallback(TrainerCallback):
    def __init__(
        self,
        val_dataset,
        processor,
        model,
        output_dir,
        retriever: RAGRetriever,
        eval_every_n_steps=50,
        early_stopping_patience=3,
        early_stopping_min_delta=0.001,
        num_final_examples_to_log=10,
        eval_batch_size=4,
    ):
        self.val_dataset = val_dataset
        self.processor = processor
        self.model = model
        self.eval_every_n_steps = eval_every_n_steps
        self.eval_batch_size = eval_batch_size
        self.output_dir = output_dir
        self.retriever = retriever

        # Параметр для логирования финальных примеров
        self.num_final_examples_to_log = num_final_examples_to_log

        self.patience = early_stopping_patience
        self.min_delta = early_stopping_min_delta
        self.best_loss = None
        self.es_counter = 0

        self.train_losses: list = []
        self.eval_steps_list: list = []
        self.metrics_history: Dict[str, list] = {
            "bertscore_f1": [],
            "bleu": [],
            "rouge1": [],
            "rouge2": [],
            "rougeL": [],
            "name_accuracy": [],
            "eval_loss": [],
        }
        self.first_visualization = True

    def _generate_prediction(self, image_path: str) -> Optional[str]:
        """Генерирует ответ модели по изображению с RAG-контекстом."""
        try:
            pil_image = Image.open(image_path).convert("RGB")
            pil_image.thumbnail((448, 448))
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

        retrieved = self.retriever.search(pil_image, top_k=3)
        context_text = self.retriever.format_context(retrieved)
        prompt_text = make_rag_prompt(context_text)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]
        }]
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            images=pil_image, text=text_input, return_tensors="pt"
        ).to(self.model.device, torch.float16)

        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )

        # Декодируем только новые токены (обрезаем входной промпт)
        new_tokens = generated_ids[:, input_len:]
        return self.processor.tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0]

    def _save_final_examples_to_mlflow(self, total_steps: int):
        """
        Сохраняет примеры предсказаний модели в формате JSON
        после окончания обучения.
        Формат: landmark_id, ground_truth, model_prediction
        """
        n = self.num_final_examples_to_log
        print(f"\n💾 Сохранение {n} финальных примеров в MLflow...")

        self.model.eval()

        # Выбираем случайные примеры из валидации
        indices = random.sample(
            range(len(self.val_dataset)),
            min(n, len(self.val_dataset)),
        )
        
        examples_list = []
        examples_dir = os.path.join(self.output_dir, "final_examples")
        os.makedirs(examples_dir, exist_ok=True)
        
        with torch.no_grad():
            for idx in indices:
                # Читаем исходный элемент датасета
                raw_item = self.val_dataset.data[idx]
                image_path = os.path.join(
                    self.val_dataset.image_dir, raw_item["image_path"]
                )

                landmark_id = raw_item.get(
                    "landmark_id", raw_item.get("id", idx)
                )
                true_name = raw_item.get(
                    "name_ru", raw_item.get("name_en", "")
                )
                true_desc = raw_item.get("ground_truth", "").strip()

                # Генерация предсказания
                raw_pred = self._generate_prediction(image_path)
                if not raw_pred:
                    continue
                    
                parsed = parse_json_response(raw_pred)
                pred_name = parsed.get("name", "")
                pred_desc = parsed.get("description", "")
                
                # Проверка совпадения имени (для удобства анализа)
                p_low = pred_name.lower().strip()
                t_low = true_name.lower().strip()
                ratio = difflib.SequenceMatcher(
                    None, p_low, t_low
                ).ratio()
                name_match = (
                    p_low == t_low
                    or t_low in p_low
                    or p_low in t_low
                    or ratio > 0.85
                )
                
                # Формируем структуру примера
                example_data = {
                    "landmark_id": landmark_id,
                    "image_path": raw_item["image_path"],
                    "ground_truth": {
                        "name": true_name,
                        "description": true_desc
                    },
                    "model_prediction": {
                        "name": pred_name,
                        "description": pred_desc,
                        "raw_output": raw_pred  # Сырой вывод для отладки
                    },
                    "name_match": name_match,
                    "step": total_steps
                }
                
                examples_list.append(example_data)
                
                # Сохраняем каждый пример в отдельный JSON файл
                example_json_path = os.path.join(
                    examples_dir, f"example_{landmark_id}.json"
                )
                with open(example_json_path, "w", encoding="utf-8") as f:
                    json.dump(example_data, f, ensure_ascii=False, indent=2)
                
                # Логирование в MLflow
                try:
                    mlflow.log_artifact(
                        example_json_path,
                        artifact_path="final_examples",
                    )
                except Exception as e:
                    print(
                        f"⚠️ Не удалось залогировать пример "
                        f"{landmark_id}: {e}"
                    )
        
        # Сохраняем сводный JSON со всеми примерами
        if examples_list:
            summary_json_path = os.path.join(examples_dir, "all_examples.json")
            summary_data = {
                "total_steps": total_steps,
                "num_examples": len(examples_list),
                "examples": examples_list
            }
            with open(summary_json_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            
            try:
                mlflow.log_artifact(
                    summary_json_path,
                    artifact_path="final_examples",
                )
                print(
                    f"✅ Финальные примеры сохранены: {summary_json_path}"
                )
            except Exception as e:
                print(f"⚠️ Не удалось залогировать сводный JSON: {e}")
            
            # Логируем метрику доли правильных названий на этих примерах
            name_match_count = sum(
                1 for ex in examples_list if ex["name_match"]
            )
            name_accuracy = name_match_count / len(examples_list)
            mlflow.log_metric(
                "final_examples_name_accuracy", name_accuracy
            )
            print(
                f"📊 Accuracy на финальных примерах: "
                f"{name_accuracy:.2%} "
                f"({name_match_count}/{len(examples_list)})"
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            step = state.global_step
            train_loss = logs['loss']
            self.train_losses.append((step, train_loss))

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_every_n_steps != 0:
            return

        print(f"\n🔍 Вычисление метрик на шаге {state.global_step}...")

        # Уменьшенная подвыборка для скорости eval во время обучения.
        # Для финальной оценки используй evaluation_model.py.
        subset_size = min(len(self.val_dataset), 20)
        indices = random.sample(range(len(self.val_dataset)), subset_size)

        all_preds_raw = []
        all_true_names = []
        all_true_descs = []
        total_loss = 0.0

        # Собираем данные для батчевой генерации
        items_for_gen = []   # (pil_image, text_gen, true_name, true_desc)
        items_for_loss = []  # (pil_image, text_loss)

        self.model.eval()
        with torch.no_grad():
            # --- Шаг 1: подготовка текстов (без GPU) ---
            for idx in indices:
                item = self.val_dataset[idx]
                pil_image = item["image"]

                retrieved = self.retriever.search(pil_image, top_k=3)
                context_text = self.retriever.format_context(retrieved)
                prompt_text = make_rag_prompt(context_text)

                messages_gen = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ]
                }]
                target_json = json.dumps({
                    "name": item["true_name"],
                    "description": item["true_desc"]
                }, ensure_ascii=False)
                messages_loss = messages_gen + [
                    {"role": "assistant", "content": target_json}
                ]
                text_gen = self.processor.apply_chat_template(
                    messages_gen, tokenize=False, add_generation_prompt=True
                )
                text_loss = self.processor.apply_chat_template(
                    messages_loss, tokenize=False, add_generation_prompt=False
                )
                items_for_gen.append(
                    (pil_image, text_gen, item["true_name"], item["true_desc"])
                )
                items_for_loss.append((pil_image, text_loss))

            # --- Шаг 2: батчевый расчёт Loss ---
            # Обрабатываем по одному (processor не поддерживает батч для loss
            # из-за разных длин последовательностей без паддинга labels)
            valid_indices_for_gen = []
            for i, (pil_image, text_loss) in enumerate(items_for_loss):
                try:
                    inp = self.processor(
                        images=pil_image, text=text_loss, return_tensors="pt"
                    ).to(self.model.device, torch.float16)
                    out = self.model(
                        pixel_values=inp["pixel_values"],
                        image_grid_thw=inp["image_grid_thw"],
                        input_ids=inp["input_ids"],
                        attention_mask=inp["attention_mask"],
                        labels=inp["input_ids"].clone()
                    )
                    total_loss += out.loss.item()
                    valid_indices_for_gen.append(i)
                except Exception as e:
                    print(f"⚠️ Loss error idx={i}: {e}")

            # --- Шаг 3: батчевая генерация ---
            # Генерируем только для примеров с успешным loss
            valid_items = [items_for_gen[i] for i in valid_indices_for_gen]

            for batch_start in range(
                0, len(valid_items), self.eval_batch_size
            ):
                batch = valid_items[
                    batch_start: batch_start + self.eval_batch_size
                ]
                batch_images = [b[0] for b in batch]
                batch_texts = [b[1] for b in batch]

                try:
                    inp_batch = self.processor(
                        images=batch_images,
                        text=batch_texts,
                        return_tensors="pt",
                        padding=True,
                    ).to(self.model.device, torch.float16)

                    gen_ids = self.model.generate(
                        **inp_batch,
                        max_new_tokens=300,  # сокращено для скорости eval
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                    # Декодируем только сгенерированные токены
                    input_len = inp_batch["input_ids"].shape[1]
                    decoded = self.processor.tokenizer.batch_decode(
                        gen_ids[:, input_len:], skip_special_tokens=True
                    )
                    for text_out, (_, _, true_name, true_desc) in zip(
                        decoded, batch
                    ):
                        all_preds_raw.append(text_out.strip())
                        all_true_names.append(true_name)
                        all_true_descs.append(true_desc)
                except Exception as e:
                    print(f"⚠️ Generation error batch {batch_start}: {e}")

        if not all_preds_raw:
            print("⚠️ Не удалось сгенерировать ни одного ответа для метрик.")
            return

        # Делим на количество успешно вычисленных loss-значений
        n_valid = len(valid_indices_for_gen)
        avg_loss = total_loss / n_valid if n_valid else 0.0

        # Вычисление метрик
        metrics = compute_metrics(
            all_preds_raw, all_true_names, all_true_descs
        )

        # Логирование в консоль
        print(f"📉 Eval Loss: {avg_loss:.4f}")
        print(
            f"📈 Name Accuracy: {metrics['name_accuracy']:.2%}"
        )
        print(
            f"📝 BERTScore F1: {metrics['bertscore_f1']:.4f},"
            f" ROUGE-L: {metrics['rougeL']:.4f}"
        )

        if self.first_visualization:
            print("\n--- Пример предсказания ---")
            parsed = parse_json_response(all_preds_raw[0])
            print(f"Pred Name: {parsed['name']}")
            print(f"True Name: {all_true_names[0]}")
            print(f"Pred Desc: {parsed['description'][:100]}...")
            print(f"True Desc: {all_true_descs[0][:100]}...")
            print("---------------------------")
            self.first_visualization = False

        # Early Stopping logic
        if (
            self.best_loss is None
            or avg_loss < self.best_loss - self.min_delta
        ):
            self.best_loss = avg_loss
            self.es_counter = 0
            print("✅ Улучшение eval_loss!")
        else:
            self.es_counter += 1
            print(f"ES Counter: {self.es_counter}/{self.patience}")
            if self.es_counter >= self.patience:
                print("🛑 Ранняя остановка!")
                control.should_training_stop = True

        # Сохранение истории
        self.eval_steps_list.append(state.global_step)
        self.metrics_history["eval_loss"].append(avg_loss)
        for k, v in metrics.items():
            self.metrics_history[k].append(v)

        # MLflow
        try:
            mlflow.log_metrics({
                "eval_loss": avg_loss,
                "name_accuracy": metrics["name_accuracy"],
                "bertscore_f1": metrics["bertscore_f1"],
                "bleu": metrics["bleu"],
                "rouge1": metrics["rouge1"],
                "rouge2": metrics["rouge2"],
                "rougeL": metrics["rougeL"]
            }, step=state.global_step)
        except Exception as e:
            print(f"MLflow logging error: {e}")

        # Визуализация
        clear_output(wait=True)
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Plot Loss
        ax1 = axes[0]
        if self.train_losses:
            steps_t, losses_t = zip(*self.train_losses)
            ax1.plot(steps_t, losses_t, 'b-', label='Train Loss', alpha=0.6)
        if self.eval_steps_list:
            ax1.plot(
                self.eval_steps_list,
                self.metrics_history["eval_loss"],
                'r-o',
                label='Eval Loss',
            )
        ax1.set_title('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot Metrics
        ax2 = axes[1]
        colors = ['g', 'm', 'c', 'orange', 'purple']
        keys_to_plot = ['name_accuracy', 'bertscore_f1', 'rougeL']
        for i, key in enumerate(keys_to_plot):
            if key in self.metrics_history and self.metrics_history[key]:
                ax2.plot(
                    self.eval_steps_list,
                    self.metrics_history[key],
                    '-o',
                    color=colors[i],
                    label=key,
                )
        ax2.set_title('Metrics')
        ax2.legend()
        ax2.grid(True)
        ax2.set_ylim(0, 1.1)

        plt.tight_layout()
        plt.show()

    def on_train_end(self, args, state, control, **kwargs):
        print("\n🏁 Обучение завершено.")
        best_loss_str = (
            f"{self.best_loss:.4f}" if self.best_loss is not None
            else "N/A (eval не запускался)"
        )
        print(
            f"🏆 Лучший eval_loss: {best_loss_str} "
            f"(счётчик ES в конце: {self.es_counter})"
        )

        # === Сохранение финальных примеров в MLflow (JSON) ===
        self._save_final_examples_to_mlflow(state.global_step)

        final_mlflow_metrics = {
            "final_best_eval_loss": self.best_loss,
            "final_ES_counter_at_end": self.es_counter,
            "final_total_steps": state.global_step,
        }
        mlflow.log_metrics(final_mlflow_metrics)

        os.makedirs(self.output_dir, exist_ok=True)
        plot_path = os.path.join(self.output_dir, "final_training_curves.png")

        if self.eval_steps_list:
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))

            ax1 = axes[0]
            if self.train_losses:
                steps_train, losses_train = zip(*self.train_losses)
                ax1.plot(
                    steps_train, losses_train, 'b-',
                    label='Train Loss', marker='o', markersize=3,
                    alpha=0.6,
                    markevery=max(1, len(steps_train) // 10),
                )
            if (
                self.eval_steps_list
                and len(self.metrics_history["eval_loss"]) > 0
            ):
                ax1.plot(
                    self.eval_steps_list,
                    self.metrics_history["eval_loss"],
                    'r-', label='Eval Loss', marker='s', markersize=4,
                )
            ax1.set_xlabel('Шаг обучения', fontsize=11)
            ax1.set_ylabel('Loss', fontsize=11)
            ax1.set_title(
                'Лосс на трейне и валидации (финал)',
                fontsize=12, fontweight='bold',
            )
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2 = axes[1]
            colors = iter([
                'g', 'm', 'c', 'orange', 'brown',
                'pink', 'gray', 'olive', 'purple', 'navy'
            ])
            for key in self.metrics_history:
                if key != "eval_loss" and len(self.metrics_history[key]) > 0:
                    color = next(colors)
                    label = (
                        key.replace("bertscore_", "BERTScore ")
                        .replace("_", " ")
                        .upper()
                    )
                    if 'f1' in key:
                        marker = '^'
                    elif 'bleu' in key:
                        marker = 'd'
                    else:
                        marker = 'o'
                    ax2.plot(
                        self.eval_steps_list,
                        self.metrics_history[key],
                        '-',
                        label=label,
                        color=color,
                        marker=marker,
                        markersize=4,
                    )
            ax2.set_xlabel('Шаг обучения', fontsize=11)
            ax2.set_ylabel('Значение метрики', fontsize=11)
            ax2.set_title(
                'Метрики качества (финал)',
                fontsize=12, fontweight='bold',
            )
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.show()
            print(f"📈 Графики сохранены: {plot_path}")
            mlflow.log_artifact(plot_path)
        else:
            print(
                "⚠️ Не было вычислено метрик для построения графиков "
                "(eval_steps_list пуст)."
            )

        torch.cuda.empty_cache()
        gc.collect()


class MultimodalTrainer(Trainer):
    """
    Кастомный Trainer для Qwen2-VL с поддержкой label smoothing.

    Стандартный Trainer.compute_loss применяет label_smoothing_factor
    только через свой внутренний CrossEntropyLoss. Здесь мы реализуем
    label smoothing явно, чтобы он работал с кастомными inputs.
    """

    def __init__(self, *args, label_smoothing: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._label_smoothing = label_smoothing

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        pixel_values = inputs.pop("pixel_values")
        image_grid_thw = inputs.pop("image_grid_thw")
        labels = inputs.pop("labels")
        # true_name / true_desc нужны только колбэку — удаляем из inputs
        inputs.pop("true_name", None)
        inputs.pop("true_desc", None)

        outputs = model(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **inputs,
            labels=labels,
        )

        if self._label_smoothing > 0.0:
            logits = outputs.logits  # (B, T, V)
            # Сдвиг: предсказываем следующий токен
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Маска: игнорируем -100
            active = shift_labels != -100
            if active.any():
                log_probs = F.log_softmax(shift_logits, dim=-1)
                # NLL на правильных токенах
                nll = -log_probs.gather(
                    dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)
                ).squeeze(-1)
                # Равномерное распределение (smoothing)
                smooth = -log_probs.mean(dim=-1)
                loss_per_token = (
                    (1.0 - self._label_smoothing) * nll
                    + self._label_smoothing * smooth
                )
                loss = loss_per_token[active].mean()
            else:
                loss = outputs.loss
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


def run_experiment(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=None,  # по умолчанию None — задаётся при вызове
    batch_size=1,
    learning_rate=2e-4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    early_stopping_patience=3,
    early_stopping_min_delta=0.001,
    augment=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    # Label smoothing: сглаживает уверенность модели, снижает галлюцинации.
    # Рекомендуемые значения: 0.0 (выкл), 0.05, 0.1.
    # При 0.1 модель не может быть уверена > 90% в любом токене.
    label_smoothing=0.1,
    exp_name_suffix="",
    seed=42,
    max_rag_facts: int = 3
):
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    set_seed(seed)

    lr_str = f"{learning_rate:.0e}".replace("+", "").replace("-0", "-")
    exp_name = (
        f"exp_r{r}_alpha{lora_alpha}_lr{lr_str}_{exp_name_suffix}"
    )
    output_dir = os.path.join(OUTPUT_BASE_DIR, exp_name)

    print(f"=== Запуск эксперимента: {exp_name} ===")

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=exp_name):
        mlflow.log_params({
            "r": r, "lora_alpha": lora_alpha, "lr": learning_rate,
            "epochs": num_train_epochs, "batch_size": batch_size,
            "seed": seed, "lora_dropout": lora_dropout,
            "target_modules": str(target_modules)
        })

        torch.cuda.empty_cache()
        gc.collect()

        # Определяем precision до загрузки модели:
        # bf16 предпочтительнее fp16 на Ampere+ GPU (меньше потерь точности)
        use_bf16 = (
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        )
        model_dtype = torch.bfloat16 if use_bf16 else torch.float16
        use_fp16 = not use_bf16

        # use_fast=True — fast processor (новый дефолт)
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", use_fast=True
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            dtype=model_dtype,
            device_map="auto",
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

        print("\n🔍 Загрузка RAGRetriever...")
        retriever = RAGRetriever(
            index_path=INDEX_PATH,
            facts_db_path=FACTS_DB_PATH,
            top_k=max_rag_facts,
        )

        train_dataset = RAGMultimodalDataset(
            data_json=TRAIN_DATASET_FILE,
            image_dir=IMAGE_DIR,
            processor=processor,
            retriever=retriever,
            augment=augment,
            max_rag_facts=max_rag_facts,
        )
        val_dataset = RAGMultimodalDataset(
            data_json=VAL_DATASET_FILE,
            image_dir=IMAGE_DIR,
            processor=processor,
            retriever=retriever,
            augment=False,
            max_rag_facts=max_rag_facts,
        )

        trainer = MultimodalTrainer(
            label_smoothing=label_smoothing,
            model=model,
            args=TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                num_train_epochs=num_train_epochs,
                save_steps=500,
                logging_steps=20,
                learning_rate=learning_rate,
                fp16=use_fp16,
                bf16=use_bf16,
                # Gradient checkpointing: экономит VRAM ценой ~20% скорости,
                # позволяет увеличить grad_accum или использовать больше LoRA.
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                # Параллельная загрузка данных
                dataloader_num_workers=4,
                dataloader_pin_memory=True,
                # Оптимизация памяти
                optim="adamw_torch_fused",  # fused AdamW быстрее на GPU
                report_to=[],
                disable_tqdm=False,
                # true_name/true_desc передаются через collate_fn
                remove_unused_columns=False,
                lr_scheduler_type=lr_scheduler_type,
                warmup_ratio=warmup_ratio,
                weight_decay=weight_decay,
                # label_smoothing_factor здесь не используется —
                # smoothing реализован в MultimodalTrainer.compute_loss
                label_smoothing_factor=0.0,
            ),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            # make_collate_fn вызывает processor на всём батче — единственный
            # корректный способ для batch_size > 1 в Qwen2-VL
            data_collator=make_collate_fn(processor),
        )

        callback = MetricsCallback(
            val_dataset, processor, model, output_dir,
            retriever=retriever,  
            eval_every_n_steps=50,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta
        )
        trainer.add_callback(callback)

        trainer.train()

        # Сохранение финальных весов и процессора
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        # Логируем только финальные веса, а не всю директорию с чекпоинтами
        mlflow.log_artifacts(output_dir, artifact_path="final_model")

    print(f"✅ Эксперимент {exp_name} завершён.")


if __name__ == "__main__":
    # Параметры baseline для NVIDIA T4 (24 GB VRAM), Qwen2-VL-2B fp16
    # ---------------------------------------------------------------
    # VRAM-бюджет:
    #   модель fp16               ~4.5 GB
    #   LoRA все модули (r=16)    ~0.5 GB
    #   оптимизатор fused AdamW   ~1.5 GB
    #   активации (grad_ckpt)     ~2.0 GB
    #   батч 2 + паддинг          ~3.0 GB
    #   BERTScore eval (GPU)      ~1.5 GB (временно, во время eval)
    #   итого                     ~13 GB  → безопасно при 24 GB
    # ---------------------------------------------------------------
    run_experiment(
        r=16,                 
        lora_alpha=32,        
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],  
        batch_size=3,         
        gradient_accumulation_steps=4,  
        learning_rate=5e-5, 
        num_train_epochs=5,                  
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.05,
        label_smoothing=0.05,
        early_stopping_patience=20, 
        early_stopping_min_delta=0.005,  
        augment=False,   
        exp_name_suffix="lora24_patience20",
        max_rag_facts=10 
    )