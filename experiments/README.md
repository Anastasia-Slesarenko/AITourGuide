# Qwen2-VL Reranking Model Training

Обучение модели Qwen2-VL с LoRA для задачи reranking достопримечательностей.

## 📋 Содержание

- [Быстрый старт](#быстрый-старт)
- [Требования](#требования)
- [Структура проекта](#структура-проекта)
- [Конфигурация](#конфигурация)
- [Обучение](#обучение)
- [Оценка](#оценка)
- [Мониторинг](#мониторинг)
- [Troubleshooting](#troubleshooting)

## 🚀 Быстрый старт

```bash
# 1. Подготовьте данные (см. DATA_FORMAT.md)
python scripts/data_preparation/step6_setup_dataset.py

# 2. Запустите обучение с дефолтными параметрами
python experiments/train.py

# 3. Оцените модель
python experiments/eval.py --checkpoint checkpoints/qwen2vl-rerank-lora/best_model
```

## 📦 Требования

### Системные требования

- **GPU**: NVIDIA GPU с минимум 16GB VRAM (рекомендуется 24GB+)
- **RAM**: Минимум 32GB
- **Диск**: 50GB+ свободного места

### Python зависимости

```bash
pip install -r requirements.txt
```

Основные пакеты:
- `transformers >= 4.37.0`
- `torch >= 2.1.0`
- `peft >= 0.8.0`
- `mlflow >= 2.9.0`
- `pillow >= 10.0.0`
- `tqdm >= 4.66.0`

## 📁 Структура проекта

```
experiments/
├── train.py              # Основной скрипт обучения
├── train_config.py       # Конфигурации экспериментов
├── eval.py              # Скрипт оценки модели
├── DATA_FORMAT.md       # Документация формата данных
└── README.md            # Этот файл

data/
├── processed/
│   ├── train.json       # Обучающая выборка
│   └── val.json         # Валидационная выборка
└── images/              # Изображения

checkpoints/             # Сохраненные модели
└── qwen2vl-rerank-lora/
    └── experiment_name/
        ├── adapter_config.json
        ├── adapter_model.bin
        └── ...
```

## ⚙️ Конфигурация

### Использование предустановленных конфигураций

```python
from train_config import (
    get_default_config,
    get_large_lora_config,
    get_aggressive_training_config,
    get_conservative_training_config
)

# Базовая конфигурация
config = get_default_config()

# Большой LoRA rank для лучшего качества
config = get_large_lora_config()

# Агрессивное обучение (быстрая сходимость)
config = get_aggressive_training_config()

# Консервативное обучение (стабильность)
config = get_conservative_training_config()
```

### Кастомизация параметров

```python
from train_config import get_default_config

config = get_default_config()

# Изменение параметров обучения
config.training.learning_rate = 1e-4
config.training.batch_size = 4
config.training.num_train_epochs = 15

# Изменение LoRA параметров
config.lora.r = 32
config.lora.lora_alpha = 64

# Изменение путей к данным
config.data.train_dataset_file = "path/to/train.json"
config.data.image_dir = "path/to/images"
```

### Основные параметры

#### LoRA параметры

- **`r`**: Rank LoRA матриц (8, 16, 32, 64)
  - Меньше = быстрее, меньше параметров
  - Больше = лучше качество, больше памяти
  
- **`lora_alpha`**: Scaling factor (обычно 2×r)
  
- **`lora_dropout`**: Dropout для LoRA слоев (0.05-0.1)

- **`target_modules`**: Какие слои модифицировать
  - Минимум: `["q_proj", "v_proj"]`
  - Рекомендуется: `["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

#### Параметры обучения

- **`batch_size`**: Размер батча на GPU (1-4)
- **`gradient_accumulation_steps`**: Накопление градиентов (4-16)
- **`learning_rate`**: Learning rate (1e-5 до 1e-4)
- **`num_train_epochs`**: Количество эпох (5-15)
- **`max_grad_norm`**: Gradient clipping (0.5-1.0)

## 🎓 Обучение

### Базовое обучение

```python
python experiments/train.py
```

### С кастомными параметрами

Отредактируйте параметры в конце `train.py`:

```python
if __name__ == "__main__":
    run_experiment(
        r=32,                    # LoRA rank
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "o_proj"],
        batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        num_train_epochs=10,
        exp_name_suffix="my_experiment",
        early_stopping_patience=10,
        eval_every_n_steps=100,
    )
```

### Множественные эксперименты

```python
# Запуск нескольких экспериментов с разными параметрами
for lr in [1e-5, 5e-5, 1e-4]:
    for r in [16, 32]:
        run_experiment(
            r=r,
            learning_rate=lr,
            exp_name_suffix=f"lr{lr}_r{r}",
        )
```

## 📊 Оценка

### Метрики

Модель оценивается по следующим метрикам:

- **Hit@1**: Процент запросов, где правильный кандидат на 1-м месте
- **MRR (Mean Reciprocal Rank)**: Средний обратный ранг правильного ответа
- **None Accuracy**: Точность на "none-of-the-above" запросах
- **Hard/Semi-hard/Easy Accuracy**: Точность по типам сложности

### Во время обучения

Метрики вычисляются каждые `eval_every_n_steps` шагов на подмножестве валидационной выборки (50 сэмплов с stratified sampling).

### Полная оценка

```bash
python experiments/eval.py \
    --checkpoint checkpoints/qwen2vl-rerank-lora/experiment_name \
    --val_dataset data/processed/val.json \
    --image_dir images \
    --batch_size 4
```

## 📈 Мониторинг

### MLflow

Все эксперименты логируются в MLflow:

```bash
# Запуск MLflow UI
mlflow ui

# Откройте http://localhost:5000 в браузере
```

В MLflow доступны:
- Параметры эксперимента
- Train loss (каждые logging_steps)
- Eval метрики (каждые eval_every_n_steps)
- Артефакты (сохраненные модели)

### Логи

Во время обучения выводятся:
- Train loss каждые 20 шагов
- Eval метрики каждые 100 шагов
- Прогресс-бар для evaluation
- Early stopping counter

## 🔧 Troubleshooting

### Out of Memory (OOM)

**Проблема**: `CUDA out of memory`

**Решения**:
1. Уменьшите `batch_size` (до 1)
2. Увеличьте `gradient_accumulation_steps`
3. Уменьшите `r` (LoRA rank)
4. Уменьшите количество `target_modules`
5. Включите `gradient_checkpointing` (уже включен)

### Медленное обучение

**Проблема**: Обучение идет слишком медленно

**Решения**:
1. Уменьшите `eval_every_n_steps` (реже оценка)
2. Уменьшите `dataloader_num_workers`
3. Используйте меньше `target_modules`
4. Проверьте, что используется Flash Attention 2

### Нестабильное обучение

**Проблема**: Loss скачет или растет

**Решения**:
1. Уменьшите `learning_rate`
2. Увеличьте `warmup_ratio`
3. Уменьшите `max_grad_norm`
4. Увеличьте `weight_decay`

### Переобучение

**Проблема**: Train loss падает, но eval метрики не растут

**Решения**:
1. Увеличьте `weight_decay`
2. Увеличьте `lora_dropout`
3. Уменьшите `num_train_epochs`
4. Используйте меньший `r` (LoRA rank)
5. Добавьте больше данных

### Ошибки загрузки данных

**Проблема**: `FileNotFoundError` или `ValueError`

**Решения**:
1. Проверьте пути в конфигурации
2. Убедитесь, что данные в правильном формате (см. DATA_FORMAT.md)
3. Проверьте, что все изображения существуют и валидны

## 📝 Best Practices

### Для начала экспериментов

1. Начните с дефолтной конфигурации
2. Обучите на небольшом subset данных (100-500 сэмплов)
3. Убедитесь, что модель может переобучиться на малых данных
4. Постепенно увеличивайте размер данных

### Для production

1. Используйте полный датасет
2. Настройте learning rate через grid search
3. Используйте early stopping
4. Сохраняйте несколько чекпоинтов
5. Оценивайте на отдельном test set

### Оптимизация гиперпараметров

Рекомендуемый порядок:
1. Learning rate (самый важный)
2. LoRA rank (r)
3. Batch size / gradient accumulation
4. Weight decay
5. Warmup ratio

## 📚 Дополнительные ресурсы

- [DATA_FORMAT.md](DATA_FORMAT.md) - Формат данных
- [train_config.py](train_config.py) - Конфигурации
- [Qwen2-VL Documentation](https://github.com/QwenLM/Qwen2-VL)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## 🤝 Поддержка

При возникновении проблем:
1. Проверьте этот README и DATA_FORMAT.md
2. Посмотрите логи MLflow
3. Проверьте системные требования
4. Создайте issue с подробным описанием проблемы
