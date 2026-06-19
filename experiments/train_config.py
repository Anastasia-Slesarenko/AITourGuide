"""
Конфигурация для обучения Qwen2-VL reranking модели.
Вынесены все параметры для удобного управления экспериментами.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class DataConfig:
    """Конфигурация данных"""
    train_dataset_file: str = "data/processed/train.json"
    val_dataset_file: str = "data/processed/val.json"
    image_dir: str = "images"


@dataclass
class ModelConfig:
    """Конфигурация модели"""
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    use_flash_attention: bool = True
    torch_dtype: str = "bfloat16"


@dataclass
class LoRAConfig:
    """Конфигурация LoRA"""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


@dataclass
class TrainingConfig:
    """Конфигурация обучения"""
    # Основные параметры
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 10
    learning_rate: float = 5e-5
    
    # Scheduler
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    
    # Регуляризация
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.05
    
    # Оптимизация
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch_fused"
    
    # Логирование и сохранение
    logging_steps: int = 20
    save_steps: int = 500
    eval_every_n_steps: int = 100
    
    # Early stopping
    early_stopping_patience: int = 10
    
    # Другое
    seed: int = 42
    dataloader_num_workers: int = 2


@dataclass
class ExperimentConfig:
    """Полная конфигурация эксперимента"""
    experiment_name: str = "Qwen2-VL-Rerank-Finetuning"
    output_base_dir: str = "checkpoints/qwen2vl-rerank-lora"
    exp_name_suffix: str = "default"
    
    data: DataConfig = None
    model: ModelConfig = None
    lora: LoRAConfig = None
    training: TrainingConfig = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.lora is None:
            self.lora = LoRAConfig()
        if self.training is None:
            self.training = TrainingConfig()


# Предустановленные конфигурации для разных экспериментов

def get_default_config() -> ExperimentConfig:
    """Базовая конфигурация"""
    return ExperimentConfig(
        exp_name_suffix="default"
    )


def get_large_lora_config() -> ExperimentConfig:
    """Конфигурация с большим LoRA rank"""
    config = ExperimentConfig(exp_name_suffix="large_lora")
    config.lora.r = 32
    config.lora.lora_alpha = 64
    return config


def get_aggressive_training_config() -> ExperimentConfig:
    """Агрессивная конфигурация обучения"""
    config = ExperimentConfig(exp_name_suffix="aggressive")
    config.training.learning_rate = 1e-4
    config.training.gradient_accumulation_steps = 16
    config.training.warmup_ratio = 0.05
    return config


def get_conservative_training_config() -> ExperimentConfig:
    """Консервативная конфигурация обучения"""
    config = ExperimentConfig(exp_name_suffix="conservative")
    config.training.learning_rate = 1e-5
    config.training.weight_decay = 0.1
    config.training.warmup_ratio = 0.2
    return config
