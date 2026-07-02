# src/core/logging.py
"""Настройка логирования для AITourGuide."""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Форматтер для структурированного JSON-логирования."""

    def format(self, record: logging.LogRecord) -> str:
        """Форматирует запись лога в JSON."""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "correlation_id"):
            log_data["correlation_id"] = record.correlation_id

        return json.dumps(log_data, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Форматтер с цветным выводом для консоли."""

    COLORS = {
        "DEBUG": "\033[36m",  # Голубой
        "INFO": "\033[32m",  # Зелёный
        "WARNING": "\033[33m",  # Жёлтый
        "ERROR": "\033[31m",  # Красный
        "CRITICAL": "\033[35m",  # Пурпурный
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Форматирует запись лога с цветом уровня."""
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_format: str = "text",
    log_file: str | None = None,
    enable_colors: bool = True,
) -> None:
    """
    Настраивает логирование приложения.

    Args:
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Формат вывода ("text" или "json")
        log_file: Путь к файлу лога (опционально)
        enable_colors: Цветной вывод в консоль (только для text-формата)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()

    # Консольный обработчик
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    fmt = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]"
    )
    if log_format == "json":
        console_handler.setFormatter(JSONFormatter())
    elif enable_colors and sys.stdout.isatty():
        console_handler.setFormatter(ColoredFormatter(fmt))
    else:
        console_handler.setFormatter(logging.Formatter(fmt))

    root_logger.addHandler(console_handler)

    # Файловый обработчик (если указан путь)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)

        if log_format == "json":
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - "
                    "%(message)s [%(filename)s:%(lineno)d]"
                )
            )

        root_logger.addHandler(file_handler)

    # Снижаем шум от сторонних библиотек
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    root_logger.info(
        f"Логирование настроено: level={level}, "
        f"format={log_format}, file={log_file or 'None'}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Возвращает логгер с указанным именем.

    Args:
        name: Имя логгера (обычно __name__)
    """
    return logging.getLogger(name)
