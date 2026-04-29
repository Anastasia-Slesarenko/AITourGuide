# 🎡 AI Tour Guide

**Интеллектуальный гид для распознавания достопримечательностей по фотографиям.**

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

</div>

---

## 🚀 Быстрый старт

### Требования

| Ресурс | Минимум | Рекомендуется |
|--------|---------|---------------|
| **CPU** | 2 ядра | 4+ ядра |
| **RAM** | 4 GB | 8 GB |
| **Диск** | 5 GB | 10 GB |
| **GPU** | ❌ Не требуется | Опционально |

### Установка за 5 минут

```bash
# 1. Клонируйте репозиторий
git clone https://github.com/your-org/AITourGuide.git
cd AITourGuide

# 2. Создайте виртуальное окружение
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Установите зависимости
pip install -r requirements.txt

# 4. Настройте переменные окружения
cp .env.example .env
# Отредактируйте .env и добавьте свои ключи API

# 5. Соберите FAISS индекс
python scripts/build_index.py

# 6. Запустите API сервер
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📄 Лицензия

Этот проект лицензирован под **[MIT License](LICENSE)** — свободно для коммерческого использования.

Все файлы в проекте распространяются в соответствии с условиями данной лицензии.