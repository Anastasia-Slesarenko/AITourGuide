# Формат данных для обучения Qwen2-VL Reranking модели

## Обзор

Данные для обучения модели reranking представлены в формате JSON и состоят из запросов с изображениями и списком кандидатов для ранжирования.

## Структура файлов

```
data/
├── processed/
│   ├── train.json          # Обучающая выборка
│   └── val.json            # Валидационная выборка
└── images/                 # Директория с изображениями
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Формат JSON

### Структура одного сэмпла

```json
{
  "query_image": "path/to/query/image.jpg",
  "candidates": [
    {
      "name": "Landmark Name",
      "caption": "Description of the landmark...",
      "candidate_type": "hard"
    },
    {
      "name": "Another Landmark",
      "caption": "Another description...",
      "candidate_type": "easy"
    }
  ],
  "target_idx": 0
}
```

### Описание полей

#### Корневой уровень

- **`query_image`** (string, обязательное)
  - Относительный путь к изображению запроса от `IMAGE_DIR`
  - Пример: `"landmarks/eiffel_tower/img_001.jpg"`

- **`candidates`** (array, обязательное)
  - Список кандидатов для ранжирования
  - Минимум 1 кандидат
  - Обычно 5-10 кандидатов на запрос

- **`target_idx`** (integer, обязательное)
  - Индекс правильного кандидата в массиве `candidates` (0-based)
  - Значение `-1` означает "none of the above" (ни один кандидат не подходит)

#### Поля кандидата

- **`name`** (string, обязательное)
  - Название достопримечательности
  - Используется в промпте для модели

- **`caption`** (string, обязательное)
  - Описание достопримечательности
  - Обрезается до 300 символов при обработке
  - Должно содержать релевантную информацию для сравнения

- **`candidate_type`** (string, опциональное)
  - Тип сложности кандидата для взвешивания loss
  - Возможные значения:
    - `"easy"` - легкий негативный пример (вес 1.0)
    - `"semi_hard"` - средней сложности (вес 1.2)
    - `"hard"` - сложный негативный пример (вес 1.5)
  - По умолчанию: `"easy"`
  - Применяется только к негативным примерам (когда это не target)

## Примеры

### Пример 1: Обычный запрос с правильным ответом

```json
{
  "query_image": "paris/eiffel_001.jpg",
  "candidates": [
    {
      "name": "Eiffel Tower",
      "caption": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel.",
      "candidate_type": "easy"
    },
    {
      "name": "Tokyo Tower",
      "caption": "Tokyo Tower is a communications and observation tower in the Shiba-koen district of Minato, Tokyo, Japan.",
      "candidate_type": "hard"
    },
    {
      "name": "CN Tower",
      "caption": "The CN Tower is a 553.3 m-high concrete communications and observation tower in Downtown Toronto, Ontario, Canada.",
      "candidate_type": "semi_hard"
    }
  ],
  "target_idx": 0
}
```

### Пример 2: None-of-the-above запрос

```json
{
  "query_image": "unknown/building_042.jpg",
  "candidates": [
    {
      "name": "Big Ben",
      "caption": "Big Ben is the nickname for the Great Bell of the striking clock at the north end of the Palace of Westminster in London.",
      "candidate_type": "easy"
    },
    {
      "name": "Leaning Tower of Pisa",
      "caption": "The Leaning Tower of Pisa is the freestanding bell tower of the cathedral of the Italian city of Pisa.",
      "candidate_type": "easy"
    }
  ],
  "target_idx": -1
}
```

## Рекомендации по подготовке данных

### Баланс классов

- **Valid queries** (target_idx != -1): 70-80% датасета
- **None-of-the-above queries** (target_idx == -1): 20-30% датасета

### Распределение сложности кандидатов

Для негативных примеров рекомендуется:
- **Easy**: 50-60% - визуально и семантически отличающиеся
- **Semi-hard**: 20-30% - похожие по категории или стилю
- **Hard**: 10-20% - очень похожие, требующие детального анализа

### Качество изображений

- Формат: JPEG, PNG
- Минимальное разрешение: 224x224 пикселей
- Рекомендуемое: 512x512 или выше
- Изображения должны быть валидными и читаемыми

### Качество описаний

- Описания должны быть информативными и точными
- Избегайте дублирования информации между кандидатами
- Включайте отличительные признаки достопримечательностей

## Валидация данных

Скрипт автоматически проверяет:

1. ✅ Существование файлов данных и директории с изображениями
2. ✅ Наличие обязательных полей в JSON
3. ✅ Непустой датасет
4. ⚠️ Валидность изображений (при загрузке в collate_fn)

## Обработка ошибок

При обнаружении проблем:

- **Отсутствующие файлы**: `FileNotFoundError` с указанием пути
- **Невалидная структура**: `ValueError` с описанием проблемы
- **Поврежденные изображения**: Пропускаются с предупреждением в логах

## Расширение формата

Для добавления новых полей:

1. Добавьте поле в JSON
2. Обновите `build_rerank_prompt()` для использования нового поля
3. Обновите валидацию в `RerankDataset.__init__()`
4. Обновите эту документацию

## Инструменты для работы с данными

См. скрипты в `scripts/data_preparation/`:
- `step6_setup_dataset.py` - создание финального датасета
- `download_dataset_s3.py` - загрузка данных из S3
- `upload_dataset_s3.py` - выгрузка данных в S3
