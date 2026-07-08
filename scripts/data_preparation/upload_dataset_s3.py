import os
import time
import json
import boto3
from botocore.exceptions import ClientError, BotoCoreError
from botocore.config import Config
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# === Настройки ===
LOCAL_FOLDER = "./images"                     # Путь к папке с изображениями
BUCKET_NAME = "ai-tour-guide"              # Имя вашего бакета
S3_PREFIX = "landmarks/"                      # Опционально: папка в бакете (оставьте "" если не нужно)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
PROGRESS_FILE = "upload_progress.json"        # Файл для сохранения прогресса
MAX_RETRIES = 5                               # Максимальное количество попыток
RETRY_DELAY = 2                               # Задержка между попытками (секунды)

# Конфигурация с увеличенными таймаутами и повторными попытками
boto_config = Config(
    retries={'max_attempts': 3, 'mode': 'adaptive'},
    connect_timeout=60,
    read_timeout=60,
    max_pool_connections=50
)

def get_s3_client():
    """Создает новый S3 клиент с настройками"""
    session = boto3.session.Session()
    return session.client(
        service_name='s3',
        endpoint_url='https://storage.yandexcloud.net',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=boto_config
    )

def check_s3_files(s3_client, bucket, prefix):
    """Проверяет какие файлы уже существуют в S3"""
    print("Проверяю существующие файлы в S3...")
    existing_files = set()
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        for page in tqdm(pages, desc="Сканирование S3"):
            if 'Contents' in page:
                for obj in page['Contents']:
                    # Извлекаем имя файла без префикса
                    filename = obj['Key'][len(prefix):] if obj['Key'].startswith(prefix) else obj['Key']
                    existing_files.add(filename)
    except Exception as e:
        print(f"Ошибка при проверке S3: {e}")
        print("Продолжаю без проверки существующих файлов...")
    
    return existing_files

def load_progress():
    """Загружает список уже загруженных файлов"""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return set(json.load(f))
        except Exception as e:
            print(f"Ошибка при загрузке прогресса: {e}")
            return set()
    return set()

def save_progress(uploaded_files):
    """Сохраняет список загруженных файлов"""
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(list(uploaded_files), f)
    except Exception as e:
        print(f"Ошибка при сохранении прогресса: {e}")

def upload_file_with_retry(s3_client, file_path, bucket, object_name, max_retries=MAX_RETRIES):
    """Загружает файл в Yandex Object Storage с повторными попытками"""
    for attempt in range(max_retries):
        try:
            s3_client.upload_file(file_path, bucket, object_name)
            return True
        except (ClientError, BotoCoreError, Exception) as e:
            if attempt < max_retries - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)  # Экспоненциальная задержка
                print(f"\nОшибка при загрузке {file_path} (попытка {attempt + 1}/{max_retries}): {e}")
                print(f"Повторная попытка через {wait_time} секунд...")
                time.sleep(wait_time)
                # Пересоздаем клиент на случай проблем с соединением
                try:
                    s3_client = get_s3_client()
                except Exception as client_error:
                    print(f"Ошибка при пересоздании клиента: {client_error}")
            else:
                print(f"\nНе удалось загрузить {file_path} после {max_retries} попыток: {e}")
                return False
    return False

def main():
    if not os.path.isdir(LOCAL_FOLDER):
        raise ValueError(f"Папка {LOCAL_FOLDER} не существует")

    # Получаем список всех файлов
    all_files = [
        f for f in os.listdir(LOCAL_FOLDER)
        if os.path.isfile(os.path.join(LOCAL_FOLDER, f))
    ]

    if not all_files:
        print("Нет файлов для загрузки.")
        return

    # Загружаем прогресс
    uploaded_files = load_progress()
    
    # Если нет локального прогресса, проверяем S3
    if not uploaded_files:
        print("\nЛокальный файл прогресса не найден.")
        response = input("Проверить существующие файлы в S3? (y/n): ").strip().lower()
        if response == 'y':
            s3_client = get_s3_client()
            uploaded_files = check_s3_files(s3_client, BUCKET_NAME, S3_PREFIX)
            if uploaded_files:
                save_progress(uploaded_files)
                print(f"Найдено {len(uploaded_files)} файлов в S3. Прогресс сохранен.")
    
    # Фильтруем уже загруженные файлы
    files_to_upload = [f for f in all_files if f not in uploaded_files]
    
    print(f"\nВсего файлов: {len(all_files)}")
    print(f"Уже загружено: {len(uploaded_files)}")
    print(f"Осталось загрузить: {len(files_to_upload)}")
    
    if not files_to_upload:
        print("Все файлы уже загружены!")
        return

    # Создаем S3 клиент
    s3_client = get_s3_client()
    
    success_count = 0
    failed_files = []
    
    # Используем tqdm с начальным значением
    with tqdm(total=len(all_files), initial=len(uploaded_files), desc="Загрузка в S3") as pbar:
        for filename in files_to_upload:
            local_path = os.path.join(LOCAL_FOLDER, filename)
            s3_key = f"{S3_PREFIX}{filename}" if S3_PREFIX else filename

            if upload_file_with_retry(s3_client, local_path, BUCKET_NAME, s3_key):
                success_count += 1
                uploaded_files.add(filename)
                
                # Сохраняем прогресс каждые 100 файлов
                if success_count % 100 == 0:
                    save_progress(uploaded_files)
            else:
                failed_files.append(filename)
            
            pbar.update(1)
    
    # Сохраняем финальный прогресс
    save_progress(uploaded_files)
    
    print(f"\n{'='*60}")
    print(f"Загружено в этой сессии: {success_count} из {len(files_to_upload)} файлов")
    print(f"Всего загружено: {len(uploaded_files)} из {len(all_files)} файлов")
    
    if failed_files:
        print(f"\nНе удалось загрузить {len(failed_files)} файлов:")
        for f in failed_files[:10]:  # Показываем первые 10
            print(f"  - {f}")
        if len(failed_files) > 10:
            print(f"  ... и еще {len(failed_files) - 10} файлов")
        print("\nЗапустите скрипт снова, чтобы повторить попытку загрузки неудачных файлов.")
    else:
        print("\nВсе файлы успешно загружены!")
        # Удаляем файл прогресса после успешной загрузки
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)

if __name__ == "__main__":
    main()