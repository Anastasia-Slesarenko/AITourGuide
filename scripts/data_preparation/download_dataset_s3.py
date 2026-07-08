#!/usr/bin/env python3
"""
Скачивание папки из Яндекс Облако S3 с прогресс-баром (tqdm)
"""

import os
import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError
from pathlib import Path
from typing import Optional
from tqdm import tqdm  # pip install tqdm


def create_s3_client(
    access_key: str,
    secret_key: str,
    endpoint_url: str = "https://storage.yandexcloud.net"
) -> BaseClient:
    """Создание S3-клиента для Яндекс Облако"""
    print(f"Создание S3 клиента...")
    print(f"  Endpoint: {endpoint_url}")
    print(f"  Region: ru-central1")
    
    client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name='ru-central1',
        config=boto3.session.Config(
            connect_timeout=10,
            read_timeout=30,
            retries={'max_attempts': 3}
        )
    )
    
    print(f"S3 клиент создан")
    return client


def get_folder_objects(
    s3_client: BaseClient,
    bucket_name: str,
    prefix: str
) -> list[dict]:
    """
    Получение списка всех объектов в папке S3 с обработкой пагинации
    """
    objects = []
    continuation_token = None
    page_count = 0
    
    try:
        while True:
            page_count += 1
            params = {'Bucket': bucket_name, 'Prefix': prefix}
            if continuation_token:
                params['ContinuationToken'] = continuation_token
            
            print(f"  Запрос страницы {page_count}...")
            response = s3_client.list_objects_v2(**params)
            
            if 'Contents' in response:
                # Фильтруем "папки" (ключи, заканчивающиеся на /)
                page_objects = [
                    obj for obj in response['Contents']
                    if not obj['Key'].endswith('/')
                ]
                objects.extend(page_objects)
                print(f"  Найдено объектов на странице: {len(page_objects)}")
            else:
                print(f"  Страница {page_count} не содержит объектов")
            
            if not response.get('IsTruncated'):
                break
            continuation_token = response.get('NextContinuationToken')
        
        return objects
    
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_msg = e.response.get('Error', {}).get('Message', str(e))
        print(f"\nОшибка S3 ({error_code}): {error_msg}")
        raise
    except Exception as e:
        print(f"\nНеожиданная ошибка: {type(e).__name__}: {e}")
        raise


def download_folder(
    s3_client: BaseClient,
    bucket_name: str,
    s3_folder: str,
    local_dir: str,
    show_progress: bool = True,
    skip_existing: bool = True,
    verify_size: bool = True
) -> list[str]:
    """
    Рекурсивное скачивание папки из S3 с прогресс-баром
    
    Args:
        s3_client: boto3 S3 client
        bucket_name: имя бакета
        s3_folder: путь к папке в S3 (например, 'data/images/')
        local_dir: локальная директория для сохранения
        show_progress: показывать ли прогресс-бары
        skip_existing: пропускать уже скачанные файлы
        verify_size: проверять размер существующих файлов
    
    Returns:
        Список скачанных файлов
    """
    # Нормализуем префикс
    prefix = s3_folder if s3_folder.endswith('/') else f"{s3_folder}/"
    if prefix.startswith('/'):
        prefix = prefix[1:]
    
    # Создаем локальную директорию
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    # Проверяем доступ к бакету
    print(f"Проверка доступа к бакету '{bucket_name}'...")
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"Бакет доступен")
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == '404':
            print(f"Бакет '{bucket_name}' не найден")
        elif error_code == '403':
            print(f"Нет доступа к бакету '{bucket_name}'")
        else:
            print(f"Ошибка доступа к бакету: {error_code}")
        raise
    
    # Получаем список объектов
    print(f"Сканирование s3://{bucket_name}/{prefix}...")
    objects = get_folder_objects(s3_client, bucket_name, prefix)
    
    if not objects:
        print("Объекты не найдены")
        return []
    
    print(f"Найдено объектов: {len(objects)}")
    
    downloaded_files = []
    skipped_files = []
    
    # Общий прогресс-бар по файлам
    with tqdm(
        total=len(objects),
        desc="Файлы",
        unit="файл",
        disable=not show_progress
    ) as pbar_files:
        
        for obj in objects:
            key = obj['Key']
            file_size = obj['Size']
            
            # Вычисляем относительный путь
            relative_path = key[len(prefix):] if key.startswith(prefix) else key
            local_path = os.path.join(local_dir, relative_path)
            
            # Создаем поддиректории
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Проверяем, существует ли файл
            if skip_existing and os.path.exists(local_path):
                local_size = os.path.getsize(local_path)
                
                # Проверяем размер, если требуется
                if verify_size:
                    if local_size == file_size:
                        skipped_files.append(local_path)
                        pbar_files.set_postfix_str(
                            f"пропущен: {Path(relative_path).name[:20]}"
                        )
                        pbar_files.update(1)
                        continue
                    else:
                        print(f"\nРазмер не совпадает: {relative_path}")
                        print(f"   Локальный: {local_size}, S3: {file_size}")
                        print(f"   Перезагрузка...")
                else:
                    # Пропускаем без проверки размера
                    skipped_files.append(local_path)
                    pbar_files.set_postfix_str(
                        f"пропущен: {Path(relative_path).name[:20]}"
                    )
                    pbar_files.update(1)
                    continue
            
            try:
                # Прогресс-бар для текущего файла (если файл > 1 МБ)
                if show_progress and file_size > 1024 * 1024:  # >1MB
                    with tqdm(
                        total=file_size,
                        desc=f"{Path(relative_path).name}",
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        leave=False
                    ) as t:
                        s3_client.download_file(
                            bucket_name,
                            key,
                            local_path,
                            Callback=lambda bytes_amount: t.update(bytes_amount)
                        )
                else:
                    # Для мелких файлов — без вложенного прогресса
                    s3_client.download_file(bucket_name, key, local_path)
                
                downloaded_files.append(local_path)
                pbar_files.set_postfix_str(f"последний: {Path(relative_path).name[:20]}")
                pbar_files.update(1)
                
            except ClientError as e:
                print(f"\nОшибка при скачивании {key}: {e}")
                continue
    
    print(f"\nГотово!")
    print(f"   Скачано файлов: {len(downloaded_files)}")
    print(f"   ⏭Пропущено файлов: {len(skipped_files)}")
    print(f"   Всего обработано: {len(downloaded_files) + len(skipped_files)}")
    
    return downloaded_files


# ==================== Пример использования ====================
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()  # Загрузка переменных из .env
    

    ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
    SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    BUCKET_NAME = "ai-tour-guide"         
    S3_FOLDER = "landmarks/"
    LOCAL_DIR = "/Users/anastasiya/Documents/AITourGuide/images"
    
    if not all([ACCESS_KEY, SECRET_KEY]):
        raise ValueError("Укажите YC_ACCESS_KEY и YC_SECRET_KEY в переменных окружения или .env файле")
    
    # Создаём клиент
    s3_client = create_s3_client(ACCESS_KEY, SECRET_KEY)
    
    # Скачиваем папку с прогресс-баром
    downloaded = download_folder(
        s3_client=s3_client,
        bucket_name=BUCKET_NAME,
        s3_folder=S3_FOLDER,
        local_dir=LOCAL_DIR,
        show_progress=True,      # Прогресс-бар включён
        skip_existing=True,      # ⏭Пропускать существующие файлы
        verify_size=True         # Проверять размер файлов
    )