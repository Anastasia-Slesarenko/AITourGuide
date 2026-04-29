"""
Сервис для перевода текста через Yandex Translate API.
"""

import os
import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class YandexTranslator:
    """Переводчик через Yandex Translate API."""
    
    def __init__(
        self,
        yc_folder_id: Optional[str] = None,
        yc_api_key: Optional[str] = None
    ):
        """
        Args:
            yc_folder_id: Yandex Cloud Folder ID
            yc_api_key: Yandex Cloud API Key
        """
        self.yc_folder_id = yc_folder_id or os.getenv('YC_FOLDER_ID')
        self.yc_api_key = yc_api_key or os.getenv('YC_API_KEY')
        self.api_url = "https://translate.api.cloud.yandex.net/translate/v2/translate"
    
    def translate(
        self,
        text: str,
        target_language: str = "ru",
        source_language: str = "en"
    ) -> Optional[str]:
        """
        Переводит текст.
        
        Args:
            text: Текст для перевода
            target_language: Целевой язык (по умолчанию русский)
            source_language: Исходный язык (по умолчанию английский)
        
        Returns:
            Переведенный текст или None при ошибке
        """
        if not self.yc_folder_id or not self.yc_api_key:
            logger.warning("Yandex Translate API keys not configured")
            return None
        
        if not text or not text.strip():
            return text
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Api-Key {self.yc_api_key}"
            }
            
            data = {
                "folderId": self.yc_folder_id,
                "texts": [text],
                "targetLanguageCode": target_language,
                "sourceLanguageCode": source_language
            }
            
            response = requests.post(
                self.api_url,
                json=data,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                translations = result.get("translations", [])
                if translations:
                    translated_text = translations[0].get("text", "")
                    logger.info(
                        f"Translated {len(text)} chars: "
                        f"{text[:50]}... -> {translated_text[:50]}..."
                    )
                    return translated_text
            else:
                logger.error(
                    f"Translation API error: {response.status_code} "
                    f"{response.text}"
                )
                return None
                
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return None
    
    def translate_dict(
        self,
        data: dict,
        keys_to_translate: list = None
    ) -> dict:
        """
        Переводит указанные ключи в словаре.
        
        Args:
            data: Словарь с данными
            keys_to_translate: Список ключей для перевода
        
        Returns:
            Словарь с переведенными значениями
        """
        if keys_to_translate is None:
            keys_to_translate = ["name", "description"]
        
        result = data.copy()
        
        for key in keys_to_translate:
            if key in result and result[key]:
                translated = self.translate(result[key])
                if translated:
                    result[key] = translated
        
        return result
