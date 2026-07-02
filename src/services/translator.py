# src/services/translator.py
"""Сервис для перевода текста через Yandex Translate API."""

import logging
import os

import httpx

logger = logging.getLogger(__name__)


class YandexTranslator:
    """Переводчик через Yandex Translate API (async)."""

    _API_URL = "https://translate.api.cloud.yandex.net/translate/v2/translate"

    def __init__(
        self,
        yc_folder_id: str | None = None,
        yc_api_key: str | None = None,
    ):
        """
        Args:
            yc_folder_id: Yandex Cloud Folder ID
            yc_api_key: Yandex Cloud API Key
        """
        self.yc_folder_id = yc_folder_id or os.getenv("YC_FOLDER_ID")
        self.yc_api_key = yc_api_key or os.getenv("YC_API_KEY")

    async def translate(
        self,
        text: str,
        target_language: str = "ru",
        source_language: str = "en",
    ) -> str | None:
        """
        Переводит текст через Yandex Translate API.

        Args:
            text: Текст для перевода
            target_language: Целевой язык (по умолчанию русский)
            source_language: Исходный язык (по умолчанию английский)

        Returns:
            Переведённый текст или None при ошибке
        """
        if not self.yc_folder_id or not self.yc_api_key:
            logger.warning("Ключи Yandex Translate API не настроены")
            return None

        if not text or not text.strip():
            return text

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.yc_api_key}",
        }
        payload = {
            "folderId": self.yc_folder_id,
            "texts": [text],
            "targetLanguageCode": target_language,
            "sourceLanguageCode": source_language,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    self._API_URL, json=payload, headers=headers
                )

            if response.status_code != 200:
                logger.error(
                    f"Ошибка Translate API: {response.status_code} "
                    f"{response.text[:200]}"
                )
                return None

            translations = response.json().get("translations", [])
            if not translations:
                return None

            translated: str = translations[0].get("text", "")
            logger.info(
                f"Переведено {len(text)} символов: {text[:50]!r} → {translated[:50]!r}"
            )
            return translated

        except Exception as e:
            logger.error(f"Ошибка перевода: {e}")
            return None

    async def translate_dict(
        self,
        data: dict,
        keys_to_translate: list[str] | None = None,
    ) -> dict:
        """
        Переводит указанные ключи в словаре.

        Args:
            data: Словарь с данными
            keys_to_translate: Список ключей для перевода

        Returns:
            Словарь с переведёнными значениями
        """
        if keys_to_translate is None:
            keys_to_translate = ["name", "description"]

        result = data.copy()
        for key in keys_to_translate:
            if key in result and result[key]:
                translated = await self.translate(result[key])
                if translated:
                    result[key] = translated

        return result
