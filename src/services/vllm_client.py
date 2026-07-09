# src/services/vllm_client.py
"""HTTP-клиент для vLLM сервера (OpenAI-совместимый API)."""

import asyncio
import logging
from typing import Any, cast

import httpx

logger = logging.getLogger(__name__)


class VLLMClient:
    """HTTP-клиент для vLLM сервера (совместимый с OpenAI API)."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )
        # Отдельный клиент с маленьким пулом для health-check, чтобы инференс
        # под нагрузкой не блокировал liveness-проверки.
        self.health_client = httpx.AsyncClient(
            timeout=httpx.Timeout(5.0),
            limits=httpx.Limits(max_keepalive_connections=1, max_connections=2),
        )
        logger.info(f"vLLM клиент инициализирован: {base_url}")

    async def health_check(self) -> bool:
        """Проверяет доступность vLLM сервера."""
        try:
            # vLLM healthcheck на /health (без /v1 префикса)
            base = self.base_url.rstrip("/")
            if base.endswith("/v1"):
                base = base[:-3]
            response = await self.health_client.get(f"{base}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"vLLM health-check не прошёл: {e}")
            return False

    async def chat_completion(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
    ) -> dict:
        """
        Отправляет запрос к vLLM серверу через OpenAI API.

        Args:
            messages: Список сообщений в формате OpenAI
            max_tokens: Максимум новых токенов
            temperature: Температура генерации
            logprobs: Возвращать ли logprobs
            top_logprobs: Количество top logprobs

        Returns:
            Ответ от vLLM сервера
        """
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        if logprobs:
            payload["logprobs"] = True
            if top_logprobs:
                payload["top_logprobs"] = top_logprobs

        last_exception: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = await self.client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                return cast(dict, response.json())

            except httpx.HTTPStatusError as e:
                last_exception = e
                logger.warning(
                    f"vLLM HTTP ошибка "
                    f"(попытка {attempt + 1}/{self.max_retries}): "
                    f"{e.response.status_code}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)

            except httpx.RequestError as e:
                last_exception = e
                logger.warning(
                    f"vLLM ошибка запроса "
                    f"(попытка {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)

        raise RuntimeError(
            f"vLLM запрос не выполнен после {self.max_retries} попыток: "
            f"{last_exception}"
        )

    async def close(self):
        """Закрывает HTTP-клиенты."""
        await self.client.aclose()
        await self.health_client.aclose()
