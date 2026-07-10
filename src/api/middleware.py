# src/api/middleware.py
"""Middleware для AITourGuide API."""

import time
from collections import defaultdict

from fastapi import HTTPException, Request


class RateLimiter:
    """Простой rate limiter на основе in-memory хранилища."""

    def __init__(self, calls: int, period: int, enabled: bool = True):
        """
        Args:
            calls: Максимальное количество запросов
            period: Период в секундах
            enabled: Если False — лимитер пропускает всё (для нагрузочных
                тестов: один IP Locust иначе упирается в общий бакет).
        """
        self.calls = calls
        self.period = period
        self.enabled = enabled
        self.requests: dict[str, list] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Проверяет, разрешён ли запрос для данного клиента."""
        if not self.enabled:
            return True

        now = time.time()

        # Удаляем устаревшие записи
        self.requests[client_id] = [
            t for t in self.requests[client_id]
            if now - t < self.period
        ]

        if len(self.requests[client_id]) >= self.calls:
            return False

        self.requests[client_id].append(now)
        return True

    def get_retry_after(self, client_id: str) -> int:
        """Возвращает секунды до следующего разрешённого запроса."""
        if not self.requests[client_id]:
            return 0
        oldest = min(self.requests[client_id])
        return max(0, int(self.period - (time.time() - oldest)))


async def check_rate_limit(request: Request, rate_limiter: RateLimiter):
    """
    Проверяет rate limit для входящего запроса.

    Raises:
        HTTPException 429: если лимит превышен
    """
    client_id = request.client.host if request.client else "unknown"

    if not rate_limiter.is_allowed(client_id):
        retry_after = rate_limiter.get_retry_after(client_id)
        raise HTTPException(
            status_code=429,
            detail=f"Превышен лимит запросов. Повторите через {retry_after} сек.",
            headers={"Retry-After": str(retry_after)},
        )
