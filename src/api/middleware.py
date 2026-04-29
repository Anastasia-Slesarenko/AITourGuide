# src/api/middleware.py
"""
Middleware for AITourGuide API.
"""

import time
from collections import defaultdict
from typing import Dict
from fastapi import HTTPException, Request


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, calls: int, period: int):
        """
        Initialize rate limiter.
        
        Args:
            calls: Maximum number of calls allowed
            period: Time period in seconds
        """
        self.calls = calls
        self.period = period
        self.requests: Dict[str, list] = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed for client.
        
        Args:
            client_id: Client identifier
        
        Returns:
            True if request is allowed, False otherwise
        """
        now = time.time()
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < self.period
        ]
        
        if len(self.requests[client_id]) >= self.calls:
            return False
        
        self.requests[client_id].append(now)
        return True
    
    def get_retry_after(self, client_id: str) -> int:
        """
        Get seconds until next request is allowed.
        
        Args:
            client_id: Client identifier
        
        Returns:
            Seconds until next request is allowed
        """
        if not self.requests[client_id]:
            return 0
        oldest = min(self.requests[client_id])
        return max(0, int(self.period - (time.time() - oldest)))


async def check_rate_limit(request: Request, rate_limiter: RateLimiter):
    """
    Dependency for checking rate limit.
    
    Args:
        request: FastAPI request object
        rate_limiter: RateLimiter instance
    
    Raises:
        HTTPException: If rate limit is exceeded
    """
    client_id = request.client.host if request.client else "unknown"
    
    if not rate_limiter.is_allowed(client_id):
        retry_after = rate_limiter.get_retry_after(client_id)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)}
        )
