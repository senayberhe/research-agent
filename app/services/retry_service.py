import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")

async def retry_async(
        operation: Callable[[],Awaitable[T]],
        max_retries: int = 2,
        delay: float = 1.0
) -> T:
    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except Exception:
            if attempt == max_retries:
                raise
            await asyncio.sleep(
                delay * (2 ** attempt)
            )

    raise RuntimeError("Operation failed after maximum retries")