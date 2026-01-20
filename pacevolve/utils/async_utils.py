"""
Async utilities for OpenPACEvolve.
"""

import asyncio
from typing import Any, Callable, Optional, TypeVar


T = TypeVar("T")


async def run_with_timeout(
    coro_or_func: Callable[..., T],
    timeout: float,
    *args,
    default: Optional[T] = None,
    **kwargs
) -> Optional[T]:
    """
    Run a coroutine or function with timeout.
    
    Args:
        coro_or_func: Coroutine or sync function to run.
        timeout: Timeout in seconds.
        *args: Arguments to pass.
        default: Default value if timeout.
        **kwargs: Keyword arguments.
        
    Returns:
        Result or default if timeout.
    """
    try:
        if asyncio.iscoroutinefunction(coro_or_func):
            result = await asyncio.wait_for(
                coro_or_func(*args, **kwargs),
                timeout=timeout
            )
        else:
            # Run sync function in executor
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: coro_or_func(*args, **kwargs)),
                timeout=timeout
            )
        return result
    except asyncio.TimeoutError:
        return default


async def gather_with_concurrency(
    n: int,
    *coros,
    return_exceptions: bool = False,
) -> list:
    """
    Gather with limited concurrency.
    
    Args:
        n: Maximum concurrent tasks.
        *coros: Coroutines to run.
        return_exceptions: Whether to return exceptions.
        
    Returns:
        List of results.
    """
    semaphore = asyncio.Semaphore(n)
    
    async def sem_coro(coro):
        async with semaphore:
            return await coro
    
    return await asyncio.gather(
        *(sem_coro(c) for c in coros),
        return_exceptions=return_exceptions,
    )
