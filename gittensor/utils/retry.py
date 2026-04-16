# The MIT License (MIT)
# Copyright © 2025 Entrius

"""Generic retry helper for GitHub API calls with exponential backoff."""

import time
from typing import Callable, Optional, TypeVar

import bittensor as bt
import requests

T = TypeVar('T')


def retry_request(
    fn: Callable[[], requests.Response],
    *,
    max_attempts: int = 3,
    backoff_base: float = 5.0,
    backoff_cap: float = 30.0,
    label: str = 'request',
) -> Optional[requests.Response]:
    """Execute an HTTP request with exponential backoff on failure.

    Args:
        fn: Callable that returns a requests.Response.
        max_attempts: Number of total attempts.
        backoff_base: Base seconds for backoff (base * 2^attempt).
        backoff_cap: Maximum backoff seconds.
        label: Label for log messages.

    Returns:
        The Response on the first successful (200) call, or None after exhausting attempts.
    """
    for attempt in range(max_attempts):
        try:
            response = fn()
            if response.status_code == 200:
                return response

            if attempt < max_attempts - 1:
                delay = min(backoff_base * (2 ** attempt), backoff_cap)
                bt.logging.warning(
                    f'{label} failed with status {response.status_code} '
                    f'(attempt {attempt + 1}/{max_attempts}), retrying in {delay:.0f}s...'
                )
                time.sleep(delay)
            else:
                bt.logging.error(
                    f'{label} failed with status {response.status_code} '
                    f'after {max_attempts} attempts'
                )

        except requests.exceptions.RequestException as e:
            if attempt < max_attempts - 1:
                delay = min(backoff_base * (2 ** attempt), backoff_cap)
                bt.logging.warning(
                    f'{label} error (attempt {attempt + 1}/{max_attempts}): {e}, '
                    f'retrying in {delay:.0f}s...'
                )
                time.sleep(delay)
            else:
                bt.logging.error(f'{label} failed after {max_attempts} attempts: {e}')

    return None


def paginated_github_get(
    url: str,
    headers: dict,
    *,
    max_attempts: int = 3,
    backoff_base: float = 5.0,
    backoff_cap: float = 30.0,
    per_page: int = 100,
    min_page_size: int = 10,
    reduce_on_5xx: bool = True,
    timeout: int = 15,
    label: str = 'paginated request',
) -> Optional[list]:
    """Fetch all pages from a GitHub REST list endpoint with retry.

    Handles pagination (per_page), exponential backoff, and optional
    page-size reduction on 5xx errors.

    Returns the full collected list, or None on total failure.
    """
    all_items: list = []
    page = 1
    attempt = 0
    current_page_size = per_page

    while attempt < max_attempts:
        try:
            response = requests.get(
                url,
                headers=headers,
                params={'per_page': current_page_size, 'page': page},
                timeout=timeout,
            )

            if response.status_code == 200:
                items = response.json()
                all_items.extend(items)
                if len(items) < current_page_size:
                    return all_items
                page += 1
                continue

            # Failure — prepare retry
            if reduce_on_5xx and response.status_code in (502, 503, 504):
                current_page_size = max(current_page_size // 2, min_page_size)

            all_items = []
            page = 1
            attempt += 1

            if attempt < max_attempts:
                delay = min(backoff_base * (2 ** (attempt - 1)), backoff_cap)
                bt.logging.warning(
                    f'{label} failed with status {response.status_code} '
                    f'(attempt {attempt}/{max_attempts}), per_page={current_page_size}, '
                    f'retrying in {delay:.0f}s...'
                )
                time.sleep(delay)

        except requests.exceptions.RequestException as e:
            all_items = []
            page = 1
            attempt += 1

            if attempt < max_attempts:
                delay = min(backoff_base * (2 ** (attempt - 1)), backoff_cap)
                bt.logging.warning(
                    f'{label} error (attempt {attempt}/{max_attempts}): {e}, '
                    f'retrying in {delay:.0f}s...'
                )
                time.sleep(delay)

    bt.logging.error(f'{label} failed after {max_attempts} attempts')
    return None
