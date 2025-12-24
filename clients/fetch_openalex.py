"""
fetch_openalex.py

OpenAlex API：跨学科 works 检索与摘要重建。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import requests
from loguru import logger


def reconstruct_abstract(
    abstract_inverted_index: Optional[Dict[str, List[int]]],
) -> str:
    """
    OpenAlex 的 abstract 是 inverted index（token -> positions）。
    这里把它还原为近似原始顺序的文本。
    """
    if not abstract_inverted_index:
        return ""

    pos_to_token: Dict[int, str] = {}
    for token, positions in abstract_inverted_index.items():
        for p in positions:
            pos_to_token.setdefault(int(p), token)

    if not pos_to_token:
        return ""
    tokens = [pos_to_token[i] for i in sorted(pos_to_token.keys())]
    return " ".join(tokens).strip()


def _get_mailto_param() -> Optional[str]:
    v = (os.getenv("OPENALEX_MAILTO") or "").strip()
    return v or None


def search_works(
    *,
    query: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    per_page: int = 25,
    cursor: str = "*",
    timeout_s: float = 20.0,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    OpenAlex works 搜索：
    GET https://api.openalex.org/works?search=...&filter=from_publication_date:...,to_publication_date:...&per_page=...&cursor=...

    Returns: (works, next_cursor)
    """
    if not query:
        return [], None

    base = "https://api.openalex.org/works"
    params: Dict[str, Any] = {
        "search": query,
        "per_page": max(1, min(int(per_page), 200)),
        "cursor": cursor or "*",
    }
    mailto = _get_mailto_param()
    if mailto:
        params["mailto"] = mailto

    filters: List[str] = []
    if from_date:
        filters.append(f"from_publication_date:{from_date}")
    if to_date:
        filters.append(f"to_publication_date:{to_date}")
    if filters:
        params["filter"] = ",".join(filters)

    try:
        resp = requests.get(base, params=params, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        works = data.get("results") or []
        next_cursor = (data.get("meta") or {}).get("next_cursor")
        return works, next_cursor
    except Exception as e:
        logger.error(f"OpenAlex 搜索失败: {e}")
        return [], None


