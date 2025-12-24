"""
fetch_crossref.py

Crossref：用 DOI 获取规范 BibTeX（用于统一引用导出）。
"""

from __future__ import annotations

import os
import re
from typing import Optional

import requests
from loguru import logger

_DOI_RE = re.compile(r"10\.\d{4,9}/\S+", re.IGNORECASE)


def normalize_doi(doi: str) -> Optional[str]:
    if not doi:
        return None
    doi = doi.strip()
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    doi = doi.replace("doi:", "").strip()
    m = _DOI_RE.search(doi)
    return m.group(0) if m else None


def fetch_bibtex_by_doi(doi: str, *, timeout_s: float = 15.0) -> Optional[str]:
    """
    通过 Crossref transform 获取 BibTeX：
    GET https://api.crossref.org/works/{doi}/transform/application/x-bibtex
    """
    norm = normalize_doi(doi)
    if not norm:
        return None

    url = f"https://api.crossref.org/works/{norm}/transform/application/x-bibtex"
    headers = {
        "Accept": "application/x-bibtex",
        # Crossref 建议带上 mailto 以便联系（可选）
        "User-Agent": f"unipaper/0.1 (mailto:{os.getenv('CROSSREF_MAILTO','')})".strip(),
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout_s)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        text = (resp.text or "").strip()
        return text or None
    except Exception as e:
        logger.warning(f"Crossref BibTeX 获取失败 doi={norm}: {e}")
        return None


