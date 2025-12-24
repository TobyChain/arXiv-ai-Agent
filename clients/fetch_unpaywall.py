"""
fetch_unpaywall.py

Unpaywall API：通过 DOI 补全合法 OA PDF / landing page。

需要设置环境变量：
  UNPAYWALL_EMAIL=your_email@example.com
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests
from loguru import logger

from clients.fetch_crossref import normalize_doi


def _get_email() -> Optional[str]:
    v = (os.getenv("UNPAYWALL_EMAIL") or "").strip()
    return v or None


def fetch_unpaywall_record(
    doi: str, *, timeout_s: float = 20.0
) -> Optional[Dict[str, Any]]:
    email = _get_email()
    if not email:
        return None
    norm = normalize_doi(doi)
    if not norm:
        return None
    url = f"https://api.unpaywall.org/v2/{norm}"
    try:
        resp = requests.get(url, params={"email": email}, timeout=timeout_s)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"Unpaywall 查询失败 doi={norm}: {e}")
        return None


def extract_best_oa_urls(record: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """返回 {pdf_url, landing_page_url}（尽量给出可下载的 PDF 链接）。"""
    if not record:
        return {"pdf_url": None, "landing_page_url": None}

    best = record.get("best_oa_location") or {}
    pdf_url = best.get("url_for_pdf") or best.get("url_for_landing_page")
    landing = best.get("url_for_landing_page") or record.get("doi_url")

    if not pdf_url:
        for loc in record.get("oa_locations") or []:
            pdf_url = loc.get("url_for_pdf")
            if pdf_url:
                landing = loc.get("url_for_landing_page") or landing
                break

    return {"pdf_url": pdf_url, "landing_page_url": landing}


