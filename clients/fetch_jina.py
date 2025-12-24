"""
fetch_jina.py

Jina Reader API：用于抓取/解析 ArXiv list 页面（或任意 URL）。
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class JinaReaderClient:
    """Jina Reader API 客户端"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        self.base_url = "https://r.jina.ai"

    def fetch_arxiv_list(
        self, subject: str = "cs.AI", skip: int = 0, show: int = 250
    ) -> Optional[Dict]:
        url = f"{self.base_url}/https://arxiv.org/list/{subject}/recent?skip={skip}&show={show}"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        logger.info(f"正在请求 Jina Reader API: {url}")
        try:
            response = requests.post(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Jina API 请求失败: {e}")
            return None

    def fetch_url(self, target_url: str) -> Optional[Dict]:
        url = f"{self.base_url}/{target_url}"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            response = requests.post(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Jina API 请求失败: {e}")
            return None

    def parse_arxiv_ids(self, jina_response: Dict, target_date: str) -> List[str]:
        content = jina_response.get("data", {}).get("content", "")
        if not content:
            logger.error("错误：JSON 中未找到 content 内容")
            return []

        logger.info(f"正在寻找日期: {target_date}")
        pattern = rf"### {re.escape(target_date)}(.*?)(?=###|$)"
        match = re.search(pattern, content, re.DOTALL)

        if not match:
            logger.warning(f"未找到日期 {target_date} 的内容，尝试寻找最近日期...")
            fallback_pattern = r"### \w+, \d+ \w+ \d+(.*?)(?=###|$)"
            match = re.search(fallback_pattern, content, re.DOTALL)
            if not match:
                logger.error("未找到任何日期内容")
                return []
            logger.info("已回退到最近的可用日期内容")

        section_text = match.group(1)
        id_pattern = r"arxiv\.org/abs/(\d+\.\d+)"
        ids = re.findall(id_pattern, section_text)

        unique_ids = list(dict.fromkeys(ids))
        logger.success(f"共提取到 {len(unique_ids)} 篇文献。")
        return unique_ids


_default_client: Optional[JinaReaderClient] = None


def get_client() -> JinaReaderClient:
    global _default_client
    if _default_client is None:
        _default_client = JinaReaderClient()
    return _default_client


def fetch_arxiv_papers(
    target_date: str, subject: str = "cs.AI", max_papers: int = 250
) -> List[str]:
    client = get_client()
    data = client.fetch_arxiv_list(subject=subject, skip=0, show=max_papers)
    if data:
        return client.parse_arxiv_ids(data, target_date)
    return []


