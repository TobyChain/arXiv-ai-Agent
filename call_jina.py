"""
Jina Reader API 调用工具
用于从 Jina 获取 ArXiv 论文列表等内容
"""

import os
import re
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class JinaReaderClient:
    """Jina Reader API 客户端"""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        self.base_url = "https://r.jina.ai"

    def fetch_arxiv_list(
        self, subject: str = "cs.AI", skip: int = 0, show: int = 250
    ) -> Optional[Dict]:
        """
        获取 ArXiv 指定主题的最新论文列表

        Args:
            subject: ArXiv 主题代码（如 "cs.AI", "cs.CV" 等）
            skip: 跳过的论文数量
            show: 显示的论文数量

        Returns:
            Jina Reader 返回的 JSON 数据，如果失败则返回 None
        """
        url = f"{self.base_url}/https://arxiv.org/list/{subject}/recent?skip={skip}&show={show}"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        logger.info(f"正在请求 Jina Reader API: {url}")
        try:
            response = requests.post(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Jina API 请求失败: {e}")
            return None

    def fetch_url(self, target_url: str) -> Optional[Dict]:
        """
        使用 Jina Reader 获取任意 URL 的内容

        Args:
            target_url: 目标 URL

        Returns:
            Jina Reader 返回的 JSON 数据
        """
        url = f"{self.base_url}/{target_url}"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            response = requests.post(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Jina API 请求失败: {e}")
            return None

    def parse_arxiv_ids(self, jina_response: Dict, target_date: str) -> List[str]:
        """
        从 Jina Reader 的响应中提取指定日期的 ArXiv ID

        Args:
            jina_response: Jina Reader 返回的 JSON 数据
            target_date: 目标日期字符串，格式如 "Wed, 17 Dec 2025"

        Returns:
            ArXiv ID 列表
        """
        content = jina_response.get("data", {}).get("content", "")
        if not content:
            logger.error("错误：JSON 中未找到 content 内容")
            return []

        logger.info(f"正在寻找日期: {target_date}")

        # 寻找指定日期的内容块
        pattern = rf"### {re.escape(target_date)}(.*?)(?=###|$)"
        match = re.search(pattern, content, re.DOTALL)

        if not match:
            logger.warning(f"未找到日期 {target_date} 的内容，尝试寻找最近日期...")
            # Fallback: 使用第一个可用日期的内容
            fallback_pattern = r"### \w+, \d+ \w+ \d+(.*?)(?=###|$)"
            match = re.search(fallback_pattern, content, re.DOTALL)
            if not match:
                logger.error("未找到任何日期内容")
                return []
            logger.info("已回退到最近的可用日期内容")

        section_text = match.group(1)

        # 提取 arXiv ID (格式如 2512.13510)
        id_pattern = r"arxiv\.org/abs/(\d+\.\d+)"
        ids = re.findall(id_pattern, section_text)

        # 去重并保持顺序
        unique_ids = list(dict.fromkeys(ids))
        logger.success(f"共提取到 {len(unique_ids)} 篇文献。")
        return unique_ids


# 单例模式
_default_client = None


def get_client() -> JinaReaderClient:
    """获取默认 Jina Reader 客户端"""
    global _default_client
    if _default_client is None:
        _default_client = JinaReaderClient()
    return _default_client


def fetch_arxiv_papers(
    target_date: str, subject: str = "cs.AI", max_papers: int = 250
) -> List[str]:
    """
    快捷函数：获取指定日期和主题的 ArXiv 论文 ID 列表

    Args:
        target_date: 目标日期字符串
        subject: ArXiv 主题代码（如 "cs.AI"）
        max_papers: 最多获取的论文数量

    Returns:
        ArXiv ID 列表
    """
    client = get_client()
    data = client.fetch_arxiv_list(subject=subject, skip=0, show=max_papers)
    if data:
        return client.parse_arxiv_ids(data, target_date)
    return []
