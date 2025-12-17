"""
大模型调用工具
用于调用 LLM API 进行文本翻译等任务
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class LLMClient:
    """LLM 客户端封装"""

    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv("BASE_URL")
        self.model = model or os.getenv("MODEL_NAME")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def translate_abstract(self, abstract: str, domain: str = "AI") -> str:
        """
        调用 LLM 翻译论文摘要

        Args:
            abstract: 原文摘要
            domain: 论文领域，默认为 AI

        Returns:
            中文翻译文本
        """
        prompt = (
            f"你是一个专业的学术翻译助手。请将以下计算机科学（{domain}方向）论文的摘要翻译成通顺、专业的中文。"
            "直接输出翻译内容，不要包含任何额外的解释或开场白。\n\n"
            f"原文：\n{abstract}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful academic translator.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"翻译失败: {e}")
            return "翻译服务暂时不可用。"

    def chat(self, messages: list, temperature: float = 0.7) -> str:
        """
        通用聊天接口

        Args:
            messages: 消息列表，格式 [{"role": "user", "content": "..."}]
            temperature: 温度参数

        Returns:
            回复文本
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return ""


# 单例模式，方便直接导入使用
_default_client = None


def get_client() -> LLMClient:
    """获取默认 LLM 客户端"""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def translate_text(text: str, domain: str = "AI") -> str:
    """快捷翻译函数"""
    return get_client().translate_abstract(text, domain)
