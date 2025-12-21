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
        调用 LLM 翻译论文摘要并进行分析

        Args:
            abstract: 原文摘要
            domain: 论文领域，默认为 AI

        Returns:
            JSON 格式的分析结果字符串
        """
        prompt = (
            f"你是一个专业的学术助手。请分析以下计算机科学（{domain}方向）论文的摘要，并提供以下信息：\n"
            f"1. 中文摘要：将摘要翻译成通顺、专业的中文。\n"
            f"2. 概括词：提供3个能够概括该论文核心内容的中文关键词。\n"
            f"3. 子主题：识别该论文所属的细分领域（如：LLM, CV, NLP, RL等）。\n"
            f"4. 推荐程度：作为一名大模型算法工程师，请给出推荐程度（极度推荐、很推荐、推荐、一般推荐、不推荐）。\n"
            f"   - 特别注意：如果涉及医学LLM（Medical LLM）的进展，或来自头部互联网厂商（如Google, Meta, OpenAI, DeepMind, 百度, 腾讯, 阿里, 字节等）的研究，请标记为“极度推荐”。\n\n"
            f"原文：\n{abstract}\n\n"
            "请严格按以下 JSON 格式输出，不要包含任何额外的解释或开场白：\n"
            "{\n"
            '  "trans_abs": "中文摘要内容",\n'
            '  "keywords": ["关键词1", "关键词2", "关键词3"],\n'
            '  "sub_topic": "子主题",\n'
            '  "recommendation": "推荐程度"\n'
            "}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful academic assistant that outputs JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            content = response.choices[0].message.content.strip()
            # 移除可能存在的 Markdown 代码块标记
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
            return content
        except Exception as e:
            logger.error(f"分析失败: {e}")
            return '{"trans_abs": "翻译服务暂时不可用。", "keywords": ["未知", "未知", "未知"], "sub_topic": "未知", "recommendation": "一般推荐"}'

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
