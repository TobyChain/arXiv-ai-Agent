"""
call_llm.py

大模型调用工具：用于调用 OpenAI 兼容 API，对摘要做结构化分析/翻译。
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

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
        调用 LLM 翻译论文摘要并进行分析，返回 JSON 字符串。
        """
        prompt = (
            f"你是一个专业的学术助手。请分析以下计算机科学（{domain}方向）论文的摘要，并提供以下信息：\n"
            f"1. 中文摘要：将摘要翻译成通顺、专业的中文。\n"
            f"2. 中文压缩：基于原文摘要内容，输出2-3句话的中文文本压缩（更短、更凝练，覆盖：研究问题/方法/结果或贡献）。\n"
            f"3. 概括词：提供3个能够概括该论文核心内容的中文关键词。\n"
            f"4. 子主题：识别该论文所属的细分领域（如：LLM, CV, NLP, RL等）。\n"
            f"5. 推荐程度：作为一名大模型算法工程师，请给出推荐程度（极度推荐、很推荐、推荐、一般推荐、不推荐）。\n"
            f"   - 特别注意：如果涉及医学LLM（Medical LLM）的进展，或来自头部互联网厂商（如Google, Meta, OpenAI, DeepMind, 百度, 腾讯, 阿里, 字节等）的研究，请标记为“极度推荐”。\n\n"
            f"原文：\n{abstract}\n\n"
            "请严格按以下 JSON 格式输出，不要包含任何额外的解释或开场白：\n"
            "{\n"
            '  "trans_abs": "中文摘要内容",\n'
            '  "compressed": "2-3句话的中文文本压缩",\n'
            '  "keywords": ["关键词1", "关键词2", "关键词3"],\n'
            '  "sub_topic": "子主题",\n'
            '  "recommendation": "推荐程度"\n'
            "}"
        )

        try:
            # 尽可能启用 JSON 模式强制输出可解析 JSON；不支持则回退。
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
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                logger.warning(f"JSON 模式不可用，回退到普通模式：{e}")
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
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
            return content
        except Exception as e:
            logger.error(f"分析失败: {e}")
            return '{"trans_abs": "翻译服务暂时不可用。", "compressed": "暂时不可用。", "keywords": ["未知", "未知", "未知"], "sub_topic": "未知", "recommendation": "一般推荐"}'

    def chat(self, messages: list, temperature: float = 0.7) -> str:
        """通用聊天接口"""
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


