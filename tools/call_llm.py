"""兼容层：保留 tools.call_llm 的导入路径，实际实现迁移至 clients.call_llm。"""

from clients.call_llm import LLMClient, get_client, translate_text  # noqa: F401

__all__ = ["LLMClient", "get_client", "translate_text"]
