"""兼容层：保留 tools.openalex 的导入路径，实际实现迁移至 clients.fetch_openalex。"""

from clients.fetch_openalex import reconstruct_abstract, search_works  # noqa: F401

__all__ = ["reconstruct_abstract", "search_works"]


