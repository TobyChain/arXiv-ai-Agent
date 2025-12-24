"""兼容层：保留 tools.call_jina 的导入路径，实际实现迁移至 clients.fetch_jina。"""

from clients.fetch_jina import JinaReaderClient, fetch_arxiv_papers, get_client  # noqa: F401

__all__ = ["JinaReaderClient", "get_client", "fetch_arxiv_papers"]
