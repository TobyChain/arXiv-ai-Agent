"""兼容层：保留 tools.crossref 的导入路径，实际实现迁移至 clients.fetch_crossref。"""

from clients.fetch_crossref import fetch_bibtex_by_doi, normalize_doi  # noqa: F401

__all__ = ["normalize_doi", "fetch_bibtex_by_doi"]


