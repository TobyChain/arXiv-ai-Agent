"""兼容层：保留 tools.unpaywall 的导入路径，实际实现迁移至 clients.fetch_unpaywall。"""

from clients.fetch_unpaywall import (  # noqa: F401
    extract_best_oa_urls,
    fetch_unpaywall_record,
)

__all__ = ["fetch_unpaywall_record", "extract_best_oa_urls"]


