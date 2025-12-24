"""兼容层：保留 tools.report2md 的导入路径，实际实现迁移至 renderers.render_markdown。"""

from renderers.render_markdown import papers_to_markdown  # noqa: F401

__all__ = ["papers_to_markdown"]
