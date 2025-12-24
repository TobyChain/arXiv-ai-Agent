"""
render_markdown.py

把论文列表渲染为简洁 Markdown（展示层）。
"""

from __future__ import annotations

from typing import Any, Dict, List


def _safe_join(items: List[str]) -> str:
    return ", ".join([i for i in items if i])


def papers_to_markdown(date_str: str, papers: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    # 直接以分割线开始，不保留顶部总标题和论文数
    lines.append("---")

    for idx, p in enumerate(papers, start=1):
        title = (p.get("title") or "").strip()
        authors = p.get("authors") or []
        abs_url = p.get("abs_url") or ""
        pdf_url = p.get("pdf_url") or ""
        sub_topic = p.get("sub_topic") or "未知"
        recommendation = p.get("recommendation") or "一般推荐"
        keywords = p.get("keywords") or []
        trans_abs = (p.get("trans_abs") or "").strip()

        lines.append(f"## {idx}. {title}")
        lines.append("")
        if authors:
            lines.append(f"- 作者：{_safe_join(authors)}")
        lines.append(f"- 子主题：{sub_topic}")
        lines.append(f"- 推荐：{recommendation}")
        if keywords:
            lines.append(f"- 关键词：{_safe_join(list(map(str, keywords)))}")
        if abs_url:
            lines.append(f"- Abstract：{abs_url}")
        if pdf_url:
            lines.append(f"- PDF：{pdf_url}")

        lines.append("")
        if trans_abs:
            lines.append("**中文摘要**")
            lines.append("")
            lines.append(trans_abs)
            lines.append("")
        lines.append("---")

    return "\n".join(lines).rstrip() + "\n"


