"""md_report.py

把 arxiv_daily 处理结果渲染为简洁 Markdown。
"""

from __future__ import annotations

from typing import Any, Dict, List


def _safe_join(items: List[str]) -> str:
    return ", ".join([i for i in items if i])


def papers_to_markdown(date_str: str, papers: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append(f"# ArXiv AI Daily Report ({date_str})")
    lines.append("")
    lines.append(f"- 论文数：{len(papers)}")
    lines.append("")
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
        compressed = (p.get("compressed") or "").strip()

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
        if compressed:
            lines.append("**中文压缩**")
            lines.append("")
            lines.append(compressed)
            lines.append("")
        if trans_abs:
            lines.append("**中文摘要**")
            lines.append("")
            lines.append(trans_abs)
            lines.append("")
        lines.append("---")

    return "\n".join(lines).rstrip() + "\n"
